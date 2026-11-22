import os
import struct
import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from google.cloud.storage.fileio import BlobReader
import soundfile as sf

from gc_files import GCStore, AudioExtenstion

logger = logging.getLogger(__name__)

@dataclass
class Segment:
    start: pd.Timestamp  # time of first sample
    byte_loc: int  # absolute byte offset of first sample (WAV only)
    n_frames: int


@dataclass
class HarpMetadata:
    site: str
    experiment: str
    longitude: float
    latitude: float
    depth: float
    n_raw: bytearray


@dataclass
class AudioData:
    nb_channels: int
    sample_rate: int
    block_align: int = None
    bits: int = None
    scale: dict = None
    float_fmt: int = None
    bytes_per: bytearray = None
    segments: list[Segment] = None
    sound_file: Optional[sf.SoundFile] = None

@dataclass
class TimePeriod:
    start: pd.Timestamp
    end: pd.Timestamp

class AudioFile:
    """Random-access reader. For WAV/XWAV reads PCM bytes directly so that
    duty-cycled or sub-sampled reads only fetch the bytes needed from GCS."""

    def __init__(self, filename: str, store: GCStore):
        self.filename: str = filename
        self.name: str = os.path.basename(filename)
        self.store: GCStore = store
        logger.debug("%s: opening GCS stream", self.name)
        self.reader: BlobReader = store.open_file(filename=filename)
        self.kind: AudioExtenstion = GCStore.audio_file_extention(filename)
        self.timestamp: pd.Timestamp | None = GCStore.time_from_name(filename)
        if self.timestamp is None:
            logger.warning("%s: no timestamp in file name, segments start at epoch", self.name)
        self.harp = None
        self.harp_meta: Optional[HarpMetadata] = None
        try:
            self.audio_data = self.parse()
        except Exception:
            logger.exception("%s: header parsing failed", self.name)
            self.reader.close()
            raise
        self._log_summary()

    def _log_summary(self) -> None:
        ad = self.audio_data
        frames = sum(s.n_frames for s in ad.segments)
        logger.info(
            "%s: %s, %d Hz, %d ch%s, %d segment(s), %.1f min of audio, starts %s%s",
            self.name,
            self.kind.value,
            ad.sample_rate,
            ad.nb_channels,
            f", {ad.bits}-bit{' float' if ad.float_fmt else ''}" if ad.bits else "",
            len(ad.segments),
            frames / ad.sample_rate / 60,
            ad.segments[0].start if ad.segments else None,
            f", HARP site={self.harp_meta.site} depth={self.harp_meta.depth} m"
            if self.harp_meta else "",
        )

    # -- open / close
    @property
    def closed(self) -> bool:
        return self.reader is None or self.reader.closed

    def open(self) -> "AudioFile":
        """Reopen the GCS stream if it was closed. Parsed metadata is kept,
        so only the stream (and the SoundFile wrapper for non-WAV) is rebuilt."""
        if self.closed:
            logger.debug("%s: reopening GCS stream", self.name)
            self.reader = self.store.open_file(filename=self.filename)
            if self.kind != AudioExtenstion.WAVE:
                self.audio_data.sound_file = sf.SoundFile(self.reader)
        return self

    def close(self) -> None:
        if not self.closed:
            logger.debug("%s: closing GCS stream", self.name)
        if self.audio_data.sound_file is not None and not self.audio_data.sound_file.closed:
            self.audio_data.sound_file.close()
        if self.reader is not None and not self.reader.closed:
            self.reader.close()

    def __enter__(self) -> "AudioFile":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def parse(self) -> AudioData:
        match self.kind:
            case AudioExtenstion.WAVE:
                return self._parse_wav()
            case _:
                return self._parse_sf()

    def _parse_sf(self) -> AudioData:
        _sf = sf.SoundFile(self.reader)
        audio_data = AudioData(
            nb_channels=_sf.channels, sample_rate=_sf.samplerate, sound_file=_sf
        )
        audio_data.scale = {
            "PCM_16": 2**15,
            "PCM_24": 2**23,
            "PCM_32": 2**31,
        }.get(_sf.subtype, 1.0)
        start = self.timestamp or pd.Timestamp(0)
        audio_data.segments = [Segment(start, 0, _sf.frames)]
        return audio_data

    # -- RIFF parsing
    def _parse_wav(self) -> AudioData:
        data = self.reader
        head = data.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError(f"{self.filename}: not a RIFF/WAVE file")
        try:
            file_size = (
                data.size
                if hasattr(data, "size") and not callable(data.size)
                else os.fstat(data.fileno()).st_size
            )
        except Exception:
            file_size = None

        pos = 12
        fmt = None
        data_loc = data_len = None
        while True:
            data.seek(pos)
            hdr = data.read(8)
            if len(hdr) < 8:
                break
            cid, csize = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            logger.debug("%s: RIFF chunk %r at %d, %d bytes", self.name, cid, pos, csize)
            if cid == b"fmt ":
                fmt = data.read(csize)
            elif cid == b"harp":
                self.harp = data.read(csize)
            elif cid == b"data":
                data_loc = pos + 8
                data_len = csize
                if file_size and (csize == 0xFFFFFFFF or data_loc + csize > file_size):
                    data_len = file_size - data_loc
                    logger.warning(
                        "%s: data chunk size %d exceeds file, truncated to %d bytes",
                        self.name, csize, data_len,
                    )
                break
            pos += 8 + csize + (csize & 1)
        if fmt is None or data_loc is None:
            logger.error("%s: fmt=%s data=%s", self.name, fmt is not None, data_loc is not None)
            raise ValueError(f"{self.filename}: missing fmt or data chunk")

        tag, nch, fs, _, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
        audio_data = AudioData(
            nb_channels=nch, sample_rate=fs, block_align=block_align, bits=bits
        )
        if tag == 0xFFFE and len(fmt) >= 26:
            tag = struct.unpack("<H", fmt[24:26])[0]
        audio_data.float_fmt = tag == 3
        audio_data.bytes_per = audio_data.bits // 8
        segs = (
            self._parse_harp(data_loc, data_len, audio_data.block_align)
            if self.harp
            else None
        )
        if not segs:
            start = self.timestamp or pd.Timestamp(0)
            segs = [Segment(start, data_loc, data_len // audio_data.block_align)]
        audio_data.segments = segs
        return audio_data

    def _parse_harp(self, data_loc, data_len, block_align):
        """Triton XWAV 'harp' subchunk: 56-byte header + 32 bytes per raw file."""
        h = self.harp
        try:
            n_raw = struct.unpack("<H", h[36:38])[0]
            self.harp_meta = HarpMetadata(
                site=h[15:19].decode("ascii", "ignore").strip("\x00 "),
                experiment=h[19:27].decode("ascii", "ignore").strip("\x00 "),
                longitude=struct.unpack("<i", h[38:42])[0] / 1e5,
                latitude=struct.unpack("<i", h[42:46])[0] / 1e5,
                depth=struct.unpack("<h", h[46:48])[0],
                n_raw=n_raw,
            )
            segs = []
            for i in range(n_raw):
                o = 56 + 32 * i
                yy, mo, dd, hh, mi, ss = h[o : o + 6]
                (ticks,) = struct.unpack("<H", h[o + 6 : o + 8])
                byte_loc, byte_len, _wl, sr = struct.unpack("<IIII", h[o + 8 : o + 24])
                t = pd.Timestamp(2000 + yy, mo, dd, hh, mi, ss) + pd.Timedelta(
                    milliseconds=ticks
                )
                if not (data_loc <= byte_loc <= data_loc + data_len) or sr <= 0:
                    raise ValueError("raw-file entry outside data chunk")
                segs.append(Segment(t, byte_loc, byte_len // block_align))
            logger.debug(
                "%s: HARP %s/%s lat=%.5f lon=%.5f, %d raw file(s)",
                self.name, self.harp_meta.experiment, self.harp_meta.site,
                self.harp_meta.latitude, self.harp_meta.longitude, n_raw,
            )
            return segs
        except Exception as e:  # malformed -> fall back to filename time
            logger.warning("%s: harp chunk not usable (%s); using filename time", self.name, e)
            return None

    def read(self, seg: Segment, frame_off: int, n: int) -> np.ndarray:
        """Return first channel as float64 in ADC counts (int formats)."""
        n = max(0, min(n, seg.n_frames - frame_off))
        if n == 0:
            return np.zeros(0)
        self.open()  # no-op if already open
        logger.debug("%s: read %d frames at offset %d of segment %s", self.name, n, frame_off, seg.start)
        if self.kind != AudioExtenstion.WAVE:
            self.audio_data.sound_file.seek(seg.byte_loc + frame_off)
            dtype = "float64"
            x = (
                self.audio_data.sound_file.read(n, dtype=dtype, always_2d=True)[:, 0]
                * self.audio_data.scale
            )
            return x
        self.reader.seek(seg.byte_loc + frame_off * self.audio_data.block_align)
        want = n
        raw = self.reader.read(n * self.audio_data.block_align)
        n = len(raw) // self.audio_data.block_align
        if n < want:
            logger.warning("%s: short read, got %d of %d frames", self.name, n, want)
        raw = raw[: n * self.audio_data.block_align]
        if self.audio_data.bytes_per == 2:
            x = np.frombuffer(raw, "<i2")
        elif self.audio_data.bytes_per == 4:
            x = np.frombuffer(raw, "<f4" if self.audio_data.float_fmt else "<i4")
        elif self.audio_data.bytes_per == 3:
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
            x = (
                b[:, 0].astype(np.int32)
                | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int8).astype(np.int32) << 16)
            )
        else:
            raise ValueError(f"unsupported bit depth {self.audio_data.bits}")
        return x.reshape(-1, self.audio_data.nb_channels)[:, 0].astype(np.float64)


    def time_range(self)->TimePeriod:
        fs = self.audio_data.sample_rate
        segs = self.audio_data.segments
        start = min(s.start for s in segs)
        end = max(s.start + pd.Timedelta(seconds=s.n_frames / fs) for s in segs)
        return TimePeriod(start=start, end=end)