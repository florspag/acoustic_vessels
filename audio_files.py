import os
import re
import struct
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from google.cloud import storage
from google.cloud.storage.fileio import BlobReader
import soundfile as sf

_TIME_PATTERNS = [
    # 20190908T060000 / 20190908_060000 / 20190908-060000
    (
        re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[T_\-]?(\d{2})(\d{2})(\d{2})(?!\d)"),
        "Y",
    ),
    # HARP style ..._190908_060000.x.wav
    (re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})[_\-](\d{2})(\d{2})(\d{2})(?!\d)"), "y"),
    # SoundTrap 6789.190908060000.wav
    (re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?!\d)"), "y"),
]

logger = logging.getLogger(__name__)


class AudioExtenstion(StrEnum):
    FLAC = "flac"
    WAVE = "wav"
    XWAVE = "x.wave"
    AIF = "aif"
    AIFF = "aiff"


class GCStore:

    def __init__(self, bucket: str):
        self.client: storage.Client = storage.Client.create_anonymous_client()
        self.bucket: storage.Bucket = self.client.bucket(bucket)

    def list_folders(self, prefix: str):
        """List the 'subfolders' directly under a prefix (like clicking in the console)."""
        it = self.client.list_blobs(self.bucket, prefix=prefix, delimiter="/")
        files = [(b.name, b.size) for b in it]  # must consume the iterator first
        return sorted(it.prefixes), files

    def list_files(
        self, prefix: str, suffix: Optional[str] = None, limit: Optional[int] = None
    ):
        """Recursively list every file under a prefix, optionally filtered by extension."""
        out = []
        for blob in self.client.list_blobs(self.bucket, prefix=prefix):
            if suffix and not blob.name.lower().endswith(suffix):
                continue
            out.append(blob.name)
            if limit and len(out) >= limit:
                break
        return out

    def download(self, filename: str, dest_dir="downloads"):
        """Download one file, keeping its file name."""
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(filename))
        self.bucket.blob(filename).download_to_filename(dest)
        return dest

    def download_files(self, filenames: list[str], dest_dir="downloads"):
        for filename in filenames:
            self.download(filename=filename, dest_dir=dest_dir)

    def download_to_memory(
        self, blob, chunk_size=32 * 1024 * 1024, workers=8
    ) -> bytearray:
        blob.reload()  # get blob.size
        size = blob.size
        buf = bytearray(size)  # allocate once
        view = memoryview(buf)

        def fetch(start):
            end = min(start + chunk_size, size) - 1  # inclusive
            data = blob.download_as_bytes(start=start, end=end, checksum=None)
            view[start : start + len(data)] = data

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(fetch, range(0, size, chunk_size)))  # list() surfaces errors

        return buf  # bytearray; use bytes(buf) if you need immutable bytes

    def open_file(self, filename) -> BlobReader:
        blob = self.bucket.blob(filename)
        # return self.download_to_memory(blob=blob)
        blob.reload()  # fetch size + generation so the reader doesn't re-query
        return blob.open("rb", chunk_size=2 * 1024 * 1024)  # 2 MB per request

    @staticmethod
    def filter_files(
        filenames: list[str],
        extentions: Optional[list[AudioExtenstion]] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ):
        if not extentions:
            return filenames
        files_kept = []
        for filename in filenames:
            file_date = GCStore.time_from_name(filename)
            ext = filename.split(".")[-1]
            if extentions and ext not in extentions:
                continue
            if start_date is not None and file_date < start_date:
                continue
            if end_date is not None and file_date > end_date:
                continue
            files_kept.append(filename)
        return files_kept

    @staticmethod
    def audio_file_extention(filename: str) -> AudioExtenstion:
        ext = filename.split(".")[-1]
        if ext not in AudioExtenstion:
            raise ValueError(f"{ext} is not an audio extension")
        return AudioExtenstion(ext)

    @staticmethod
    def time_from_name(name: str) -> pd.Timestamp | None:
        base = os.path.basename(name)
        for rx, kind in _TIME_PATTERNS:
            for m in rx.finditer(base):
                g = [int(x) for x in m.groups()]
                year = g[0] if kind == "Y" else 2000 + g[0]
                try:
                    return pd.Timestamp(year, g[1], g[2], g[3], g[4], g[5])
                except ValueError:
                    continue
        return None


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
class AUDIO_DATA:
    nb_channels: int
    sample_rate: int
    block_align: int = None
    bits: int = None
    scale: dict = None
    float_fmt: int = None
    bytes_per: bytearray = None
    segments: list[Segment] = None
    sound_file: Optional[sf.SoundFile] = None


class AudioFile:
    """Random-access reader. For WAV/XWAV reads PCM bytes directly so that
    duty-cycled or sub-sampled reads only fetch the bytes needed from GCS."""

    def __init__(self, filename: str, store: GCStore):
        self.filename: str = filename
        self.store: GCStore = store
        self.reader: BlobReader = store.open_file(filename=filename)
        self.kind: AudioExtenstion = GCStore.audio_file_extention(filename)
        self.timestamp: pd.Timestamp | None = GCStore.time_from_name(filename)
        self.harp = None
        self.harp_meta: Optional[HarpMetadata] = None
        self.audio_data = self.parse()

    # -- open / close
    @property
    def closed(self) -> bool:
        return self.reader is None or self.reader.closed

    def open(self) -> "AudioFile":
        """Reopen the GCS stream if it was closed. Parsed metadata is kept,
        so only the stream (and the SoundFile wrapper for non-WAV) is rebuilt."""
        if self.closed:
            self.reader = self.store.open_file(filename=self.filename)
            if self.kind != AudioExtenstion.WAVE:
                self.audio_data.sound_file = sf.SoundFile(self.reader)
        return self

    def close(self) -> None:
        if self.audio_data.sound_file is not None and not self.audio_data.sound_file.closed:
            self.audio_data.sound_file.close()
        if self.reader is not None and not self.reader.closed:
            self.reader.close()

    def __enter__(self) -> "AudioFile":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def parse(self) -> AUDIO_DATA:
        match self.kind:
            case AudioExtenstion.WAVE:
                return self._parse_wav()
            case _:
                return self._parse_sf()

    def _parse_sf(self) -> AUDIO_DATA:
        _sf = sf.SoundFile(self.reader)
        audio_data = AUDIO_DATA(
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
    def _parse_wav(self) -> AUDIO_DATA:
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
            if cid == b"fmt ":
                fmt = data.read(csize)
            elif cid == b"harp":
                self.harp = data.read(csize)
            elif cid == b"data":
                data_loc = pos + 8
                data_len = csize
                if file_size and (csize == 0xFFFFFFFF or data_loc + csize > file_size):
                    data_len = file_size - data_loc
                break
            pos += 8 + csize + (csize & 1)
        if fmt is None or data_loc is None:
            raise ValueError(f"{self.filename}: missing fmt or data chunk")

        tag, nch, fs, _, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
        audio_data = AUDIO_DATA(
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
            return segs
        except Exception as e:  # malformed -> fall back to filename time
            raise ValueError(
                f"  ! {os.path.basename(self.filename)}: harp chunk not usable ({e}); using filename time"
            )

    def read(self, seg: Segment, frame_off: int, n: int) -> np.ndarray:
        """Return first channel as float64 in ADC counts (int formats)."""
        n = max(0, min(n, seg.n_frames - frame_off))
        if n == 0:
            return np.zeros(0)
        self.open()  # no-op if already open
        if self.kind != AudioExtenstion.WAVE:
            self.audio_data.sound_file.seek(seg.byte_loc + frame_off)
            dtype = "float64"
            x = (
                self.audio_data.sound_file.read(n, dtype=dtype, always_2d=True)[:, 0]
                * self.audio_data.scale
            )
            return x
        self.reader.seek(seg.byte_loc + frame_off * self.audio_data.block_align)
        raw = self.reader.read(n * self.audio_data.block_align)
        n = len(raw) // self.audio_data.block_align
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


def _decimator(fs_in: int, fs_out: int):
    if fs_in <= fs_out:
        return fs_in, (lambda x: x)
    r = Fraction(fs_out, fs_in).limit_denominator(2000)
    fs_eff = fs_in * r.numerator / r.denominator
    return fs_eff, (lambda x: signal.resample_poly(x, r.numerator, r.denominator))


def load_calibration(tf_path, freqs, gain_db):
    cal = np.full(freqs.shape, float(gain_db or 0.0))
    if tf_path:
        tf = np.loadtxt(tf_path, comments=("#", "%"), delimiter=None, usecols=(0, 1))
        cal += np.interp(freqs, tf[:, 0], tf[:, 1])
    return cal


def extract_spectral(
    audio_file: AudioFile,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    seconds_per_minute: float = 10.0,
    fmax: float = 1000.0,
    tf=None,
    gain_db: float = 0.0,
) -> tuple[list[pd.Timestamp], np.ndarray, list[np.ndarray]]:

    # Setup decimation
    fs_target = int(2 * fmax)
    times, rows = [], []
    freqs = None
    fs_eff, dec = _decimator(audio_file.audio_data.sample_rate, fs_target)

    # FFT setup (length)
    nper = int(round(fs_eff))  # 1-s FFT -> 1 Hz bins
    n_used = 0
    for seg in audio_file.audio_data.segments:
        seg_dur = seg.n_frames / audio_file.audio_data.sample_rate
        seg_end = seg.start + pd.Timedelta(seconds=seg_dur)
        if seg_end < t0 or seg.start > t1:
            continue
        k = 0
        while k * 60.0 < seg_dur:
            tk = seg.start + pd.Timedelta(seconds=60.0 * k)
            if t0 <= tk <= t1:
                take = min(seconds_per_minute, seg_dur - 60.0 * k)
                if take >= 2.0:
                    x = audio_file.read(
                        seg,
                        int(
                            60.0 * k * audio_file.audio_data.sample_rate,
                        ),
                        int(
                            take * audio_file.audio_data.sample_rate,
                        ),
                    )
                    # Removes mean (DC offset) => no acoustic information
                    x = dec(x - x.mean())
                    if len(x) >= nper:
                        # Welch's method
                        # Power spectral density
                        # f in sample freq and pxx power spectrum of x.
                        f, pxx = signal.welch(
                            x,
                            fs=fs_eff,
                            window="hann",
                            nperseg=nper,
                            noverlap=nper // 2,
                            scaling="density",
                        )
                        keep = (f >= 1) & (f <= fmax)
                        if freqs is None:
                            freqs = f[keep]
                        if keep.sum() == len(freqs):
                            times.append(tk)
                            # Conversion to db
                            rows.append(10 * np.log10(pxx[keep] + 1e-20))
                            n_used += 1
            k += 1
    logger.info(
        f" {os.path.basename(audio_file.filename)}  fs={audio_file.audio_data.sample_rate}  "
        f"{len(audio_file.audio_data.segments)} segment(s)  {n_used} minutes"
    )
    return times, freqs, rows


def process_audio(
    audio_files: list[AudioFile],
    t0,
    t1,
    seconds_per_minute=10.0,
    fmax=1000.0,
    tf=None,
    gain_db=0.0,
) -> np.ndarray[pd.Timestamp]:
    times: list[pd.Timestamp]=[]
    rows:list[np.ndarray[float]] = []
    freqs = None
    for i, p in enumerate(audio_files):
        tms, frqs, rws = extract_spectral(
            audio_file=p,
            t0=t0,
            t1=t1,
            seconds_per_minute=seconds_per_minute,
            fmax=fmax,
            tf=tf,
            gain_db=gain_db,
        )
        p.close()  # free the GCS stream; read() reopens it if needed later
        logger.info(f"  [{i + 1}/{len(audio_files)}]")
        times += tms
        rows += rws
        if freqs is None:
            freqs = frqs
    if not rows:
        raise SystemExit("No audio inside the requested time window.")
    order = np.argsort(np.array(times, dtype="datetime64[ns]"))
    t = np.array(times, dtype="datetime64[ns]")[order]
    psd = np.vstack(rows)[order]

    cal = load_calibration(tf, freqs, gain_db)
    psd = (psd + cal).astype(np.float32)
    calibrated = bool(tf) or bool(gain_db)
    return t, freqs.astype(np.float32), psd, calibrated


def plot_spectral(freqs: list[float], psd: np.array, calibrated:bool):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(freqs, psd)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB re 1 µPa²/Hz)" if calibrated else "PSD (dB, uncalibrated)")
    ax.set_title("Spectrum")
    ax.grid(True, which="both", alpha=0.3)
    plt.show()