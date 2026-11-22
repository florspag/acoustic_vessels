"""Deployment metadata published with NOAA passive acoustic datasets.

Each deployment folder in the ``noaa-passive-bioacoustic`` bucket holds an
``audio/`` folder, an ``ancillary/`` folder (calibration transfer functions,
LTSAs) and a ``metadata/`` folder with one JSON file describing the
deployment: position and depth, deployment and recovery times, sampling,
duty cycle and data-quality periods. For example::

    sefsc/audio/gomex/ef/sefsc_gomex_ef_01/
        audio/GOM_EF_01_160716_170502.x.flac
        ancillary/calibration/828_160707_B_HARP.tf
        metadata/SEFSC_GOMx_EF_01.json

All times are UTC. Numeric values are stored as strings in the JSON and are
converted here.

Examples
--------
>>> meta = DeploymentMetadata.for_audio_file(store, files[0])
>>> meta.latitude, meta.longitude
(27.73308, -92.95133)
>>> meta.is_deployed(pd.Timestamp(2016, 7, 18))   # deck test before deployment
False
>>> box = meta.bbox(radius_km=20)                  # for AIS().load_ais(...)
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from gc_files import GCStore, Store

logger = logging.getLogger(__name__)


def _ts(value) -> Optional[pd.Timestamp]:
    return pd.Timestamp(value) if value else None


def _num(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class SamplingPeriod:
    """Sample rate and bit depth over a period.

    Attributes
    ----------
    start, end : pandas.Timestamp
    sample_rate_hz : float
        Sample rate in Hz. The JSON gives it in kHz (``"2"`` for 2 kHz).
    bits : int, optional
        Bits per sample.
    """

    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    sample_rate_hz: Optional[float]
    bits: Optional[int]


@dataclass
class DutyCycle:
    """Recording schedule over a period.

    ``duration`` is recorded, then nothing for ``interval``; ``interval`` 0
    means continuous recording. Values are as published (HARP datasets give
    ``DURATION`` 60 / ``INTERVAL`` 0 for continuous recording).
    """

    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    duration: Optional[float]
    interval: Optional[float]

    @property
    def continuous(self) -> bool:
        """``True`` if there is no pause between recordings."""
        return not self.interval


@dataclass
class QualityPeriod:
    """Data quality for a frequency band over a period.

    Attributes
    ----------
    quality : str
        ``"Good"``, ``"Unverified"`` (usually: calibration not validated in
        that band) or ``"Unusable"`` (e.g. recording on deck after recovery).
    low_freq, high_freq : float
        Frequency band, in Hz.
    start, end : pandas.Timestamp
    comments : str
    channels : str
        Channel(s) the entry applies to.
    """

    quality: str
    low_freq: Optional[float]
    high_freq: Optional[float]
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    comments: str
    channels: str

    def covers(self, t: pd.Timestamp) -> bool:
        """``True`` if ``t`` is inside this period."""
        return (self.start is None or t >= self.start) and (self.end is None or t <= self.end)


@dataclass
class DeploymentMetadata:
    """Metadata of one deployment, from its ``metadata/*.json`` file.

    Attributes
    ----------
    collection_name : str
        E.g. ``"SEFSC_GOMx_EF_01"``.
    project, site, deployment_name : str
    site_aliases : list[str]
        E.g. ``["Flower Garden East"]``.
    platform, instrument_type, instrument_id : str
        E.g. ``"Mooring"``, ``"HARP"``, ``"27-160721"``.
    latitude, longitude : float
        Deployment position, in decimal degrees.
    instrument_depth, bottom_depth : float
        In metres, positive downwards (the JSON stores them as negative).
    sea_area : str
    deployment_time, recovery_time : pandas.Timestamp
        When the instrument reached and left its position. Audio outside
        this window is deck tests or recovery noise.
    audio_start, audio_end : pandas.Timestamp
        First and last recorded sample.
    sampling : list[SamplingPeriod]
    duty_cycle : list[DutyCycle]
    quality : list[QualityPeriod]
    preamp_ids : list[str]
        Preamplifier IDs of the audio sensors; calibration files in
        ``ancillary/calibration/`` are named after them.
    path : str
        Blob name of the JSON file.
    raw : dict
        The full JSON, for fields not mapped here (abstract, people, ...).
    """

    collection_name: str
    project: str
    site: str
    deployment_name: str
    site_aliases: list[str]
    platform: str
    instrument_type: str
    instrument_id: str
    latitude: Optional[float]
    longitude: Optional[float]
    instrument_depth: Optional[float]
    bottom_depth: Optional[float]
    sea_area: str
    deployment_time: Optional[pd.Timestamp]
    recovery_time: Optional[pd.Timestamp]
    audio_start: Optional[pd.Timestamp]
    audio_end: Optional[pd.Timestamp]
    sampling: list[SamplingPeriod] = field(default_factory=list)
    duty_cycle: list[DutyCycle] = field(default_factory=list)
    quality: list[QualityPeriod] = field(default_factory=list)
    preamp_ids: list[str] = field(default_factory=list)
    path: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    # -- construction
    @classmethod
    def from_dict(cls, d: dict, path: str = "") -> "DeploymentMetadata":
        """Build from the parsed JSON of a deployment metadata file."""
        dep = d.get("DEPLOYMENT", {}) or {}
        sampling, duty = [], []
        for ch in (d.get("CHANNELS") or {}).values():
            details = ch.get("SAMPLING_DETAILS", {}) or {}
            for s in details.get("SAMPLING", []) or []:
                khz = _num(s.get("SAMPLE_RATE"))
                bits = _num(s.get("SAMPLE_BITS"))
                sampling.append(SamplingPeriod(
                    start=_ts(s.get("START")),
                    end=_ts(s.get("END")),
                    sample_rate_hz=khz * 1000 if khz is not None else None,
                    bits=int(bits) if bits is not None else None,
                ))
            for c in details.get("DUTY_CYCLE", []) or []:
                duty.append(DutyCycle(
                    start=_ts(c.get("START")),
                    end=_ts(c.get("END")),
                    duration=_num(c.get("DURATION")),
                    interval=_num(c.get("INTERVAL")),
                ))
        quality = [
            QualityPeriod(
                quality=q.get("quality", ""),
                low_freq=_num(q.get("low_freq")),
                high_freq=_num(q.get("high_freq")),
                start=_ts(q.get("start")),
                end=_ts(q.get("end")),
                comments=q.get("comments", ""),
                channels=str(q.get("channels", "")),
            )
            for q in (d.get("QUALITY_DETAILS") or {}).get("quality_details", []) or []
        ]
        depth = _num(dep.get("DEPLOY_INSTRUMENT_DEPTH"))
        bottom = _num(dep.get("DEPLOY_BOTTOM_DEPTH"))
        return cls(
            collection_name=d.get("DATA_COLLECTION_NAME", ""),
            project=d.get("PROJECT_NAME", ""),
            site=d.get("SITE", ""),
            deployment_name=d.get("DEPLOYMENT_NAME", ""),
            site_aliases=list(d.get("SITE_ALIASES", []) or []),
            platform=d.get("PLATFORM_NAME", ""),
            instrument_type=d.get("INSTRUMENT_TYPE", ""),
            instrument_id=d.get("INSTRUMENT_ID", ""),
            latitude=_num(dep.get("DEPLOY_LAT")),
            longitude=_num(dep.get("DEPLOY_LON")),
            instrument_depth=abs(depth) if depth is not None else None,
            bottom_depth=abs(bottom) if bottom is not None else None,
            sea_area=dep.get("SEA_AREA", ""),
            deployment_time=_ts(dep.get("DEPLOYMENT_TIME")),
            recovery_time=_ts(dep.get("RECOVERY_TIME")),
            audio_start=_ts(dep.get("AUDIO_START")),
            audio_end=_ts(dep.get("AUDIO_END")),
            sampling=sampling,
            duty_cycle=duty,
            quality=quality,
            preamp_ids=[s["PRE_ID"] for s in (d.get("SENSORS") or {}).get("AUDIO", []) if s.get("PRE_ID")],
            path=path,
            raw=d,
        )

    @classmethod
    def from_store(cls, store: Store, path: str) -> "DeploymentMetadata":
        """Load a metadata JSON from the bucket.

        Parameters
        ----------
        store : Store
            :class:`GCStore` or :class:`LocalStore`.
        path : str
            Blob name, e.g.
            ``"sefsc/audio/gomex/ef/sefsc_gomex_ef_01/metadata/SEFSC_GOMx_EF_01.json"``.
        """
        meta = cls.from_dict(store.read_json(path), path=path)
        logger.info(
            "%s: %s at %.5f, %.5f, %s m deep, deployed %s -> %s",
            meta.collection_name, meta.instrument_type, meta.latitude or float("nan"),
            meta.longitude or float("nan"), meta.instrument_depth, meta.deployment_time, meta.recovery_time,
        )
        return meta

    @classmethod
    def for_audio_file(cls, store: Store, audio_filename: str) -> "DeploymentMetadata":
        """Load the metadata of the deployment an audio file belongs to.

        The deployment folder is the part of the path before ``/audio/``; its
        ``metadata/`` folder must hold exactly one JSON file.

        Raises
        ------
        FileNotFoundError
            If the path has no ``/audio/`` folder or no metadata JSON is found.
        ValueError
            If several JSON files are found.
        """
        root = deployment_folder(audio_filename)
        found = store.list_files(prefix=root + "metadata/", suffix=".json")
        if not found:
            raise FileNotFoundError(f"no metadata JSON under {root}metadata/")
        if len(found) > 1:
            raise ValueError(f"several metadata JSON files under {root}metadata/: {found}")
        return cls.from_store(store, found[0])

    # -- helpers
    @property
    def folder(self) -> str:
        """Deployment folder in the bucket, ending with ``/``."""
        return self.path.rsplit("metadata/", 1)[0] if "metadata/" in self.path else ""

    @property
    def sample_rate_hz(self) -> Optional[float]:
        """Sample rate of the first sampling period, in Hz."""
        return self.sampling[0].sample_rate_hz if self.sampling else None

    def is_deployed(self, t: pd.Timestamp) -> bool:
        """``True`` if ``t`` is between deployment and recovery.

        Audio outside that window was recorded on deck (tests before
        deployment, or after recovery) and should not be analysed.
        """
        return (self.deployment_time is None or t >= self.deployment_time) and (
            self.recovery_time is None or t <= self.recovery_time
        )

    def quality_at(self, t: pd.Timestamp, freq_hz: Optional[float] = None) -> list[QualityPeriod]:
        """Quality entries that apply at time ``t`` (and frequency ``freq_hz``, if given)."""
        out = [q for q in self.quality if q.covers(t)]
        if freq_hz is not None:
            out = [
                q for q in out
                if (q.low_freq is None or freq_hz >= q.low_freq) and (q.high_freq is None or freq_hz <= q.high_freq)
            ]
        return out

    def bbox(self, radius_km: float):
        """Bounding box around the deployment position, for :meth:`ais_data.AIS.load_ais`."""
        from ais_data import BBox  # local import: AIS support is optional

        if self.latitude is None or self.longitude is None:
            raise ValueError(f"{self.collection_name}: no deployment position in metadata")
        return BBox.around(self.latitude, self.longitude, radius_km)

    def calibration_files(self, store: Store) -> list[str]:
        """Calibration transfer-function files (``*.tf``) of this deployment.

        These are bucket paths. ``recording.Catalog.calibration_filepath``
        picks the right one and returns a local path, which is easier than
        downloading one by hand.
        """
        if not self.folder:
            return []
        return store.list_files(prefix=self.folder + "ancillary/calibration/", suffix=".tf")


def deployment_folder(audio_filename: str) -> str:
    """Deployment folder of an audio blob: the path before ``/audio/``, ending with ``/``.

    Raises
    ------
    FileNotFoundError
        If the path has no ``/audio/`` component after the dataset prefix.
    """
    head, sep, _ = audio_filename.rpartition("/audio/")
    if not sep:
        raise FileNotFoundError(f"cannot find the deployment folder of {os.path.basename(audio_filename)}")
    return head + "/"


def download_with_metadata(
    store: GCStore,
    audio_files: list[str],
    dest_dir: str = "downloads",
    calibration: bool = True,
    workers: int = 4,
    overwrite: bool = False,
) -> list[str]:
    """Download audio files together with their deployments' metadata.

    For every deployment the audio files belong to, also downloads its
    ``metadata/*.json`` and, if ``calibration``, its
    ``ancillary/calibration/*.tf`` files. Bucket paths are kept under
    ``dest_dir``, so the copy can be read offline with ``LocalStore(dest_dir)``
    exactly as the bucket is read with ``GCStore``. Files already downloaded
    with the right size are skipped.

    Parameters
    ----------
    store : GCStore
        Bucket to download from.
    audio_files : list[str]
        Audio blob names, e.g. from ``GCStore.filter_files``.
    dest_dir : str, default "downloads"
        Local folder standing for the bucket root.
    calibration : bool, default True
        Also download the calibration transfer functions.
    workers : int, default 4
        Files downloaded in parallel. Audio files are 250-300 MB each.
    overwrite : bool, default False
        Download again even if a complete local copy exists.

    Returns
    -------
    list[str]
        Local paths of the audio files, in input order.

    Examples
    --------
    >>> download_with_metadata(store, files[:3])
    >>> local = LocalStore("downloads")
    >>> meta = DeploymentMetadata.for_audio_file(local, files[0])
    >>> aud = AudioFile(files[0], local)
    """
    folders = sorted({deployment_folder(f) for f in audio_files})
    extra = []
    for root in folders:
        extra += store.list_files(prefix=root + "metadata/", suffix=".json")
        if calibration:
            extra += store.list_files(prefix=root + "ancillary/calibration/", suffix=".tf")
    logger.info(
        "Downloading %d audio file(s) and %d metadata/calibration file(s) from %d deployment(s) to %s",
        len(audio_files), len(extra), len(folders), dest_dir,
    )
    store.download_files(extra, dest_dir=dest_dir, keep_path=True, overwrite=overwrite)
    return store.download_files(
        audio_files, dest_dir=dest_dir, keep_path=True, overwrite=overwrite, workers=workers
    )
