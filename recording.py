"""An audio file together with its deployment metadata and calibration.

:class:`Catalog` reads everything from one :class:`~gc_files.Store` (the
bucket or a downloaded copy), so the audio, the metadata and the calibration
file of a :class:`Recording` always belong together. Metadata and
calibration are loaded once per deployment and reused for all its files.

Examples
--------
>>> catalog = Catalog(LocalStore("downloads"))          # or Catalog(GCStore(...))
>>> times, freqs, psd, calibrated = extract_spectrum_multiple_files(
...     files, t0, t1, catalog=catalog, workers=4)      # one merged time series

One recording at a time, e.g. to inspect metadata:

>>> for rec in catalog.recordings(files, deployed_only=True):
...     with rec:
...         print(rec.audio.name, rec.deployed_range, rec.meta.site)
"""
import os
import logging
from dataclasses import dataclass
from typing import Iterator, Optional

from gc_files import Store
from audio_files import AudioFile, TimePeriod
from deployment_metadata import DeploymentMetadata, deployment_folder

logger = logging.getLogger(__name__)


@dataclass
class Recording:
    """One audio file with the metadata and calibration of its deployment.

    Attributes
    ----------
    audio : AudioFile
        The audio file; its GCS stream is open until :meth:`close`.
    meta : DeploymentMetadata
        Metadata of the deployment the file belongs to.
    tf : str or None
        Local path of the calibration transfer function, ready to pass as
        ``tf_filepath=`` to ``extract_spectral_distribution``; ``None`` if the
        deployment has none. ``extract_spectrum_multiple_files(..., catalog=...)``
        picks it up by itself.
    """

    audio: AudioFile
    meta: DeploymentMetadata
    tf: Optional[str]

    @property
    def filename(self) -> str:
        """Blob name of the audio file."""
        return self.audio.filename

    @property
    def time_range(self) -> TimePeriod:
        """Start and end of the audio."""
        return self.audio.time_range()

    @property
    def deployed_range(self) -> Optional[TimePeriod]:
        """Part of the audio recorded while the instrument was deployed.

        ``None`` if the whole file is a deck test or post-recovery recording.
        """
        r = self.time_range
        start = max(r.start, self.meta.deployment_time) if self.meta.deployment_time else r.start
        end = min(r.end, self.meta.recovery_time) if self.meta.recovery_time else r.end
        return TimePeriod(start, end) if start < end else None

    def close(self) -> None:
        """Close the audio stream."""
        self.audio.close()

    def __enter__(self) -> "Recording":
        self.audio.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Catalog:
    """Opens recordings from one store, caching metadata per deployment.

    Parameters
    ----------
    store : Store
        :class:`~gc_files.GCStore` or :class:`~gc_files.LocalStore`. Audio,
        metadata and calibration are all read from it.
    tf_dir : str, default "downloads"
        With a ``GCStore``, where calibration files are downloaded (keeping
        their bucket path). Ignored with a ``LocalStore``.
    """

    def __init__(self, store: Store, tf_dir: str = "downloads"):
        self.store = store
        self.tf_dir = tf_dir
        self._meta: dict[str, DeploymentMetadata] = {}
        self._tf: dict[str, Optional[str]] = {}

    def metadata(self, audio_filename: str) -> DeploymentMetadata:
        """Metadata of the deployment ``audio_filename`` belongs to (cached)."""
        folder = deployment_folder(audio_filename)
        if folder not in self._meta:
            self._meta[folder] = DeploymentMetadata.for_audio_file(self.store, audio_filename)
        return self._meta[folder]

    def calibration_filepath(self, audio_filename: str) -> Optional[str]:
        """Local path of the deployment's calibration file (cached).

        When the deployment has several ``.tf`` files, the one named after
        the hydrophone preamplifier (``meta.preamp_ids``) is used. Returns
        ``None``, with a warning, when there is none.
        """
        folder = deployment_folder(audio_filename)
        if folder not in self._tf:
            meta = self.metadata(audio_filename)
            self._tf[folder] = self._pick_tf(meta)
        return self._tf[folder]

    def _pick_tf(self, meta: DeploymentMetadata) -> Optional[str]:
        found = meta.calibration_files(self.store)
        if not found:
            logger.warning("%s: no calibration file, levels will be uncalibrated", meta.collection_name)
            return None
        matching = [f for f in found if any(os.path.basename(f).startswith(p + "_") for p in meta.preamp_ids)]
        chosen = (matching or found)[0]
        if len(matching or found) > 1:
            logger.warning(
                "%s: %d calibration files, using %s",
                meta.collection_name, len(matching or found), os.path.basename(chosen),
            )
        local = self.store.download(chosen, dest_dir=self.tf_dir, keep_path=True)
        logger.info("%s: calibration %s", meta.collection_name, os.path.basename(chosen))
        return local

    def open(self, audio_filename: str) -> Recording:
        """Open one audio file with its metadata and calibration."""
        return Recording(
            audio=AudioFile(filename=audio_filename, store=self.store),
            meta=self.metadata(audio_filename),
            tf=self.calibration_filepath(audio_filename),
        )

    def recordings(self, audio_files: list[str], deployed_only: bool = False) -> Iterator[Recording]:
        """Open recordings one at a time.

        Each recording is opened when the loop reaches it; close it (or use
        ``with rec:``) before moving on, so only one stream is open at once.

        Parameters
        ----------
        audio_files : list[str]
            Audio blob names.
        deployed_only : bool, default False
            Skip files recorded entirely on deck, before deployment or after
            recovery.
        """
        for f in audio_files:
            rec = self.open(f)
            if deployed_only and rec.deployed_range is None:
                logger.info("%s: recorded on deck, skipped", os.path.basename(f))
                rec.close()
                continue
            yield rec
