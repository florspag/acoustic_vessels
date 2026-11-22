"""Google Cloud Storage access: list, filter, download and stream audio files.

Also parses recording start times from file names (ISO, HARP and SoundTrap
naming styles).
"""
import io
import os
import re
import json
import time
import logging
import threading
from typing import Optional, Protocol, runtime_checkable
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
import pandas as pd
from google.cloud import storage
from google.cloud.storage.fileio import BlobReader

logger = logging.getLogger(__name__)

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

class AudioExtenstion(StrEnum):
    """Audio file extensions the readers understand.

    Only the text after the last dot is compared, so Triton XWAV files
    (``*.x.wav``) are :attr:`WAVE`; :class:`AudioFile` detects their ``harp``
    chunk from the file content.
    """

    FLAC = "flac"
    WAVE = "wav"
    AIF = "aif"
    AIFF = "aiff"

@runtime_checkable
class Store(Protocol):
    """Read access to files laid out like the bucket, wherever they are.

    :class:`GCStore` (the bucket) and :class:`LocalStore` (a downloaded copy)
    both follow this protocol, so any code typed with ``Store`` works on
    either. File names are always bucket-style paths with ``/``, e.g.
    ``"sefsc/audio/gomex/ef/sefsc_gomex_ef_01/audio/GOM_EF_01_160723_162002.x.flac"``.

    Any class with these methods is a ``Store``; no inheritance needed.
    ``isinstance(x, Store)`` checks that the methods exist (not their
    signatures).
    """

    def open_file(self, filename: str) -> io.BufferedIOBase:
        """Open a file as a seekable binary stream."""
        ...

    def read_json(self, filename: str) -> dict:
        """Read and parse a JSON file."""
        ...

    def list_files(
        self, prefix: str, suffix: Optional[str] = None, limit: Optional[int] = None
    ) -> list[str]:
        """Recursively list file names starting with ``prefix``."""
        ...

    def list_folders(self, prefix: str) -> tuple[list[str], list[tuple[str, int]]]:
        """Sub-folders and ``(name, size)`` of files directly under ``prefix``."""
        ...

    def download(
        self,
        filename: str,
        dest_dir: str = "downloads",
        keep_path: bool = False,
        overwrite: bool = False,
    ) -> str:
        """Make a file available locally and return its local path."""
        ...


class GCStore:
    """Read-only access to a public Google Cloud Storage bucket.

    Uses an anonymous client, so no credentials are needed. Clients are not
    guaranteed thread-safe: create one ``GCStore`` per thread.

    Parameters
    ----------
    bucket : str
        Bucket name, e.g. ``"noaa-passive-bioacoustic"``.

    Attributes
    ----------
    client : google.cloud.storage.Client
    bucket : google.cloud.storage.Bucket
    """

    def __init__(self, bucket: str):
        self.client: storage.Client = storage.Client.create_anonymous_client()
        self.bucket: storage.Bucket = self.client.bucket(bucket)

    def list_folders(self, prefix: str) -> tuple[list[str], list[tuple[str, int]]]:
        """List what sits directly under a prefix, like browsing the GCS console.

        Parameters
        ----------
        prefix : str
            Folder-like prefix, ending with ``/`` (e.g. ``"sefsc/"``).

        Returns
        -------
        folders : list[str]
            Sorted sub-prefixes one level down.
        files : list[tuple[str, int]]
            ``(blob name, size in bytes)`` of files directly under the prefix.
        """
        it = self.client.list_blobs(self.bucket, prefix=prefix, delimiter="/")
        files = [(b.name, b.size) for b in it]  # must consume the iterator first
        return sorted(it.prefixes), files

    def list_files(
        self, prefix: str, suffix: Optional[str] = None, limit: Optional[int] = None
    ):
        """Recursively list every file under a prefix.

        Parameters
        ----------
        prefix : str
            Bucket prefix to search under.
        suffix : str, optional
            Keep only names ending with this (lower-case) suffix, e.g. ``".flac"``.
        limit : int, optional
            Stop after this many matches.

        Returns
        -------
        list[str]
            Full blob names, in bucket listing order.
        """
        out = []
        for blob in self.client.list_blobs(self.bucket, prefix=prefix):
            if suffix and not blob.name.lower().endswith(suffix):
                continue
            out.append(blob.name)
            if limit and len(out) >= limit:
                break
        return out

    def download(
        self,
        filename: str,
        dest_dir: str = "downloads",
        keep_path: bool = False,
        overwrite: bool = False,
    ) -> str:
        """Download one blob to ``dest_dir``.

        The file is written to ``<dest>.part`` and renamed when complete, so an
        interrupted download is never mistaken for a finished one.

        Parameters
        ----------
        filename : str
            Full blob name in the bucket.
        dest_dir : str, default "downloads"
            Local directory, created if missing.
        keep_path : bool, default False
            ``True`` keeps the blob's full path under ``dest_dir``, so the
            local copy mirrors the bucket and can be read with
            ``LocalStore(dest_dir)``. ``False`` keeps only the base name.
        overwrite : bool, default False
            Download again even if a file of the same size is already there.

        Returns
        -------
        str
            Local path of the downloaded file.
        """
        rel = filename if keep_path else os.path.basename(filename)
        dest = os.path.join(dest_dir, rel)
        blob = self.bucket.blob(filename)
        if not overwrite and os.path.exists(dest):
            blob.reload()
            if blob.size == os.path.getsize(dest):
                logger.debug("%s: already downloaded", os.path.basename(dest))
                return dest
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        t_start = time.perf_counter()
        part = dest + ".part"
        try:
            blob.download_to_filename(part)
        except BaseException:
            if os.path.exists(part):
                os.remove(part)
            raise
        os.replace(part, dest)
        logger.info(
            "%s: downloaded %.0f MB in %.0f s",
            os.path.basename(dest), os.path.getsize(dest) / 1e6, time.perf_counter() - t_start,
        )
        return dest

    def download_files(
        self,
        filenames: list[str],
        dest_dir: str = "downloads",
        keep_path: bool = False,
        overwrite: bool = False,
        workers: int = 1,
    ) -> list[str]:
        """Download several blobs with :meth:`download`.

        With ``workers > 1``, files are downloaded in parallel threads, each
        with its own GCS client. Returns the local paths, in input order.
        """
        if workers <= 1:
            return [self.download(f, dest_dir, keep_path, overwrite) for f in filenames]
        local = threading.local()
        bucket = self.bucket.name

        def one(f):
            if not hasattr(local, "store"):
                local.store = GCStore(bucket)
            return local.store.download(f, dest_dir, keep_path, overwrite)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, filenames))

    def download_to_memory(
        self, blob, chunk_size=32 * 1024 * 1024, workers=8
    ) -> bytearray:
        """Download a whole blob into memory with parallel ranged requests.

        Parameters
        ----------
        blob : google.cloud.storage.Blob
            Blob to fetch.
        chunk_size : int, default 32 MiB
            Size of each ranged request.
        workers : int, default 8
            Number of concurrent requests.

        Returns
        -------
        bytearray
            The blob's content.
        """
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

    def read_json(self, filename: str) -> dict:
        """Download a JSON blob and parse it.

        Parameters
        ----------
        filename : str
            Full blob name, e.g. a deployment's ``metadata/*.json``.

        Returns
        -------
        dict
            The parsed JSON.
        """
        return json.loads(self.bucket.blob(filename).download_as_bytes())

    def open_file(self, filename) -> BlobReader:
        """Open a blob as a seekable binary stream.

        Data is fetched on demand in 2 MB requests, so random access (e.g.
        reading a few seconds per minute) downloads only what is read.

        Parameters
        ----------
        filename : str
            Full blob name.

        Returns
        -------
        google.cloud.storage.fileio.BlobReader
            File-like object opened in ``"rb"`` mode. Close it when done.
        """
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
    ) -> list[str]:
        """Filter blob names by extension and date, and sort them by date.

        The date of each file is parsed from its name with :meth:`time_from_name`.

        Parameters
        ----------
        filenames : list[str]
            Blob names, e.g. from :meth:`list_files`.
        extentions : list[AudioExtenstion], optional
            Keep only these extensions (compared with the text after the last
            dot). ``None`` or empty keeps every extension.
        start_date, end_date : pandas.Timestamp, optional
            Keep files whose name date is within ``[start_date, end_date]``.
            This is the file's *start* time: a file starting just before
            ``start_date`` is dropped even if it covers it. When either bound
            is set, files with no date in their name are dropped.

        Returns
        -------
        list[str]
            Kept names, oldest first; files with no date in their name go last.
        """
        files_kept = []
        for filename in filenames:
            file_date = GCStore.time_from_name(filename)
            ext = filename.split(".")[-1]
            if extentions and ext not in extentions:
                continue
            if file_date is None and (start_date is not None or end_date is not None):
                continue  # can't place it in the window
            if start_date is not None and file_date < start_date:
                continue
            if end_date is not None and file_date > end_date:
                continue
            files_kept.append((file_date, filename))
        files_kept.sort(key=lambda df: (df[0] is None, df[0] or pd.Timestamp.min, df[1]))
        return [f for _, f in files_kept]

    @staticmethod
    def audio_file_extention(filename: str) -> AudioExtenstion:
        """Return the :class:`AudioExtenstion` of a file name.

        Raises
        ------
        ValueError
            If the text after the last dot is not a known audio extension.
        """
        ext = filename.split(".")[-1]
        if ext not in AudioExtenstion:
            raise ValueError(f"{ext} is not an audio extension")
        return AudioExtenstion(ext)

    @staticmethod
    def time_from_name(name: str) -> pd.Timestamp | None:
        """Parse a recording start time from a file name.

        Recognised formats, tried in this order on the base name:

        - ``20190908T060000``, ``20190908_060000``, ``20190908-060000``
        - HARP: ``..._190908_060000.x.wav``
        - SoundTrap: ``6789.190908060000.wav``

        Two-digit years are read as 20YY.

        Returns
        -------
        pandas.Timestamp or None
            The first valid date found, or ``None`` if none matches.
        """
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


class LocalStore:
    """Read files from a local folder that mirrors a bucket.

    Has the same reading interface as :class:`GCStore` (``open_file``,
    ``list_files``, ``list_folders``, ``read_json``), so :class:`AudioFile`,
    ``DeploymentMetadata`` and the spectrum functions work on downloaded data
    without network access. Blob names are paths relative to ``root``; use
    ``GCStore.download(..., keep_path=True)`` or ``download_with_metadata`` to
    build such a folder. Safe to share between threads.

    Parameters
    ----------
    root : str
        Local folder standing for the bucket root, e.g. ``"downloads"``.
    """

    def __init__(self, root: str = "downloads"):
        self.root = root

    def _path(self, filename: str) -> str:
        return os.path.join(self.root, filename)

    def open_file(self, filename: str) -> io.BufferedIOBase:
        """Open a local file in binary mode."""
        return open(self._path(filename), "rb")

    def read_json(self, filename: str) -> dict:
        """Read and parse a local JSON file."""
        with open(self._path(filename), "rb") as f:
            return json.load(f)

    def list_files(
        self, prefix: str, suffix: Optional[str] = None, limit: Optional[int] = None
    ) -> list[str]:
        """Recursively list files whose relative path starts with ``prefix``.

        Same arguments and result as :meth:`GCStore.list_files`; names use
        ``/`` and are relative to ``root``. Unfinished ``.part`` downloads are
        skipped.
        """
        base = os.path.dirname(prefix) if not prefix.endswith("/") else prefix
        start = self._path(base)
        out = []
        if not os.path.isdir(start):
            return out
        for dirpath, _, names in os.walk(start):
            for n in sorted(names):
                rel = os.path.relpath(os.path.join(dirpath, n), self.root).replace(os.sep, "/")
                if not rel.startswith(prefix) or rel.endswith(".part"):
                    continue
                if suffix and not rel.lower().endswith(suffix):
                    continue
                out.append(rel)
                if limit and len(out) >= limit:
                    return sorted(out)
        return sorted(out)

    def list_folders(self, prefix: str) -> tuple[list[str], list[tuple[str, int]]]:
        """Sub-folders and files directly under ``prefix``, like :meth:`GCStore.list_folders`."""
        path = self._path(prefix)
        if not os.path.isdir(path):
            return [], []
        folders, files = [], []
        for n in sorted(os.listdir(path)):
            rel = prefix + n
            if os.path.isdir(os.path.join(path, n)):
                folders.append(rel + "/")
            else:
                files.append((rel, os.path.getsize(os.path.join(path, n))))
        return folders, files

    def download(
        self,
        filename: str,
        dest_dir: str = "downloads",
        keep_path: bool = False,
        overwrite: bool = False,
    ) -> str:
        """Return the local path of ``filename``.

        The file is already local, so nothing is copied and the other
        arguments are ignored; they are there to match :class:`Store`.
        """
        return self._path(filename)
