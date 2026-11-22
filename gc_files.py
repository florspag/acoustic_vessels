import os
import re
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
import pandas as pd
from google.cloud import storage
from google.cloud.storage.fileio import BlobReader

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

