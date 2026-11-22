import os
import time
import logging
from dataclasses import dataclass, fields
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

@dataclass
class AISData:
    """One AIS position report.

    Fields are the MarineCadastre columns in snake_case (``BaseDateTime`` ->
    ``base_date_time``, ``LAT`` -> ``latitude``, ...); the DataFrames returned
    by :class:`AIS` use the same names as columns.

    Attributes
    ----------
    mmsi : str
        Maritime Mobile Service Identity, the vessel's radio ID.
    base_date_time : pandas.Timestamp
        Time of the report, UTC.
    latitude, longitude : float
        Position in decimal degrees. 91 / 181 mean "not available" in AIS;
        they are kept as is, but any bounding box drops them.
    sog : float, optional
        Speed over ground, in knots (0-102.2; 102.2 means 102.2 kn or more).
        ``None`` / NaN when not available (102.3 in the source).
    cog : float, optional
        Course over ground, in degrees (0-359.9). ``None`` / NaN when not
        available (360 in the source).
    heading : float, optional
        True heading, in degrees (0-359). ``None`` / NaN when not available
        (511 in the source; common on small vessels with no gyro compass
        connected to the transponder).

    vessel_name, imo, call_sign : str, optional
        Identification, when broadcast.
    vessel_type : int, optional
        Ship-type code. Files up to 2017 often use MarineCadastre's own
        codes above 1000 (e.g. 1004 freight ship); see the vessel type code
        table on MarineCadastre.gov. Later files use the standard AIS codes:
        30 fishing, 31-32 towing, 35 military, 36 sailing, 37 pleasure craft,
        52 tug, 60-69 passenger, 70-79 cargo, 80-89 tanker.
    status : int, optional
        Navigational status: 0 under way using engine, 1 at anchor,
        5 moored, 7 fishing, 15 undefined, ...
    length, width, draft : float, optional
        Dimensions, in metres.
    cargo : float, optional
        Cargo / hazard code.

    Notes
    -----
    AIS messages (ITU-R M.1371) encode each field as a fixed-size integer, so
    a missing value cannot be left blank: the transponder sends a reserved
    "not available" code instead. MarineCadastre keeps these codes as numbers.

    ========= ========================= ============ ================
    Field     Encoding                  Valid range  "Not available"
    ========= ========================= ============ ================
    heading   9 bits, whole degrees     0-359        511 (2**9 - 1)
    cog       12 bits, 0.1 degree       0-3599       3600 (360.0 deg)
    sog       10 bits, 0.1 knot         0-1022       1023 (102.3 kn)
    latitude  27 bits, 1/10000 minute   -90 to 90    91 deg
    longitude 28 bits, 1/10000 minute   -180 to 180  181 deg
    ========= ========================= ============ ================

    Left in, these codes would read as real values: a mean heading pulled
    towards 511 deg, or a missing speed counted as 102 kn. :meth:`AIS.read_ais_day`
    turns the heading, course and speed codes into NaN so pandas skips them.
    """

    mmsi: str
    base_date_time: pd.Timestamp
    latitude: float
    longitude: float
    sog: Optional[float]
    cog: Optional[float]
    heading: Optional[float]
    vessel_name: Optional[str]
    imo: Optional[str]
    call_sign: Optional[str]
    vessel_type: Optional[int]
    status: Optional[int]
    length: Optional[float]
    width: Optional[float]
    draft: Optional[float]
    cargo: Optional[float]

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> list["AISData"]:
        """Turn a DataFrame from :func:`load_ais` into ``AISData`` objects.

        Missing values become ``None``.
        """
        names = [f.name for f in fields(cls)]
        df = df[names].astype(object).where(df[names].notna(), None)
        return [cls(**row) for row in df.to_dict("records")]


@dataclass(frozen=True)
class BBox:
    """Geographic bounding box, in decimal degrees."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @classmethod
    def around(cls, lat: float, lon: float, radius_km: float) -> "BBox":
        """Box containing every point within ``radius_km`` of ``(lat, lon)``."""
        dlat = radius_km / 111.32
        dlon = radius_km / (111.32 * max(np.cos(np.radians(lat)), 1e-6))
        return cls(lat - dlat, lat + dlat, lon - dlon, lon + dlon)

    def contains(self, lat: pd.Series, lon: pd.Series) -> pd.Series:
        """Boolean mask of the points inside the box."""
        return lat.between(self.lat_min, self.lat_max) & lon.between(self.lon_min, self.lon_max)

class AIS:
    __AIS_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"
    __DEFAULT_DIR = os.path.join("downloads", "ais")
    # Source column -> AISData field name
    __RENAME = {
        "MMSI": "mmsi",
        "BaseDateTime": "base_date_time",
        "LAT": "latitude",
        "LON": "longitude",
        "SOG": "sog",
        "COG": "cog",
        "Heading": "heading",
        "VesselName": "vessel_name",
        "IMO": "imo",
        "CallSign": "call_sign",
        "VesselType": "vessel_type",
        "Status": "status",
        "Length": "length",
        "Width": "width",
        "Draft": "draft",
        "Cargo": "cargo",
        "TranscieverClass": "transceiver_class",
    }

    __DTYPES = {
        "MMSI": "string",
        "LAT": "float64",
        "LON": "float64",
        "SOG": "float64",
        "COG": "float64",
        "Heading": "float64",
        "VesselName": "string",
        "IMO": "string",
        "CallSign": "string",
        "VesselType": "Int64",
        "Status": "Int64",  # source column names, before renaming
        "Length": "float64",
        "Width": "float64",
        "Draft": "float64",
        "Cargo": "float64",
        "TranscieverClass": "string",  # spelled this way in the source, from 2018
    }


    def __ais_url(self, day: pd.Timestamp) -> str:
        """URL of the MarineCadastre daily file for ``day``."""
        return self.__AIS_URL.format(year=day.year, month=day.month, day=day.day)


    def __download_ais_day(
            self,
            day: pd.Timestamp,
            dest_dir: str = __DEFAULT_DIR,
            overwrite: bool = False,
            timeout: float = 60.0,
            retries: int = 3,
        ) -> str:
        """Download one day of AIS data, unless it is already cached.

        The file is written to ``<name>.part`` and renamed when complete, so an
        interrupted download is never mistaken for a cached file.

        Parameters
        ----------
        day : pandas.Timestamp
            Day to download (the time of day is ignored).
        dest_dir : str, default "downloads/ais"
            Cache directory, created if missing.
        overwrite : bool, default False
            Download again even if the file is cached.
        timeout : float, default 60.0
            Seconds to wait for the server between two received chunks.
        retries : int, default 3
            Attempts before giving up.

        Returns
        -------
        str
            Local path of the zip file.

        Raises
        ------
        FileNotFoundError
            If NOAA has no file for that day (HTTP 404).
        requests.HTTPError
            For other HTTP errors, after ``retries`` attempts.
        """
        day = pd.Timestamp(day).normalize()
        url = self.__ais_url(day)
        dest = os.path.join(dest_dir, os.path.basename(url))
        if os.path.exists(dest) and not overwrite:
            logger.debug("%s: cached", os.path.basename(dest))
            return dest
        os.makedirs(dest_dir, exist_ok=True)
        part = dest + ".part"
        for attempt in range(1, retries + 1):
            t_start = time.perf_counter()
            try:
                with requests.get(url, stream=True, timeout=timeout) as r:
                    if r.status_code == 404:
                        raise FileNotFoundError(f"no AIS file for {day.date()}: {url}")
                    r.raise_for_status()
                    size = int(r.headers.get("content-length", 0))
                    logger.info("%s: downloading %.0f MB", os.path.basename(dest), size / 1e6)
                    got = 0
                    with open(part, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                            got += len(chunk)
                if size and got != size:
                    raise IOError(f"incomplete download: {got} of {size} bytes")
                os.replace(part, dest)
                logger.info(
                    "%s: done (%.0f MB in %.0f s)",
                    os.path.basename(dest), got / 1e6, time.perf_counter() - t_start,
                )
                return dest
            except FileNotFoundError:
                raise
            except (requests.RequestException, IOError) as e:
                if os.path.exists(part):
                    os.remove(part)
                if attempt == retries:
                    logger.error("%s: download failed after %d attempts", os.path.basename(dest), retries)
                    raise
                wait = 5 * attempt
                logger.warning("%s: attempt %d failed (%s), retrying in %d s", os.path.basename(dest), attempt, e, wait)
                time.sleep(wait)


    def __read_ais_day(
            self,
            path: str,
            bbox: Optional[BBox] = None,
            t0: Optional[pd.Timestamp] = None,
            t1: Optional[pd.Timestamp] = None,
            mmsi: Optional[list[str]] = None,
            chunksize: int = 1_000_000,
        ) -> pd.DataFrame:
        """Read one daily AIS zip, keeping only the rows that match the filters.

        The CSV is streamed in chunks, so memory use is bounded by ``chunksize``
        rows plus the rows kept. AIS "not available" codes are turned into NaN:
        heading 511, course 360 or more (3600 tenths of a degree; higher values
        are not allowed), speed 102.3 knots or more (1023 tenths of a knot).
        See the Notes of :class:`AISData` for why these codes exist.

        Parameters
        ----------
        path : str
            Zip file from :func:`download_ais_day`.
        bbox : BBox, optional
            Keep positions inside this box.
        t0, t1 : pandas.Timestamp, optional
            Keep reports with ``t0 <= base_date_time <= t1`` (UTC, timezone-naive).
        mmsi : list[str], optional
            Keep only these vessels.
        chunksize : int, default 1_000_000
            Rows read at a time.

        Returns
        -------
        pandas.DataFrame
            Columns named as the :class:`AISData` fields, plus
            ``transceiver_class`` for files that have it. Not sorted.
        """
        t_start = time.perf_counter()
        kept, n_read = [], 0
        mmsi_set = set(map(str, mmsi)) if mmsi is not None else None
        reader = pd.read_csv(
            path,
            compression="zip",
            chunksize=chunksize,
            dtype={k: v for k, v in self.__DTYPES.items() if k != "BaseDateTime"},
            usecols=lambda c: c in self.__DTYPES or c == "BaseDateTime",
        )
        for chunk in reader:
            n_read += len(chunk)
            m = pd.Series(True, index=chunk.index)
            if bbox is not None:
                m &= bbox.contains(chunk["LAT"], chunk["LON"])
            if mmsi_set is not None:
                m &= chunk["MMSI"].isin(mmsi_set)
            chunk = chunk[m]
            if chunk.empty:
                continue
            chunk = chunk.assign(BaseDateTime=pd.to_datetime(chunk["BaseDateTime"]))
            if t0 is not None:
                chunk = chunk[chunk["BaseDateTime"] >= t0]
            if t1 is not None:
                chunk = chunk[chunk["BaseDateTime"] <= t1]
            kept.append(chunk)
        df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=list(self.__DTYPES) + ["BaseDateTime"])
        df = df.rename(columns=self.__RENAME)
        df["heading"] = df["heading"].where(df["heading"] != 511)
        df["cog"] = df["cog"].where(df["cog"] < 360)
        df["sog"] = df["sog"].where(df["sog"] < 102.3)
        logger.info(
            "%s: kept %d of %d reports (%d vessels) in %.0f s",
            os.path.basename(path), len(df), n_read, df["mmsi"].nunique(), time.perf_counter() - t_start,
        )
        return df


    def _load_ais_one_day(
            self,
            day: pd.Timestamp,
            dest_dir: str,
            bbox: Optional[BBox],
            start: pd.Timestamp,
            end: pd.Timestamp,
            mmsi: Optional[list[str]],
        ) -> Optional[pd.DataFrame]:
        """Download and read one day; ``None`` if NOAA has no file for it."""
        try:
            path = self.__download_ais_day(day, dest_dir=dest_dir)
        except FileNotFoundError as e:
            logger.warning("%s", e)
            return None
        return self.__read_ais_day(path, bbox=bbox, t0=start, t1=end, mmsi=mmsi)


    def load_ais(
            self,
            start: pd.Timestamp,
            end: pd.Timestamp,
            bbox: Optional[BBox] = None,
            mmsi: Optional[list[str]] = None,
            dest_dir: str = __DEFAULT_DIR,
            workers: int = 2,
        ) -> pd.DataFrame:
        """Download (if needed) and read AIS reports between ``start`` and ``end``.

        Parameters
        ----------
        start, end : pandas.Timestamp
            Time window, UTC. Every day it touches is downloaded.
        bbox : BBox, optional
            Area to keep, e.g. ``BBox.around(lat, lon, radius_km=20)`` around a
            hydrophone. Without it, a day keeps millions of rows.
        mmsi : list[str], optional
            Keep only these vessels.
        dest_dir : str, default "downloads/ais"
            Cache directory for the daily zips.
        workers : int, default 2
            Days downloaded and read in parallel. Keep it small: each day is a
            120 MB download from a public server.

        Returns
        -------
        pandas.DataFrame
            Reports sorted by ``mmsi`` then ``base_date_time``, columns as in
            :func:`read_ais_day`. Days with no file on the server are skipped
            with a warning.
        """
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        days = pd.date_range(start.normalize(), end.normalize(), freq="D")
        logger.info("Loading AIS for %d day(s), %s -> %s, bbox=%s", len(days), start, end, bbox)

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(days)))) as ex:
            one_day = partial(
                self._load_ais_one_day, dest_dir=dest_dir, bbox=bbox, start=start, end=end, mmsi=mmsi
            )
            frames = [f for f in ex.map(one_day, days) if f is not None]
        if not frames:
            raise ValueError(f"No AIS data between {start} and {end}")
        df = pd.concat(frames, ignore_index=True).sort_values(["mmsi", "base_date_time"], ignore_index=True)
        logger.info("AIS: %d reports from %d vessels", len(df), df["mmsi"].nunique())
        return df
