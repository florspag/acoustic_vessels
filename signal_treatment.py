"""Spectral analysis of underwater recordings for ship-noise monitoring.

Per-minute Welch power spectral densities, calibration, integration over the
ship-noise frequency bands, plots, and detection of loud events.
"""
import os
import time
import threading
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from audio_files import AudioFile, logger
from gc_files import GCStore, Store

if TYPE_CHECKING:
    from recording import Catalog

def _decimator(fs_in: int, fs_out: int):
    """Build a resampler from ``fs_in`` down to about ``fs_out`` Hz.

    Returns ``(fs_eff, fn)``: the exact output rate and a function applying a
    polyphase resample. If ``fs_in <= fs_out`` the signal is left unchanged.
    """
    if fs_in <= fs_out:
        return fs_in, (lambda x: x)
    r = Fraction(fs_out, fs_in).limit_denominator(2000)
    fs_eff = fs_in * r.numerator / r.denominator
    return fs_eff, (lambda x: signal.resample_poly(x, r.numerator, r.denominator))


def load_calibration(tf_filepath:str, freqs:np.ndarray[float], gain_db:float):
    """Build the dB correction to add to a PSD.

    Parameters
    ----------
    tf_filepath : str or None
        Local text file with two columns: frequency (Hz) and correction (dB).
        Lines starting with ``#`` or ``%`` are ignored. Interpolated onto
        ``freqs``.
    freqs : numpy.ndarray
        Frequencies of the PSD bins, in Hz.
    gain_db : float
        Constant offset added at all frequencies.

    Returns
    -------
    numpy.ndarray
        Correction in dB, same shape as ``freqs``.
    """
    cal = np.full(freqs.shape, float(gain_db or 0.0))
    if tf_filepath:
        tf = np.loadtxt(tf_filepath, comments=("#", "%"), delimiter=None, usecols=(0, 1))
        cal += np.interp(freqs, tf[:, 0], tf[:, 1])
    return cal

def extract_spectral_distribution(
    audio_file: AudioFile,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    seconds_per_minute: float = 10.0,
    fmax: float = 1000.0,
    tf_filepath:str=None,
    gain_db: float = 0.0,
) -> tuple[list[pd.Timestamp], np.ndarray[float], list[np.ndarray[float]]]:
    """Compute one power spectral density per minute of a single file.

    At each minute in ``[t0, t1]`` (counted from the start of each segment),
    the first ``seconds_per_minute`` seconds are read, the DC offset is
    removed, the signal is decimated to ``2 * fmax`` and a Welch PSD is
    computed (1 s Hann window, 50 % overlap, so 1 Hz bins). Readings shorter
    than 2 s are skipped.

    Parameters
    ----------
    audio_file : AudioFile
        File to analyse.
    t0, t1 : pandas.Timestamp
        Time window.
    seconds_per_minute : float, default 10.0
        Seconds of audio analysed at the start of each minute.
    fmax : float, default 1000.0
        Highest frequency kept, in Hz.
    tf_filepath : str, optional
        Local calibration file, see :func:`load_calibration`, e.g.
        ``Catalog.calibration_filepath(audio_file.filename)``.
    gain_db : float, default 0.0
        Constant calibration offset, in dB.

    Returns
    -------
    times : list[pandas.Timestamp]
        Start time of each analysed minute.
    freqs : numpy.ndarray or None
        Bin frequencies from 1 Hz to ``fmax``; ``None`` if nothing was analysed.
    rows : list[numpy.ndarray]
        One PSD per minute, in dB, aligned with ``freqs``. Calibrated
        (dB re 1 µPa²/Hz) if ``tf_filepath`` or ``gain_db`` is given, else
        relative.
    """

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
        " %s  fs=%s  %d segment(s)  %d minutes",
        os.path.basename(audio_file.filename),
        audio_file.audio_data.sample_rate,
        len(audio_file.audio_data.segments),
        n_used,
    )
    if rows and (tf_filepath or gain_db):
        cal = load_calibration(tf_filepath, freqs, gain_db)
        rows = [r + cal for r in rows]
    return times, freqs, rows


def extract_spectrum_multiple_files(
    audio_files: list[AudioFile | str],
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    store: Optional[Store] = None,
    seconds_per_minute: float = 10.0,
    fmax: float = 1000.0,
    tf=None,
    gain_db: float = 0.0,
    workers: int = 1,
    catalog: Optional["Catalog"] = None,
    deployed_only: bool = True,
) -> tuple[np.ndarray[pd.Timestamp], np.ndarray[float], np.ndarray[np.ndarray[float]], bool]:
    """Compute per-minute spectra over several files, merged and sorted by time.

    Runs :func:`extract_spectral_distribution` on each file, calibrates it,
    checks that all files give the same frequency grid, and merges the
    results into one time series.

    With a ``catalog``, each file is calibrated with the transfer function of
    its own deployment, and (with ``deployed_only``) its window is trimmed to
    the time the instrument was on the seafloor, so files from several
    deployments can be mixed.

    Parameters
    ----------
    audio_files : list[AudioFile or str]
        Files to analyse, as ``AudioFile`` objects or blob names. Blob names
        are opened with ``store`` and closed as soon as the file is done;
        prefer them, so streams are not held open in advance.
    t0, t1 : pandas.Timestamp
        Time window.
    store : Store, optional
        :class:`GCStore` or :class:`LocalStore`; required when
        ``audio_files`` contains blob names.
    seconds_per_minute : float, default 10.0
        Seconds of audio analysed at the start of each minute.
    fmax : float, default 1000.0
        Highest frequency kept, in Hz.
    tf : str, optional
        Calibration file for all files, see :func:`load_calibration`. Not
        allowed with ``catalog``, which gives each file its own.
    gain_db : float, default 0.0
        Constant calibration offset, in dB, added to every file.
    workers : int, default 1
        Number of files processed in parallel threads. At most ``workers``
        GCS streams are open at once, and each thread gets its own GCS client.
    catalog : recording.Catalog, optional
        Gives each file's deployment metadata and calibration file. Its store
        is used when ``store`` is not given.
    deployed_only : bool, default True
        With ``catalog``: analyse only between deployment and recovery time,
        skipping deck tests. Ignored without ``catalog``.

    Returns
    -------
    times : numpy.ndarray of datetime64[ns]
        Start time of each analysed minute, sorted.
    freqs : numpy.ndarray of float32
        Bin frequencies, in Hz.
    psd : numpy.ndarray of float32
        PSD in dB, shape ``(n_times, n_freqs)``.
    calibrated : bool
        ``True`` if every file was calibrated (``tf``, ``gain_db`` or the
        catalog's transfer functions).

    Raises
    ------
    ValueError
        If blob names are given without ``store`` or ``catalog``, if both
        ``tf`` and ``catalog`` are given, if files give different frequency
        grids (sample rate below ``2 * fmax``), if calibrated and uncalibrated
        files would be mixed, or if no file has audio in the window.

    Examples
    --------
    >>> catalog = Catalog(LocalStore("downloads"))
    >>> times, freqs, psd, calibrated = extract_spectrum_multiple_files(
    ...     files, t0, t1, catalog=catalog, workers=4)
    """
    n = len(audio_files)
    names = [os.path.basename(p if isinstance(p, str) else p.filename) for p in audio_files]
    if catalog is not None:
        if tf is not None:
            raise ValueError("pass either `tf` or `catalog`, not both: the catalog gives each file its own")
        store = store or catalog.store
        # Load metadata and calibration once per deployment, here, before threads start
        for p in audio_files:
            catalog.calibration_filepath(p if isinstance(p, str) else p.filename)
    if store is None and any(isinstance(p, str) for p in audio_files):
        raise ValueError("pass `store` or `catalog` when audio_files contains paths")
    workers = max(1, min(workers, n))
    t_start = time.perf_counter()
    logger.info(
        "Extracting spectra from %d file(s) between %s and %s (%.0f s/min, fmax=%.0f Hz, %d worker(s))",
        n, t0, t1, seconds_per_minute, fmax, workers,
    )

    local = threading.local()

    def thread_store():
        # google-cloud-storage clients are not guaranteed thread-safe: one per thread
        if workers == 1 or not isinstance(store, GCStore):
            return store  # LocalStore opens plain files: safe to share
        if not hasattr(local, "store"):
            local.store = GCStore(bucket=store.bucket.name)
        return local.store

    def file_settings(audio_file: AudioFile | str)->tuple[pd.Timestamp, pd.Timestamp, str|None]:
        """(t0, t1, tf) for one file: the catalog's deployment window and calibration."""
        if catalog is None:
            return t0, t1, tf
        blob = audio_file if isinstance(audio_file, str) else audio_file.filename
        start, end = t0, t1
        if deployed_only:
            meta = catalog.metadata(blob)
            if meta.deployment_time is not None:
                start = max(start, meta.deployment_time)
            if meta.recovery_time is not None:
                end = min(end, meta.recovery_time)
        return start, end, catalog.calibration_filepath(blob)

    def process(i: int, audio_file: AudioFile | str)->tuple[str, np.ndarray[pd.Timestamp], np.ndarray[float], list[np.ndarray[float]], str|None ]:
        name = names[i]
        start, end, file_tf = file_settings(audio_file)
        file_cal = bool(file_tf) or bool(gain_db)
        if start > end:
            logger.info("[%d/%d] %s: outside deployment period, skipped", i + 1, n, name)
            if not isinstance(audio_file, str):
                audio_file.close()
            return name, [], None, [], file_cal
        t_file = time.perf_counter()
        logger.debug("[%d/%d] opening %s", i + 1, n, name)
        try:
            if isinstance(audio_file, str):
                audio_file = AudioFile(filename=audio_file, store=thread_store())
            try:
                times, freqs, psds = extract_spectral_distribution(
                    audio_file=audio_file,
                    t0=start,
                    t1=end,
                    seconds_per_minute=seconds_per_minute,
                    fmax=fmax,
                    tf_filepath=file_tf,
                    gain_db=gain_db,
                )
            finally:
                audio_file.close()  # free the GCS stream; read() reopens it if needed later
        except Exception:
            logger.exception("[%d/%d] %s: extraction failed", i + 1, n, name)
            raise
        elapsed = time.perf_counter() - t_file
        if not psds:
            logger.warning("[%d/%d] %s: no audio in window, skipped (%.1f s)", i + 1, n, name, elapsed)
        else:
            logger.info(
                "[%d/%d] %s: %d minute(s), %s -> %s (%.1f s)",
                i + 1, n, name, len(psds), times[0], times[-1], elapsed,
            )
        return name, times, freqs, psds, file_cal

    if workers == 1:
        results = [process(i, p) for i, p in enumerate(audio_files)]
    else:
        ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="spectrum")
        try:
            results = list(ex.map(process, range(n), audio_files))  # keeps input order
        except BaseException:
            ex.shutdown(wait=True, cancel_futures=True)  # don't start remaining files
            raise
        ex.shutdown(wait=True)

    times: list[pd.Timestamp] = []
    rows: list[np.ndarray] = []
    freqs = None
    skipped: list[str] = []
    cal_flags: set[bool] = set()
    for name, tms, frqs, rws, file_cal in results:
        if not rws:
            skipped.append(name)
            continue
        cal_flags.add(file_cal)
        if freqs is None:
            freqs = frqs
        elif len(frqs) != len(freqs) or not np.allclose(frqs, freqs):
            raise ValueError(
                f"{name}: frequency grid differs from previous files "
                f"({len(frqs)} vs {len(freqs)} bins); sample rate below 2*fmax?"
            )
        times += tms
        rows += rws
    if not rows:
        logger.error("No audio between %s and %s in %d file(s)", t0, t1, n)
        raise ValueError(f"No audio between {t0} and {t1} in {n} file(s).")
    order = np.argsort(np.array(times, dtype="datetime64[ns]"))
    t = np.array(times, dtype="datetime64[ns]")[order]
    psd = np.vstack(rows)[order].astype(np.float32)

    if len(cal_flags) > 1:
        raise ValueError(
            "some files are calibrated and others are not (a deployment has no "
            "calibration file?); their levels cannot be compared"
        )
    calibrated = cal_flags == {True}
    logger.info(
        "Done: %d spectra x %d bins from %d/%d file(s)%s, %s, %.1f s total",
        psd.shape[0], psd.shape[1], n - len(skipped), n,
        f" ({len(skipped)} skipped)" if skipped else "",
        "calibrated" if calibrated else "uncalibrated",
        time.perf_counter() - t_start,
    )
    return t, freqs.astype(np.float32), psd, calibrated


def band_levels(freqs, psd_db, bands)->dict[str, float]:
    """Integrate a PSD over frequency bands.

    Power is summed over each band in linear units, weighted by bin width,
    then converted back to dB.

    Parameters
    ----------
    freqs : numpy.ndarray
        Bin frequencies, in Hz.
    psd_db : numpy.ndarray
        PSD in dB, shape ``(n_freqs,)`` or ``(n_times, n_freqs)``.
    bands : dict[str, tuple[float, float]]
        Band name -> ``(low, high)`` in Hz; ``low`` is included, ``high``
        excluded. See :func:`standard_bands`.

    Returns
    -------
    dict[str, float or numpy.ndarray]
        Band level in dB per band: a float for a single spectrum, an array of
        shape ``(n_times,)`` for a 2D input. ``nan`` if no bin is in the band.
    """
    p = 10 ** (psd_db.astype(np.float64) / 10)
    df = np.gradient(freqs)
    out = {}
    for name, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi)
        out[name] = 10 * np.log10((p[..., m] * df[m]).sum(axis=-1) + 1e-20) if m.any() else np.nan
    return out


def standard_bands(fmax) -> dict[str, tuple[float, float]]:
    """Frequency bands used to monitor ship noise.

    Returns a broadband 20–500 Hz band plus the third-octave bands centred on
    63, 125, 250 and 500 Hz (EU MSFD ship-noise indicators). Bands above
    ``fmax`` are left out and the broadband band is capped at ``fmax``.

    Parameters
    ----------
    fmax : float
        Highest frequency available, in Hz (usually ``freqs[-1]``).

    Returns
    -------
    dict[str, tuple[float, float]]
        Band name -> ``(low, high)`` in Hz, e.g. ``"tol_63Hz"``.
    """
    b = {"broadband_20_500Hz": (20, min(500, fmax))}
    for fc in (63, 125, 250, 500):           # MSFD ship-noise third-octave bands
        if fc * 2 ** (1 / 6) <= fmax:
            b[f"tol_{fc}Hz"] = (fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6))
    return b


def plot_spectral(freqs: list[float], psd: np.array, calibrated:bool):
    """Plot one spectrum on a logarithmic frequency axis.

    Parameters
    ----------
    freqs : array-like
        Bin frequencies, in Hz.
    psd : numpy.ndarray
        PSD in dB, same length as ``freqs``.
    calibrated : bool
        Label the y-axis in dB re 1 µPa²/Hz if ``True``, else as uncalibrated.
    """
    _, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(freqs, psd)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB re 1 µPa²/Hz)" if calibrated else "PSD (dB, uncalibrated)")
    ax.set_title("Spectrum")
    ax.grid(True, which="both", alpha=0.3)
    plt.show()


def plot_bands_spectrum(freqs: list[float], psd: np.array, bands_def: dict, bands_levels: dict):
    """Plot one spectrum with the third-octave bands shaded and labelled.

    The broadband band is not drawn, as it would cover the others.

    Parameters
    ----------
    freqs : array-like
        Bin frequencies, in Hz.
    psd : numpy.ndarray
        PSD in dB, same length as ``freqs``.
    bands_def : dict[str, tuple[float, float]]
        Bands to draw, from :func:`standard_bands`.
    bands_levels : dict[str, float]
        Level of each band for this spectrum, from :func:`band_levels`;
        shown in each band's label.
    """
    _, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(freqs, psd, color="0.3", lw=1, label="PSD")
    for i, (name, (lo, hi)) in enumerate(bands_def.items()):
        if name.startswith("broadband"):
            continue  # it would hide the third-octave bands
        c = f"C{i}"
        ax.axvspan(lo, hi, color=c, alpha=0.2)
        ax.text(np.sqrt(lo * hi), ax.get_ylim()[1], f"{name}\n{bands_levels[name]:.1f} dB",
                ha="center", va="top", fontsize=8, color=c)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB, uncalibrated)")
    ax.grid(True, which="both", alpha=0.3)
    plt.show()


def plot_bands_timeserie(times: list[pd.Timestamp], freqs: list[float], psds: list[np.array], bands_def: dict, instant: Optional[int]=None):
    """Plot the level of each band over time.

    Parameters
    ----------
    times : array-like of pandas.Timestamp
        Time of each spectrum.
    freqs : array-like
        Bin frequencies, in Hz.
    psds : list[numpy.ndarray] or numpy.ndarray
        Spectra in dB, one per time, or a ``(n_times, n_freqs)`` array.
    bands_def : dict[str, tuple[float, float]]
        Bands to plot, from :func:`standard_bands`.
    instant : int, optional
        Index of a time to mark with a dashed vertical line.

    Raises
    ------
    ValueError
        If ``instant`` is past the end of ``times``.
    """
    P = np.vstack(psds)                      # (n_times, n_freqs)
    bt = band_levels(freqs=freqs, psd_db=P, bands=bands_def)   # each value is an array of shape (n_times,)

    df_bands = pd.DataFrame(bt, index=pd.to_datetime(times))

    ax = df_bands.plot(figsize=(10, 4), lw=1)
    if instant is not None:
        if not -len(times) <= instant < len(times):
            raise ValueError(f"instant {instant} not in times (len={len(times)})")
        ax.axvline(df_bands.index[instant], color="k", ls="--", lw=0.8)  # marks the instant from plot 1
    ax.set_ylabel("Band level (dB, uncalibrated)")
    ax.grid(True, alpha=0.3)
    plt.show()

def acoustic_events(times: list[pd.Timestamp], band_level:float, baseline_hours=6, thresh_db=6.0, min_minutes=3)-> tuple[pd.DataFrame, np.ndarray[float],np.ndarray[float] ]:
    """Detect loud events, such as passing ships, in a band-level time series.

    The baseline is a centred rolling median over ``baseline_hours``. An event
    is a run of points at least ``thresh_db`` above the baseline, with no gap
    longer than 5 minutes between consecutive points, lasting at least
    ``min_minutes`` (counted from the first to the last point's start time,
    plus one minute).

    Parameters
    ----------
    times : array-like of pandas.Timestamp
        Time of each value, sorted.
    band_level : array-like
        Band level in dB at each time, e.g. ``band_levels(...)["tol_63Hz"]``.
    baseline_hours : float, default 6
        Width of the rolling-median window.
    thresh_db : float, default 6.0
        Minimum excess over the baseline, in dB.
    min_minutes : int, default 3
        Minimum event duration, in minutes.

    Returns
    -------
    events : pandas.DataFrame
        One row per event, with columns ``start``, ``end``, ``peak_time``,
        ``peak_level`` (dB) and ``peak_excess`` (dB over baseline).
        Empty (no columns) if nothing is detected.
    baseline : numpy.ndarray
        Rolling-median baseline at each time, in dB.
    excess : numpy.ndarray
        ``band_level - baseline`` at each time, in dB.
    """
    s = pd.Series(band_level, index=pd.DatetimeIndex(times))
    win = f"{int(baseline_hours * 60)}min"
    base = s.rolling(win, center=True, min_periods=5).median()
    excess = s - base
    above = (excess >= thresh_db).values
    ev, i, n = [], 0, len(s)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1] and (s.index[j + 1] - s.index[j]) <= pd.Timedelta(minutes=5):
                j += 1
            if (s.index[j] - s.index[i]) >= pd.Timedelta(minutes=min_minutes - 1):
                seg = excess.iloc[i:j + 1]
                k = seg.values.argmax()
                ev.append(dict(start=s.index[i], end=s.index[j], peak_time=seg.index[k],
                               peak_level=s.iloc[i + k], peak_excess=seg.iloc[k]))
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(ev), base.values, excess.values