import os
import numpy as np
import pandas as pd
from fractions import Fraction
import matplotlib.pyplot as plt
from scipy import signal

from audio_files import AudioFile, logger

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

def extract_spectral_distribution(
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
        " %s  fs=%s  %d segment(s)  %d minutes",
        os.path.basename(audio_file.filename),
        audio_file.audio_data.sample_rate,
        len(audio_file.audio_data.segments),
        n_used,
    )
    return times, freqs, rows


def extract_spectrum_multiple_files(
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
        tms, frqs, rws = extract_spectral_distribution(
            audio_file=p,
            t0=t0,
            t1=t1,
            seconds_per_minute=seconds_per_minute,
            fmax=fmax,
            tf=tf,
            gain_db=gain_db,
        )
        p.close()  # free the GCS stream; read() reopens it if needed later
        logger.info("  [%d/%d]", i + 1, len(audio_files))
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


def band_levels(freqs, psd_db, bands)->dict[str, float]:
    p = 10 ** (psd_db.astype(np.float64) / 10)
    df = np.gradient(freqs)
    out = {}
    for name, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi)
        out[name] = 10 * np.log10((p[..., m] * df[m]).sum(axis=-1) + 1e-20) if m.any() else np.nan
    return out


def standard_bands(fmax) -> dict[str, tuple[float, float]]:
    b = {"broadband_20_500Hz": (20, min(500, fmax))}
    for fc in (63, 125, 250, 500):           # MSFD ship-noise third-octave bands
        if fc * 2 ** (1 / 6) <= fmax:
            b[f"tol_{fc}Hz"] = (fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6))
    return b


def plot_spectral(freqs: list[float], psd: np.array, calibrated:bool):
    _, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(freqs, psd)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB re 1 µPa²/Hz)" if calibrated else "PSD (dB, uncalibrated)")
    ax.set_title("Spectrum")
    ax.grid(True, which="both", alpha=0.3)
    plt.show()


def plot_bands_spectrum(freqs: list[float], psd: np.array, bands_def: dict, bands_levels: dict):
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
    P = np.vstack(psds)                      # (n_times, n_freqs)
    bt = band_levels(freqs=freqs, psd_db=P, bands=bands_def)   # each value is an array of shape (n_times,)

    df_bands = pd.DataFrame(bt, index=pd.to_datetime(times))

    ax = df_bands.plot(figsize=(10, 4), lw=1)
    if instant is not None:
        if instant > len(times):
            raise ValueError(f"instant {instant} not in times (len={len(times)})")
        ax.axvline(df_bands.index[instant], color="k", ls="--", lw=0.8)  # marks the instant from plot 1
    ax.set_ylabel("Band level (dB, uncalibrated)")
    ax.grid(True, alpha=0.3)
    plt.show()