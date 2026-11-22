# acoustic_vessels

Tools to measure underwater ship noise from passive acoustic recordings. The code streams audio files from NOAA's public bioacoustic bucket on Google Cloud Storage. For each minute of recording it computes a power spectral density (PSD), then integrates the PSD into the frequency bands used to monitor vessel noise.

## Install

Requires Python 3.11 or later, because `StrEnum` arrived in 3.11.

```bash
conda create -n acoustic python=3.11
conda activate acoustic
pip install numpy pandas scipy matplotlib soundfile google-cloud-storage jupyter
```

The bucket is public, so `GCStore` uses an anonymous client and needs no credentials.

## Project layout

| File | Contents |
|---|---|
| `gc_files.py` | `GCStore`: lists, filters and streams files from a GCS bucket. Also the `AudioExtenstion` enum and parsing of timestamps from file names (ISO, HARP and SoundTrap styles). |
| `audio_files.py` | `AudioFile`: random-access reader over a GCS stream. It parses WAV/RIFF headers and Triton XWAV `harp` chunks (duty-cycled segments, site metadata), so a read fetches only the bytes it needs. FLAC, AIFF and other formats go through `soundfile`. |
| `signal_treatment.py` | Spectral analysis: decimation, Welch PSD, calibration, band levels and plots. |
| `vessel.ipynb` | Worked example from start to finish. |
| `downloads/` | Local copies of files fetched with `GCStore.download`. Not tracked in git. |

## Usage

```python
import pandas as pd
from gc_files import GCStore, AudioExtenstion
from audio_files import AudioFile
from signal_treatment import (
    extract_spectral_distribution, standard_bands, band_levels,
    plot_spectral, plot_bands_spectrum, plot_bands_timeserie,
)

store = GCStore(bucket="noaa-passive-bioacoustic")
files = store.list_files(prefix="sefsc")
files = GCStore.filter_files(
    files,
    extentions=[AudioExtenstion.FLAC],
    start_date=pd.Timestamp(2016, 7, 31),
    end_date=pd.Timestamp(2016, 8, 1),
)

aud = AudioFile(filename=files[0], store=store)
t0 = aud.timestamp
t1 = t0 + pd.Timedelta(hours=3)

times, freqs, psds = extract_spectral_distribution(aud, t0=t0, t1=t1)

# Spectrum at one instant, with the ship-noise bands overlaid
i = 80
bands_def = standard_bands(fmax=freqs[-1])
levels = band_levels(freqs, psds[i], bands_def)
plot_spectral(freqs, psds[i], calibrated=False)
plot_bands_spectrum(freqs, psds[i], bands_def, levels)

# Band levels over the whole window (a passing ship shows up as a rise in the 63/125 Hz bands)
plot_bands_timeserie(times, freqs, psds, bands_def, instant=i)
```

To process several files, use `extract_spectrum_multiple_files(audio_files, t0, t1, ...)`. It concatenates and sorts the results by time, applies calibration, and returns `(times, freqs, psd, calibrated)`.

## How the spectra are computed

- **Duty cycle**: the code reads `seconds_per_minute` seconds (10 s by default) at the start of each minute in `[t0, t1]`. A reading shorter than 2 s is skipped.
- **Decimation**: the signal is resampled to `2 × fmax` with `scipy.signal.resample_poly` (`fmax` defaults to 1000 Hz).
- **PSD**: Welch's method with a 1 s Hann window and 50 % overlap, which gives 1 Hz bins. The DC offset is removed first. Only bins from 1 Hz to `fmax` are kept, and the result is in dB.
- **Bands** (`standard_bands`): one broadband 20–500 Hz band, plus the third-octave bands centred on 63, 125, 250 and 500 Hz. These are the ship-noise indicators of the EU Marine Strategy Framework Directive (MSFD). `band_levels` integrates linear power over each band and returns dB. It accepts one spectrum or a 2D `(time, freq)` array.

## Calibration

Without calibration, levels are relative: dB re 1 ADC count²/Hz. To get absolute levels in dB re 1 µPa²/Hz, pass either or both of the following to `extract_spectrum_multiple_files`:

- `gain_db`: a constant offset for the whole system.
- `tf`: the path to a two-column text file (frequency in Hz, correction in dB). The correction is interpolated onto the PSD frequencies.

## Data

The data comes from the NOAA Passive Bioacoustic archive on GCS (bucket `noaa-passive-bioacoustic`). The `sefsc/` prefix holds the Southeast Fisheries Science Center recordings, for example the HARP deployments `GOM_EF_01` and `GOM_WF_01` in the Gulf of Mexico.
