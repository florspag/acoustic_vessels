# acoustic_vessels

Tools to measure underwater ship noise from passive acoustic recordings and to link it to the vessels that made it. The code streams audio files from NOAA's public bioacoustic bucket on Google Cloud Storage. For each minute of recording it computes a power spectral density (PSD) and integrates it into the frequency bands used to monitor vessel noise. It then detects loud passages in those band levels. Separately, it downloads AIS vessel positions from MarineCadastre.gov, so detections can be matched to known ships.

## Install

Requires Python 3.11 or later, because `StrEnum` arrived in 3.11.

```bash
conda create -n acoustic python=3.11
conda activate acoustic
pip install numpy pandas scipy matplotlib soundfile google-cloud-storage requests jupyter
```

No credentials are needed. The GCS bucket is public, so `GCStore` uses an anonymous client, and the AIS files are public downloads.

## Project layout

| File | Contents |
| --- | --- |
| `gc_files.py` | `GCStore`: lists, filters and streams files from a GCS bucket. Also the `AudioExtenstion` enum and parsing of timestamps from file names (ISO, HARP and SoundTrap styles). |
| `audio_files.py` | `AudioFile`: random-access reader over a GCS stream. It parses WAV/RIFF headers and Triton XWAV `harp` chunks (duty-cycled segments, site metadata), so a read fetches only the bytes it needs. FLAC, AIFF and other formats go through `soundfile`. |
| `signal_treatment.py` | Spectral analysis: decimation, Welch PSD, calibration, band levels, plots, and detection of loud events. |
| `ais_data.py` | `AIS`: downloads and filters daily AIS files from MarineCadastre.gov. `AISData` is one position report, `BBox` a search area. |
| `vessel.ipynb` | Worked example from start to finish. |
| `downloads/` | Local copies of audio files fetched with `GCStore.download`, and the AIS cache in `downloads/ais/`. Not tracked in git. |

## Usage

### Spectra of one file

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
files = GCStore.filter_files(          # sorted by the date in the file name
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

To keep only one site, filter on the base name, e.g. `fnmatch(os.path.basename(f), "GOM_EF*")`. To get the start and end of a file's audio, use `aud.time_range()`.

### Several files, in parallel

```python
from signal_treatment import extract_spectrum_multiple_files

times, freqs, psd, calibrated = extract_spectrum_multiple_files(
    audio_files=files,      # bucket paths, opened and closed one at a time
    t0=t0, t1=t1,
    store=store,
    workers=8,              # files processed in parallel threads
)
```

- **Output**: results are merged and sorted by time, then calibrated. `psd` has shape `(n_times, n_freqs)`.
- **Skipped files**: files with no audio in the window are skipped with a warning.
- **Frequency grids**: if the files give different grids (sample rate below `2 × fmax`), the function raises a `ValueError` that names the file.
- **Threads**: each thread uses its own GCS client, and at most `workers` streams are open at once. `workers=4` to `8` is a reasonable start.

`filter_files` compares each file's *start* time to the date range. A file that starts just before `t0` but runs past it is dropped, so set `start_date` one file length earlier than `t0`.

### Detecting passages

```python
from signal_treatment import acoustic_events

lvl = band_levels(freqs, psd, standard_bands(freqs[-1]))["tol_63Hz"]
events, baseline, excess = acoustic_events(times, lvl, baseline_hours=6, thresh_db=6.0, min_minutes=3)
```

The baseline is a centred rolling median of the band level over `baseline_hours`. An event is a run of minutes at least `thresh_db` above that baseline, with gaps of at most 5 minutes, lasting at least `min_minutes`. `events` has one row per event, with `start`, `end`, `peak_time`, `peak_level` and `peak_excess`.

### AIS vessel positions

```python
from ais_data import AIS, AISData, BBox

box = BBox.around(lat=27.5, lon=-85.4, radius_km=20)   # around the hydrophone
ais = AIS().load_ais(t0, t1, bbox=box, workers=2)      # DataFrame, sorted by mmsi then time
records = AISData.from_frame(ais)                      # list[AISData], if you want objects
```

`load_ais` downloads every day the window touches. Each day is one zip of about 120 MB for all U.S. waters, cached in `downloads/ais/`. The CSV is then read in chunks of a million rows, keeping only reports inside the box and the time window, so a whole day (700 MB unzipped) never has to fit in memory.

Columns are the `AISData` fields: the MarineCadastre column names in snake_case (`BaseDateTime` → `base_date_time`, `LAT` → `latitude`, …).

## How the spectra are computed

- **Duty cycle**: the code reads `seconds_per_minute` seconds (10 s by default) at the start of each minute in `[t0, t1]`. A reading shorter than 2 s is skipped.
- **Decimation**: the signal is resampled to `2 × fmax` with `scipy.signal.resample_poly` (`fmax` defaults to 1000 Hz).
- **PSD**: Welch's method with a 1 s Hann window and 50 % overlap, which gives 1 Hz bins. The DC offset is removed first. Only bins from 1 Hz to `fmax` are kept, and the result is in dB.
- **Bands** (`standard_bands`): one broadband 20–500 Hz band, plus the third-octave bands centred on 63, 125, 250 and 500 Hz. These are the ship-noise indicators of the EU Marine Strategy Framework Directive (MSFD). `band_levels` integrates linear power over each band and returns dB. It accepts one spectrum or a 2D `(time, freq)` array.

These settings suit noise monitoring. Telling vessel types apart needs more: continuous audio at the full sample rate around each passage, and frequencies well above 1 kHz, where propeller cavitation is.

## Calibration

Without calibration, levels are relative: dB re 1 ADC count²/Hz. To get absolute levels in dB re 1 µPa²/Hz, pass either or both of the following to `extract_spectral_distribution` or `extract_spectrum_multiple_files`:

- `gain_db`: a constant offset for the whole system.
- `tf`: the path to a two-column text file (frequency in Hz, correction in dB). The correction is interpolated onto the PSD frequencies.

## AIS data notes

- **Times are UTC.** Check that the audio file-name timestamps are UTC too before matching AIS reports to acoustic events. HARP files normally are.
- **"Not available" codes.** AIS (ITU-R M.1371) encodes every field as a fixed-size integer, so a missing value is sent as a reserved code instead of a blank. `load_ais` turns these into NaN:

  | Field | Valid range | "Not available" in the source |
  | --- | --- | --- |
  | `heading` | 0–359° | 511. This is common: many small vessels have no gyro compass connected to the transponder. |
  | `cog` | 0–359.9° | 360 |
  | `sog` | 0–102.2 kn (102.2 means 102.2 kn or more) | 102.3 |

  Latitude 91° and longitude 181° also mean "not available". They are kept as they are, but any bounding box drops them.
- **Vessel type codes.** Files up to 2017 often use MarineCadastre's own codes above 1000 (e.g. 1004 = freight ship). Later files use the standard AIS codes: 30 fishing, 52 tug, 60–69 passenger, 70–79 cargo, 80–89 tanker, … See the vessel type code table on MarineCadastre.gov.

## Logging

Each module logs to its own logger: `audio_files` (also used by `signal_treatment`) and `ais_data`. To see progress in a notebook:

```python
import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
logging.getLogger("audio_files").setLevel(logging.INFO)
logging.getLogger("ais_data").setLevel(logging.INFO)
```

At INFO you get one summary line per audio file, per-file progress and timing for multi-file runs, and download and filtering progress for AIS. DEBUG adds stream open and close events, RIFF chunks and every read, which is noisy.

## Data

- **Audio**: the NOAA Passive Bioacoustic archive on GCS, bucket `noaa-passive-bioacoustic`. The `sefsc/` prefix holds the Southeast Fisheries Science Center recordings, for example the HARP deployments `GOM_EF_01` and `GOM_WF_01` in the Gulf of Mexico.
- **AIS**: MarineCadastre.gov (NOAA / BOEM), daily files for U.S. waters, at `https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{mm}_{dd}.zip`.
