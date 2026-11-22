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
| `gc_files.py` | `GCStore` (the GCS bucket) and `LocalStore` (a downloaded copy) list, filter and open files. Both follow the `Store` protocol, so the rest of the code works on either. Also the `AudioExtenstion` enum and parsing of timestamps from file names (ISO, HARP and SoundTrap styles). |
| `audio_files.py` | `AudioFile`: random-access reader over a GCS stream. It parses WAV/RIFF headers and Triton XWAV `harp` chunks (duty-cycled segments, site metadata), so a read fetches only the bytes it needs. FLAC, AIFF and other formats go through `soundfile`. |
| `signal_treatment.py` | Spectral analysis: decimation, Welch PSD, calibration, band levels, plots, and detection of loud events. |
| `deployment_metadata.py` | `DeploymentMetadata`: each deployment's metadata JSON (position, depth, deployment and recovery times, sample rate, duty cycle, data-quality periods), plus its calibration files. `download_with_metadata` downloads audio files together with their metadata and calibration. |
| `recording.py` | `Catalog`: opens audio files from one store together with their deployment's metadata and calibration file, cached per deployment. `Recording` is one such bundle. |
| `ais_data.py` | `AIS`: downloads and filters daily AIS files from MarineCadastre.gov. `AISData` is one position report, `BBox` a search area. |
| `vessel.ipynb` | Worked example from start to finish. |
| `downloads/` | Local copy of the bucket (audio, metadata, calibration, keeping bucket paths) for `LocalStore`, and the AIS cache in `downloads/ais/`. Not tracked in git. |

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

### Several files, calibrated, in parallel

```python
from recording import Catalog
from signal_treatment import extract_spectrum_multiple_files

catalog = Catalog(store)    # or Catalog(LocalStore("downloads")), see "Working offline"
times, freqs, psd, calibrated = extract_spectrum_multiple_files(
    audio_files=files,      # bucket paths, opened and closed one at a time
    t0=t0, t1=t1,
    catalog=catalog,        # metadata and calibration of each file's deployment
    workers=8,              # files processed in parallel threads
)
```

- **Output**: one time series for all files, merged and sorted by time. `psd` has shape `(n_times, n_freqs)`.
- **Calibration**: each file is calibrated with its own deployment's transfer function, so the list can span several deployments. If some deployments have a calibration file and others don't, the function raises a `ValueError` rather than mix absolute and relative levels.
- **Deck tests**: with `deployed_only=True` (the default), each file's window is trimmed to the time between deployment and recovery.
- **Without a catalog**: pass `store=store`, plus optionally `tf=` for one calibration file applied to all files. `tf` and `catalog` can't be used together.
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

### Deployment metadata

Each deployment folder in the bucket has an `audio/` folder, an `ancillary/` folder (calibration files, LTSAs) and a `metadata/` folder with one JSON file:

```text
sefsc/audio/gomex/ef/sefsc_gomex_ef_01/
    audio/GOM_EF_01_160716_170502.x.flac
    ancillary/calibration/828_160707_B_HARP.tf
    metadata/SEFSC_GOMx_EF_01.json
```

```python
from deployment_metadata import DeploymentMetadata

meta = DeploymentMetadata.for_audio_file(store, files[0])  # finds <deployment>/metadata/*.json
meta.latitude, meta.longitude      # 27.73308, -92.95133
meta.instrument_depth              # 241.0 (metres, positive downwards)
meta.sample_rate_hz                # 2000.0
meta.is_deployed(t)                # False for deck tests before deployment or after recovery
meta.quality_at(t, freq_hz=63)     # [QualityPeriod(quality='Good', low_freq=50, high_freq=600, ...)]
```

`DeploymentMetadata.from_store(store, path)` loads a given JSON instead. Numbers and times stored as text in the JSON are converted, and the sample rate goes from kHz to Hz. The full JSON stays in `meta.raw`, for fields that aren't mapped (abstract, people, ...).

Before analysing a deployment, check two things:

- **Deck tests.** Audio can start days before the instrument reaches the seafloor. For EF_01, audio starts on 2016-07-16 but the deployment is on 2016-07-21 06:00, so the first file was recorded on the ship. Skip any time where `meta.is_deployed(t)` is `False`.
- **Quality by frequency.** `meta.quality` lists periods and frequency bands rated `Good`, `Unverified` or `Unusable`. For EF_01, calibration is only `Good` from 50 to 600 Hz. The third-octave bands at 63–500 Hz are fine, but the broadband 20–500 Hz band starts in the unverified range.

### Working offline

`download_with_metadata` downloads audio files together with each deployment's metadata JSON and calibration files. It keeps their bucket paths under `dest_dir`, so `LocalStore` reads the copy exactly as `GCStore` reads the bucket:

```python
from gc_files import LocalStore
from deployment_metadata import download_with_metadata

download_with_metadata(store, files, dest_dir="downloads", workers=4)   # skips files already there

catalog = Catalog(LocalStore("downloads"))   # same code as above, no network needed
times, freqs, psd, calibrated = extract_spectrum_multiple_files(files, t0, t1, catalog=catalog, workers=4)
```

```text
downloads/sefsc/audio/gomex/ef/sefsc_gomex_ef_01/
    audio/GOM_EF_01_160723_162002.x.flac
    metadata/SEFSC_GOMx_EF_01.json
    ancillary/calibration/828_160707_B_HARP.tf
```

Each audio file is 250–300 MB, so a whole EF_01 deployment is about 50 GB. Filter by date before downloading. Downloads go to a `.part` file that is renamed once complete, so an interrupted download is never taken for a complete file.

`GCStore`, `LocalStore` and anything else with the same methods (`open_file`, `read_json`, `list_files`, `list_folders`, `download`) satisfy the `Store` protocol in `gc_files.py`. A small fake store for tests counts too.

### One recording at a time

`Catalog.recordings` yields a `Recording` per file: the `AudioFile`, its `DeploymentMetadata`, and the local path of its calibration file.

```python
for rec in catalog.recordings(files, deployed_only=True):   # skips files recorded entirely on deck
    with rec:                                                # closes the audio stream
        print(rec.audio.name, rec.deployed_range, rec.meta.site, rec.tf)
```

Use it to inspect files or to run a custom analysis on each one. To get spectra over time, use `extract_spectrum_multiple_files(..., catalog=catalog)` rather than looping, so the results are merged. For a single file, the calibration path is `catalog.calibration_filepath(filename)`, which you pass to `extract_spectral_distribution(..., tf_filepath=...)`.

### AIS vessel positions

```python
from ais_data import AIS, AISData

box = meta.bbox(radius_km=20)                          # around the hydrophone
ais = AIS().load_ais(t0, t1, bbox=box, workers=2)      # DataFrame, sorted by mmsi then time
records = AISData.from_frame(ais)                      # list[AISData], if you want objects
```

Without metadata, build the box by hand with `BBox.around(lat, lon, radius_km)` from `ais_data`.

`load_ais` downloads every day the window touches. Each day is one zip of about 120 MB for all U.S. waters, cached in `downloads/ais/`. The CSV is then read in chunks of a million rows, keeping only reports inside the box and the time window, so a whole day (700 MB unzipped) never has to fit in memory.

Columns are the `AISData` fields: the MarineCadastre column names in snake_case (`BaseDateTime` → `base_date_time`, `LAT` → `latitude`, …).

## How the spectra are computed

- **Duty cycle**: the code reads `seconds_per_minute` seconds (10 s by default) at the start of each minute in `[t0, t1]`. A reading shorter than 2 s is skipped.
- **Decimation**: the signal is resampled to `2 × fmax` with `scipy.signal.resample_poly` (`fmax` defaults to 1000 Hz).
- **PSD**: Welch's method with a 1 s Hann window and 50 % overlap, which gives 1 Hz bins. The DC offset is removed first. Only bins from 1 Hz to `fmax` are kept, and the result is in dB.
- **Bands** (`standard_bands`): one broadband 20–500 Hz band, plus the third-octave bands centred on 63, 125, 250 and 500 Hz. These are the ship-noise indicators of the EU Marine Strategy Framework Directive (MSFD). `band_levels` integrates linear power over each band and returns dB. It accepts one spectrum or a 2D `(time, freq)` array.

These settings suit noise monitoring. Telling vessel types apart needs more: continuous audio at the full sample rate around each passage, and frequencies well above 1 kHz, where propeller cavitation is. Check `meta.sample_rate_hz` first. EF_01 is recorded at 2 kHz, so 1 kHz is the highest frequency it holds.

## Calibration

Without calibration, levels are relative: dB re 1 ADC count²/Hz. Absolute levels in dB re 1 µPa²/Hz come from a transfer function: a two-column text file (frequency in Hz, correction in dB) interpolated onto the PSD frequencies.

Each deployment's transfer functions are in its `ancillary/calibration/` folder, named after the hydrophone preamplifier (`meta.preamp_ids`), e.g. `828_160707_B_HARP.tf`. There are three ways to apply one:

| How | Calibration used |
| --- | --- |
| `extract_spectrum_multiple_files(files, t0, t1, catalog=catalog)` | Each file's own deployment. Recommended. |
| `extract_spectrum_multiple_files(files, t0, t1, store=store, tf=path)` | One local `.tf` file for all files |
| `extract_spectral_distribution(audio, t0, t1, tf_filepath=path)` | One file |

`Catalog.calibration_filepath(filename)` returns the local path of the right `.tf` file. With a `GCStore`, it first downloads the file into `downloads/`. A constant `gain_db` offset can be added in all three cases.

For EF_01, the transfer function adds about +70 dB between 63 and 500 Hz. Check `meta.quality` for the bands where the calibration is valid.

## AIS data notes

- **Times are UTC**, like the SEFSC recordings (the deployment metadata says so), so AIS reports and acoustic events can be compared directly.
- **"Not available" codes.** AIS (ITU-R M.1371) encodes every field as a fixed-size integer, so a missing value is sent as a reserved code instead of a blank. `load_ais` turns these into NaN:

  | Field | Valid range | "Not available" in the source |
  | --- | --- | --- |
  | `heading` | 0–359° | 511. This is common: many small vessels have no gyro compass connected to the transponder. |
  | `cog` | 0–359.9° | 360 |
  | `sog` | 0–102.2 kn (102.2 means 102.2 kn or more) | 102.3 |

  Latitude 91° and longitude 181° also mean "not available". They are kept as they are, but any bounding box drops them.
- **Vessel type codes.** Files up to 2017 often use MarineCadastre's own codes above 1000 (e.g. 1004 = freight ship). Later files use the standard AIS codes: 30 fishing, 52 tug, 60–69 passenger, 70–79 cargo, 80–89 tanker, … See the vessel type code table on MarineCadastre.gov.

## Logging

Each module logs to its own logger: `audio_files` (also used by `signal_treatment`), `gc_files` (downloads), `deployment_metadata`, `recording` and `ais_data`. To see progress in a notebook:

```python
import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
logging.getLogger("audio_files").setLevel(logging.INFO)
for name in ("gc_files", "deployment_metadata", "recording"):
    logging.getLogger(name).setLevel(logging.INFO)
logging.getLogger("ais_data").setLevel(logging.INFO)
```

At INFO you get one summary line per audio file, per-file progress and timing for multi-file runs, and download and filtering progress for AIS. DEBUG adds stream open and close events, RIFF chunks and every read, which is noisy.

## Data

- **Audio**: the NOAA Passive Bioacoustic archive on GCS, bucket `noaa-passive-bioacoustic`. The `sefsc/` prefix holds the Southeast Fisheries Science Center recordings, for example the HARP deployments `GOM_EF_01` and `GOM_WF_01` in the Gulf of Mexico. The Gulf of Mexico sites are under `sefsc/audio/gomex/` (`ef`, `ei`, `ep`, `gi`, `wf`), with one folder per deployment.
- **AIS**: MarineCadastre.gov (NOAA / BOEM), daily files for U.S. waters, at `https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{mm}_{dd}.zip`.
