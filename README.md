# Monk — Bioimpedance Analysis

A Python toolkit for chicken tissue bioimpedance analysis and storage-age estimation, with an interactive results dashboard. The measurement format supports calibrated frequency sweeps for an **ESP32 + AD5933** instrument.

The included demo uses synthetic readings. Age-derived scores and Fresh / Moderate / Spoiled categories are illustrative; they do not establish food safety.

## Try the interface

Open `monk-demo.html` in a modern desktop or mobile browser. Select a sample, switch magnitude/phase charts, explore storage age, run a fictional sweep, or export simulated results. No Python or hardware is required. Internet access is optional for Google Fonts; local fallback fonts remain available.

The interface is independent of the Python models. Its example signals and predictions are hand-built illustrations, not trained-model outputs. Responsive layouts are implemented, but physical-phone visual verification is pending.

## Files

- `monk.py`: data validation, feature extraction, model training and prediction.
- `monk-demo.html`: interactive dashboard with synthetic readings.
- `tests/test_monk.py`: regression checks for measurements, calculations and prediction.

Generated datasets, models, charts and reports are written to `result_output/`.

## Setup

Use Python 3.11 or later. The checked environment used Python 3.14 with the exact dependency versions in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
# Fast first run: train Ridge on synthetic data, without charts.
python monk.py --full --models Monk-LR --no-plots

# Predict three new synthetic specimens with the saved model.
python monk.py --demo

# Compare all three models and create five evaluation charts.
python monk.py --full

# Train on an existing CSV without regenerating it.
python monk.py --train --csv measurements.csv

# Predict on an unknown specimen: elapsed_hours is optional.
python monk.py --predict unknown.csv

# Optional: separate output folder and parallel CV workers.
python monk.py --full --output-dir experiments/run-01 --jobs 2

# Regression checks.
python -m unittest discover -s tests -v
```

Running the script without a mode shows usage instead of regenerating data. Running `--full` deliberately regenerates the specified training CSV and outputs; use `--train` for collected measurements. Existing reports and model files are replaced when their commands run.

When using a custom output folder, `--demo` and `--predict` should receive the same `--output-dir`, or specify `--model` explicitly. Prediction results are written into the output folder, not alongside the input CSV. Only load your own or otherwise trusted `.joblib` files: that format uses Python pickle.

## Measurement CSV

Required columns:

- `sample_id`: specimen identifier; this groups the training/validation split.
- `temperature_c`: one sample temperature per sweep, repeated on its frequency rows.
- `freq_hz`: positive integer frequency in Hz.
- `magnitude`: calibrated impedance magnitude in ohms, greater than zero.
- `phase`: calibrated impedance phase in degrees, between -180 and 180.
- `elapsed_hours`: nonnegative known storage age, required only for training.

Recommended columns:

- `sweep_id`: measurement identifier within a specimen. Required to distinguish repeated measurements when `elapsed_hours` is absent or multiple sweeps share the same age.
- `data_source`: for example `synthetic` or `experimental`; retained in outputs.
- `real`, `imag`: optional calibrated impedance components; not currently used as prediction features.

```csv
sample_id,sweep_id,temperature_c,freq_hz,magnitude,phase,data_source
UNKNOWN-01,sweep-001,24.5,10000,400.0,-10.0,synthetic
UNKNOWN-01,sweep-001,24.5,20000,350.0,-15.0,synthetic
```

This two-point example demonstrates the schema only. Predictions must use **exactly the frequency set used to train the saved model**, normally the ten frequencies 10–100 kHz. All sweeps must be complete, contain no duplicate frequencies, and use consistent temperature and age metadata. The pipeline rejects ambiguous measurements instead of averaging them silently. Without `sweep_id`, training uses `(sample_id, elapsed_hours)` and unknown-age prediction assumes one sweep per specimen.

**Raw AD5933 real/imaginary register contents are not calibrated impedance values.** Convert and calibrate before supplying measurements. The actual tissue range depends on the electrodes and setup; the simulated few-hundred-ohm range does not establish the experimental range.

## How the pipeline works

1. Simulate 12 specimens at 12 ages with ten frequency points: 1,440 readings, 144 sweeps.
2. Extract magnitude/phase per frequency, log-frequency spectral slope, means, ranges and temperature: 26 features for the default sweep.
3. Compare Ridge regression, Random Forest and Gradient Boosting with leave-one-specimen-out cross-validation.
4. Report MAE, RMSE and R² against a grouped mean-age baseline.
5. Select the lowest-MAE model and fit it on all development data.
6. Predict age, then map it to an illustrative 0–100 index and age bins.

The simulator follows assumed Cole–Cole parameter changes; both generated training and demo data follow those assumptions. Their performance cannot demonstrate generalization to real tissue. Model selection uses the same cross-validation folds whose metrics are reported, so a reserved experimental batch is still needed for final evaluation. Negative or above-range regression predictions remain visible in `pred_hours`; only the index and category are bounded.

The freshness index is `100 × (1 − predicted_hours / 48)`, clipped to 0–100. Category boundaries are 16 and 32 hours; exactly 16 remains Fresh, exactly 32 remains Moderate. These are code choices, not validated spoilage thresholds.

## References

- [AD5933 datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/ad5933.pdf).
- [Analog Devices CN0217 signal-conditioning reference](https://www.analog.com/en/resources/reference-designs/circuits-from-the-lab/CN0217.html).
- [Grouped cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html).

Downloaded reference PDFs stay local; link their publishers rather than distributing those files with the project.

## Repository contents

The `.gitignore` excludes environments, caches, generated models/results, local credential folders and downloaded reference PDFs.
