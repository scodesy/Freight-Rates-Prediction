# Freight Rates Prediction

This repository contains a Python solution for the freight rate prediction assessment. It validates the provided freight data, builds time-based train/validation splits, trains baseline and tree-based models, and writes the final `validation_predictions.csv` submission file.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Data

The assessment files are expected under `data/raw/`:

```text
data/raw/Train_Test.csv
data/raw/Validation.csv
data/raw/Validation_Predictions_Template.csv
data/raw/December_Chart_Inputs.csv
```

## Validation approach

I used chronological expanding-window validation to avoid training on future loads:

| Fold | Training data | Validation data |
|---|---|---|
| July | Jan-Jun | July |
| August | Jan-Jul | August |
| September | Jan-Aug | September |

October was held out until after model selection and used as a final sanity check. The selected model was an Extra Trees regressor trained on a log-transformed target using the `december_compatible` feature set, so the same feature path works for both the validation loads and the fixed December chart inputs.

## Run checks

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/ruff check src scripts tests
```

## Run experiments

Baselines:

```bash
.venv/bin/python scripts/run_baselines.py \
  --data data/raw/Train_Test.csv \
  --artifact-root experiments
```

Candidate models:

```bash
.venv/bin/python scripts/run_candidates.py \
  --data data/raw/Train_Test.csv \
  --artifact-root experiments \
  --seed 42 \
  --workers 4
```

Freeze a chosen winner and comparator, then evaluate October once:

```bash
.venv/bin/python scripts/freeze_selection.py \
  --artifact-root experiments \
  --winner-run-id <winner-run-id> \
  --comparator-run-id <comparator-run-id>

.venv/bin/python scripts/evaluate_holdout.py \
  --data data/raw/Train_Test.csv \
  --lock experiments/selection/final_locked.json \
  --access-marker experiments/holdout/october_access.json \
  --artifact-root experiments
```

## Final model result

The selected development winner was:

```text
ExtraTreesRegressor
target: log1p(posted_rate)
n_estimators: 160
min_samples_leaf: 6
random_state: 42
feature_view: december_compatible
```

Development mean across July, August, and September:

| Model | MAE | RMSE | WMAPE |
|---|---:|---:|---:|
| Extra Trees + log target | 153.44 | 632.23 | 6.43% |
| Ridge baseline | 195.18 | 637.47 | 8.15% |

October holdout check:

| Model | MAE | RMSE | WMAPE |
|---|---:|---:|---:|
| Extra Trees + log target | 132.24 | 655.55 | 5.56% |
| Ridge baseline | 145.11 | 654.87 | 6.10% |

The final submission file is committed as:

```text
validation_predictions.csv
```

It contains exactly:

```text
load_id,predicted_rate
```

with 12,000 prediction rows.

## Scorer

The provided scorer can be used to validate the final prediction and December chart files:

```bash
MPLCONFIGDIR=/tmp/freight-rate-matplotlib \
.venv/bin/python scorer.py \
  --predictions validation_predictions.csv \
  --december-predictions December_Chart_Predictions.csv \
  --output-dir scorer_results
```

Expected output:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```
