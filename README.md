# CS-433 Project 1 — Predicting coronary heart disease (MICHD) from BRFSS

Binary classification of MICHD (heart attack / angina / coronary heart disease)
from 321 answers of the 2015 Behavioral Risk Factor Surveillance System survey,
using **NumPy only**. Final model: L2-regularised logistic regression trained
with Newton's method on a 818-dimensional engineered feature set, with an
F1-optimal decision threshold.

## Repository layout

| File | Content |
|---|---|
| `implementations.py` | The 6 required methods (`mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression`) + loss/gradient helpers |
| `preprocessing.py` | `Preprocessor` class: feature cleaning / encoding pipeline (fit on train, applied to test) |
| `models.py` | Newton logistic regression with sample weights, F1 / accuracy / AUC, F1-optimal threshold, stratified k-fold CV |
| `run.py` | **Reproduces our best submission** → `submission.csv` |
| `experiments.py` | All cross-validation experiments of the report (baselines, λ sweep, ablation) → `results/results.json`, `figures/` |
| `helpers.py` | Course-provided CSV loading / submission writing |
| `report/` | LaTeX source and PDF of the 2-page report |

## How to run

Requirements: Python ≥ 3.9 and NumPy (matplotlib only for the figures).

1. Download `x_train.csv`, `y_train.csv`, `x_test.csv` from AIcrowd and put
   them in `data/dataset/` (do not rename them).
2. Generate the submission (≈ 3–4 min, mostly CSV parsing):
   ```bash
   python run.py
   ```
   This writes `submission.csv` (labels in {-1, 1}). The procedure is fully
   deterministic.
3. (Optional) Re-run every experiment of the report (≈ 1 h on 2 cores):
   ```bash
   python experiments.py           # results/results.json + figures/
   python experiments.py --figures # only redraw the figures
   ```

## Pipeline (details in the report)

1. **Column selection** — drop survey-administration columns (interview date,
   sequence numbers, sampling weights, phone-line questions) and raw answers
   that use mixed unit encodings when the CDC provides a cleaned computed
   version (e.g. `WEIGHT2` → `WTKG3`, `FRUIT1` → `FRUTDA1_`); drop columns
   with > 90 % missing values.
2. **Categorical columns** (≤ 15 distinct values, plus `_STATE`) → one-hot
   encoding, NaN as its own level, levels with < 50 occurrences dropped.
   "Don't know" (7) and "Refused" (9) thus become indicators instead of large
   numbers.
3. **Continuous columns** → decode BRFSS special codes (88 = "none" → 0;
   77/99/900/99900 → missing), `log1p` of non-negative columns with skewness
   > 2, clipping at the 1st/99th training percentiles, median imputation +
   missing indicator, standardisation.
4. **Model** — logistic regression, λ = 1e-4 (bias not penalised), solved by
   Newton's method (≤ 15 steps).
5. **Threshold** — the logit threshold maximising F1 on the training set
   (class prior ≈ 8.8 % positives, so 0.5 is far from optimal for F1).

All statistics (medians, percentiles, means, levels, threshold) are learned on
the training data only; in cross-validation they are re-learned inside each
training fold.

## Cross-validation protocol

Stratified 5-fold CV (seed 0); the threshold is tuned on the training folds and
evaluated on the held-out fold. Reported metrics: F1 (competition metric),
accuracy and ROC-AUC, mean ± std over folds.
