# Reporting-aware prediction of biomass pyrolysis product yields

This repository provides the literature-derived input dataset, Python analysis scripts,
software requirements and run instructions for the research article
*Reporting-aware prediction of biomass pyrolysis product yields under mass-closure constraints*.

## Input dataset

`data/model_ready_dataset.xlsx` contains 442 experimental observations from 42 fixed-bed
pyrolysis studies, organized into 69 study–feedstock–composition validation groups.
Its `Model_Ready_Data` worksheet contains the variables used by the scripts;
`Column_Dictionary` describes their meanings. Source publication identifiers and
experimental reporting descriptors are included for traceability.

The accompanying Supplementary Data workbook contains the source-linked
experimental records, original and closed observed yields, and leave-one-study-out
predictions.

## Requirements and setup

The reference environment used Python 3.13.0. Direct Python package requirements
are listed in `requirements.txt`.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/00_validate_dataset.py
python -m unittest discover -s tests -v
```

## Reproduce the analyses

Run the standard workflow from the repository root:

```bash
python run_all.py
```

Or choose steps, for example:

```bash
python run_all.py --only 00,02,04
```

The grouped feature-ablation analysis (06b) and the study-level audit (10) use
results produced by step 04; run step 04 first. Some full analyses are computationally expensive.

| Step | Analysis |
|---|---|
| 00 | Input validation and dataset checks |
| 01 | Model and target-handling screen (Table 1) |
| 02 | Feature-engineering sensitivity (Supplementary Table S3) |
| 03 | Product-specific cross-validation (Fig. 2) |
| 04 | Random-row, grouped and leave-one-study-out validation (Fig. 3) |
| 05 | Feature importance (Fig. 4) |
| 06a, 06b | Feature-group ablation (Table 2) |
| 07 | Additional model and target-handling comparisons |
| 08 | Process-detail sensitivity (Supplementary Fig. S1) |
| 09 | Correlation and collinearity diagnostics (Supplementary Fig. S2) |
| 10 | Study-level leave-one-study-out diagnostics |
| 11 | Yield-sum acceptance-window sensitivity (Supplementary Table S4) |

### Study-disjoint reporting-descriptor ablation

Supplementary Table S9 compares three feature sets at seeds 42, 7 and 123,
using identical study exclusions and model settings. Run:

```bash
python scripts/13_study_disjoint_reporting_ablation.py --smoke
python scripts/13_study_disjoint_reporting_ablation.py
```

The `--smoke` option checks execution using a single study holdout.
The complete run compares the full predictor set, a predictor set excluding two
composition sums, and one further excluding seven reporting/basis categories
while retaining the pyrolysis-regime descriptor. The final contrast is
conditional on removing the composition sums.

### Four-model comparison

Supplementary Table S10 compares XGBoost, ExtraTrees, Random Forest and
CatBoost under the same reference predictors and validation partitions:

```bash
python scripts/12_harmonized_model_comparison.py --smoke
python scripts/12_harmonized_model_comparison.py
```

The smoke test checks execution.
Run the complete comparison separately from `run_all.py`.

## Interpretation and reuse

Predicted product fractions are normalized to 100 wt.% in the closure-aware
analyses. This does not correct incomplete physical mass recovery in source
experiments. Random-row validation does not constitute independent prospective
validation. Interpret numerical results using the corresponding feature set,
target treatment and validation design in the manuscript and Supplementary
Information.

Publication citations are listed in the manuscript. Reuse of extracted source data
remains subject to rights associated with the original publications.