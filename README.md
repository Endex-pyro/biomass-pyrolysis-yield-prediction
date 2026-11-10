# Reporting-aware prediction of biomass pyrolysis product yields

This repository provides the model-ready dataset and essential Python analysis code supporting the research article **Reporting-aware prediction of biomass pyrolysis product yields under mass-closure constraints**.

**Please cite the associated research article when using the dataset, methods, or scientific findings.** The article remains the primary reference for scientific results and their interpretation. Full bibliographic information will be linked here when available. Software attribution is given in [CITATION.cff](CITATION.cff).

## Dataset

`data/model_ready_dataset.xlsx` contains 442 observations from 42 published fixed-bed pyrolysis studies, organized into 69 study–feedstock–composition groups. The workbook provides the `Model_Ready_Data` sheet for calculations and variable definitions in `Column_Dictionary`.

Publication identifiers and validation-group assignments are retained for traceability. The published studies use heterogeneous yield definitions and measurement conventions; predictions should not be interpreted as independent prospective experimental validation.

## Run analyses

The reference analysis was run with Python 3.13.0. `requirements.txt` pins the direct package versions from that environment; results may vary on other platforms or dependency builds.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Validate the dataset and run preflight tests:

```bash
python scripts/00_validate_dataset.py
python -m unittest discover -s tests -v
```

Run the analyses in dependency order:

```bash
python run_all.py
```

Or select specific steps:

```bash
python run_all.py --only 00,04,06b
```

The grouped feature-ablation step (`06b`) uses the validation splits created by step `04`; run the latter first. Outputs are written to `outputs/` locally and are not tracked by Git. Some analyses require substantial computation.

The package supports dataset validation, model screening, product-specific prediction, random-row and grouped validation, leave-one-study-out assessment, feature attribution, reporting-feature ablation, model comparison and sensitivity to optional process variables. The article and its supporting information contain the full scientific results, figures and interpretation. No model outputs, diagnostic workbooks or generated figures are included in this repository.

### Harmonized ensemble-model comparison

The matched four-model comparison reported in Supplementary Table S11 is reproduced by `scripts/12_harmonized_model_comparison.py`. It compares XGBoost, ExtraTrees, Random Forest and CatBoost using the reference feature set and identical held-out partitions across models: 20 random-row splits, 20 grouped splits and 42 leave-one-study-out folds.

Run this computationally intensive analysis separately from `run_all.py`:

```bash
python scripts/12_harmonized_model_comparison.py --smoke
python scripts/12_harmonized_model_comparison.py
```

The smoke test only verifies the execution path; its results are not comparable to the full analysis. The full run writes split-level results, summary metrics and checks against the reference XGBoost validation to `outputs/`, which is excluded from Git.

## Use and attribution

Source publication identifiers, model settings, split definitions and evaluation procedures are retained to support reproducibility. No license is granted solely by making the repository public; reuse of compiled source data may depend on rights associated with the original publications.
