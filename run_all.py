"""Run the analysis scripts in dependency-safe order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
OUTPUTS = ROOT / "outputs"
ANALYSES = OrderedDict([
    ("00", ("00_validate_dataset.py", "00_dataset_audit.csv")),
    ("01", ("01_baseline_model_screen.py", "01_baseline_model_screen_results.xlsx")),
    ("02", ("02_feature_engineering_sensitivity.py", "02_feature_engineering_sensitivity_results.xlsx")),
    ("03", ("03_product_specific_oof_diagnostics.py", "03_product_specific_oof_diagnostics_results.xlsx")),
    ("04", ("04_validation_ladder.py", "04_validation_ladder_results.xlsx")),
    ("05", ("05_feature_importance.py", "05_feature_importance_results.xlsx")),
    ("06a", ("06a_random_row_ablation.py", "06a_random_row_ablation_results.xlsx")),
    ("06b", ("06b_grouped_holdout_ablation.py", "06b_grouped_holdout_ablation_results.xlsx")),
    ("07", ("07_model_selection_robustness.py", "07_model_selection_robustness_results.xlsx")),
    ("08", ("08_process_detail_sensitivity.py", "08_process_detail_sensitivity_results.xlsx")),
    ("09", ("09_collinearity_diagnostics.py", "09_collinearity_diagnostics_results.xlsx")),
    ("10", ("10_loso_audit_and_environment.py", "10_loso_audit_and_environment_results.xlsx")),
    ("11", ("11_closure_window_sensitivity.py", "11_closure_window_sensitivity_results.xlsx")),
])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Comma-separated analysis identifiers; default all.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.only:
        steps = [step.strip() for step in args.only.split(",") if step.strip()]
        unknown = [step for step in steps if step not in ANALYSES]
        if unknown:
            parser.error(f"Unknown analysis step(s): {unknown}; expected {list(ANALYSES)}")
    else:
        steps = list(ANALYSES)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for step in steps:
        script, output = ANALYSES[step]
        output_path = OUTPUTS / output
        if args.skip_existing and output_path.exists():
            print(f"[skip {step}] {output}")
            continue
        command = [sys.executable, str(SCRIPTS / script)]
        print(f"[run {step}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
