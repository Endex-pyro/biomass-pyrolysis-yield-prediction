"""Run the analyses in dependency-safe order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

ANALYSES = OrderedDict([
    ("00", ("00_validate_dataset.py", "00_dataset_audit.csv")),
    ("01", ("01_baseline_model_screen.py", "01_baseline_model_screen_results.xlsx")),
    ("03", ("03_product_specific_oof_diagnostics.py", "03_product_specific_oof_diagnostics_results.xlsx")),
    ("04", ("04_validation_ladder.py", "04_validation_ladder_results.xlsx")),
    ("05", ("05_feature_importance.py", "05_feature_importance_results.xlsx")),
    ("06a", ("06a_random_row_ablation.py", "06a_random_row_ablation_results.xlsx")),
    ("06b", ("06b_grouped_holdout_ablation.py", "06b_grouped_holdout_ablation_results.xlsx")),
    ("07", ("07_model_selection_robustness.py", "07_model_selection_robustness_results.xlsx")),
    ("08", ("08_process_detail_sensitivity.py", "08_process_detail_sensitivity_results.xlsx")),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all analyses, or a comma-separated subset such as "
            "'00,04,06b'. Step 06b requires step 04 output."
        )
    )
    parser.add_argument(
        "--only",
        help="Comma-separated step identifiers. Default: run every step.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a step when its expected output file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser.parse_args()


def selected_steps(only: str | None) -> list[str]:
    if not only:
        return list(ANALYSES)

    requested = [item.strip() for item in only.split(",") if item.strip()]
    unknown = [item for item in requested if item not in ANALYSES]
    if unknown:
        raise SystemExit(
            f"Unknown step identifiers: {unknown}. Valid choices: {list(ANALYSES)}"
        )
    return requested


def main() -> None:
    args = parse_args()
    steps = selected_steps(args.only)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for step in steps:
        script_name, output_name = ANALYSES[step]
        script = SCRIPTS_DIR / script_name
        output = OUTPUT_DIR / output_name

        if args.skip_existing and output.exists():
            print(f"[skip {step}] {output_name} already exists")
            continue

        command = [sys.executable, str(script)]
        print(f"[run {step}] {' '.join(command)}")
        if args.dry_run:
            continue

        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
