"""Fast preflight tests that do not fit the machine-learning models."""

from __future__ import annotations

import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA_FILE = ROOT / "data" / "model_ready_dataset.xlsx"
DATA_SHEET = "Model_Ready_Data"

ANALYSIS_SCRIPTS = {
    "00_validate_dataset.py",
    "01_baseline_model_screen.py",
    "03_product_specific_oof_diagnostics.py",
    "04_validation_ladder.py",
    "05_feature_importance.py",
    "06a_random_row_ablation.py",
    "06b_grouped_holdout_ablation.py",
    "07_model_selection_robustness.py",
    "08_process_detail_sensitivity.py",
    "12_harmonized_model_comparison.py",
}

FORBIDDEN_PATH_TOKENS = {
    "Raw_Literature_dataset",
    "/mnt/data/",
}


class PackagePreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_excel(DATA_FILE, sheet_name=DATA_SHEET)

    def test_expected_analysis_scripts_are_present(self) -> None:
        observed = {p.name for p in SCRIPTS.glob("*.py")}
        self.assertTrue(ANALYSIS_SCRIPTS.issubset(observed))

    def test_all_python_files_compile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for index, path in enumerate(sorted(ROOT.rglob("*.py"))):
                with self.subTest(path=path.name):
                    py_compile.compile(
                        path,
                        cfile=str(Path(tmpdir) / f"{index:02d}_{path.stem}.pyc"),
                        doraise=True,
                    )

    def test_no_legacy_absolute_or_working_directory_paths(self) -> None:
        for path in sorted(SCRIPTS.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN_PATH_TOKENS:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, text)

    def test_dataset_identity(self) -> None:
        self.assertEqual(len(self.df), 442)
        self.assertEqual(self.df["study_id"].nunique(), 42)
        self.assertEqual(self.df["validation_group_id"].nunique(), 69)

    def test_closure_window_counts(self) -> None:
        yield_sum = pd.to_numeric(self.df["yield_sum_wt_pct"], errors="coerce")
        mask_90 = yield_sum.between(90.0, 110.0, inclusive="both")
        mask_95 = yield_sum.between(95.0, 105.0, inclusive="both")

        self.assertEqual(int(mask_90.sum()), 356)
        self.assertEqual(self.df.loc[mask_90, "study_id"].nunique(), 38)
        self.assertEqual(int(mask_95.sum()), 303)
        self.assertEqual(self.df.loc[mask_95, "study_id"].nunique(), 34)

    def test_process_detail_counts(self) -> None:
        optional = [
            "holding_time_min_parsed",
            "particle_size_mm_parsed",
            "sample_mass_g_parsed",
        ]
        self.assertEqual(self.df[optional[0]].notna().sum(), 364)
        self.assertEqual(self.df[optional[1]].notna().sum(), 410)
        self.assertEqual(self.df[optional[2]].notna().sum(), 388)

        complete = self.df[optional].notna().all(axis=1)
        self.assertEqual(int(complete.sum()), 322)
        self.assertEqual(self.df.loc[complete, "study_id"].nunique(), 29)

    def test_provided_group_ids_match_canonical_partition(self) -> None:
        composition = [
            "volatile_matter_wt_pct",
            "fixed_carbon_wt_pct",
            "ash_wt_pct",
            "carbon_wt_pct",
            "hydrogen_wt_pct",
            "nitrogen_wt_pct",
            "oxygen_wt_pct",
        ]
        canonical = (
            self.df["study_id"].astype(str)
            + " | "
            + self.df["feedstock_name"].astype(str)
        )
        for column in composition:
            canonical = (
                canonical
                + " | "
                + pd.to_numeric(self.df[column], errors="coerce").map(
                    lambda value: f"{value:.3f}"
                )
            )

        mapping = pd.DataFrame(
            {
                "provided": self.df["validation_group_id"].astype(str),
                "canonical": canonical,
            }
        )
        self.assertEqual(
            mapping.groupby("provided")["canonical"].nunique().max(), 1
        )
        self.assertEqual(
            mapping.groupby("canonical")["provided"].nunique().max(), 1
        )

    def test_runner_dry_run(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "run_all.py"),
                "--only",
                "00,08",
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("[run 00]", completed.stdout)
        self.assertIn("[run 08]", completed.stdout)


if __name__ == "__main__":
    unittest.main()
