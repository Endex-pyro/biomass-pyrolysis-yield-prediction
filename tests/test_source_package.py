"""Source-package checks that do not fit the machine-learning models."""
import gzip
import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

class SourcePackageChecks(unittest.TestCase):
    def test_packed_feature_engineering_source(self):
        source = gzip.decompress(
            (SCRIPTS / "02_feature_engineering_sensitivity.py.gz").read_bytes()
        ).decode("utf-8")
        ast.parse(source)
        self.assertIn("RANDOM_SEEDS = list(range(20))", source)

    def test_reporting_ablation_source(self):
        source = (SCRIPTS / "13_study_disjoint_reporting_ablation.py").read_text()
        ast.parse(source)
        self.assertIn("no_sum_no_reporting", source)

    def test_extended_runner_dry_run(self):
        p = subprocess.run(
            [sys.executable, str(ROOT / "run_all.py"), "--only", "00,02,09,11", "--dry-run"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        for step in ("00", "02", "09", "11"):
            self.assertIn(f"[run {step}]", p.stdout)

if __name__ == "__main__":
    unittest.main()
