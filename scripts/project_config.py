"""Shared paths for the public reproducibility package."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def env_path(name: str, default: Path) -> Path:
    """Resolve environment overrides relative to the repository root."""
    raw_value = os.environ.get(name)
    path = Path(raw_value).expanduser() if raw_value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DATA_FILE = env_path(
    "PYROLYSIS_DATA_FILE",
    PROJECT_ROOT / "data" / "model_ready_dataset.xlsx",
)

DATA_SHEET = os.environ.get("PYROLYSIS_DATA_SHEET", "Model_Ready_Data")

OUTPUT_DIR = env_path("PYROLYSIS_OUTPUT_DIR", PROJECT_ROOT / "outputs")


def output_file(filename: str) -> Path:
    """Return a repository-relative output path and create its directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename
