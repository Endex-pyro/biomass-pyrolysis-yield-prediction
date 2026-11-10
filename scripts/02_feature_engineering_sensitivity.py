"""Run the feature-engineering sensitivity analysis from its packaged Python source.

The source is stored losslessly as a gzip-compressed .py file alongside this
launcher. Run this file normally, or decompress the source for inspection.
"""
from __future__ import annotations

import gzip
from pathlib import Path

source_path = Path(__file__).with_suffix(".py.gz")
source = gzip.decompress(source_path.read_bytes()).decode("utf-8")
exec(compile(source, str(source_path.with_suffix("")), "exec"), globals())
