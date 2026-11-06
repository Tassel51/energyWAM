#!/usr/bin/env python
"""Run the EnergyWAM test suite.  See tests/harness.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.harness import run  # noqa: E402

if __name__ == "__main__":
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(run(filt))
