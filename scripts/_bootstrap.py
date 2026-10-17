"""Make ``src/`` importable when the package is not pip-installed.

Every script imports this first, so `python scripts/run_baseline.py ...` works
from a fresh clone without `pip install -e .`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PROJECT_ROOT = _ROOT
