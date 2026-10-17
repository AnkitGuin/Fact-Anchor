#!/usr/bin/env python3
"""Download and normalise the supported datasets (never substitutes another dataset).

Examples
--------
python scripts/download_data.py --list
python scripts/download_data.py --dataset wikibio_hallucination --max-examples 50
python scripts/download_data.py --dataset fever --inspect --dry-run
python scripts/download_data.py --all --out data/processed
"""

from __future__ import annotations

import sys

from _bootstrap import PROJECT_ROOT  # noqa: F401

from factanchor.cli import main

if __name__ == "__main__":
    argv = ["datasets"]
    if "--config" not in sys.argv:
        argv += ["--config", "configs/default.yaml"]
    sys.exit(main(argv + sys.argv[1:]))
