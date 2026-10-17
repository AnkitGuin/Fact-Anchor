#!/usr/bin/env python3
"""Environment / model / knowledge feasibility report (Phase 0 exit criteria).

python scripts/doctor.py --config configs/default.yaml --check-model --check-knowledge
"""

from __future__ import annotations

import sys

from _bootstrap import PROJECT_ROOT  # noqa: F401

from factanchor.cli import main

if __name__ == "__main__":
    argv = ["doctor"]
    if "--config" not in sys.argv:
        argv += ["--config", "configs/default.yaml"]
    sys.exit(main(argv + sys.argv[1:]))
