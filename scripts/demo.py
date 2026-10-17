#!/usr/bin/env python3
"""Console demo: baseline vs FactAnchor on the same prompt.

python scripts/demo.py --config configs/smoke_toy.yaml --prompt "Who was Marie Curie?"
"""

from __future__ import annotations

import sys

from _bootstrap import PROJECT_ROOT  # noqa: F401

from factanchor.cli import main

if __name__ == "__main__":
    argv = ["demo"]
    if "--config" not in sys.argv:
        argv += ["--config", "configs/default.yaml"]
    sys.exit(main(argv + sys.argv[1:]))
