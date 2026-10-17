#!/usr/bin/env python3
"""Run FactAnchor (entity-grounded contrastive decoding) generation.

Examples
--------
python scripts/run_factanchor.py --config configs/default.yaml --prompt "Who was Marie Curie?"
python scripts/run_factanchor.py --config configs/smoke_toy.yaml --prompt "Who was Ada Lovelace?" --json
"""

from __future__ import annotations

import sys

from _bootstrap import PROJECT_ROOT  # noqa: F401

from factanchor.cli import main

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/default.yaml"])
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "factanchor"])
    if any(flag in sys.argv for flag in ("--dataset", "--limit", "--run-id", "--condition")):
        sys.argv = [sys.argv[0], "run", *sys.argv[1:]]
    else:
        sys.argv = [sys.argv[0], "generate", *sys.argv[1:]]
    sys.exit(main())
