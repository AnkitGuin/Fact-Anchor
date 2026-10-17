#!/usr/bin/env python3
"""Run baseline (no-contrast) generation for one prompt or a small example file.

Examples
--------
python scripts/run_baseline.py --config configs/baseline.yaml --prompt "Who was Marie Curie?"
python scripts/run_baseline.py --config configs/smoke_toy.yaml \
    --dataset data/processed/dev_smoke.jsonl --limit 3
"""

from __future__ import annotations

import sys

from _bootstrap import PROJECT_ROOT  # noqa: F401  (sets sys.path)

from factanchor.cli import main

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/baseline.yaml"])
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "baseline"])
    if any(flag in sys.argv for flag in ("--dataset", "--limit", "--run-id")):
        # dataset mode: delegate to `factanchor run`
        sys.argv = [sys.argv[0], "run", *sys.argv[1:]]
    else:
        sys.argv = [sys.argv[0], "generate", *sys.argv[1:]]
    sys.exit(main())
