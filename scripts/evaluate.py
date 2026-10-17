#!/usr/bin/env python3
"""Evaluate saved generations and write metrics.json.

Examples
--------
python scripts/evaluate.py --input artifacts/runs/<run_id> --output artifacts/metrics/evaluation.json
python scripts/evaluate.py --input artifacts/runs/<run_id>/generations.jsonl \
    --dataset data/processed/dev_smoke.jsonl --out artifacts/metrics/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401

from factanchor.cli import main


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--judge", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    parser, _ = parser.parse_known_args()
    return parser


if __name__ == "__main__":
    args = parse()
    argv = ["factanchor", "evaluate", "--input", args.input, "--config", args.config]
    if args.judge:
        argv += ["--judge", args.judge]
    if args.dataset:
        argv += ["--dataset", args.dataset]
    if args.split:
        argv += ["--split", args.split]
    if args.limit:
        argv += ["--limit", args.limit]
    for override in args.set:
        argv += ["--set", override]
    if args.json:
        argv += ["--json"]
    # The CLI writes metrics.json next to the generations; --output copies it to
    # the requested path so the documented interface keeps working.
    code = main(argv[1:])
    if code == 0 and args.output:
        source = Path(args.input)
        if source.is_file():
            source = source.parent
        metrics = source / "metrics.json"
        target = Path(args.output)
        if metrics.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(metrics, target)
            print(f"copied metrics to {target}")
        else:
            print(f"WARNING: {metrics} was not produced", file=sys.stderr)
    sys.exit(code)
