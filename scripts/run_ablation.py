#!/usr/bin/env python3
"""Run the ablation grid defined in an ablation config file.

Two spellings are supported:

1. Spec-style ``--config`` pointing at the ablation file::

       python scripts/run_ablation.py --config configs/experiments.yaml

   The ablation file names its own template through its ``base:`` key.

2. Explicit ``--ablation`` plus an optional plain ``--config`` template::

       python scripts/run_ablation.py --ablation configs/experiments.yaml \\
           --dataset data/processed/dev_smoke.jsonl --limit 4

Any other flag is forwarded unchanged to ``factanchor ablate``.
"""

from __future__ import annotations

import sys

from _bootstrap import PROJECT_ROOT  # noqa: F401  (sets sys.path)

from factanchor.cli import main

_DEFAULT_ABLATION = "configs/experiments.yaml"
_DEFAULT_TEMPLATE = "configs/default.yaml"


def _take_flag(args: list[str], flag: str) -> str | None:
    """Remove ``--flag value`` / ``--flag=value`` from ``args`` and return value."""
    for index, item in enumerate(args):
        if item == flag:
            if index + 1 < len(args):
                value = args[index + 1]
                del args[index : index + 2]
            else:
                del args[index : index + 1]
                value = None
            return value
        if item.startswith(f"{flag}="):
            value = item.split("=", 1)[1]
            del args[index]
            return value
    return None


def build_argv(argv: list[str]) -> list[str]:
    """Translate the documented script interface into a ``factanchor ablate`` call."""
    rest = [item for item in argv if item not in {"-h", "--help"}]
    wants_help = len(rest) != len(argv)
    config = _take_flag(rest, "--config")
    ablation = _take_flag(rest, "--ablation")
    template = config if (config and ablation) else _DEFAULT_TEMPLATE
    ablation = ablation or config or _DEFAULT_ABLATION
    out = ["ablate", "--config", template, "--ablation", ablation, *rest]
    if wants_help:
        out.append("--help")
    return out


if __name__ == "__main__":
    sys.exit(main(build_argv(sys.argv[1:])))
