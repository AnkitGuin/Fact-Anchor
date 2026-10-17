"""Command line interface (``factanchor`` / ``python -m factanchor``).

Subcommands mirror the scripts in ``scripts/`` (which are thin wrappers around
this module), so the two entry points can never diverge:

``generate``   one prompt, baseline or FactAnchor, prints the answer + metadata
``run``        a set of examples (JSONL or a supported dataset) -> run directory
``evaluate``   score saved generations (``artifacts/.../generations.jsonl``)
``ablate``     run the ablation grid from a config
``datasets``   show dataset availability/licence info; optionally download
``doctor``     environment/model feasibility report (Phase 0 exit criteria)
``demo``       two-mode console comparison, no files written unless asked
``cache``      inspect/clear the knowledge cache

Design rules: ``--help`` everywhere, ``--set key=value`` overrides, JSON output
via ``--json``, exit codes 0 ok / 1 error / 2 configuration error / 3 nothing ran,
and no secrets echoed to the terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import Config, ConfigError, coerce_scalar, load_config
from .utils import (
    collect_environment,
    get_logger,
    json_default,
    setup_logging,
    flush_all_handlers,
)

logger = get_logger("cli")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_NOTHING = 3


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="factanchor",
        description=(
            "FactAnchor: entity-grounded contrastive decoding for hallucination suppression "
            "(research prototype). It does not guarantee factual output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"factanchor {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Options shared by every subcommand.  They live on the parent parser (not on
    # the top-level one) so that `factanchor generate --set ...` works: a top-level
    # default would be clobbered by the subparser's own default.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.yaml", help="path to a YAML config")
    common.add_argument("--json", action="store_true", help="print machine-readable JSON instead of prose")
    common.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (-v: INFO, -vv: DEBUG)",
    )
    common.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override any config path (repeatable)"
    )

    # -- generate -----------------------------------------------------------
    generate = subparsers.add_parser(
        "generate", parents=[common], help="generate for a single prompt (baseline or factanchor)"
    )
    generate.add_argument("--prompt", help="the prompt text")
    generate.add_argument("--prompt-file", help="read the prompt from a file (preferred for long prompts)")
    generate.add_argument("--mode", choices=["baseline", "factanchor"], help="override decoding.mode")
    generate.add_argument("--alpha", type=float, help="override decoding.alpha")
    generate.add_argument("--max-new-tokens", type=int, help="override decoding.max_new_tokens")
    generate.add_argument("--temperature", type=float, help="override decoding.temperature")
    generate.add_argument("--top-p", type=float, help="override decoding.top_p")
    generate.add_argument("--do-sample", action="store_true", help="override decoding.do_sample=true")
    generate.add_argument("--seed", type=int, help="override run.seed")
    generate.add_argument("--corruption-rate", type=float, help="override corruption.corruption_rate")
    generate.add_argument(
        "--corruption-strategy",
        choices=["object_swap", "attribute_perturbation", "entity_substitution", "omission", "none"],
        help="override corruption.strategy",
    )
    generate.add_argument("--model-id", help="override model.model_id (e.g. 'toy' for offline)")
    generate.add_argument("--run-id", help="explicit run id for --save-run (never overwrites an existing directory)")
    generate.add_argument("--save-run", action="store_true", help="also write a run directory with artifacts")
    generate.add_argument("--show-contexts", action="store_true", help="print both contexts")

    # -- run ----------------------------------------------------------------
    run = subparsers.add_parser("run", parents=[common], help="run a condition over a dataset subset")
    run.add_argument("--dataset", default="data/processed/dev_smoke.jsonl", help="JSONL file or dataset key")
    run.add_argument("--split", help="dataset split (HF loaders)")
    run.add_argument("--limit", type=int, help="maximum number of examples")
    run.add_argument("--condition", default="", help="condition name recorded in the artifacts")
    run.add_argument("--run-id", help="explicit run id (never overwrites an existing directory)")
    run.add_argument("--no-warmup", action="store_true", help="skip the warm-up generation")
    run.add_argument("--quiet", action="store_true", help="suppress console logging")

    # -- evaluate -----------------------------------------------------------
    evaluate = subparsers.add_parser("evaluate", parents=[common], help="evaluate saved generations")
    evaluate.add_argument("--input", required=True, help="run directory (or a generations.jsonl file)")
    evaluate.add_argument("--examples", help="JSONL with Example records (evidence/reference)")
    evaluate.add_argument("--dataset", help="dataset key when --examples is not given")
    evaluate.add_argument("--split", help="dataset split")
    evaluate.add_argument("--limit", type=int, help="maximum examples to load")
    evaluate.add_argument("--output", help="where to write metrics.json (default: inside the run dir)")
    evaluate.add_argument(
        "--judge",
        choices=["heuristic_lexical", "gold_reference", "model_judge"],
        help="override evaluation.claim_judge",
    )

    # -- ablate -------------------------------------------------------------
    ablate = subparsers.add_parser("ablate", parents=[common], help="run the ablation grid")
    ablate.add_argument("--ablation", default="configs/experiments.yaml", help="ablation config file")
    ablate.add_argument("--dataset", default="data/processed/dev_smoke.jsonl", help="JSONL file or dataset key")
    ablate.add_argument("--limit", type=int, help="maximum number of examples")
    ablate.add_argument("--output", help="where to write ablation_results.json")
    ablate.add_argument("--no-warmup", action="store_true", help="skip warm-up generations")

    # -- datasets -----------------------------------------------------------
    datasets = subparsers.add_parser("datasets", parents=[common], help="dataset availability and licence info")
    datasets.add_argument("--dataset", help="dataset key to download")
    datasets.add_argument("--all", action="store_true", help="download every supported dataset")
    datasets.add_argument("--list", action="store_true", help="list supported datasets (default)")
    datasets.add_argument("--inspect", action="store_true", help="print the first raw record's fields")
    datasets.add_argument("--max-examples", type=int, default=50, help="examples to convert")
    datasets.add_argument("--out", default="data/processed", help="output directory")
    datasets.add_argument("--dry-run", action="store_true", help="report what would be downloaded")

    # -- doctor -------------------------------------------------------------
    doctor = subparsers.add_parser("doctor", parents=[common], help="environment and model feasibility report")
    doctor.add_argument("--check-model", action="store_true", help="actually load the model (slow)")
    doctor.add_argument("--check-knowledge", action="store_true", help="probe the knowledge API")

    # -- demo ---------------------------------------------------------------
    demo = subparsers.add_parser("demo", parents=[common], help="compare baseline and FactAnchor on one prompt")
    demo.add_argument("--prompt", help="prompt text")
    demo.add_argument("--prompt-file", help="read the prompt from a file")
    demo.add_argument("--alpha", type=float, default=None, help="contrast strength for the FactAnchor run")
    demo.add_argument("--max-new-tokens", type=int, default=None)
    # `--json` comes from the shared parent parser (no duplicate definition)

    # -- cache --------------------------------------------------------------
    cache = subparsers.add_parser("cache", parents=[common], help="knowledge cache utilities")
    cache.add_argument("--stats", action="store_true", help="show cache statistics")
    cache.add_argument("--clear", action="store_true", help="delete cached knowledge responses")
    cache.add_argument("--namespace", help="restrict --clear to one namespace")

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(args: argparse.Namespace, extra_overrides: dict[str, Any] | None = None) -> Config:
    overrides: list[str] = list(getattr(args, "set", []) or [])
    for key, value in (extra_overrides or {}).items():
        if value is not None:
            overrides.append(f"{key}={value}")
    return load_config(args.config, overrides=overrides)


def _prompt_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "prompt_file", None):
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        if not text.strip():
            raise ConfigError(f"prompt file {args.prompt_file} is empty")
        return text.strip()
    prompt = (getattr(args, "prompt", None) or "").strip()
    if not prompt:
        if sys.stdin.isatty():
            raise ConfigError("provide --prompt or --prompt-file (stdin is a terminal)")
        text = sys.stdin.read().strip()
        if not text:
            raise ConfigError("no prompt supplied on the command line or via stdin")
        return text
    return prompt


def _print(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default))
    elif isinstance(payload, str):
        print(payload)


def _build_knowledge(config: Config):
    if not config.retrieval.enabled:
        return None
    from .knowledge.pipeline import KnowledgePipeline, build_provider

    return KnowledgePipeline(config, provider=build_provider(config))


def _load_model_with_timing(config: Config):
    from .model_adapter import build_adapter

    started = time.monotonic()
    adapter = build_adapter(config)
    return adapter, time.monotonic() - started


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    # `--mode baseline` means "no contrastive term": alpha=0 is implied when the
    # user did not pass an explicit alpha, so the recorded run is unambiguous.
    implied_alpha = args.alpha
    if args.mode == "baseline" and args.alpha is None:
        implied_alpha = 0.0
    config = _load_config(
        args,
        {
            "decoding.mode": args.mode,
            "decoding.alpha": implied_alpha,
            "decoding.max_new_tokens": args.max_new_tokens,
            "decoding.temperature": args.temperature,
            "decoding.top_p": args.top_p,
            "decoding.do_sample": True if args.do_sample else None,
            "run.seed": args.seed,
            "corruption.corruption_rate": args.corruption_rate,
            "corruption.strategy": args.corruption_strategy,
            "model.model_id": args.model_id,
        },
    )
    setup_logging("DEBUG" if args.verbose > 1 else config.run.log_level)
    prompt = _prompt_from_args(args)

    adapter, load_s = _load_model_with_timing(config)
    knowledge = _build_knowledge(config)
    from .generation import generate_example

    run_dir = None
    run_id = ""
    run_context = None
    if args.save_run:
        from .experiment import create_run

        run_context = create_run(config, condition=config.run.name, run_id=args.run_id, dataset_label="cli", split_label="-")
        run_id = run_context.run_id
        run_dir = str(run_context.run_dir)

    record = generate_example(
        example_id="cli::0",
        prompt=prompt,
        config=config,
        adapter=adapter,
        run_id=run_id,
        condition=config.run.name,
        dataset="cli",
        split="-",
        knowledge=knowledge,
        model_load_s=load_s,
    )
    if run_context is not None:
        run_context.append_generation(record)

    payload = {
        "run_id": run_id,
        "run_dir": run_dir,
        "mode": record.mode,
        "variant": record.variant,
        "alpha": record.alpha,
        "model_id": record.model_id,
        "is_toy_model": record.is_toy_model,
        "status": record.status.value,
        "degraded_reason": record.degraded_reason,
        "prompt": record.prompt,
        "answer": record.answer,
        "stop_reason": record.stop_reason,
        "output_tokens": record.output_tokens,
        "generation_s": record.latency.generation_s,
        "ms_per_token": record.ms_per_token,
        "model_load_s": record.latency.model_load_s,
        "retrieval_s": record.latency.fact_retrieval_s,
        "entities": [entity.model_dump() for entity in record.entities],
        "n_facts": len(record.facts),
        "facts": [fact.render() for fact in record.facts],
        "corruption": record.corruption.summary if record.corruption else None,
        "contrast_applied": record.contrast_applied,
    }
    if args.show_contexts:
        payload["base_context"] = record.base_context
        payload["contrast_context"] = record.contrast_context
    if record.is_toy_model:
        payload["warning"] = (
            "model.model_id='toy' uses randomly initialised weights: the text is not meaningful. "
            "Use it to check the plumbing only."
        )
    if args.json:
        _print(payload, as_json=True)
    else:
        _print(
            "\n".join(
                [
                    f"mode            : {record.mode} ({record.variant}), alpha={record.alpha}",
                    f"model           : {record.model_id}{' [TOY - not meaningful text]' if record.is_toy_model else ''}",
                    f"status          : {record.status.value}{f' ({record.degraded_reason})' if record.degraded_reason else ''}",
                    f"entities        : {', '.join(e.display for e in record.entities) or 'none linked'}",
                    f"facts           : {len(record.facts)} retrieved"
                    + (f" / {record.corruption.changed_count} corrupted in the contrast branch" if record.corruption else ""),
                    f"tokens          : {record.output_tokens} in {record.latency.generation_s:.2f}s "
                    f"({record.ms_per_token or 0:.1f} ms/token)",
                    "",
                    "answer:",
                    record.answer or "[empty]",
                ]
            ),
            as_json=False,
        )
        if run_dir:
            print(f"\nartifacts: {run_dir}")
        print(
            "\nNOTE: generated text requires independent verification; experimental corrupted facts "
            "are never evidence."
        )
    return EXIT_OK if record.status.value != "error" else EXIT_ERROR


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(
        args,
        {
            "evaluation.max_examples": args.limit,
            "evaluation.split": args.split,
        },
    )
    setup_logging("DEBUG" if args.verbose > 1 else config.run.log_level, quiet=args.quiet)
    from .evaluation import load_examples
    from .experiment import run_condition

    examples = load_examples(args.dataset, split=args.split, max_examples=args.limit or config.evaluation.max_examples)
    if not examples:
        logger.error("no examples loaded from %s", args.dataset)
        return EXIT_NOTHING
    adapter, load_s = _load_model_with_timing(config)
    knowledge = _build_knowledge(config)
    summary = run_condition(
        config,
        examples,
        condition_name=args.condition or config.run.name,
        adapter=adapter,
        model_load_s=load_s,
        knowledge=knowledge,
        warmup=not args.no_warmup,
        run_id=args.run_id,
    )
    payload = summary.describe()
    # `summary.metrics` is the full metrics document; the console summary reports
    # this condition's slice of it.
    payload["metrics"] = summary.metrics.get("by_condition", {}).get(summary.condition, {})
    if args.json:
        _print(payload, as_json=True)
    else:
        metrics = payload["metrics"] or {}
        print(
            f"condition {summary.condition}: {summary.n_examples} example(s), "
            f"{summary.n_ok} ok, {summary.n_degraded} degraded, {summary.n_errors} error(s)"
        )
        if metrics:
            print(
                f"  claim support={metrics.get('mean_claim_support_rate')} "
                f"hallucination={metrics.get('mean_hallucination_rate')} "
                f"claims={metrics.get('n_total_claims')}"
            )
        print(f"  median ms/token={summary.latency.get('ms_per_token_median')}")
        print(f"artifacts: {summary.run_dir}")
    return EXIT_OK if summary.n_ok else EXIT_ERROR


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = _load_config(args, {"evaluation.claim_judge": args.judge})
    setup_logging("DEBUG" if args.verbose > 1 else config.run.log_level)
    from .evaluation import evaluate_run_dir

    run_path = Path(args.input)
    if run_path.is_file():
        if run_path.name != "generations.jsonl":
            logger.error("%s is not a generations.jsonl file nor a run directory", run_path)
            return EXIT_CONFIG
        run_dir = run_path.parent
        # evaluate a bare generations file by pointing at its directory
        temporary = run_dir
    else:
        temporary = run_path
    output = evaluate_run_dir(
        temporary,
        config=config,
        examples_path=args.examples,
        dataset=args.dataset,
        split=args.split,
        max_examples=args.limit,
        output_path=args.output,
    )
    payload = output.aggregate
    _print(payload, as_json=args.json)
    if not args.json:
        for condition, metrics in (payload.get("by_condition") or {}).items():
            print(
                f"{condition}: support={metrics.get('mean_claim_support_rate')} "
                f"hallucination={metrics.get('mean_hallucination_rate')} "
                f"n_claims={metrics.get('n_total_claims')} "
                f"median_ms/token={metrics.get('latency', {}).get('ms_per_token_median')}"
            )
        print("\nReminder: heuristic_lexical judging is a proxy, not entailment.")
    return EXIT_OK


def cmd_ablate(args: argparse.Namespace) -> int:
    config = _load_config(args, {"evaluation.max_examples": args.limit})
    setup_logging("DEBUG" if args.verbose > 1 else config.run.log_level)
    from .evaluation import load_examples
    from .experiment import run_ablation

    examples = load_examples(args.dataset, max_examples=args.limit or config.evaluation.max_examples)
    if not examples:
        logger.error("no examples loaded from %s", args.dataset)
        return EXIT_NOTHING
    results = run_ablation(
        args.ablation,
        examples=examples,
        base_overrides={item.split("=", 1)[0]: coerce_scalar(item.split("=", 1)[1]) for item in args.set},
        results_path=args.output or (config.output_dir / "ablation_results.json"),
        warmup=not args.no_warmup,
    )
    condensed = {
        "reference_condition": results.get("reference_condition"),
        "conditions": {
            name: {
                "summary": entry["summary"],
                "metrics": {
                    key: entry["metrics"].get(key)
                    for key in ("mean_claim_support_rate", "mean_hallucination_rate", "n_total_claims")
                },
                "overhead_vs_reference": entry.get("overhead_vs_reference"),
            }
            for name, entry in results["conditions"].items()
        },
        "warnings": results["warnings"],
        "note": results["comparison_note"],
    }
    plots_dir = config.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"ablation_{Path(args.ablation).stem}.png"
    from .experiment import write_ablation_plot

    if write_ablation_plot(results, plot_path):
        logger.info("wrote ablation plot to %s", plot_path)
    if args.json:
        _print(condensed, as_json=True)
    else:
        print(f"reference condition: {results.get('reference_condition')}")
        for name, entry in condensed["conditions"].items():
            metrics = entry["metrics"]
            overhead = (entry.get("overhead_vs_reference") or {}).get("generation", {})
            print(
                f"{name:28s} support={metrics.get('mean_claim_support_rate')} "
                f"hallucination={metrics.get('mean_hallucination_rate')} "
                f"claims={metrics.get('n_total_claims')} "
                f"overhead_median={overhead.get('overhead_pct_median')}%"
            )
        if results["warnings"]:
            print("warnings:")
            for warning in results["warnings"]:
                print(f"  - {warning}")
        print(f"full results: {args.output or (config.output_dir / 'ablation_results.json')}")
        print("Reminder: heuristic judging is a proxy and small subsets are not significance tests.")
    return EXIT_OK


def cmd_datasets(args: argparse.Namespace) -> int:
    from .evaluation import DATASET_SPECS, available_datasets, load_hf_examples, load_jsonl_examples

    if args.dataset is None and not args.all:
        rows = []
        for name in available_datasets():
            spec = DATASET_SPECS[name]
            rows.append(
                {
                    "key": name,
                    "hf_path": spec.hf_path,
                    "task": spec.task,
                    "licence": spec.licence,
                    "description": spec.description,
                    "citation": spec.citation,
                    "notes": spec.notes,
                }
            )
        _print(rows, as_json=True if args.json else False)
        if not args.json:
            for row in rows:
                print(f"{row['key']:24s} {row['task']:20s} {row['licence']}")
                print(f"    {row['description']}")
        return EXIT_OK

    targets = available_datasets() if args.all else [args.dataset]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for name in targets:
        if DATASET_SPECS[name].hf_path is None:
            logger.info("%s is a local JSONL format; nothing to download", name)
            results.append({"dataset": name, "status": "local-only"})
            continue
        if args.dry_run:
            results.append({"dataset": name, "status": "dry-run", "hf_path": DATASET_SPECS[name].hf_path})
            continue
        try:
            examples = load_hf_examples(name, max_examples=args.max_examples, inspect=args.inspect)
        except Exception as exc:
            logger.error("could not prepare %s: %s", name, exc)
            results.append({"dataset": name, "status": "error", "error": str(exc)})
            continue
        target = out_dir / f"{name}.jsonl"
        with open(target, "w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(example.model_dump_json() + "\n")
        results.append({"dataset": name, "status": "ok", "n_examples": len(examples), "path": str(target)})
    _print(results, as_json=True if args.json else False)
    return EXIT_OK if all(item["status"] in {"ok", "local-only", "dry-run"} for item in results) else EXIT_ERROR


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _load_config(args)
    try:
        setup_logging(config.run.log_level)
    except Exception:
        setup_logging("INFO")
    report: dict[str, Any] = {"factanchor_version": __version__, "config": str(args.config)}
    report["environment"] = collect_environment()
    report["config_valid"] = True
    report["model"] = {
        "model_id": config.model.model_id,
        "revision": config.model.revision,
        "is_toy": config.model.model_id.strip().lower() == "toy",
        "device_setting": config.run.device,
        "dtype_setting": config.run.dtype,
        "max_context_tokens": config.model.max_context_tokens,
    }
    checks: list[dict[str, Any]] = []

    try:
        import torch  # noqa: F401

        checks.append({"check": "torch", "status": "ok"})
    except ImportError:
        checks.append(
            {
                "check": "torch",
                "status": "missing",
                "hint": "pip install -e .[hf] (or use model.model_id='toy' with configs/smoke_toy.yaml)",
            }
        )
    try:
        import transformers  # noqa: F401

        checks.append({"check": "transformers", "status": "ok"})
    except ImportError:
        checks.append({"check": "transformers", "status": "missing", "hint": "pip install -e .[hf]"})
    checks.append({"check": "datasets", "status": "ok" if _module_available("datasets") else "missing",
                   "hint": "pip install -e .[datasets] for WikiBio/HaluEval/FEVER loaders"})
    checks.append({"check": "pandas", "status": "ok" if _module_available("pandas") else "missing"})
    checks.append({"check": "matplotlib", "status": "ok" if _module_available("matplotlib") else "missing"})
    checks.append({"check": "streamlit", "status": "ok" if _module_available("streamlit") else "missing",
                   "hint": "pip install -e .[demo] for app.py"})
    checks.append({"check": "spacy", "status": "ok" if _module_available("spacy") else "optional-missing",
                   "hint": "ner.backend='rule' is the default and needs no extra"})

    if args.check_knowledge:
        try:
            from .knowledge.pipeline import KnowledgePipeline, build_provider

            pipeline = KnowledgePipeline(config, provider=build_provider(config))
            started = time.monotonic()
            candidates = pipeline.provider.search_entities("Marie Curie", limit=2)
            elapsed = time.monotonic() - started
            checks.append(
                {
                    "check": "knowledge_api",
                    "status": "ok" if candidates else "no-results",
                    "elapsed_s": round(elapsed, 3),
                    "candidates": [candidate.to_dict() for candidate in candidates],
                }
            )
            if candidates:
                facts = pipeline.provider.get_facts(candidates[0].entity_id, max_facts=3)
                checks.append(
                    {
                        "check": "knowledge_facts",
                        "status": "ok" if facts else "no-facts",
                        "n_facts": len(facts),
                        "sample": [fact.render() for fact in facts],
                    }
                )
        except Exception as exc:
            checks.append({"check": "knowledge_api", "status": "error", "error": str(exc)})

    if args.check_model:
        try:
            from .model_adapter import build_adapter

            started = time.monotonic()
            adapter = build_adapter(config)
            load_s = time.monotonic() - started
            logits = adapter.logits_for_prefix(adapter.encode(adapter.render("Hello")))
            checks.append(
                {
                    "check": "model_load",
                    "status": "ok" if logits is not None else "error",
                    "load_s": round(load_s, 2),
                    "device": adapter.device,
                    "dtype": adapter.dtype,
                    "vocab_size": adapter.vocab_size,
                    "eos_token_ids": adapter.eos_token_ids,
                }
            )
        except Exception as exc:
            checks.append({"check": "model_load", "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    report["checks"] = checks
    _print(report, as_json=True)  # stdout stays machine-readable
    print("Notes:", file=sys.stderr)
    print(
        "- Llama-3-8B-Instruct is gated: accept the licence on Hugging Face and set model.model_id/revision.",
        file=sys.stderr,
    )
    print(
        "- Datasets are not committed; run `factanchor datasets --all` (needs network + `[datasets]`).",
        file=sys.stderr,
    )
    print("- No metric is reported unless it was actually computed.", file=sys.stderr)
    return EXIT_OK


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def cmd_demo(args: argparse.Namespace) -> int:
    config = _load_config(
        args,
        {"decoding.alpha": args.alpha, "decoding.max_new_tokens": args.max_new_tokens},
    )
    setup_logging("DEBUG" if args.verbose > 1 else config.run.log_level)
    prompt = _prompt_from_args(args) if (args.prompt or args.prompt_file) else "Who was Marie Curie?"
    from .generation import generate_example
    from .model_adapter import build_adapter

    adapter, load_s = _load_model_with_timing(config)
    knowledge = _build_knowledge(config)
    baseline_config = config.with_section("decoding", {"mode": "baseline", "alpha": 0.0})
    factanchor_config = config.with_section(
        "decoding", {"mode": "factanchor", "alpha": config.decoding.alpha if config.decoding.alpha > 0 else 0.5}
    )
    results: dict[str, Any] = {"prompt": prompt, "model_id": adapter.model_id, "load_s": load_s, "runs": {}}
    for label, run_config in (("baseline", baseline_config), ("factanchor", factanchor_config)):
        record = generate_example(
            example_id=f"demo::{label}",
            prompt=prompt,
            config=run_config,
            adapter=adapter,
            condition=label,
            knowledge=knowledge,
            model_load_s=load_s if label == "baseline" else 0.0,
        )
        results["runs"][label] = {
            "answer": record.answer,
            "status": record.status.value,
            "degraded_reason": record.degraded_reason,
            "alpha": record.alpha,
            "contrast_applied": record.contrast_applied,
            "output_tokens": record.output_tokens,
            "generation_s": record.latency.generation_s,
            "total_s": record.latency.total_s,
            "ms_per_token": record.ms_per_token,
            "entities": [entity.display for entity in record.entities],
            "facts": [fact.render() for fact in record.facts],
            "corruption": record.corruption.summary if record.corruption else None,
        }
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False, default=json_default))
    else:
        print(f"prompt: {prompt}\nmodel : {adapter.model_id}\n")
        for label, payload in results["runs"].items():
            print(f"--- {label} (alpha={payload['alpha']}) ---")
            print(payload["answer"] or "[empty]")
            print(
                f"    status={payload['status']} tokens={payload['output_tokens']} "
                f"generation={payload['generation_s']:.2f}s total={payload['total_s']:.2f}s"
            )
        facts = results["runs"]["factanchor"]
        print(f"\nlinked entities: {', '.join(facts['entities']) or 'none'}")
        print(f"retrieved facts: {len(facts['facts'])}")
        if facts["corruption"]:
            print(f"corruption summary: {facts['corruption']}")
        print(
            "\nWARNING: corrupted facts are an internal experimental condition and are NOT evidence. "
            "Generated text requires independent verification."
        )
    return EXIT_OK


def cmd_cache(args: argparse.Namespace) -> int:
    config = _load_config(args)
    from .utils import DiskCache

    cache = DiskCache(config.cache_dir, ttl_seconds=config.retrieval.cache_ttl_seconds)
    if args.clear:
        removed = cache.clear(args.namespace)
        print(f"removed {removed} cached file(s) from {cache.root}")
        return EXIT_OK
    stats = cache.stats()
    namespaces = sorted(path.name for path in cache.root.iterdir() if path.is_dir()) if cache.root.exists() else []
    print(json.dumps({"cache_dir": str(cache.root), "stats": stats, "namespaces": namespaces}, indent=2))
    return EXIT_OK


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "generate": cmd_generate,
        "run": cmd_run,
        "evaluate": cmd_evaluate,
        "ablate": cmd_ablate,
        "datasets": cmd_datasets,
        "doctor": cmd_doctor,
        "demo": cmd_demo,
        "cache": cmd_cache,
    }
    handler = commands.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces the choice
        parser.error(f"unknown command {args.command!r}")
        return EXIT_CONFIG
    try:
        return int(handler(args))
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return EXIT_CONFIG
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.warning("interrupted")
        return EXIT_ERROR
    except Exception as exc:
        logger.exception("unhandled error: %s", exc)
        return EXIT_ERROR
    finally:
        flush_all_handlers()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
