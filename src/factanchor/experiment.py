"""Experiment runner: run directories, manifests, conditions and ablations.

Responsibilities (spec Sections 13, 16):

* create a **unique** run directory per experiment and never overwrite results;
* write ``config.yaml``, ``environment.json``, ``generations.jsonl``,
  ``metrics.json``, ``errors.jsonl``, ``logs.txt`` and plots;
* run one named condition over a list of examples, with warm-up, per-example
  seeds, incremental JSONL saving and error isolation (one bad example cannot
  destroy a run);
* build ablation conditions from a grid (cartesian product) plus named extras,
  and report conditions that could not be built instead of skipping them quietly.

Latency policy: model loading is timed once and recorded separately; retrieval,
corruption and context building are timed per example; the token loop is timed
inside :mod:`factanchor.generation`.  Warm-up generations run before the measured
loop and are excluded from the latency statistics (documented in the manifest).
"""

from __future__ import annotations

import copy
import itertools
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .config import Config, ConfigError, Condition, deep_merge, load_config, set_by_path
from .evaluation import evaluate_records, load_jsonl_examples, write_manual_review_sample
from .generation import generate_example
from .metrics import latency_overhead, summarise_latency
from .knowledge.base import resolve_source_label
from .schemas import Example, GenerationRecord, RunManifest, Status
from .utils import (
    append_jsonl,
    collect_environment,
    flush_all_handlers,
    get_logger,
    git_info,
    json_default,
    new_run_id,
    read_json,
    resolve_run_dir,
    set_global_seed,
    sha256_file,
    setup_logging,
    write_json,
)

__all__ = [
    "RunContext",
    "RunSummary",
    "create_run",
    "run_condition",
    "build_conditions",
    "run_ablation",
    "warmup_model",
    "write_run_plots",
]

logger = get_logger("experiment")

WARMUP_POLICY = (
    "One warm-up generation (max_new_tokens=1, same context shape) runs before the measured "
    "loop so that lazy kernel/config initialisation is not counted in generation latency. "
    "Warm-up happens once per process; on CUDA, timings are taken with torch.cuda.synchronize()."
)


@dataclass
class RunContext:
    """Filesystem + logging context of one run directory."""

    run_id: str
    config: Config
    run_dir: Path
    condition: str = ""
    warnings: list[str] = field(default_factory=list)
    config_path: Path | None = None
    _started: float = field(default_factory=time.monotonic)

    # -- paths --------------------------------------------------------------
    @property
    def generations_path(self) -> Path:
        return self.run_dir / "generations.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def errors_path(self) -> Path:
        return self.run_dir / "errors.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def logs_path(self) -> Path:
        return self.run_dir / "logs.txt"

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    # -- writers ------------------------------------------------------------
    def write_manifest(self, manifest: RunManifest) -> Path:
        return write_json(self.manifest_path, manifest.model_dump(mode="json"))

    def write_environment(self) -> Path:
        payload = collect_environment({"run_id": self.run_id, "condition": self.condition})
        payload["argv"] = sys.argv
        return write_json(self.run_dir / "environment.json", payload)

    def append_generation(self, record: GenerationRecord) -> None:
        append_jsonl(self.generations_path, [record])

    def append_error(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.errors_path, [{**payload, "run_id": self.run_id}])


@dataclass
class RunSummary:
    """Outcome of running one condition over a set of examples."""

    run_id: str
    condition: str
    run_dir: Path
    n_examples: int = 0
    n_ok: int = 0
    n_degraded: int = 0
    n_errors: int = 0
    records: list[GenerationRecord] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    model_load_s: float = 0.0
    warmup_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def generations_path(self) -> Path:
        return self.run_dir / "generations.jsonl"

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "condition": self.condition,
            "run_dir": str(self.run_dir),
            "n_examples": self.n_examples,
            "n_ok": self.n_ok,
            "n_degraded": self.n_degraded,
            "n_errors": self.n_errors,
            "model_load_s": round(self.model_load_s, 3),
            "warmup_s": round(self.warmup_s, 3),
            "wall_s": round(self.wall_s, 3),
            "latency": self.latency,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Run creation
# ---------------------------------------------------------------------------


def create_run(
    config: Config,
    *,
    condition: str = "",
    run_id: str | None = None,
    dataset_label: str = "",
    split_label: str = "",
    quiet: bool = False,
) -> RunContext:
    """Create the run directory and write config/environment/logs."""
    resolved_id = run_id or config.run.run_id or new_run_id(config.run.name or condition or "run")
    run_dir, warnings = resolve_run_dir(
        config.output_dir,
        resolved_id,
        overwrite=config.run.overwrite,
        allow_reuse=bool(config.run.run_id) or config.run.allow_run_dir_reuse,
    )
    context = RunContext(run_id=resolved_id, config=config, run_dir=run_dir, condition=condition, warnings=warnings)
    for warning in warnings:
        logger.warning(warning)

    config_path = run_dir / "config.yaml"
    if config_path.exists() and not config.run.overwrite:
        # keep the previous config of a reused directory: never silently replace it
        alternate = run_dir / f"config.{time.strftime('%H%M%S')}.yaml"
        config_path = alternate
        context.warnings.append(f"existing config.yaml preserved; this run's config written to {alternate.name}")
    config_path.write_text(config.to_yaml(), encoding="utf-8")
    context.config_path = config_path
    context.write_environment()
    setup_logging(config.run.log_level, log_file=context.logs_path, quiet=quiet)

    manifest = RunManifest(
        run_id=resolved_id,
        run_name=config.run.name,
        condition=condition,
        git_hash=(git_info().get("hash") if config.run.git_hash else None),
        git_dirty=(git_info().get("dirty") if config.run.git_hash else None),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        hardware={
            "cpu_count": _cpu_count(),
            "cpu": platform.processor() or platform.machine(),
        },
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        tokenizer_id=config.model.tokenizer_id or config.model.model_id,
        tokenizer_revision=config.model.tokenizer_revision or config.model.revision,
        is_toy_model=config.run.is_toy_model or config.model.model_id.strip().lower() == "toy",
        device=config.run.device,
        dtype=config.run.dtype,
        seed=config.run.seed,
        dataset=dataset_label,
        dataset_split=split_label,
        knowledge_provider=config.retrieval.provider if config.retrieval.enabled else "none",
        knowledge_source=(
            resolve_source_label(config.retrieval.provider, config.retrieval.source_name)
            if config.retrieval.enabled
            else "none"
        ),
        corruption_strategy=config.corruption.strategy,
        corruption_rate=config.corruption.corruption_rate,
        config_path=str(config_path),
        config_hash=sha256_file(config_path),
        warnings=list(context.warnings),
        notes=[WARMUP_POLICY, "contrast and baseline branches always share device, dtype and weights"],
    )
    context.write_manifest(manifest)
    # Created up front (empty) so every run directory has the documented layout and
    # consumers can always open errors.jsonl. Zero lines == no failures recorded.
    if not context.errors_path.exists():
        context.errors_path.write_text("", encoding="utf-8")
    logger.info("run %s created at %s (condition=%s)", resolved_id, run_dir, condition or "unnamed")
    return context


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


def warmup_model(
    adapter,
    config: Config,
    *,
    prompt: str = "Warm-up: name one element.",
    runs: int = 1,
) -> float:
    """Run a short generation to warm caches; returns the elapsed seconds."""
    from .generation import TextGenerator

    generator = TextGenerator(adapter, config)
    started = time.monotonic()
    for _ in range(max(1, runs)):
        try:
            generator.generate(prompt, contrast_prompt=prompt if config.decoding.mode == "factanchor" else None, max_new_tokens=1)
        except Exception as exc:  # pragma: no cover - warm-up must not be fatal
            logger.warning("warm-up generation failed (%s); continuing", exc)
            break
    return time.monotonic() - started


# ---------------------------------------------------------------------------
# Condition execution
# ---------------------------------------------------------------------------


def run_condition(
    config: Config,
    examples: Sequence[Example],
    *,
    condition_name: str = "",
    adapter=None,
    model_load_s: float = 0.0,
    knowledge=None,
    run: RunContext | None = None,
    warmup: bool = True,
    warmup_runs: int = 1,
    run_id: str | None = None,
    evaluate: bool = True,
    reporter=None,
    skip_errors: bool = False,
) -> RunSummary:
    """Run one condition over ``examples`` and save every artifact.

    ``skip_errors=False`` keeps going after an example-level failure (recording it
    in ``errors.jsonl``); the summary always reports how many failed.
    """
    seed_report = set_global_seed(config.run.seed, deterministic=config.run.deterministic_algorithms)
    logger.info("seed report: %s", seed_report)

    if run is None:
        run = create_run(
            config,
            condition=condition_name,
            run_id=run_id,
            dataset_label=examples[0].dataset if examples else "",
            split_label=examples[0].split if examples else "",
        )
    if model_load_s:
        manifest = RunManifest.model_validate(read_json(run.manifest_path))
        manifest.notes.append(
            f"model load took {model_load_s:.2f}s (timed separately from generation; "
            "in-process runs loaded the model once)"
        )
        run.write_manifest(manifest)

    if adapter is None:
        from .model_adapter import build_adapter

        load_started = time.monotonic()
        adapter = build_adapter(config)
        model_load_s = time.monotonic() - load_started
    if knowledge is None and config.retrieval.enabled:
        from .knowledge.pipeline import build_provider, KnowledgePipeline

        knowledge = KnowledgePipeline(config, provider=build_provider(config))

    warmup_s = warmup_model(adapter, config, runs=warmup_runs) if warmup else 0.0
    if warmup:
        logger.info("warm-up completed in %.2fs (excluded from latency statistics)", warmup_s)

    # A brand-new run directory has no generations.jsonl; if the file exists the
    # directory was reused deliberately, so we append and say so loudly -- previous
    # results are never deleted.
    generations_path = run.generations_path
    if generations_path.exists():
        warning = (
            f"{generations_path} already exists: new records will be APPENDED. "
            "Previous results are preserved; use a new run id for a clean comparison."
        )
        logger.warning(warning)
        run.warnings.append(warning)

    summary = RunSummary(
        run_id=run.run_id,
        condition=condition_name or config.run.name,
        run_dir=run.run_dir,
        n_examples=len(examples),
        model_load_s=model_load_s,
        warmup_s=warmup_s,
        warnings=list(run.warnings),
    )
    wall_started = time.monotonic()
    for index, example in enumerate(examples):
        per_example_seed = config.run.seed if not config.decoding.do_sample else config.run.seed + index
        example_config = config.with_section("run", {"seed": per_example_seed})
        try:
            record = generate_example(
                example_id=example.example_id,
                prompt=example.prompt,
                config=example_config,
                adapter=adapter,
                run_id=run.run_id,
                condition=condition_name or config.run.name,
                dataset=example.dataset,
                split=example.split,
                task=example.task,
                reference=example.reference,
                entity_hint=example.entity_hint,
                knowledge=knowledge,
                reporter=reporter,
                model_load_s=model_load_s if index == 0 else 0.0,
            )
        except Exception as exc:
            logger.exception("example %s failed", example.example_id)
            payload = {
                "example_id": example.example_id,
                "condition": condition_name or config.run.name,
                "error": f"{type(exc).__name__}: {exc}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            run.append_error(payload)
            summary.errors.append(payload)
            summary.n_errors += 1
            if not skip_errors:
                continue
            raise
        record.metadata.setdefault("dataset", example.dataset)
        record.metadata["per_example_seed"] = per_example_seed
        run.append_generation(record)
        summary.records.append(record)
        if record.status is Status.OK:
            summary.n_ok += 1
        elif record.status is Status.DEGRADED:
            summary.n_degraded += 1
        else:
            summary.n_errors += 1
        if reporter is not None:
            reporter(
                "example_done",
                {
                    "index": index,
                    "total": len(examples),
                    "example_id": example.example_id,
                    "status": record.status.value,
                    "answer_chars": len(record.answer),
                    "generation_s": record.latency.generation_s,
                },
            )
    summary.wall_s = time.monotonic() - wall_started
    summary.latency = summarise_latency(summary.records)
    summary.latency["wall_s"] = summary.wall_s

    if evaluate and summary.records:
        try:
            output = evaluate_records(summary.records, config=config, examples=examples)
            payload = output.to_dict()
            payload.update(
                {
                    "run_id": run.run_id,
                    "condition": summary.condition,
                    "run_dir": str(run.run_dir),
                    "dataset": examples[0].dataset if examples else "",
                    "split": examples[0].split if examples else "",
                    "n_examples_loaded": len(examples),
                    "latency": summary.latency,
                    "model_load_s": summary.model_load_s,
                    "warmup_s": summary.warmup_s,
                    "wall_s": summary.wall_s,
                    "evaluation_settings": config.evaluation.model_dump(mode="json"),
                    "warmup_policy": WARMUP_POLICY,
                }
            )
            summary.metrics = payload
            write_json(run.metrics_path, payload)
            if config.evaluation.manual_review_sample:
                write_manual_review_sample(
                    output,
                    summary.records,
                    run.run_dir / "manual_review.jsonl",
                    sample_size=config.evaluation.manual_review_sample,
                )
            plots = write_run_plots(summary, run.plots_dir)
            if plots:
                logger.info("wrote %d plot(s) to %s", len(plots), run.plots_dir)
        except Exception as exc:  # evaluation must not silently pass
            logger.exception("evaluation failed for %s", summary.condition)
            summary.errors.append({"stage": "evaluation", "error": f"{type(exc).__name__}: {exc}"})
    elif evaluate:
        logger.warning("no records were produced; metrics.json not written")

    manifest = RunManifest.model_validate(read_json(run.manifest_path))
    manifest.n_examples = len(examples)
    manifest.notes.append(
        f"finished: {summary.n_ok} ok, {summary.n_degraded} degraded, {summary.n_errors} error(s) "
        f"in {summary.wall_s:.1f}s"
    )
    manifest.warnings.extend(summary.warnings)
    run.write_manifest(manifest)
    flush_all_handlers()
    logger.info("condition %s finished: %s", summary.condition, summary.describe())
    return summary


# ---------------------------------------------------------------------------
# Ablation grid
# ---------------------------------------------------------------------------


def build_conditions(ablation_path: Path | str, *, base_overrides: dict[str, Any] | None = None) -> list[Condition]:
    """Build named conditions from an ablation config file.

    File format (see ``configs/experiments.yaml``)::

        base: default.yaml            # template config
        knobs: {path: value, ...}     # applied to every condition
        grid:  {path: [v1, v2, ...]}  # cartesian product
        conditions: [{name, note, overrides}, ...]

    Returns :class:`~factanchor.config.Condition` objects.  Invalid combinations
    raise :class:`ConfigError` naming the offending condition, rather than being
    dropped silently.
    """
    import yaml

    path = Path(ablation_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping")
    base_name = raw.get("base", "default.yaml")
    base_path = Path(base_name)
    if not base_path.is_absolute() and not base_path.exists():
        candidate = path.parent / base_name
        if candidate.exists():
            base_path = candidate
    base_config = load_config(base_path)
    knobs: dict[str, Any] = raw.get("knobs") or {}
    grid: dict[str, list[Any]] = raw.get("grid") or {}
    named: list[dict[str, Any]] = raw.get("conditions") or []
    if not isinstance(knobs, dict) or not isinstance(grid, dict) or not isinstance(named, list):
        raise ConfigError(f"{path}: 'knobs' and 'grid' must be mappings and 'conditions' a list")

    conditions: list[Condition] = []
    failures: list[str] = []
    if grid:
        keys = list(grid)
        for combination in itertools.product(*(grid[key] for key in keys)):
            overrides = dict(zip(keys, combination))
            name = "-".join(f"{_short_key(key)}{_slug(value)}" for key, value in overrides.items())
            merged = {**knobs, **overrides}
            try:
                conditions.append(
                    Condition(
                        name=name,
                        note="grid condition",
                        config=base_config.apply_overrides(merged),
                        overrides=merged,
                    )
                )
            except ConfigError as exc:
                failures.append(f"grid condition {name}: {exc}")
    for entry in named:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ConfigError(f"{path}: every entry in 'conditions' needs a 'name'")
        overrides = {**knobs, **(entry.get("overrides") or {})}
        try:
            conditions.append(
                Condition(
                    name=str(entry["name"]),
                    note=str(entry.get("note", "")),
                    config=base_config.apply_overrides(overrides),
                    overrides=overrides,
                )
            )
        except ConfigError as exc:
            failures.append(f"condition {entry.get('name')}: {exc}")
    if base_overrides:
        rebuilt: list[Condition] = []
        for condition in conditions:
            try:
                rebuilt.append(
                    Condition(
                        name=condition.name,
                        note=condition.note,
                        config=condition.config.apply_overrides(base_overrides),
                        overrides={**condition.overrides, **base_overrides},
                    )
                )
            except ConfigError as exc:
                failures.append(f"condition {condition.name} (CLI overrides): {exc}")
        conditions = rebuilt
    if failures:
        message = "some conditions are invalid and were NOT run:\n  - " + "\n  - ".join(failures)
        logger.error(message)
        raise ConfigError(message)
    if not conditions:
        raise ConfigError(f"{path} defines no conditions (need a non-empty 'grid' or 'conditions')")
    return conditions


def _short_key(path: str) -> str:
    tail = path.split(".")[-1]
    return {"corruption_rate": "rate", "max_facts_per_entity": "facts", "max_new_tokens": "tokens"}.get(tail, tail)


def _slug(value: Any) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return text


def run_ablation(
    ablation_path: Path | str,
    *,
    examples: Sequence[Example],
    base_overrides: dict[str, Any] | None = None,
    results_path: Path | str | None = None,
    warmup: bool = True,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Run every condition in an ablation file on the *same* examples.

    The same adapter (single model load) and the same example list are used for
    every condition, which is what makes the comparison paired (spec 11.5).
    """
    conditions = build_conditions(ablation_path, base_overrides=base_overrides)
    if not examples:
        raise ValueError("run_ablation needs at least one example")

    load_started = time.monotonic()
    first_config = conditions[0].config
    from .model_adapter import build_adapter

    adapter = build_adapter(first_config)
    model_load_s = time.monotonic() - load_started

    results: dict[str, Any] = {
        "ablation": str(ablation_path),
        "n_examples": len(examples),
        "model_id": first_config.model.model_id,
        "model_load_s": model_load_s,
        "warmup_policy": WARMUP_POLICY,
        "conditions": {},
    }
    shared_run_id = new_run_id(first_config.run.name or "ablation")
    summaries: dict[str, RunSummary] = {}
    for condition in conditions:
        config = condition.config
        if config.run.run_id is None:
            config = config.with_section("run", {"run_id": None})
        run = create_run(
            config,
            condition=condition.name,
            run_id=f"{shared_run_id}-{condition.name}",
            dataset_label=examples[0].dataset,
            split_label=examples[0].split,
        )
        logger.info("=== condition %s: %s", condition.name, condition.note)
        summary = run_condition(
            config,
            examples,
            condition_name=condition.name,
            adapter=adapter,
            model_load_s=model_load_s,
            run=run,
            warmup=warmup,
            evaluate=True,
            skip_errors=skip_errors,
        )
        summaries[condition.name] = summary
        results["conditions"][condition.name] = {
            "note": condition.note,
            "overrides": condition.overrides,
            "summary": summary.describe(),
            "metrics": summary.metrics.get("by_condition", {}).get(condition.name, {}),
            "run_dir": str(summary.run_dir),
            "generations": str(summary.generations_path),
        }

    # paired latency overhead of every condition against the baseline-like one
    reference = _pick_reference_condition(conditions, summaries)
    if reference:
        reference_records = summaries[reference].records
        for name, summary in summaries.items():
            if name == reference or not summary.records:
                continue
            results["conditions"][name]["overhead_vs_reference"] = {
                "reference": reference,
                "generation": latency_overhead(summary.records, reference_records),
                "total": latency_overhead(summary.records, reference_records, key=lambda r: r.latency.total_s),
                "factuality": _factuality_delta(summary, summaries[reference]),
            }
    results["reference_condition"] = reference
    results["comparison_note"] = (
        "Every condition ran on the same examples with the same model weights, device and "
        "decoding settings except for the documented intervention. Factuality deltas are "
        "descriptive: with a small development subset they are not significance tests. "
        "See 'warnings' for conditions that were degraded or errored."
    )
    results["warnings"] = [
        f"{name}: {len(summary.errors)} error(s)" for name, summary in summaries.items() if summary.errors
    ]
    if results_path:
        write_json(results_path, results)
        logger.info("wrote ablation results to %s", results_path)
    return results


def _pick_reference_condition(conditions: Sequence[Condition], summaries: dict[str, RunSummary]) -> str | None:
    """Choose the condition that latency overhead is measured against.

    Preference order: a conventional baseline name, then any condition whose
    decoding mode is ``baseline`` (alpha=0 by validation), then the first
    condition.  This keeps "overhead vs baseline" meaningful even for grids whose
    condition names are generated.
    """
    for name in ("base_model_no_context", "baseline"):
        if name in summaries:
            return name
    for condition in conditions:
        if condition.config.decoding.mode == "baseline" and condition.name in summaries:
            return condition.name
    return next(iter(summaries)) if summaries else None


def _factuality_delta(condition: RunSummary, reference: RunSummary) -> dict[str, float | None]:
    left = condition.metrics.get("by_condition", {}).get(condition.condition, {})
    right = reference.metrics.get("by_condition", {}).get(reference.condition, {})
    out: dict[str, float | None] = {}
    for key in ("mean_claim_support_rate", "mean_hallucination_rate", "mean_evidence_coverage", "mean_factscore"):
        a, b = left.get(key), right.get(key)
        out[key] = (a - b) if (a is not None and b is not None) else None
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def write_run_plots(summary: RunSummary, plots_dir: Path) -> list[Path]:
    """Write latency/factuality plots for one condition (skipped if matplotlib is absent)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        logger.info("matplotlib is not available; skipping plots")
        return []
    plots_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    records = summary.records
    if not records:
        return []

    # 1) per-example latency
    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.plot(range(len(records)), [r.latency.generation_s for r in records], marker="o", markersize=3, label="generation")
    axis.plot(range(len(records)), [r.latency.total_s for r in records], marker="x", markersize=3, label="end-to-end")
    axis.set_xlabel("example index")
    axis.set_ylabel("seconds")
    axis.set_title(f"{summary.condition}: latency per example")
    axis.legend()
    figure.tight_layout()
    path = plots_dir / "latency_per_example.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    created.append(path)

    # 2) tokens vs latency scatter
    figure, axis = plt.subplots(figsize=(4.5, 3.5))
    axis.scatter([r.output_tokens for r in records], [1000 * r.latency.generation_s for r in records], s=14)
    axis.set_xlabel("generated tokens")
    axis.set_ylabel("generation time (ms)")
    axis.set_title(f"{summary.condition}: tokens vs time")
    figure.tight_layout()
    path = plots_dir / "tokens_vs_time.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    created.append(path)

    # 3) factuality summary if available
    metrics = summary.metrics.get("by_condition", {}).get(summary.condition, {})
    values = [
        metrics.get("mean_claim_support_rate"),
        metrics.get("mean_hallucination_rate"),
        metrics.get("mean_evidence_coverage"),
    ]
    if any(value is not None for value in values):
        figure, axis = plt.subplots(figsize=(4.5, 3.5))
        labels = ["claim support", "hallucination", "evidence coverage"]
        axis.bar(labels, [value if value is not None else 0.0 for value in values])
        axis.set_ylim(0, 1)
        axis.set_title(f"{summary.condition}: claim metrics")
        figure.tight_layout()
        path = plots_dir / "claim_metrics.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        created.append(path)
    return created


def write_ablation_plot(results: dict[str, Any], path: Path | str) -> Path | None:
    """Plot factuality and latency across conditions of an ablation run."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return None
    conditions = list(results.get("conditions", {}))
    if not conditions:
        return None
    support = []
    hallucination = []
    latency = []
    for name in conditions:
        entry = results["conditions"][name]
        metrics = entry.get("metrics", {}) or {}
        support.append(metrics.get("mean_claim_support_rate"))
        hallucination.append(metrics.get("mean_hallucination_rate"))
        latency.append(entry.get("summary", {}).get("latency", {}).get("generation_s_median"))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    positions = range(len(conditions))
    axes[0].bar([p - 0.2 for p in positions], [v if v is not None else 0 for v in support], width=0.4, label="claim support")
    axes[0].bar([p + 0.2 for p in positions], [v if v is not None else 0 for v in hallucination], width=0.4, label="hallucination")
    axes[0].set_xticks(list(positions))
    axes[0].set_xticklabels(conditions, rotation=45, ha="right", fontsize=7)
    axes[0].legend(fontsize=7)
    axes[0].set_title("factuality by condition")
    axes[1].bar(list(positions), [v if v is not None else 0 for v in latency])
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels(conditions, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("median generation time (s)")
    axes[1].set_title("latency by condition")
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=140)
    plt.close(figure)
    return target


def iter_examples_from_file(path: Path | str, *, max_examples: int | None = None) -> Iterator[Example]:
    """Stream examples from a JSONL file (used by the CLI ``--dataset`` flag)."""
    for example in load_jsonl_examples(path, max_examples=max_examples):
        yield example
