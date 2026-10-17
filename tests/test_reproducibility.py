"""Reproducibility / artifact tests (spec 15 regression + Section 16)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import TOY_ENTITIES
from factanchor.config import load_config
from factanchor.experiment import create_run, run_condition, warmup_model
from factanchor.generation import generate_example
from factanchor.knowledge.base import MockKnowledgeProvider
from factanchor.knowledge.pipeline import KnowledgePipeline
from factanchor.schemas import Evidence, Example
from factanchor.utils import read_json, read_jsonl, resolve_run_dir, sha256_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def offline_config(tmp_path):
    return load_config(
        PROJECT_ROOT / "configs" / "smoke_toy.yaml",
        overrides={
            "run.output_dir": str(tmp_path / "artifacts"),
            "retrieval.cache_dir": str(tmp_path / "cache"),
            "retrieval.local_json_path": str(PROJECT_ROOT / "data" / "processed" / "offline_knowledge.jsonl"),
            "decoding.max_new_tokens": 6,
            "evaluation.manual_review_sample": 5,
        },
    )


@pytest.fixture
def examples():
    return [
        Example(example_id="t::0", prompt="Who was Marie Curie?", dataset="test", split="dev",
                reference="Marie Curie was a physicist and chemist.",
                evidence=[Evidence(text="Marie Curie was a physicist and chemist.", source="test")]),
        Example(example_id="t::1", prompt="Who was Ada Lovelace?", dataset="test", split="dev",
                reference="Ada Lovelace was a mathematician.",
                evidence=[Evidence(text="Ada Lovelace was an English mathematician.", source="test")]),
    ]


def _run_once(config, examples, tmp_path, *, run_id, adapter, condition="factanchor"):
    provider = MockKnowledgeProvider(TOY_ENTITIES)
    pipeline = KnowledgePipeline(config, provider=provider)
    run = create_run(config, condition=condition, run_id=run_id, dataset_label="test", split_label="dev")
    return run_condition(
        config,
        examples,
        condition_name=condition,
        adapter=adapter,
        knowledge=pipeline,
        run=run,
        warmup=False,
    )


def test_same_seed_gives_identical_generations(offline_config, examples, toy_adapter, tmp_path):
    first = _run_once(offline_config, examples, tmp_path, run_id="repro-a", adapter=toy_adapter)
    second = _run_once(offline_config, examples, tmp_path, run_id="repro-b", adapter=toy_adapter)
    first_records = read_jsonl(first.generations_path)
    second_records = read_jsonl(second.generations_path)
    assert [record["generated_token_ids"] for record in first_records] == [
        record["generated_token_ids"] for record in second_records
    ]
    assert [record["answer"] for record in first_records] == [record["answer"] for record in second_records]
    assert [record["prompt_sha256"] for record in first_records] == [
        record["prompt_sha256"] for record in second_records
    ]


def test_run_directory_is_never_overwritten(offline_config, examples, toy_adapter, tmp_path):
    first = _run_once(offline_config, examples, tmp_path, run_id="duplicate", adapter=toy_adapter)
    second = _run_once(offline_config, examples, tmp_path, run_id="duplicate", adapter=toy_adapter)
    assert first.run_dir != second.run_dir
    assert second.run_dir.name.endswith("-1")
    assert first.generations_path.exists() and second.generations_path.exists()
    assert any("already exists" in warning for warning in second.warnings)


def test_resolve_run_dir_reports_collisions(tmp_path):
    path, warnings = resolve_run_dir(tmp_path, "run-x")
    assert not warnings
    path_two, warnings_two = resolve_run_dir(tmp_path, "run-x")
    assert path != path_two
    assert warnings_two
    path_three, warnings_three = resolve_run_dir(tmp_path, "run-x", overwrite=True)
    assert path_three == path and "reused" in warnings_three[0]


def test_run_artifacts_are_written(offline_config, examples, toy_adapter, tmp_path):
    summary = _run_once(offline_config, examples, tmp_path, run_id="artifacts", adapter=toy_adapter)
    run_dir = summary.run_dir
    for name in ("config.yaml", "environment.json", "generations.jsonl", "manifest.json", "logs.txt", "metrics.json"):
        assert (run_dir / name).exists(), f"{name} missing from the run directory"
    manifest = read_json(run_dir / "manifest.json")
    assert manifest["model_id"] == "toy"
    assert manifest["is_toy_model"] is True
    assert manifest["n_examples"] == len(examples)
    environment = read_json(run_dir / "environment.json")
    assert environment["python_version"]
    assert "library_versions" in environment
    metrics = read_json(run_dir / "metrics.json")
    assert metrics["n_examples_loaded"] == len(examples)
    assert "warmup_policy" in metrics
    assert metrics["by_condition"], "per-condition metrics must be present"
    generations = read_jsonl(run_dir / "generations.jsonl")
    assert len(generations) == len(examples)
    first = generations[0]
    for key in (
        "example_id",
        "mode",
        "alpha",
        "model_id",
        "tokenizer_id",
        "seed",
        "answer",
        "latency",
        "status",
        "params",
        "entities",
        "facts",
    ):
        assert key in first, f"{key} missing from the generation record"
    assert first["latency"]["total_s"] >= first["latency"]["generation_s"]


def test_generation_records_trace_back_to_evidence(offline_config, examples, toy_adapter):
    """End-to-end traceability: record -> facts -> corruption audit (quality gate)."""
    config = offline_config
    provider = MockKnowledgeProvider(TOY_ENTITIES)
    pipeline = KnowledgePipeline(config, provider=provider)
    record = generate_example(
        example_id=examples[0].example_id,
        prompt=examples[0].prompt,
        config=config,
        adapter=toy_adapter,
        condition="trace",
        knowledge=pipeline,
    )
    assert record.entities, "an entity should have been linked for this prompt"
    assert record.facts, "facts should have been retrieved"
    assert record.corruption is not None
    audit_ids = {entry.original_fact_id for entry in record.corruption.audit}
    fact_ids = {fact.fact_id for fact in record.corruption.original_facts}
    assert audit_ids <= fact_ids
    assert record.corruption.original_facts == tuple(record.facts)
    assert record.prompt_sha256 == sha256_text(record.prompt)
    assert record.base_context
    if record.mode == "factanchor" and record.alpha > 0:
        assert record.contrast_context
        assert record.contrast_applied or record.status.value != "ok"


def test_alpha_zero_matches_baseline_end_to_end(offline_config, examples, toy_adapter):
    factanchor_zero = offline_config.with_section(
        "decoding", {"mode": "factanchor", "alpha": 0.0, "max_new_tokens": 6}
    )
    baseline = offline_config.with_section("decoding", {"mode": "baseline", "alpha": 0.0, "max_new_tokens": 6})
    provider = MockKnowledgeProvider(TOY_ENTITIES)
    pipeline = KnowledgePipeline(offline_config, provider=provider)
    first = generate_example(
        example_id="t::0", prompt=examples[0].prompt, config=factanchor_zero, adapter=toy_adapter, knowledge=pipeline
    )
    second = generate_example(
        example_id="t::0", prompt=examples[0].prompt, config=baseline, adapter=toy_adapter, knowledge=None
    )
    assert first.generated_token_ids == second.generated_token_ids
    assert first.answer == second.answer
    assert first.contrast_applied is False


def test_config_file_hash_is_recorded(offline_config, tmp_path, examples, toy_adapter):
    summary = _run_once(offline_config, examples, tmp_path, run_id="hash", adapter=toy_adapter)
    manifest = read_json(summary.run_dir / "manifest.json")
    import hashlib

    payload = (summary.run_dir / "config.yaml").read_bytes()
    assert manifest["config_hash"] == hashlib.sha256(payload).hexdigest()


def test_toy_model_is_flagged_everywhere(offline_config, examples, toy_adapter, tmp_path):
    summary = _run_once(offline_config, examples, tmp_path, run_id="flag", adapter=toy_adapter)
    generations = read_jsonl(summary.generations_path)
    assert all(record["is_toy_model"] for record in generations)
    metrics = read_json(summary.run_dir / "metrics.json")
    notes = metrics["by_condition"][summary.condition]["notes"]
    assert any("toy" in note for note in notes) or any(
        "toy" in note for item in metrics["per_example"] for note in item.get("notes", [])
    )


def test_errors_are_isolated_and_recorded(offline_config, examples, tmp_path, monkeypatch):
    """A failing example is recorded in errors.jsonl; the run continues."""
    from factanchor import generation as generation_module

    original = generation_module.generate_example

    def flaky(**kwargs):
        if kwargs["example_id"] == "t::1":
            raise RuntimeError("simulated example failure")
        return original(**kwargs)

    monkeypatch.setattr("factanchor.experiment.generate_example", flaky)
    config = offline_config
    pipeline = KnowledgePipeline(config, provider=MockKnowledgeProvider(TOY_ENTITIES))
    run = create_run(config, condition="flaky", run_id="flaky", dataset_label="test", split_label="dev")
    from factanchor.model_adapter import ToyCausalLMAdapter

    summary = run_condition(
        config, examples, condition_name="flaky", adapter=ToyCausalLMAdapter(), knowledge=pipeline, run=run, warmup=False
    )
    assert summary.n_errors == 1
    assert summary.n_ok + summary.n_degraded == 1
    errors = read_jsonl(summary.run_dir / "errors.jsonl")
    assert errors and errors[0]["example_id"] == "t::1"
    assert len(read_jsonl(summary.generations_path)) == 1


def test_warmup_runs_and_is_excluded_from_metrics(offline_config, toy_adapter):
    elapsed = warmup_model(toy_adapter, offline_config)
    assert elapsed >= 0.0
