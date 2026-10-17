"""Integration tests: whole pipeline with a mocked knowledge provider (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import FlakyProvider, TOY_ENTITIES
from factanchor.config import load_config
from factanchor.evaluation import evaluate_records, evaluate_run_dir, load_jsonl_examples
from factanchor.experiment import create_run, run_condition
from factanchor.generation import generate_example
from factanchor.knowledge.base import MockKnowledgeProvider
from factanchor.knowledge.pipeline import KnowledgePipeline
from factanchor.schemas import Evidence, Example, Status
from factanchor.utils import read_json, read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_SMOKE = PROJECT_ROOT / "data" / "processed" / "dev_smoke.jsonl"


@pytest.fixture
def config(tmp_path):
    return load_config(
        PROJECT_ROOT / "configs" / "smoke_toy.yaml",
        overrides={
            "run.output_dir": str(tmp_path / "artifacts"),
            "retrieval.cache_dir": str(tmp_path / "cache"),
            "retrieval.local_json_path": str(PROJECT_ROOT / "data" / "processed" / "offline_knowledge.jsonl"),
            "decoding.max_new_tokens": 5,
            "evaluation.manual_review_sample": 3,
        },
    )


def test_dev_smoke_examples_load_and_validate():
    examples = load_jsonl_examples(DEV_SMOKE)
    assert len(examples) == 8
    assert all(example.prompt for example in examples)
    assert examples[0].example_id == "dev::0"
    assert any(example.reference is None for example in examples)


def test_factanchor_completes_with_a_mocked_provider(config, toy_adapter):
    pipeline = KnowledgePipeline(config, provider=MockKnowledgeProvider(TOY_ENTITIES))
    record = generate_example(
        example_id="it::0",
        prompt="Who was Marie Curie?",
        config=config,
        adapter=toy_adapter,
        condition="factanchor",
        knowledge=pipeline,
    )
    assert record.mode == "factanchor"
    assert record.status in {Status.OK, Status.DEGRADED}
    assert record.answer is not None
    assert record.entities and record.facts
    assert record.corruption is not None and record.corruption.strategy == "object_swap"
    assert record.contrast_context is not None
    assert record.latency.generation_s > 0
    assert record.latency.recompute_total() > 0
    assert record.output_tokens == len(record.generated_token_ids)


def test_baseline_mode_never_builds_a_contrast_context(config, toy_adapter):
    baseline = config.with_section("decoding", {"mode": "baseline", "alpha": 0.0})
    pipeline = KnowledgePipeline(config, provider=MockKnowledgeProvider(TOY_ENTITIES))
    record = generate_example(
        example_id="it::1", prompt="Who was Marie Curie?", config=baseline, adapter=toy_adapter, knowledge=pipeline
    )
    assert record.contrast_context is None
    assert record.contrast_applied is False


def test_no_facts_path_is_recorded_as_degraded(config, toy_adapter):
    empty_pipeline = KnowledgePipeline(config, provider=MockKnowledgeProvider([]))
    record = generate_example(
        example_id="it::2",
        prompt="Who was Marie Curie?",
        config=config,
        adapter=toy_adapter,
        condition="factanchor",
        knowledge=empty_pipeline,
    )
    assert "no_contrast_facts" in (record.degraded_reason or "")
    assert record.status is Status.DEGRADED
    assert record.contrast_applied is False


def test_empty_contrast_block_control_runs_the_contrast_branch(config, toy_adapter):
    config = config.with_section("decoding", {"on_missing_contrast": "empty_block"})
    record = generate_example(
        example_id="it::3",
        prompt="Who was Marie Curie?",
        config=config,
        adapter=toy_adapter,
        condition="empty_block",
        knowledge=KnowledgePipeline(config, provider=MockKnowledgeProvider([])),
    )
    assert record.contrast_context is not None
    assert record.contrast_applied is True
    assert "empty_contrast_block" in (record.degraded_reason or "")


def test_retrieval_error_policy_records_an_error_without_generating(config, toy_adapter):
    failing = config.with_section("retrieval", {"fallback_on_unavailable": "error"})
    pipeline = KnowledgePipeline(failing, provider=FlakyProvider(fail_facts=True))
    record = generate_example(
        example_id="it::4",
        prompt="Who was Marie Curie?",
        config=failing,
        adapter=toy_adapter,
        condition="error-policy",
        knowledge=pipeline,
    )
    assert record.status is Status.ERROR
    assert record.answer == ""
    assert record.error
    assert "retrieval_failed" in (record.degraded_reason or "")


def test_no_facts_fallback_policy_continues_generation(config, toy_adapter):
    config = config.with_section("retrieval", {"fallback_on_unavailable": "no_facts"})
    pipeline = KnowledgePipeline(config, provider=FlakyProvider(fail_facts=True))
    record = generate_example(
        example_id="it::5",
        prompt="Who was Marie Curie?",
        config=config,
        adapter=toy_adapter,
        condition="no-facts",
        knowledge=pipeline,
    )
    assert record.status is Status.DEGRADED
    assert record.answer != ""
    assert record.facts == []


def test_disabled_retrieval_skips_the_knowledge_step(config, toy_adapter):
    config = config.with_section("retrieval", {"enabled": False})
    record = generate_example(
        example_id="it::6",
        prompt="Who was Marie Curie?",
        config=config,
        adapter=toy_adapter,
        condition="no-retrieval",
        knowledge=None,
    )
    assert record.entities == [] and record.facts == []
    assert record.latency.entity_linking_s == 0.0


def test_full_run_writes_metrics_and_consumes_them(config, toy_adapter, tmp_path):
    examples = load_jsonl_examples(DEV_SMOKE, max_examples=4)
    pipeline = KnowledgePipeline(config, provider=MockKnowledgeProvider(TOY_ENTITIES))
    run = create_run(config, condition="integration", run_id="integration", dataset_label="dev_smoke", split_label="test")
    summary = run_condition(
        config,
        examples,
        condition_name="integration",
        adapter=toy_adapter,
        knowledge=pipeline,
        run=run,
        warmup=False,
    )
    assert summary.n_ok + summary.n_degraded == len(examples)
    generations = [record for record in read_jsonl(summary.generations_path)]
    assert len(generations) == len(examples)
    metrics_path = summary.run_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = read_json(metrics_path)
    condition_metrics = metrics["by_condition"]["integration"]
    assert condition_metrics["n_examples"] == len(examples)
    assert "definitions" in condition_metrics
    assert condition_metrics["judge"] == "heuristic_lexical"
    # metrics are reproducible from the saved artifacts alone
    from factanchor.schemas import GenerationRecord

    records = [GenerationRecord.model_validate(item) for item in generations]
    recomputed = evaluate_records(records, config=config, examples=examples)
    assert recomputed.aggregate["by_condition"]["integration"]["n_total_claims"] == condition_metrics["n_total_claims"]


def test_evaluate_run_dir_cli_path(config, toy_adapter, tmp_path):
    examples = load_jsonl_examples(DEV_SMOKE, max_examples=3)
    run = create_run(config, condition="evaluate-path", run_id="evaluate-path", dataset_label="dev", split_label="test")
    run_condition(
        config,
        examples,
        condition_name="evaluate-path",
        adapter=toy_adapter,
        knowledge=KnowledgePipeline(config, provider=MockKnowledgeProvider(TOY_ENTITIES)),
        run=run,
        warmup=False,
    )
    output = evaluate_run_dir(run.run_dir, config=config, examples_path=DEV_SMOKE)
    assert output.per_example
    payload = read_json(run.run_dir / "metrics.json")
    assert payload["run_dir"] == str(run.run_dir)
    assert payload["judge"] == "heuristic_lexical" or "judge" in payload


def test_evaluation_without_evidence_reports_unevidenced_claims(config, toy_adapter):
    record = generate_example(
        example_id="it::7",
        prompt="Who was Marie Curie?",
        config=config,
        adapter=toy_adapter,
        condition="no-evidence",
        knowledge=KnowledgePipeline(config, provider=MockKnowledgeProvider(TOY_ENTITIES)),
    )
    output = evaluate_records([record], config=config, examples=[])
    assert output.notes
    metrics = output.per_example[0]
    if metrics.n_claims:
        assert metrics.evidence_coverage == 0.0


def test_gold_reference_judge_on_wikibio_style_evidence(config, toy_adapter):
    """Evidence with factual/nonfactual labels drives the gold-reference judge."""
    config = config.with_section("evaluation", {"claim_judge": "gold_reference"})
    record = generate_example(
        example_id="it::8",
        prompt="Who was Marie Curie?",
        config=config,
        adapter=toy_adapter,
        condition="gold",
        knowledge=KnowledgePipeline(config, provider=MockKnowledgeProvider(TOY_ENTITIES)),
    )
    record = record.model_copy(update={"answer": "Marie Curie was a physicist and chemist."})
    examples = [
        Example(
            example_id="it::8",
            prompt="Who was Marie Curie?",
            dataset="test",
            evidence=[
                Evidence(text="Marie Curie was a physicist and chemist.", source="gold", label="factual"),
                Evidence(text="Marie Curie was an astronaut.", source="gold", label="nonfactual"),
            ],
        )
    ]
    output = evaluate_records([record], config=config, examples=examples)
    assert output.per_example[0].n_claims >= 1
    assert output.by_condition["gold"]["judge"] == "gold_reference"


def test_offline_config_uses_the_local_knowledge_file(config, toy_adapter):
    """The end-to-end offline path (json_file provider) must work without network."""
    pipeline = KnowledgePipeline(config)
    record = generate_example(
        example_id="it::9",
        prompt="Who was Albert Einstein?",
        config=config,
        adapter=toy_adapter,
        condition="offline",
        knowledge=pipeline,
    )
    assert record.entities, "the offline knowledge file should provide a link"
    assert record.facts, "the offline knowledge file should provide facts"
    # the json_file provider must label its facts "LocalJSON", never "Wikidata"
    assert all(fact.source == "LocalJSON" for fact in record.facts)
