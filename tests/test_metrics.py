"""Metric tests: hand-computed expectations, empty inputs, missing evidence (spec 15)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from factanchor.metrics import (
    METRIC_DEFINITIONS,
    LexicalEvidenceJudge,
    aggregate_metrics,
    bootstrap_ci,
    exact_match,
    extract_claims,
    judge_claims,
    latency_overhead,
    rouge_l,
    score_record,
    summarise_latency,
    token_f1,
)
from factanchor.schemas import (
    Evidence,
    ExampleMetrics,
    GenerationRecord,
    LatencyBreakdown,
    Status,
)


def _record(
    answer: str,
    *,
    reference: str | None = None,
    generation_s: float = 1.0,
    tokens: int = 10,
    example_id: str = "t::0",
) -> GenerationRecord:
    latency = LatencyBreakdown(generation_s=generation_s, total_s=generation_s + 0.5)
    return GenerationRecord(
        example_id=example_id,
        prompt="Who was Marie Curie?",
        answer=answer,
        reference=reference,
        output_tokens=tokens,
        latency=latency,
        condition="test",
        model_id="toy",
        status=Status.OK,
    )


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


def test_empty_answer_yields_no_claims():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_claims_are_extracted_from_factual_sentences():
    claims = extract_claims(
        "Marie Curie was a physicist. She won the Nobel Prize in 1903. I hope this helps!"
    )
    texts = [claim.text for claim in claims]
    assert any("physicist" in text for text in texts)
    assert any("1903" in text for text in texts)
    assert not any("hope this helps" in text for text in texts)
    assert all(claim.claim_id for claim in claims)


def test_claims_capture_numbers_and_entities():
    claims = extract_claims("The population of the United States was 331,449,281 in 2020.")
    assert claims
    assert "331,449,281" in claims[0].numbers
    assert "United" in claims[0].entities or "United States" in claims[0].text


def test_model_claim_extraction_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError, match="not implemented in v1"):
        extract_claims("Marie Curie was a physicist.", extractor="model")


def test_claim_extraction_is_deterministic():
    answer = "Ada Lovelace was a mathematician. She worked on the Analytical Engine in 1843."
    first = [claim.claim_id for claim in extract_claims(answer)]
    second = [claim.claim_id for claim in extract_claims(answer)]
    assert first == second


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------


def test_missing_evidence_is_reported_as_no_evidence():
    claims = extract_claims("Marie Curie was a physicist.")
    judgement = judge_claims(claims, [], judge="heuristic_lexical")
    assert judgement.verdicts
    assert all(verdict.label == "no_evidence" for verdict in judgement.verdicts)


def test_lexical_judge_supports_a_matching_claim():
    claim = extract_claims("Marie Curie was a Polish physicist who studied radioactivity.")[0]
    evidence = [
        Evidence(
            text="Marie Curie was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.",
            source="test",
        )
    ]
    verdict = LexicalEvidenceJudge().judge(claim, evidence)
    assert verdict.label == "supported"
    assert verdict.score >= 0.35


def test_lexical_judge_flags_conflicting_numbers_as_contradicted():
    claim = extract_claims("Marie Curie won the Nobel Prize in Physics in 1921.")[0]
    evidence = [Evidence(text="Marie Curie won the Nobel Prize in Physics in 1903.", source="test")]
    verdict = LexicalEvidenceJudge().judge(claim, evidence)
    assert verdict.label == "contradicted"
    assert "conflicting figures" in verdict.rationale


def test_lexical_judge_marks_unrelated_evidence_unsupported():
    claim = extract_claims("Albert Einstein developed the theory of relativity.")[0]
    evidence = [Evidence(text="Mount Everest is 8,848 metres tall.", source="test")]
    verdict = LexicalEvidenceJudge().judge(claim, evidence)
    assert verdict.label == "unsupported"


def test_model_judge_requires_an_adapter():
    with pytest.raises(ValueError, match="judge_model_id"):
        judge_claims([], [], judge="model_judge", adapter=None)


def test_unknown_judge_raises():
    with pytest.raises(ValueError, match="unknown claim judge"):
        judge_claims([], [], judge="vibes")


def test_gold_reference_judge_uses_labels():
    claim = extract_claims("Marie Curie won the Nobel Prize in Physics in 1903.")[0]
    factual = [Evidence(text="Marie Curie won the Nobel Prize in Physics in 1903.", source="gold", label="factual")]
    nonfactual = [Evidence(text="Marie Curie won the Nobel Prize in Physics in 1903.", source="gold", label="nonfactual")]
    from factanchor.metrics import GoldReferenceJudge

    judge = GoldReferenceJudge()
    assert judge.judge(claim, factual).label == "supported"
    assert judge.judge(claim, nonfactual).label == "contradicted"


# ---------------------------------------------------------------------------
# Per-example scoring (hand-computed)
# ---------------------------------------------------------------------------


def test_score_record_hand_computed_rates():
    answer = "Marie Curie was a physicist. Marie Curie was born in 1867."
    evidence = [
        Evidence(text="Marie Curie was a physicist and chemist.", source="test"),
        Evidence(text="Marie Curie was born in 1867 in Warsaw.", source="test"),
    ]
    record = _record(answer)
    metrics, judgement = score_record(record, evidence=evidence)
    assert metrics.n_claims == len(judgement.claims)
    assert metrics.n_supported + metrics.n_unsupported + metrics.n_contradicted + metrics.n_no_evidence == metrics.n_claims
    judged = metrics.n_supported + metrics.n_unsupported + metrics.n_contradicted
    if judged:
        assert metrics.claim_support_rate == metrics.n_supported / judged
        expected_hallucination = (
            metrics.n_unsupported + metrics.n_contradicted
        ) / judged
        assert math.isclose(metrics.hallucination_rate, expected_hallucination)
    else:
        assert metrics.claim_support_rate is None
        assert metrics.hallucination_rate is None
    assert metrics.status == "ok"


def test_score_record_without_evidence_marks_all_claims_unevidenced():
    record = _record("Marie Curie was a physicist who discovered radium.")
    metrics, _ = score_record(record, evidence=[])
    if metrics.n_claims:
        assert metrics.evidence_coverage == 0.0
        assert metrics.hallucination_rate is None
        assert metrics.factscore is None  # nothing was judged


def test_score_record_empty_answer_is_none_not_zero():
    record = _record("")
    metrics, _ = score_record(record, evidence=[])
    assert metrics.n_claims == 0
    assert metrics.claim_support_rate is None
    assert metrics.hallucination_rate is None
    assert metrics.factscore is None
    assert metrics.notes


def test_score_record_skips_semantic_metrics_without_reference():
    record = _record("Marie Curie was a physicist.", reference=None)
    metrics, _ = score_record(record, evidence=[])
    assert metrics.token_f1 is None and metrics.rouge_l is None and metrics.exact_match is None


def test_score_record_flags_toy_models():
    record = _record("Marie Curie was a physicist.")
    record = record.model_copy(update={"is_toy_model": True})
    metrics, _ = score_record(record, evidence=[])
    assert any("toy model" in note for note in metrics.notes)


def test_error_records_produce_no_claims():
    record = _record("some text", ).model_copy(update={"status": Status.ERROR, "error": "boom"})
    metrics, judgement = score_record(record, evidence=[])
    assert metrics.n_claims == 0
    assert judgement.claims == []
    assert any("status=error" in note for note in metrics.notes)


# ---------------------------------------------------------------------------
# Similarity metrics (hand-computed)
# ---------------------------------------------------------------------------


def test_token_f1_hand_computed():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("a b", "c d") == 0.0
    assert token_f1("", "a") is None
    assert math.isclose(token_f1("a b c d", "a b"), 2 * (2 / 4) * (2 / 2) / ((2 / 4) + (2 / 2)))


def test_rouge_l_hand_computed():
    assert rouge_l("the cat sat", "the cat sat") == 1.0
    assert rouge_l("a b c d", "a c") == pytest.approx(2 * (2 / 4) * (2 / 2) / ((2 / 4) + (2 / 2)))
    assert rouge_l("", "a") is None


def test_exact_match_normalises_case_and_punctuation():
    assert exact_match("Marie Curie!", "marie curie") == 1.0
    assert exact_match("Marie Curie", "Albert Einstein") == 0.0
    assert exact_match("", "x") is None


# ---------------------------------------------------------------------------
# Aggregation and statistics
# ---------------------------------------------------------------------------


def test_bootstrap_ci_needs_two_values():
    assert bootstrap_ci([0.5]) is None
    assert bootstrap_ci([]) is None
    interval = bootstrap_ci([0.2, 0.4, 0.6, 0.8, 1.0], samples=500, seed=0)
    assert interval is not None and interval[0] < interval[1]


def test_aggregate_metrics_handles_empty_input():
    summary = aggregate_metrics([])
    assert summary["n_examples"] == 0
    assert summary["notes"]
    assert summary["definitions"] is METRIC_DEFINITIONS


def test_aggregate_metrics_computes_means_and_counts():
    items = [
        ExampleMetrics(example_id="a", condition="c", n_claims=2, n_supported=2, claim_support_rate=1.0, hallucination_rate=0.0, evidence_coverage=1.0, factscore=1.0, status="ok"),
        ExampleMetrics(example_id="b", condition="c", n_claims=2, n_supported=0, n_unsupported=2, claim_support_rate=0.0, hallucination_rate=1.0, evidence_coverage=1.0, factscore=0.0, status="ok"),
        ExampleMetrics(example_id="c", condition="c", n_claims=0, claim_support_rate=None, hallucination_rate=None, status="error"),
    ]
    summary = aggregate_metrics(items, condition="c", judge="heuristic_lexical")
    assert summary["n_examples"] == 3
    assert summary["n_ok"] == 2 and summary["n_errors"] == 1
    assert summary["n_with_claims"] == 2
    assert summary["mean_claim_support_rate"] == 0.5
    assert summary["mean_hallucination_rate"] == 0.5
    assert "claim_support_rate" in summary["ci"]


def test_aggregate_metrics_ignores_none_values():
    items = [
        ExampleMetrics(example_id="a", condition="c", n_claims=1, n_supported=1, claim_support_rate=1.0, status="ok"),
        ExampleMetrics(example_id="b", condition="c", n_claims=0, claim_support_rate=None, status="ok"),
    ]
    summary = aggregate_metrics(items, condition="c")
    assert summary["mean_claim_support_rate"] == 1.0
    assert summary["mean_hallucination_rate"] is None


def test_summarise_latency_reports_medians_and_spread():
    records = [
        _record("answer", generation_s=1.0, tokens=10),
        _record("answer", generation_s=3.0, tokens=30),
    ]
    summary = summarise_latency(records)
    assert summary["generation_s_median"] == 2.0
    assert summary["ms_per_token_median"] == 100.0
    assert summary["n"] == 2.0


def test_latency_overhead_is_paired_by_example_id():
    baseline = [
        _record("a", generation_s=1.0, example_id="e0"),
        _record("b", generation_s=2.0, example_id="e1"),
    ]
    condition = [
        _record("a", generation_s=2.0, example_id="e0"),  # +100%
        _record("b", generation_s=3.0, example_id="e1"),  # +50%
    ]
    overhead = latency_overhead(condition, baseline)
    assert overhead["n_pairs"] == 2
    # deltas are +100% and +50%: the median of two values is their mean
    assert overhead["overhead_pct_median"] == pytest.approx(75.0)
    assert overhead["overhead_pct_mean"] == pytest.approx(75.0)
    assert overhead["absolute_delta_s_median"] == pytest.approx(1.0)


def test_latency_overhead_without_matching_examples():
    overhead = latency_overhead([_record("a")], [])
    assert overhead["n_pairs"] == 0
    assert overhead["overhead_pct_median"] is None


def test_metric_definitions_cover_reported_keys():
    for key in ("claim_support_rate", "hallucination_rate", "factscore", "latency_overhead_pct"):
        assert key in METRIC_DEFINITIONS
        assert len(METRIC_DEFINITIONS[key]) > 20
