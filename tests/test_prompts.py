"""Context builder tests (spec Section 9: separation, wording, safeguards)."""

from __future__ import annotations

import pytest

from factanchor.config import ContextConfig, load_config
from factanchor.corruption import corrupt_facts
from factanchor.schemas import Fact
from factanchor.prompts import (
    EXPERIMENTAL_HEADER,
    NEUTRAL_HEADER,
    ContextBuilder,
    ReferenceLeakError,
    build_contexts,
)


def test_base_context_contains_only_the_question(config):
    pair = build_contexts("Who was Marie Curie?", config=config)
    assert "Who was Marie Curie?" in pair.base_context
    assert EXPERIMENTAL_HEADER not in pair.base_context
    assert "physicist" not in pair.base_context


def test_contrast_context_holds_corrupted_facts_and_base_does_not(config, sample_facts):
    result = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=1.0, seed=3)
    pair = build_contexts("Who was Marie Curie?", config=config, retrieval=result)
    assert pair.contrast_context is not None
    assert EXPERIMENTAL_HEADER in pair.contrast_context
    assert pair.has_contrast_facts
    corrupted_labels = {
        fact.object_label for fact in result.corrupted_facts if fact.metadata.get("corrupted_from")
    }
    assert corrupted_labels, "the fixture should produce at least one corrupted label"
    for label in corrupted_labels:
        assert label in pair.contrast_context
        assert label not in pair.base_context


def test_original_facts_never_reach_the_contrast_block_when_swapped(config, sample_facts):
    """A swapped fact must not also appear in its original form in the same block."""
    result = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=0.5, seed=11)
    pair = build_contexts("Who was Marie Curie?", config=config, retrieval=result)
    for record in result.audit:
        if record.is_corrupted and record.corruption_type == "object_swap":
            assert f"| {record.original_object}" not in pair.contrast_context


def test_contexts_are_identical_except_for_the_fact_block(config, sample_facts):
    result = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=0.5, seed=5)
    pair = build_contexts("Who was Marie Curie?", config=config, retrieval=result)
    assert pair.base_context.strip() == "Who was Marie Curie?"
    assert pair.contrast_context.startswith("User question:\nWho was Marie Curie?")
    assert pair.template_version == config.context.template_version


def test_base_context_lengths_do_not_depend_on_corruption(config, sample_facts):
    first = build_contexts("Who was Marie Curie?", config=config, retrieval=corrupt_facts(sample_facts, seed=1))
    second = build_contexts("Who was Marie Curie?", config=config, retrieval=corrupt_facts(sample_facts, seed=2))
    assert first.base_context == second.base_context


def test_no_facts_means_no_contrast_context_by_default(config):
    pair = build_contexts("Who was Marie Curie?", config=config)
    assert pair.contrast_context is None
    assert pair.fact_block == ""
    assert pair.notes


def test_empty_fact_block_control_keeps_the_skeleton(config):
    config = config.with_section("context", {"include_empty_fact_block": True})
    pair = build_contexts("Who was Marie Curie?", config=config)
    assert pair.contrast_context is not None
    assert EXPERIMENTAL_HEADER in pair.contrast_context
    assert pair.n_facts_rendered == 0


def test_reference_leak_is_blocked(config):
    reference = "Marie Curie was a Polish and naturalised-French physicist who won two Nobel Prizes."
    builder = ContextBuilder.from_config(config)
    with pytest.raises(ReferenceLeakError):
        builder.build("Marie Curie was a Polish and naturalised-French physicist who won two Nobel Prizes.", reference=reference)


def test_short_references_do_not_trigger_the_leak_guard(config):
    builder = ContextBuilder.from_config(config)
    pair = builder.build("Who was Marie Curie?", reference="Marie Curie")
    assert pair.base_context.strip() == "Who was Marie Curie?"


def test_trusted_facts_require_the_explicit_flag(config, sample_facts):
    builder = ContextBuilder.from_config(config)
    with pytest.raises(ValueError, match="trusted_facts_in_base"):
        builder.build("Who was Marie Curie?", trusted_facts=sample_facts)


def test_trusted_facts_go_to_the_base_branch_only_when_enabled(config, sample_facts):
    config = config.with_section("context", {"trusted_facts_in_base": True})
    pair = build_contexts("Who was Marie Curie?", config=config)
    # build_contexts does not pass trusted facts; use the builder directly
    builder = ContextBuilder.from_config(config)
    pair = builder.build("Who was Marie Curie?", trusted_facts=sample_facts[:2])
    assert "Retrieved entity facts (trusted):" in pair.base_context
    assert "physicist" in pair.base_context


def test_fact_block_truncation_is_reported(config):
    config = config.with_section("context", {"max_fact_block_chars": 90})
    facts = corrupt_facts(
        [
            Fact(
                subject_id="Q1",
                subject_label=f"Subject {index}",
                predicate_id="P1",
                predicate_label="predicate",
                object_label="object value",
            )
            for index in range(12)
        ],
        strategy="none",
        corruption_rate=0.0,
    )
    pair = build_contexts("Anything?", config=config, retrieval=facts)
    assert pair.fact_block_was_truncated is True
    assert any("truncated" in note for note in pair.notes)
    assert len(pair.fact_block) <= 90 + 40  # header + one line tolerance


def test_prompt_ablation_switches_the_header(config):
    config = config.with_section("context", {"label_as_experimental": False})
    builder = ContextBuilder.from_config(config)
    result = corrupt_facts(
        [Fact(subject_id="Q1", subject_label="A", predicate_id="P1", object_label="B")],
        strategy="none",
        corruption_rate=0.0,
    )
    pair = builder.build("question", retrieval=result)
    assert NEUTRAL_HEADER in pair.contrast_context
    assert EXPERIMENTAL_HEADER not in pair.contrast_context


def test_template_validation_catches_missing_placeholders():
    with pytest.raises(ValueError, match="question"):
        ContextConfig(base_template="no placeholder here")
    with pytest.raises(ValueError, match="fact_block"):
        ContextConfig(contrast_template="only {question}")
    with pytest.raises(ValueError, match="fact_line_format"):
        ContextConfig(fact_line_format="- {subject} only")


def test_describe_reports_metadata_without_text(config, sample_facts):
    builder = ContextBuilder.from_config(config)
    pair = builder.build("Who was Marie Curie?", retrieval=corrupt_facts(sample_facts, seed=1))
    described = builder.describe(pair)
    assert described["n_facts_rendered"] >= 0
    assert "base_context" not in described
    assert isinstance(described["notes"], list)


def test_corrupted_fact_ids_are_listed(config, sample_facts):
    result = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=1.0, seed=7)
    pair = build_contexts("Who was Marie Curie?", config=config, retrieval=result)
    assert pair.corrupted_fact_ids
    original_ids = {fact.fact_id for fact in result.original_facts}
    assert not (set(pair.corrupted_fact_ids) & original_ids)
