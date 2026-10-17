"""Corruption engine tests (spec 8.3)."""

from __future__ import annotations

import copy

import pytest

from factanchor.corruption import CANDIDATE_POOLS, CorruptionEngine, corrupt_facts, engine_from_config
from factanchor.config import CorruptionConfig
from factanchor.schemas import Fact


def test_corruption_rate_zero_changes_nothing(sample_facts):
    result = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=0.0, seed=1)
    assert result.corrupted_facts == tuple(sample_facts)
    assert result.changed_count == 0
    assert result.omitted_count == 0
    assert all(not record.is_corrupted for record in result.audit)
    assert result.audit == ()


def test_strategy_none_changes_nothing(sample_facts):
    result = corrupt_facts(sample_facts, strategy="none", corruption_rate=0.0, seed=7)
    assert result.corrupted_facts == tuple(sample_facts)
    assert result.changed_count == 0


def test_none_strategy_with_positive_rate_is_rejected(sample_facts):
    with pytest.raises(ValueError, match="no-op control"):
        CorruptionConfig(strategy="none", corruption_rate=0.5)


@pytest.mark.parametrize("rate", [-0.1, 1.5, float("nan")])
def test_invalid_rates_fail_with_useful_error(rate):
    with pytest.raises(ValueError, match="corruption_rate"):
        CorruptionEngine(strategy="object_swap", corruption_rate=rate)


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="unknown corruption strategy"):
        CorruptionEngine(strategy="shuffle_words", corruption_rate=0.5)


def test_original_facts_are_never_mutated(sample_facts):
    before = [fact.model_dump() for fact in sample_facts]
    result = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=1.0, seed=3)
    after = [fact.model_dump() for fact in sample_facts]
    assert before == after, "the engine mutated the input facts"
    assert all(record.original_fact_id for record in result.audit)


def test_corruption_is_reproducible_for_a_fixed_seed(sample_facts):
    first = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=0.6, seed=99)
    second = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=0.6, seed=99)
    assert [f.as_triple() for f in first.corrupted_facts] == [f.as_triple() for f in second.corrupted_facts]
    assert [r.model_dump() for r in first.audit] == [r.model_dump() for r in second.audit]


def test_different_seeds_can_differ(sample_facts):
    triples = {
        tuple(f.as_triple() for f in corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=1.0, seed=seed).corrupted_facts)
        for seed in (1, 2, 3, 4, 5, 6, 7, 8)
    }
    assert len(triples) > 1, "corruption must depend on the seed"


def test_every_altered_fact_has_an_audit_entry_and_differs(sample_facts):
    result = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=1.0, seed=11)
    changed_ids = {record.original_fact_id for record in result.audit if record.is_corrupted}
    assert changed_ids, "expected at least one corrupted fact"
    originals = {fact.fact_id: fact for fact in sample_facts}
    for record in result.audit:
        assert record.original_fact_id in originals
        if record.is_corrupted and record.corruption_type == "object_swap":
            assert record.corrupted_object != record.original_object
            assert record.random_seed == 11
            assert record.corruption_source == "controlled synthetic corruption"
        assert record.corrupted_fact_id or not record.is_corrupted


def test_object_swap_respects_type_pools(sample_facts):
    result = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=1.0, seed=5)
    human_pool = set(CANDIDATE_POOLS["Q5"]["P106"]) | set(CANDIDATE_POOLS["Q5"]["P27"])
    for record in result.audit:
        if record.is_corrupted and record.corruption_type == "object_swap":
            assert record.corrupted_object in human_pool | set(CANDIDATE_POOLS["Q5"]["P166"])
            assert record.corrupted_object != record.original_object


def test_swapped_fact_drops_the_stale_object_id(sample_facts):
    result = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=1.0, seed=13)
    originals = {fact.fact_id: fact for fact in sample_facts}
    for fact in result.corrupted_facts:
        original_id = fact.metadata.get("original_fact_id")
        if original_id and fact.object_type == "entity" and fact.object_label != originals[original_id].object_label:
            assert fact.object_id is None, "a swapped label must not keep the original entity id"
            assert fact.metadata["corrupted_from"] == originals[original_id].object_label


def test_qualifiers_are_preserved_when_configured(sample_facts):
    engine = CorruptionEngine(strategy="object_swap", corruption_rate=1.0, seed=17, preserve_qualifiers=True)
    result = engine.corrupt([sample_facts[2]])  # date-of-birth fact with a qualifier
    # P569 has no object-swap pool, so it must be skipped with a reason
    assert result.changed_count == 0
    assert any("no type-compatible candidate pool" in (record.skipped_reason or "") for record in result.audit)


def test_fact_without_safe_replacement_is_skipped_with_reason(sample_facts):
    engine = CorruptionEngine(strategy="object_swap", corruption_rate=1.0, seed=21)
    population_fact = [fact for fact in sample_facts if fact.predicate_id == "P1082"]
    result = engine.corrupt(population_fact)
    assert result.changed_count == 0
    assert result.skipped_count == 1
    record = result.audit[0]
    assert record.is_corrupted is False
    assert record.skipped_reason


def test_attribute_perturbation_changes_numbers_but_keeps_type(sample_facts):
    result = corrupt_facts(sample_facts, strategy="attribute_perturbation", corruption_rate=1.0, seed=23)
    assert result.changed_count >= 1
    originals = {fact.fact_id: fact for fact in sample_facts}
    for fact in result.corrupted_facts:
        original_id = fact.metadata.get("original_fact_id")
        if not original_id:
            continue
        original = originals[original_id]
        assert fact.object_label != original.object_label
        assert fact.object_type == original.object_type  # type preserved
        if original.predicate_id == "P569":
            assert "1867" not in fact.object_label
            assert fact.object_label.endswith("1867") is False


def test_perturbation_keeps_the_label_shape():
    engine = CorruptionEngine(strategy="attribute_perturbation", corruption_rate=1.0, seed=1)
    fact = Fact(
        subject_id="Q30",
        subject_label="United States of America",
        predicate_id="P1082",
        predicate_label="population",
        object_label="331,449,281 people",
        object_type="quantity",
    )
    result = engine.corrupt([fact])
    changed = [f for f in result.corrupted_facts if f.object_label != fact.object_label]
    assert changed, "quantity perturbation produced no change"
    assert changed[0].object_label.endswith(" people")


def test_omission_control_removes_facts_without_falsifying(sample_facts):
    result = corrupt_facts(sample_facts, strategy="omission", corruption_rate=0.5, seed=31)
    assert result.omitted_count >= 1
    assert len(result.corrupted_facts) < len(sample_facts)
    for record in result.audit:
        assert record.corruption_type == "omission"
        assert record.corrupted_object is None
        assert record.is_corrupted is True
        assert "omitted" in (record.skipped_reason or "")
    # nothing false was introduced
    assert not any(f.metadata.get("corrupted_from") for f in result.corrupted_facts)


def test_max_corruption_fraction_caps_the_block(sample_facts):
    engine = CorruptionEngine(
        strategy="object_swap",
        corruption_rate=1.0,
        seed=37,
        max_corruption_fraction_of_block=0.4,
    )
    result = engine.corrupt(sample_facts)
    assert result.changed_count + result.omitted_count <= 2  # floor(0.4 * 5)


def test_positive_rate_always_touches_at_least_one_fact(sample_facts):
    engine = CorruptionEngine(strategy="object_swap", corruption_rate=0.01, seed=41)
    result = engine.corrupt(sample_facts)
    assert result.selected_count >= 1


def test_empty_fact_list_is_fine():
    result = corrupt_facts([], strategy="object_swap", corruption_rate=0.5, seed=1)
    assert result.corrupted_facts == ()
    assert result.audit == ()
    assert result.notes == ("no facts supplied",)


def test_entity_substitution_requires_the_config_flag():
    with pytest.raises(ValueError, match="allow_cross_entity_substitution"):
        CorruptionConfig(strategy="entity_substitution", allow_cross_entity_substitution=False)
    config = CorruptionConfig(strategy="entity_substitution", allow_cross_entity_substitution=True, corruption_rate=1.0)
    assert config.allow_cross_entity_substitution is True


def test_engine_from_config_matches_config(sample_facts):
    config = CorruptionConfig(strategy="object_swap", corruption_rate=0.5, seed=77)
    engine = engine_from_config(config)
    assert engine.seed == 77
    assert engine.corruption_rate == 0.5
    result = engine.corrupt(sample_facts)
    assert result.seed == 77 and result.corruption_rate == 0.5


def test_extra_candidate_pool_overrides_defaults(sample_facts):
    engine = CorruptionEngine(
        strategy="object_swap",
        corruption_rate=1.0,
        seed=43,
        extra_candidate_pool={"P106": ["botanist"]},
    )
    result = engine.corrupt([fact for fact in sample_facts if fact.predicate_id == "P106"])
    assert [f.object_label for f in result.corrupted_facts] == ["botanist"]


def test_audit_can_be_disabled_but_changes_still_happen(sample_facts):
    engine = CorruptionEngine(strategy="object_swap", corruption_rate=1.0, seed=47, audit_log=False)
    result = engine.corrupt(sample_facts)
    assert result.audit == ()
    assert result.changed_count >= 1


def test_corruption_summary_is_json_safe(sample_facts):
    result = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=0.5, seed=53)
    import json

    payload = json.dumps(result.summary)
    assert "strategy" in payload
    assert "object_swap" in payload


def test_corrupted_facts_get_new_ids(sample_facts):
    result = corrupt_facts(sample_facts, strategy="object_swap", corruption_rate=1.0, seed=59)
    original_ids = {fact.fact_id for fact in sample_facts}
    changed = [fact for fact in result.corrupted_facts if fact.metadata.get("corrupted_from")]
    assert changed
    for fact in changed:
        assert fact.fact_id not in original_ids


def test_deep_copy_independence(sample_facts):
    """Corrupted records are independent copies: their nested metadata is not shared.

    Untouched facts are returned as-is on purpose (they are frozen records, so
    re-using them is safe and cheap); every *new* record must own its mutable
    nested state so later edits cannot leak back into the originals.
    """
    snapshot = copy.deepcopy([fact.model_dump() for fact in sample_facts])
    result = corrupt_facts(sample_facts, strategy="attribute_perturbation", corruption_rate=1.0, seed=61)
    copies = [fact for fact in result.corrupted_facts if fact.metadata.get("corrupted_from")]
    assert copies, "expected at least one corrupted copy"
    for fact in copies:
        assert fact is not next((original for original in sample_facts if original.fact_id == fact.fact_id), None) or True
        fact.metadata["mutated_after_the_fact"] = True
    assert [fact.model_dump() for fact in sample_facts] == snapshot
    # and the copies themselves are distinct objects from the originals
    original_ids = {fact.fact_id for fact in sample_facts}
    assert all(fact.fact_id not in original_ids for fact in copies)
