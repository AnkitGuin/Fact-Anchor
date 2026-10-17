"""Entity extraction, linking and retrieval tests (spec 7.x, 15)."""

from __future__ import annotations

import pytest

from conftest import FlakyProvider, TOY_ENTITIES, CountingProvider

from factanchor.config import RetrievalConfig, load_config
from factanchor.entities import (
    LexiconEntityExtractor,
    RuleBasedEntityExtractor,
    build_extractor,
    deduplicate_mentions,
    extract_entities,
)
from factanchor.knowledge.base import (
    JsonFileKnowledgeProvider,
    KnowledgeUnavailable,
    MockKnowledgeProvider,
    build_candidate,
)
from factanchor.knowledge.pipeline import KnowledgePipeline, context_overlap_score, normalise_mention
from factanchor.schemas import EntityMention, EntityRecord


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_empty_and_whitespace_text_return_no_mentions():
    extractor = RuleBasedEntityExtractor()
    assert extractor.extract("") == []
    assert extractor.extract("   \n ") == []


def test_prompt_without_entities_does_not_crash():
    extractor = RuleBasedEntityExtractor()
    assert extractor.extract("what is the answer to this question") == []
    assert extractor.extract("123 456 789") == []


def test_multi_token_names_are_extracted_with_offsets():
    text = "Who was Marie Curie?"
    mentions = extract_entities(text, backend=RuleBasedEntityExtractor())
    assert mentions, "expected at least one mention"
    top = mentions[0]
    assert top.mention == "Marie Curie"
    assert text[top.start : top.end] == "Marie Curie"
    assert top.confidence > 0.5


def test_sentence_initial_short_words_are_not_entities():
    mentions = extract_entities("In 1903 she won the prize.", backend=RuleBasedEntityExtractor())
    assert all(mention.mention.lower() != "in" for mention in mentions)


def test_overlapping_mentions_are_deduplicated_keeping_the_longest(mentions):
    deduped = deduplicate_mentions(mentions)
    surfaces = [mention.mention for mention in deduped]
    assert "Marie Curie" in surfaces
    assert "Marie" not in surfaces


def test_deduplicate_mentions_detects_bad_offsets():
    mention = EntityMention(mention="Marie Curie", start=0, end=50, confidence=0.9)
    with pytest.raises(ValueError, match="outside text"):
        deduplicate_mentions([mention], text="short text")


def test_mention_offsets_are_validated():
    with pytest.raises(ValueError):
        EntityMention(mention="x", start=5, end=5)
    with pytest.raises(ValueError):
        EntityMention(mention="   ", start=0, end=3)


def test_lexicon_extractor_handles_lists_and_empty_lexicon():
    extractor = LexiconEntityExtractor(known_entities=["Marie Curie", "France"])
    found = extractor.extract("Marie Curie moved to France in 1891.")
    assert {mention.mention for mention in found} == {"Marie Curie", "France"}
    assert LexiconEntityExtractor(known_entities=[]).extract("Marie Curie") == []


def test_extractor_respects_max_entities():
    config = load_config("configs/smoke_toy.yaml", overrides={"ner.max_entities": 1})
    mentions = extract_entities(
        "Marie Curie met Albert Einstein and Douglas Adams in Paris",
        backend=build_extractor(config.ner),
    )
    assert len(mentions) <= 1


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------


def test_linking_resolves_known_entities(pipeline):
    mentions = [EntityMention(mention="Marie Curie", start=8, end=19, confidence=0.9)]
    result = pipeline.link_entities(mentions, prompt="Who was Marie Curie?")
    assert result.linked == 1
    record = result.records[0]
    assert record.entity_id == "Q7186"
    assert record.label == "Marie Curie"
    assert record.linking_confidence >= 0.75
    assert record.link_method in {"search", "dataset_hint"}


def test_linking_uses_context_to_disambiguate():
    provider = MockKnowledgeProvider(
        [
            {"entity_id": "Q41421", "label": "Jordan", "description": "country in Western Asia", "facts": []},
            {
                "entity_id": "Q19850112",
                "label": "Jordan",
                "description": "American basketball player in the National Basketball Association",
                "facts": [],
            },
        ]
    )
    config = load_config("configs/smoke_toy.yaml")
    pipeline = KnowledgePipeline(config, provider=provider)
    mention = EntityMention(
        mention="Jordan",
        start=0,
        end=6,
        confidence=0.7,
        context="Who is the basketball player Jordan who played in the NBA?",
    )
    result = pipeline.link_entities([mention], prompt=mention.context)
    assert result.records[0].entity_id == "Q19850112", "context should prefer the basketball player"
    assert result.records[0].candidates, "candidate list must be preserved for audit"


def test_low_confidence_mentions_stay_unlinked_with_candidates():
    provider = MockKnowledgeProvider(
        [{"entity_id": "Q1", "label": "Completely Different Person", "description": "unrelated thing", "facts": []}]
    )
    config = load_config("configs/smoke_toy.yaml", overrides={"retrieval.min_linking_confidence": 0.99})
    pipeline = KnowledgePipeline(config, provider=provider)
    mention = EntityMention(mention="Completely Different", start=0, end=19, confidence=0.5)
    result = pipeline.link_entities([mention])
    assert result.linked == 0
    assert result.unlinked == 1
    assert result.records[0].entity_id is None
    assert result.records[0].link_method == "below_threshold"


def test_unlinkable_mention_is_preserved_not_dropped(pipeline):
    mentions = [EntityMention(mention="Nonexistent Person", start=0, end=18, confidence=0.5)]
    result = pipeline.link_entities(mentions)
    assert len(result.records) == 1
    assert result.records[0].mention == "Nonexistent Person"


def test_linking_is_bounded_by_max_entities(pipeline):
    pipeline.retrieval.max_entities = 1
    mentions = [
        EntityMention(mention="Marie Curie", start=0, end=11, confidence=0.9),
        EntityMention(mention="United States", start=20, end=33, confidence=0.9),
    ]
    result = pipeline.link_entities(mentions)
    assert len(result.records) <= 1


def test_linking_service_unavailable_keeps_running():
    pipeline = KnowledgePipeline(
        load_config("configs/smoke_toy.yaml"), provider=FlakyProvider(fail_search=True)
    )
    result = pipeline.link_entities([EntityMention(mention="Marie Curie", start=0, end=11)])
    assert result.linked == 0
    assert result.records[0].link_method == "service_unavailable"
    assert result.errors


def test_dataset_hint_is_used_when_provided(pipeline):
    result = pipeline.link_entities([], entity_hint="Marie Curie")
    assert result.linked == 1
    assert result.records[0].entity_id == "Q7186"
    assert result.records[0].link_method == "dataset_hint"


def test_normalise_mention_handles_articles_and_punctuation():
    assert normalise_mention("The United States of America.") == "united states of america"
    assert normalise_mention("  Marie   Curie ") == "marie curie"


def test_context_overlap_score_discriminates():
    basketball = build_candidate("Q2", "Jordan", description="American basketball player")
    country = build_candidate("Q1", "Jordan", description="country in Western Asia")
    context = "Which basketball player Jordan won the championship?"
    assert context_overlap_score(context, basketball) > context_overlap_score(context, country)


def test_linking_reuses_memoised_results():
    provider = CountingProvider(TOY_ENTITIES)
    pipeline = KnowledgePipeline(load_config("configs/smoke_toy.yaml"), provider=provider)
    mentions = [EntityMention(mention="Marie Curie", start=0, end=11, confidence=0.9)]
    pipeline.link_entities(mentions)
    pipeline.link_entities(mentions)
    assert provider.search_calls == 1, "the same mention must not hit the provider twice"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_retrieval_returns_linked_facts(pipeline):
    entities = [EntityRecord(mention="Marie Curie", entity_id="Q7186", label="Marie Curie", linking_confidence=0.9)]
    result = pipeline.retrieve_facts(entities)
    assert result.facts
    assert {fact.subject_id for fact in result.facts} == {"Q7186"}
    assert all(fact.source for fact in result.facts)
    assert all(fact.retrieved_at for fact in result.facts)


def test_retrieval_respects_max_facts_per_entity(pipeline):
    entities = [EntityRecord(mention="Marie Curie", entity_id="Q7186", label="Marie Curie", linking_confidence=0.9)]
    result = pipeline.retrieve_facts(entities, max_facts_per_entity=2)
    assert len(result.facts) <= 2


def test_retrieval_deduplicates_identical_facts():
    provider = MockKnowledgeProvider(
        [
            {
                "entity_id": "Q1",
                "label": "Duplicate City",
                "facts": [
                    {"predicate_id": "P17", "predicate_label": "country", "object_label": "France"},
                    {"predicate_id": "P17", "predicate_label": "country", "object_label": "France"},
                ],
            }
        ]
    )
    pipeline = KnowledgePipeline(load_config("configs/smoke_toy.yaml"), provider=provider)
    result = pipeline.retrieve_facts([EntityRecord(mention="x", entity_id="Q1", label="Duplicate City")])
    assert len(result.facts) == 1


def test_retrieval_skips_low_confidence_facts():
    provider = MockKnowledgeProvider(
        [
            {
                "entity_id": "Q1",
                "label": "Low Confidence",
                "facts": [{"predicate_id": "P17", "object_label": "France", "confidence": 0.1}],
            }
        ]
    )
    config = load_config("configs/smoke_toy.yaml", overrides={"retrieval.min_fact_confidence": 0.5})
    pipeline = KnowledgePipeline(config, provider=provider)
    result = pipeline.retrieve_facts([EntityRecord(mention="x", entity_id="Q1", label="Low Confidence")])
    assert result.facts == []


def test_retrieval_skips_malformed_facts():
    provider = MockKnowledgeProvider(
        [
            {
                "entity_id": "Q1",
                "label": "Malformed",
                "facts": [
                    {"predicate_id": "", "object_label": "broken"},
                    {
                        "predicate_id": "P31",
                        "object_label": "human",
                        "object_type": "entity",  # entity object without an id
                    },
                ],
            }
        ]
    )
    pipeline = KnowledgePipeline(load_config("configs/smoke_toy.yaml"), provider=provider)
    result = pipeline.retrieve_facts([EntityRecord(mention="x", entity_id="Q1", label="Malformed")])
    assert result.facts == []


def test_retrieval_without_linked_entities_is_explicit(pipeline):
    result = pipeline.retrieve_facts([EntityRecord(mention="unknown", entity_id=None)])
    assert result.facts == []
    assert result.degraded_reason == "no_linked_entities"


def test_retrieval_unavailable_no_facts_policy():
    config = load_config(
        "configs/smoke_toy.yaml", overrides={"retrieval.fallback_on_unavailable": "no_facts"}
    )
    pipeline = KnowledgePipeline(config, provider=FlakyProvider(fail_facts=True))
    result = pipeline.retrieve_facts([EntityRecord(mention="x", entity_id="Q1", label="Whatever", linking_confidence=0.9)])
    assert result.facts == []
    assert result.degraded_reason == "retrieval_unavailable"
    assert result.errors


def test_retrieval_unavailable_error_policy_raises():
    config = load_config("configs/smoke_toy.yaml", overrides={"retrieval.fallback_on_unavailable": "error"})
    pipeline = KnowledgePipeline(config, provider=FlakyProvider(fail_facts=True))
    with pytest.raises(KnowledgeUnavailable):
        pipeline.retrieve_facts([EntityRecord(mention="x", entity_id="Q1", label="Whatever", linking_confidence=0.9)])


def test_retrieval_cache_only_fallback_uses_previous_results(tmp_path, monkeypatch):
    config = load_config(
        "configs/smoke_toy.yaml",
        overrides={"retrieval.fallback_on_unavailable": "cache_only", "retrieval.cache_dir": str(tmp_path / "cache")},
    )
    working = KnowledgePipeline(config, provider=MockKnowledgeProvider(TOY_ENTITIES))
    entities = [EntityRecord(mention="Marie Curie", entity_id="Q7186", label="Marie Curie", linking_confidence=0.9)]
    first = working.retrieve_facts(entities)
    assert first.facts

    # a new pipeline (fresh process) with a dead service must serve the cached facts
    broken = KnowledgePipeline(config, provider=FlakyProvider(fail_facts=True))
    second = broken.retrieve_facts(entities)
    assert second.facts, "cached facts should be used when the service is down"
    assert {fact.object_label for fact in second.facts} <= {fact.object_label for fact in first.facts}
    assert second.errors and second.cache_used


def test_retrieval_memoises_within_a_process():
    provider = CountingProvider(TOY_ENTITIES)
    pipeline = KnowledgePipeline(load_config("configs/smoke_toy.yaml"), provider=provider)
    entities = [EntityRecord(mention="Marie Curie", entity_id="Q7186", label="Marie Curie", linking_confidence=0.9)]
    pipeline.retrieve_facts(entities)
    pipeline.retrieve_facts(entities)
    assert provider.fact_calls == 1


def test_retrieval_disabled_returns_no_facts():
    config = load_config("configs/smoke_toy.yaml", overrides={"retrieval.enabled": False})
    pipeline = KnowledgePipeline(config, provider=MockKnowledgeProvider(TOY_ENTITIES))
    result = pipeline.retrieve_facts([EntityRecord(mention="x", entity_id="Q7186", label="Marie Curie")])
    assert result.facts == []
    assert result.degraded_reason == "retrieval_disabled"


# ---------------------------------------------------------------------------
# Offline provider plumbing
# ---------------------------------------------------------------------------


def test_json_file_provider_loads_records(project_root):
    provider = JsonFileKnowledgeProvider(project_root / "data" / "processed" / "offline_knowledge.jsonl")
    candidates = provider.search_entities("Marie Curie", limit=3)
    assert candidates and candidates[0].entity_id == "Q7186"
    facts = provider.get_facts("Q7186", max_facts=3)
    assert facts and all(fact.subject_id == "Q7186" for fact in facts)
    assert provider.known_entities()


def test_json_file_provider_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        JsonFileKnowledgeProvider(tmp_path / "nope.jsonl")


def test_json_file_provider_reports_bad_lines(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"entity_id": "Q1", "label": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        JsonFileKnowledgeProvider(path)


def test_mock_provider_unavailable_raises():
    provider = MockKnowledgeProvider([], available=False)
    with pytest.raises(KnowledgeUnavailable):
        provider.search_entities("anything")


def test_retrieval_config_requires_local_path_for_json_provider():
    with pytest.raises(ValueError, match="local_json_path"):
        RetrievalConfig(provider="json_file", local_json_path=None)


def test_dbpedia_provider_fails_loudly():
    from factanchor.knowledge.wikidata import SPARQL_BACKED_DBpediaProvider

    with pytest.raises(NotImplementedError, match="not implemented in v1"):
        SPARQL_BACKED_DBpediaProvider()


def test_wikidata_time_formatting_respects_precision():
    from factanchor.knowledge.wikidata import format_quantity, format_wikidata_time

    # Wikibase precision: 11..8 = sub-day..day, 7 = month, 6 = year, 5 = decade,
    # 4 = century, 3 = millennium (see wikidata.py)
    assert format_wikidata_time({"time": "+1867-11-07T00:00:00Z", "precision": 11}) == "7 November 1867"
    assert format_wikidata_time({"time": "+1867-11-07T00:00:00Z", "precision": 8}) == "7 November 1867"
    assert format_wikidata_time({"time": "+1867-11-00T00:00:00Z", "precision": 7}) == "November 1867"
    assert format_wikidata_time({"time": "+1867-00-00T00:00:00Z", "precision": 6}) == "1867"
    assert format_wikidata_time({"time": "+1867-00-00T00:00:00Z", "precision": 5}) == "1860s"
    assert format_wikidata_time({"time": "+1800-00-00T00:00:00Z", "precision": 4}) == "18th century"
    assert format_wikidata_time({"time": "+1000-00-00T00:00:00Z", "precision": 3}) == "1st millennium"
    assert format_wikidata_time({"time": "-0500-00-00T00:00:00Z", "precision": 6}) == "500 BCE"
    assert format_wikidata_time({"time": "+3000000-00-00T00:00:00Z", "precision": 2}).startswith("about 3 million")
    assert format_wikidata_time({"time": "garbage"}) is None
    assert format_quantity({"amount": "+331449281", "unit": "1"}) == "331,449,281"


def test_wikidata_object_id_parsing_skips_maintenance_items():
    from factanchor.knowledge.wikidata import WikidataProvider

    provider = WikidataProvider(api_url="https://example.invalid/w/api.php")
    parsed = provider._parse_datavalue({"type": "wikibase-entityid", "value": {"id": "Q4167410"}}, "wikibase-item")
    assert parsed is None, "disambiguation pages must not become facts"
