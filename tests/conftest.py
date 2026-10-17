"""Shared pytest fixtures.

Everything here is offline: no network, no model downloads.  The Knowledge API is
mocked and the model backend is the bundled NumPy toy transformer unless a test
explicitly opts into Hugging Face via the ``hf`` marker.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import-time convenience
    sys.path.insert(0, str(SRC))

from factanchor.config import Config, load_config  # noqa: E402
from factanchor.corruption import CorruptionEngine  # noqa: E402
from factanchor.knowledge.base import (  # noqa: E402
    EntityCandidate,
    KnowledgeProvider,
    KnowledgeUnavailable,
    MockKnowledgeProvider,
)
from factanchor.knowledge.pipeline import KnowledgePipeline  # noqa: E402
from factanchor.model_adapter import ToyCausalLMAdapter  # noqa: E402
from factanchor.schemas import EntityMention, Fact, Qualifier  # noqa: E402

TOY_ENTITIES: list[dict[str, Any]] = [
    {
        "entity_id": "Q7186",
        "label": "Marie Curie",
        "description": "Polish and naturalised-French physicist and chemist",
        "aliases": ["Madame Curie"],
        "facts": [
            {
                "predicate_id": "P106",
                "predicate_label": "occupation",
                "object_id": "Q169470",
                "object_label": "physicist",
                "object_type": "entity",
            },
            {
                "predicate_id": "P106",
                "predicate_label": "occupation",
                "object_id": "Q593644",
                "object_label": "chemist",
                "object_type": "entity",
            },
            {
                "predicate_id": "P27",
                "predicate_label": "country of citizenship",
                "object_id": "Q142",
                "object_label": "France",
                "object_type": "entity",
            },
            {
                "predicate_id": "P569",
                "predicate_label": "date of birth",
                "object_label": "7 November 1867",
                "object_type": "time",
            },
            {
                "predicate_id": "P166",
                "predicate_label": "award received",
                "object_id": "Q38104",
                "object_label": "Nobel Prize in Physics",
                "object_type": "entity",
            },
        ],
    },
    {
        "entity_id": "Q30",
        "label": "United States of America",
        "description": "country in North America",
        "aliases": ["United States", "USA"],
        "facts": [
            {
                "predicate_id": "P30",
                "predicate_label": "continent",
                "object_id": "Q49",
                "object_label": "North America",
                "object_type": "entity",
            },
            {
                "predicate_id": "P36",
                "predicate_label": "capital",
                "object_id": "Q61",
                "object_label": "Washington, D.C.",
                "object_type": "entity",
            },
            {
                "predicate_id": "P1082",
                "predicate_label": "population",
                "object_label": "331,449,281",
                "object_type": "quantity",
            },
        ],
    },
]


class FlakyProvider(KnowledgeProvider):
    """Provider that fails on demand, to exercise the unavailable-service paths."""

    name = "flaky"
    source_label = "FlakyKB"

    def __init__(self, *, fail_search: bool = False, fail_facts: bool = False) -> None:
        super().__init__(cache=None)
        self.fail_search = fail_search
        self.fail_facts = fail_facts
        self.search_calls = 0
        self.fact_calls = 0

    def search_entities(self, query: str, *, limit: int = 5, context: str | None = None) -> list[EntityCandidate]:
        self.search_calls += 1
        if self.fail_search:
            raise KnowledgeUnavailable("simulated search outage", provider=self.name)
        return [EntityCandidate(entity_id="Q1", label=query, score=0.9, source=self.source_label)]

    def get_facts(self, entity_id: str, *, max_facts: int = 10) -> list[Fact]:
        self.fact_calls += 1
        if self.fail_facts:
            raise KnowledgeUnavailable("simulated fact outage", provider=self.name)
        return []


class CountingProvider(MockKnowledgeProvider):
    """Mock provider that counts calls (used for cache/bound assertions)."""

    name = "counting"

    def __init__(self, entities: Sequence[dict[str, Any]]) -> None:
        super().__init__(entities)
        self.search_calls = 0
        self.fact_calls = 0

    def search_entities(self, query: str, *, limit: int = 5, context: str | None = None) -> list[EntityCandidate]:
        self.search_calls += 1
        return super().search_entities(query, limit=limit, context=context)

    def get_facts(self, entity_id: str, *, max_facts: int = 10) -> list[Fact]:
        self.fact_calls += 1
        return super().get_facts(entity_id, max_facts=max_facts)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def config() -> Config:
    """Default configuration, forced onto the offline toy backend."""
    return load_config(
        PROJECT_ROOT / "configs" / "smoke_toy.yaml",
        overrides={
            "run.output_dir": "artifacts",
            "run.log_prompts": True,
            "retrieval.local_json_path": str(PROJECT_ROOT / "data" / "processed" / "offline_knowledge.jsonl"),
            "evaluation.bertscore_enabled": False,
        },
    )


@pytest.fixture
def toy_adapter() -> ToyCausalLMAdapter:
    return ToyCausalLMAdapter(use_chat_template=False)


@pytest.fixture
def mock_provider() -> MockKnowledgeProvider:
    return MockKnowledgeProvider(TOY_ENTITIES)


@pytest.fixture
def pipeline(config: Config, mock_provider: MockKnowledgeProvider) -> KnowledgePipeline:
    return KnowledgePipeline(config, provider=mock_provider)


@pytest.fixture
def sample_facts() -> list[Fact]:
    """A small typed fact set covering entity, time and quantity objects."""
    return [
        Fact(
            subject_id="Q7186",
            subject_label="Marie Curie",
            predicate_id="P106",
            predicate_label="occupation",
            object_id="Q169470",
            object_label="physicist",
            object_type="entity",
            source="TestKB",
            metadata={"subject_class": "Q5"},
        ),
        Fact(
            subject_id="Q7186",
            subject_label="Marie Curie",
            predicate_id="P27",
            predicate_label="country of citizenship",
            object_id="Q142",
            object_label="France",
            object_type="entity",
            source="TestKB",
            metadata={"subject_class": "Q5"},
        ),
        Fact(
            subject_id="Q7186",
            subject_label="Marie Curie",
            predicate_id="P569",
            predicate_label="date of birth",
            object_label="7 November 1867",
            object_type="time",
            source="TestKB",
            qualifiers=(
                Qualifier(
                    predicate_id="P19",
                    predicate_label="place of birth",
                    value="Warsaw",
                    value_type="entity",
                    value_id="Q270",
                ),
            ),
            metadata={"subject_class": "Q5"},
        ),
        Fact(
            subject_id="Q7186",
            subject_label="Marie Curie",
            predicate_id="P166",
            predicate_label="award received",
            object_id="Q38104",
            object_label="Nobel Prize in Physics",
            object_type="entity",
            source="TestKB",
            metadata={"subject_class": "Q5"},
        ),
        Fact(
            subject_id="Q30",
            subject_label="United States of America",
            predicate_id="P1082",
            predicate_label="population",
            object_label="331,449,281",
            object_type="quantity",
            source="TestKB",
            metadata={"subject_class": "Q3624078"},
        ),
    ]


@pytest.fixture
def engine() -> CorruptionEngine:
    return CorruptionEngine(strategy="object_swap", corruption_rate=0.5, seed=42)


@pytest.fixture
def mentions() -> list[EntityMention]:
    return [
        EntityMention(mention="Marie Curie", start=8, end=19, confidence=0.8, source="rule"),
        EntityMention(mention="Marie", start=8, end=13, confidence=0.4, source="rule"),  # overlaps
    ]
