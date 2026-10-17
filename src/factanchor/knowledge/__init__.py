"""Knowledge access: providers, entity linking and bounded fact retrieval."""

from .base import (
    DEFAULT_SOURCE_LABELS,
    DEFAULT_USER_AGENT,
    resolve_source_label,
    resolve_user_agent,
    EntityCandidate,
    HttpJsonClient,
    JsonFileKnowledgeProvider,
    KnowledgeProvider,
    KnowledgeUnavailable,
    MockKnowledgeProvider,
    build_candidate,
)
from .pipeline import (
    KnowledgePipeline,
    LinkResult,
    RetrievalResult,
    build_provider,
    context_overlap_score,
    normalise_mention,
    retrieve_entity_context,
)
from .wikidata import (
    PROPERTY_ALLOWLIST,
    SPARQL_BACKED_DBpediaProvider,
    WikidataProvider,
    format_quantity,
    format_wikidata_time,
)

__all__ = [
    "DEFAULT_SOURCE_LABELS",
    "DEFAULT_USER_AGENT",
    "resolve_source_label",
    "resolve_user_agent",
    "EntityCandidate",
    "HttpJsonClient",
    "JsonFileKnowledgeProvider",
    "KnowledgeProvider",
    "KnowledgeUnavailable",
    "MockKnowledgeProvider",
    "build_candidate",
    "KnowledgePipeline",
    "LinkResult",
    "RetrievalResult",
    "build_provider",
    "context_overlap_score",
    "normalise_mention",
    "retrieve_entity_context",
    "PROPERTY_ALLOWLIST",
    "SPARQL_BACKED_DBpediaProvider",
    "WikidataProvider",
    "format_quantity",
    "format_wikidata_time",
]
