"""Entity linking + fact retrieval pipeline (spec 7.2, 7.3, 7.4).

The pipeline is the single place that decides *when* the knowledge service is
consulted, *how much* it is asked for, and *what happens when it fails*:

``link_entities(mentions, prompt)``
    Resolve mentions to canonical ids.  Disambiguation uses the local context
    (words around the mention) not just label equality; ambiguous or
    low-confidence mentions stay **unlinked** with their candidate list
    preserved.
``retrieve_facts(entities, max_facts_per_entity)``
    Bounded, deduplicated, cached retrieval.  On failure the configured
    fallback applies (``cache_only`` | ``no_facts`` | ``error``).

Retrieval happens **before** decoding (spec 7.4); timings are returned so the
runner can report retrieval latency separately from token-generation latency.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Sequence

from ..config import Config, RetrievalConfig
from ..entities import deduplicate_mentions
from ..schemas import EntityMention, EntityRecord, Fact
from ..utils import DiskCache, Timer, get_logger
from .base import (
    DEFAULT_USER_AGENT,
    EntityCandidate,
    resolve_source_label,
    resolve_user_agent,
    JsonFileKnowledgeProvider,
    KnowledgeProvider,
    KnowledgeUnavailable,
    MockKnowledgeProvider,
)
from .wikidata import WikidataProvider

__all__ = [
    "KnowledgePipeline",
    "LinkResult",
    "RetrievalResult",
    "build_provider",
    "normalise_mention",
    "context_overlap_score",
]

logger = get_logger("knowledge.pipeline")

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)

_GRAMMAR_STOPWORDS = frozenset(
    """
    a an the of and or in on at for with from to by is was are were be been being
    who whom whose what which where when why how did does do dat that this these those
    """.split()
)


def normalise_mention(mention: str) -> str:
    """Lower-case, drop a leading article, collapse whitespace and punctuation."""
    text = _LEADING_ARTICLE_RE.sub("", (mention or "").strip())
    text = re.sub(r"[^\w\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _content_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if token.lower() not in _GRAMMAR_STOPWORDS and len(token) > 1
    }


def context_overlap_score(mention_context: str | None, candidate: EntityCandidate, prompt: str | None = None) -> float:
    """Similarity between the surrounding text and a candidate's description.

    This is the contextual disambiguation signal: a "Jordan" in a basketball
    question overlaps with the description "American basketball player" but not
    with "country in Western Asia".  Returns a value in ``[0, 1]``.
    """
    haystack = " ".join(part for part in (mention_context or "", prompt or "") if part)
    if not haystack.strip():
        return 0.0
    context_tokens = _content_tokens(haystack)
    candidate_tokens = _content_tokens(candidate.description) | _content_tokens(candidate.label)
    if not context_tokens or not candidate_tokens:
        return 0.0
    overlap = len(context_tokens & candidate_tokens)
    jaccard = overlap / len(context_tokens | candidate_tokens)
    containment = overlap / max(1, min(len(context_tokens), len(candidate_tokens)))
    string_sim = SequenceMatcher(None, normalise_mention(candidate.label), normalise_mention(haystack)).ratio()
    return max(0.0, min(1.0, 0.6 * containment + 0.3 * jaccard + 0.1 * string_sim))


@dataclass
class LinkResult:
    """Outcome of linking: records plus bookkeeping for the run log."""

    records: list[EntityRecord] = field(default_factory=list)
    linked: int = 0
    unlinked: int = 0
    ambiguous: int = 0
    api_calls: int = 0
    elapsed_s: float = 0.0
    degraded_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def entity_ids(self) -> list[str]:
        return [r.entity_id for r in self.records if r.entity_id]


@dataclass
class RetrievalResult:
    """Outcome of fact retrieval."""

    facts: list[Fact] = field(default_factory=list)
    elapsed_s: float = 0.0
    cache_used: bool = False
    degraded_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    entities_queried: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.facts


def build_provider(config: Config | RetrievalConfig) -> KnowledgeProvider:
    """Instantiate the configured provider (with a shared on-disk cache)."""
    retrieval = config.retrieval if isinstance(config, Config) else config
    cache = DiskCache(
        config.resolve_path(retrieval.cache_dir) if isinstance(config, Config) else retrieval.cache_dir,
        ttl_seconds=retrieval.cache_ttl_seconds,
        enabled=True,
    )
    if retrieval.provider == "wikidata":
        return WikidataProvider(
            api_url=retrieval.api_url,
            cache=cache,
            user_agent=resolve_user_agent(retrieval.user_agent),
            timeout=retrieval.request_timeout_seconds,
            max_retries=retrieval.max_retries,
            rate_limit_per_second=retrieval.rate_limit_per_second,
            include_qualifiers=retrieval.include_qualifiers,
            allow_literal_values=retrieval.allow_literal_values,
            prefer_low_degree_properties=retrieval.prefer_low_degree_properties,
            max_value_chars=retrieval.max_value_chars,
            source_label=resolve_source_label("wikidata", retrieval.source_name),
            properties_allowlist=retrieval.properties_allowlist,
        )
    if retrieval.provider == "json_file":
        if not retrieval.local_json_path:
            raise ValueError("retrieval.local_json_path is required for provider='json_file'")
        path = config.resolve_path(retrieval.local_json_path) if isinstance(config, Config) else retrieval.local_json_path
        return JsonFileKnowledgeProvider(
            path, cache=cache, source_label=resolve_source_label("json_file", retrieval.source_name)
        )
    if retrieval.provider == "mock":
        # An empty mock provider is intentional: it exercises the "no facts" path.
        return MockKnowledgeProvider(
            [], cache=cache, source_label=resolve_source_label("mock", retrieval.source_name)
        )
    if retrieval.provider == "dbpedia":
        from .wikidata import SPARQL_BACKED_DBpediaProvider  # local import: raises with a clear message

        return SPARQL_BACKED_DBpediaProvider()
    raise ValueError(f"unknown retrieval.provider {retrieval.provider!r}")


class KnowledgePipeline:
    """Linking + retrieval with caching, bounds and explicit failure handling."""

    def __init__(
        self,
        config: Config | RetrievalConfig,
        provider: KnowledgeProvider | None = None,
    ) -> None:
        self.config = config
        self.retrieval: RetrievalConfig = config.retrieval if isinstance(config, Config) else config
        self.provider = provider if provider is not None else build_provider(config)
        cache_dir = None
        if isinstance(config, Config):
            cache_dir = config.resolve_path(self.retrieval.cache_dir)
        elif self.retrieval.cache_dir:
            cache_dir = self.retrieval.cache_dir
        self.cache = DiskCache(cache_dir or ".cache/factanchor", ttl_seconds=self.retrieval.cache_ttl_seconds)
        self.source_label = resolve_source_label(self.retrieval.provider, self.retrieval.source_name)
        # per-process memoisation on top of the disk cache
        self._link_memo: dict[tuple[str, str], tuple[list[EntityRecord], str | None]] = {}
        self._fact_memo: dict[tuple[str, int, str], list[Fact]] = {}

    # ------------------------------------------------------------------ link
    def link_entities(
        self,
        mentions: Sequence[EntityMention],
        *,
        prompt: str | None = None,
        entity_hint: str | None = None,
    ) -> LinkResult:
        """Link mentions to canonical entity ids (spec 7.2)."""
        result = LinkResult()
        if not self.retrieval.enabled:
            result.records = [
                EntityRecord(mention=m.mention, entity_id=None, label=None, linking_confidence=0.0, link_method="retrieval_disabled", source="none", start=m.start, end=m.end)
                for m in mentions
            ]
            result.unlinked = len(result.records)
            result.degraded_reason = "retrieval_disabled"
            return result
        if not mentions and not entity_hint:
            return result

        started = time.monotonic()
        # De-duplicate mentions by normalised form, keep the best (longest) one.
        unique_mentions = deduplicate_mentions(mentions)
        hint_record: list[EntityRecord] = []
        if entity_hint:
            hint_records = self._link_mention(
                EntityMention(mention=entity_hint, start=0, end=len(entity_hint), confidence=0.99, source="dataset_hint"),
                prompt=prompt,
                hint=entity_hint,
            )
            hint_record = [r for r in hint_records.records if r.is_linked]
            result.api_calls += hint_records.api_calls
            result.errors.extend(hint_records.errors)

        for mention in unique_mentions:
            if len(result.records) + len(hint_record) >= self.retrieval.max_entities:
                break
            linked = self._link_mention(mention, prompt=prompt, hint=entity_hint)
            result.api_calls += linked.api_calls
            result.errors.extend(linked.errors)
            result.records.extend(linked.records)

        # Dataset-provided entity label wins if it produced a confident link.
        if hint_record:
            hint_ids = {r.entity_id for r in hint_record}
            result.records = [r for r in result.records if r.entity_id not in hint_ids]
            result.records = hint_record + result.records
        result.records = result.records[: self.retrieval.max_entities]

        result.linked = sum(1 for r in result.records if r.is_linked)
        result.unlinked = sum(1 for r in result.records if not r.is_linked)
        result.ambiguous = sum(1 for r in result.records if r.candidates and not r.is_linked)
        result.elapsed_s = time.monotonic() - started
        if result.errors and result.linked == 0:
            result.degraded_reason = "linking_failed"
        return result

    def _link_mention(
        self,
        mention: EntityMention,
        *,
        prompt: str | None,
        hint: str | None,
    ) -> LinkResult:
        out = LinkResult()
        key = (normalise_mention(mention.mention), (mention.context or prompt or "")[: self.retrieval.entity_context_hint_chars])
        if key in self._link_memo:
            memo_records, memo_error = self._link_memo[key]
            out.records = [record.model_copy() for record in memo_records]
            out.errors = [memo_error] if memo_error else []
            out.linked = sum(1 for r in out.records if r.is_linked)
            out.unlinked = len(out.records) - out.linked
            return out
        try:
            candidates = self.provider.search_entities(
                mention.mention,
                limit=5,
                context=mention.context or prompt,
            )
            out.api_calls += 1
        except KnowledgeUnavailable as exc:
            logger.warning("entity linking unavailable for %r: %s", mention.mention, exc)
            out.errors.append(str(exc))
            out.records = [self._unlinked_record(mention, method="service_unavailable", candidates=[])]
            out.unlinked = 1
            self._link_memo[key] = ([r.model_copy() for r in out.records], str(exc))
            return out
        except Exception as exc:  # defensive: a provider bug must not kill the run
            logger.exception("unexpected linker failure for %r", mention.mention)
            out.errors.append(f"{type(exc).__name__}: {exc}")
            out.records = [self._unlinked_record(mention, method="linker_error", candidates=[])]
            out.unlinked = 1
            return out

        if not candidates:
            out.records = [self._unlinked_record(mention, method="no_candidates", candidates=[])]
            out.unlinked = 1
            self._link_memo[key] = ([r.model_copy() for r in out.records], None)
            return out

        scored = [
            (self._score_candidate(mention, candidate, prompt=prompt, hint=hint), candidate)
            for candidate in candidates
        ]
        scored.sort(key=lambda pair: -pair[0])
        best_score, best = scored[0]
        candidate_dicts = [candidate.to_dict() for _score, candidate in scored[:3]]
        if best_score < self.retrieval.min_linking_confidence:
            logger.debug(
                "mention %r left unlinked (best %.2f < %.2f)",
                mention.mention,
                best_score,
                self.retrieval.min_linking_confidence,
            )
            out.records = [
                self._unlinked_record(mention, method="below_threshold", candidates=candidate_dicts, confidence=best_score)
            ]
            out.unlinked = 1
            self._link_memo[key] = ([r.model_copy() for r in out.records], None)
            return out

        # Two near-tied candidates => keep the ambiguity visible in the record.
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        ambiguous = (best_score - runner_up) < 0.05 and len(scored) > 1
        record = EntityRecord(
            mention=mention.mention,
            entity_id=best.entity_id,
            label=best.label,
            description=best.description or None,
            linking_confidence=round(best_score, 4),
            link_method="dataset_hint" if hint and normalise_mention(hint) == normalise_mention(mention.mention) else "search",
            aliases=list(best.aliases),
            source=best.source,
            start=mention.start,
            end=mention.end,
            candidates=candidate_dicts,
        )
        logger.debug(
            "linked %r -> %s (%s) confidence=%.3f%s",
            mention.mention,
            best.entity_id,
            best.label,
            best_score,
            " [ambiguous]" if ambiguous else "",
        )
        out.records = [record]
        out.linked = 1
        out.ambiguous = 1 if ambiguous else 0
        self._link_memo[key] = ([r.model_copy() for r in out.records], None)
        return out

    def _score_candidate(
        self,
        mention: EntityMention,
        candidate: EntityCandidate,
        *,
        prompt: str | None,
        hint: str | None,
    ) -> float:
        """Combine label agreement, provider score and contextual overlap."""
        normalised_mention = normalise_mention(mention.mention)
        normalised_label = normalise_mention(candidate.label)
        if normalised_mention and normalised_mention == normalised_label:
            label_score = 1.0
        elif normalised_mention and (normalised_mention in normalised_label or normalised_label in normalised_mention):
            label_score = 0.85
        elif any(normalise_mention(a) == normalised_mention for a in candidate.aliases):
            label_score = 0.8
        else:
            label_score = max(0.3, SequenceMatcher(None, normalised_mention, normalised_label).ratio() * 0.7)
        context_score = context_overlap_score(mention.context, candidate, prompt)
        base_score = float(candidate.score)
        hint_bonus = 0.0
        if hint and normalised_label == normalise_mention(hint):
            hint_bonus = 0.15
        # Label agreement dominates; context breaks the ties that matter.
        score = 0.55 * label_score + 0.25 * base_score + 0.20 * context_score + hint_bonus
        return max(0.0, min(1.0, score))

    @staticmethod
    def _unlinked_record(
        mention: EntityMention,
        *,
        method: str,
        candidates: list[dict],
        confidence: float = 0.0,
    ) -> EntityRecord:
        return EntityRecord(
            mention=mention.mention,
            entity_id=None,
            label=None,
            description=None,
            linking_confidence=round(float(confidence), 4),
            link_method=method,
            source="unlinked",
            start=mention.start,
            end=mention.end,
            candidates=candidates,
        )

    # -------------------------------------------------------------- retrieve
    def retrieve_facts(
        self,
        entities: Sequence[EntityRecord],
        *,
        max_facts_per_entity: int | None = None,
    ) -> RetrievalResult:
        """Retrieve bounded, deduplicated facts for the linked entities (spec 7.3)."""
        outcome = RetrievalResult()
        if not self.retrieval.enabled:
            outcome.degraded_reason = "retrieval_disabled"
            return outcome
        per_entity = int(max_facts_per_entity or self.retrieval.max_facts_per_entity)
        linked = [record for record in entities if record.is_linked][: self.retrieval.max_entities]
        if not linked:
            outcome.degraded_reason = "no_linked_entities"
            return outcome

        started = time.monotonic()
        seen: set[tuple[str, str, str]] = set()
        for record in linked:
            assert record.entity_id is not None  # filtered above
            outcome.entities_queried.append(record.entity_id)
            memo_key = (record.entity_id, per_entity, self.source_label)
            if memo_key in self._fact_memo:
                facts = list(self._fact_memo[memo_key])
                outcome.cache_used = True
            else:
                try:
                    facts = self.provider.get_facts(record.entity_id, max_facts=per_entity)
                    self._fact_memo[memo_key] = list(facts)
                    # Persist for the ``cache_only`` fallback: if the service goes
                    # down later, we can still serve *previously retrieved* facts
                    # (and only those -- nothing is ever synthesised).
                    if self.cache is not None and facts:
                        self.cache.set(
                            "facts",
                            self._fact_cache_key(record.entity_id, per_entity),
                            [fact.model_dump(mode="json") for fact in facts],
                        )
                except KnowledgeUnavailable as exc:
                    logger.warning("fact retrieval unavailable for %s: %s", record.entity_id, exc)
                    outcome.errors.append(str(exc))
                    facts = self._fallback_facts(record)
                    outcome.cache_used = True
                except Exception as exc:  # defensive
                    logger.exception("provider failure for %s", record.entity_id)
                    outcome.errors.append(f"{type(exc).__name__}: {exc}")
                    facts = self._fallback_facts(record)
            for fact in facts:
                if fact.confidence < self.retrieval.min_fact_confidence:
                    continue
                if not self._is_well_formed(fact):
                    continue
                key = (fact.subject_id, fact.predicate_id, fact.object_id or fact.object_label)
                if key in seen:
                    continue
                seen.add(key)
                outcome.facts.append(fact)

        outcome.elapsed_s = time.monotonic() - started
        if outcome.errors:
            if outcome.facts:
                outcome.degraded_reason = "partial_retrieval_failure"
            else:
                # May raise when fallback_on_unavailable == "error".
                outcome.degraded_reason = self._apply_failure_policy(outcome)
        elif not outcome.facts:
            outcome.degraded_reason = "no_facts_returned"
        return outcome

    def _fact_cache_key(self, entity_id: str, per_entity: int) -> str:
        return f"{entity_id}|{per_entity}|{self.source_label}"

    def _is_well_formed(self, fact: Fact) -> bool:
        """Reject malformed results (spec 7.3, 8.2)."""
        if not fact.predicate_id or not fact.subject_id:
            return False
        if not fact.object_label.strip():
            return False
        if fact.object_type == "entity" and fact.object_id is None:
            return False
        if len(fact.object_label) > 600:
            return False
        return True

    def _fallback_facts(self, record: EntityRecord) -> list[Fact]:
        """Apply ``fallback_on_unavailable``; never fabricates facts."""
        policy = self.retrieval.fallback_on_unavailable
        if policy == "error":
            raise KnowledgeUnavailable(
                f"knowledge service unavailable and retrieval.fallback_on_unavailable='error' "
                f"(entity {record.entity_id})",
                provider=getattr(self.provider, "name", "unknown"),
                transient=True,
            )
        if policy == "no_facts":
            return []
        # cache_only: use the on-disk cache for this entity if present
        if self.cache is None or record.entity_id is None:
            return []
        cached = self.cache.get("facts", self._fact_cache_key(record.entity_id, self.retrieval.max_facts_per_entity))
        if not cached:
            return []
        try:
            facts = [Fact.model_validate(item) for item in cached]
        except Exception as exc:
            logger.warning("cached facts for %s are unreadable: %s", record.entity_id, exc)
            return []
        logger.info("using %d cached fact(s) for %s", len(facts), record.entity_id)
        return facts

    def _apply_failure_policy(self, outcome: RetrievalResult) -> str | None:
        if self.retrieval.fallback_on_unavailable == "error":
            raise KnowledgeUnavailable(
                "knowledge service unavailable for all requested entities and "
                "retrieval.fallback_on_unavailable='error': " + "; ".join(outcome.errors),
                provider=getattr(self.provider, "name", "unknown"),
            )
        return "retrieval_unavailable" if outcome.errors else None

    # -------------------------------------------------------------- helpers
    def known_entities(self) -> list[str]:
        """Lexicon for the NER backend, when the provider can supply one."""
        try:
            return list(self.provider.known_entities())
        except Exception:  # pragma: no cover - provider specific
            return []

    @property
    def cache_stats(self) -> dict:
        return self.cache.stats() if self.cache else {}

    def close(self) -> None:
        self.provider.close()


def retrieve_entity_context(
    prompt: str,
    *,
    config: Config,
    pipeline: KnowledgePipeline | None = None,
    extractor=None,
    entity_hint: str | None = None,
) -> tuple[list[EntityRecord], RetrievalResult, LinkResult, dict[str, float]]:
    """Convenience end-to-end knowledge step used by the runner and the demo.

    Returns ``(entities, retrieval_result, link_result, timings)`` where
    ``timings`` contains ``extraction_s``, ``linking_s``, ``retrieval_s``.
    """
    from ..entities import build_extractor  # local import avoids a cycle

    pipeline = pipeline or KnowledgePipeline(config)
    extractor = extractor or build_extractor(config.ner, known_entities=pipeline.known_entities())

    timings: dict[str, float] = {"extraction_s": 0.0, "linking_s": 0.0, "retrieval_s": 0.0}
    with Timer("extraction", synchronize=False) as extraction:
        mentions = extractor.extract(prompt)
    timings["extraction_s"] = extraction.elapsed

    with Timer("linking") as linking:
        link_result = pipeline.link_entities(mentions, prompt=prompt, entity_hint=entity_hint)
    timings["linking_s"] = linking.elapsed

    with Timer("retrieval") as retrieval:
        retrieval_result = pipeline.retrieve_facts(link_result.records)
    timings["retrieval_s"] = retrieval.elapsed
    return link_result.records, retrieval_result, link_result, timings
