"""Knowledge-provider interface, shared HTTP plumbing and the mock provider.

Every concrete provider implements :class:`KnowledgeProvider`:

* ``search_entities(query, limit, context) -> list[EntityCandidate]``
* ``get_facts(entity_id, max_facts) -> list[Fact]``

Failure policy (spec 7.3)
-------------------------
A provider **never** fabricates a fact.  When the backend is unreachable it
raises :class:`KnowledgeUnavailable`, and the caller
(:class:`factanchor.knowledge.pipeline.KnowledgePipeline`) applies the
configured fallback: ``cache_only`` (use whatever is cached), ``no_facts``
(continue with an empty fact list) or ``error`` (fail loudly).
"""

from __future__ import annotations

import json
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from ..schemas import Fact, Qualifier, utc_now_iso
from ..utils import DiskCache, RateLimiter, get_logger

__all__ = [
    "KnowledgeUnavailable",
    "resolve_user_agent",
    "resolve_source_label",
    "DEFAULT_SOURCE_LABELS",
    "EntityCandidate",
    "KnowledgeProvider",
    "HttpJsonClient",
    "MockKnowledgeProvider",
    "JsonFileKnowledgeProvider",
    "build_candidate",
    "DEFAULT_USER_AGENT",
]

logger = get_logger("knowledge")

DEFAULT_USER_AGENT = "FactAnchor/0.1 (research prototype; +https://example.org/factanchor)"

#: Fallback source label per provider.  Facts carry this label into every record and
#: audit entry, so an offline fixture can never be presented as a real knowledge
#: source.
DEFAULT_SOURCE_LABELS: dict[str, str] = {
    "wikidata": "Wikidata",
    "dbpedia": "DBpedia",
    "json_file": "LocalJSON",
    "mock": "MockKB",
}


def resolve_source_label(provider: str, configured: str | None = None) -> str:
    """Return the label used for facts from ``provider`` (spec 6.2 provenance)."""
    if configured and configured.strip():
        return configured.strip()
    return DEFAULT_SOURCE_LABELS.get(provider, provider)


def resolve_user_agent(configured: str | None = None, env: dict[str, str] | None = None) -> str:
    """Resolve the HTTP User-Agent.

    Wikimedia requires a descriptive agent (https://meta.wikimedia.org/wiki/User-Agent_policy).
    Precedence: explicit config value > ``FACTANCHOR_USER_AGENT`` > a string built
    from ``FACTANCHOR_CONTACT`` > the packaged default.  Never read silently from
    an arbitrary variable -- only these documented names are used.
    """
    import os as _os

    env = dict(_os.environ if env is None else env)
    if configured and configured.strip():
        return configured.strip()
    from_env = (env.get("FACTANCHOR_USER_AGENT") or "").strip()
    if from_env:
        return from_env
    contact = (env.get("FACTANCHOR_CONTACT") or "").strip()
    if contact:
        return f"FactAnchor/0.1 (research prototype; contact: {contact})"
    return DEFAULT_USER_AGENT


class KnowledgeUnavailable(RuntimeError):
    """Raised when a knowledge backend cannot answer (network, HTTP, timeout, 5xx)."""

    def __init__(self, message: str, *, provider: str = "unknown", transient: bool = True) -> None:
        super().__init__(message)
        self.provider = provider
        self.transient = transient


@dataclass(frozen=True)
class EntityCandidate:
    """One candidate returned by an entity search (used for linking/disambiguation)."""

    entity_id: str
    label: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    concept_uri: str | None = None
    score: float = 0.0
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "description": self.description,
            "aliases": list(self.aliases),
            "score": self.score,
            "source": self.source,
        }


def build_candidate(
    entity_id: str,
    label: str,
    *,
    description: str = "",
    aliases: Sequence[str] = (),
    score: float = 0.0,
    source: str = "unknown",
) -> EntityCandidate:
    return EntityCandidate(
        entity_id=entity_id,
        label=label,
        description=description or "",
        aliases=tuple(a for a in aliases if a),
        score=float(score),
        source=source,
    )


# ---------------------------------------------------------------------------
# HTTP helper with timeouts, retries, rate limiting and caching
# ---------------------------------------------------------------------------


@dataclass
class HttpJsonClient:
    """Minimal JSON-over-HTTP client honouring the spec's networking rules.

    * bounded retries with exponential backoff for transient errors
      (429/5xx/timeouts);
    * client-side rate limiting;
    * explicit connect/read timeouts;
    * on-disk response cache;
    * a descriptive User-Agent (Wikimedia requires one).
    """

    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = 10.0
    max_retries: int = 3
    rate_limit_per_second: float = 5.0
    cache: DiskCache | None = None
    namespace: str = "http"
    backoff_base: float = 1.0
    _limiter: RateLimiter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.rate_limit_per_second)

    # -- public -------------------------------------------------------------
    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
        cache_key: str | None = None,
        use_cache: bool = True,
    ) -> Any:
        full_url = self._build_url(url, params)
        if self.cache is not None and use_cache:
            cached = self.cache.get(self.namespace, cache_key or full_url)
            if cached is not None:
                return cached

        payload = self._request_with_retries(full_url, headers or {})
        if self.cache is not None and use_cache:
            self.cache.set(self.namespace, cache_key or full_url, payload)
        return payload

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _build_url(url: str, params: dict[str, Any] | None) -> str:
        if not params:
            return url
        query = urlparse.urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
        joiner = "&" if "?" in url else "?"
        return f"{url}{joiner}{query}"

    def _request_with_retries(self, url: str, headers: dict[str, str]) -> Any:
        last_error: Exception | None = None
        attempts = max(1, int(self.max_retries) + 1)
        request_headers = {"User-Agent": self.user_agent, "Accept": "application/json", **headers}
        for attempt in range(attempts):
            self._limiter.acquire()
            try:
                request = urlrequest.Request(url, headers=request_headers, method="GET")
                with urlrequest.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - fixed scheme
                    body = response.read().decode("utf-8", errors="replace")
                return json.loads(body)
            except urlerror.HTTPError as exc:
                retryable = exc.code in {408, 425, 429, 500, 502, 503, 504}
                detail = f"HTTP {exc.code} from {url}"
                last_error = exc
                if not retryable:
                    raise KnowledgeUnavailable(detail, transient=False) from exc
                logger.warning("%s (attempt %d/%d)", detail, attempt + 1, attempts)
            except (urlerror.URLError, socket.timeout, TimeoutError) as exc:
                last_error = exc
                logger.warning("network error for %s (attempt %d/%d): %s", url, attempt + 1, attempts, exc)
            except json.JSONDecodeError as exc:
                # Malformed body: treat as unavailable; a retry is unlikely to help.
                raise KnowledgeUnavailable(f"invalid JSON from {url}: {exc}", transient=False) from exc
            except OSError as exc:  # DNS failures, connection refused, ...
                last_error = exc
                logger.warning("OS error for %s (attempt %d/%d): %s", url, attempt + 1, attempts, exc)

            if attempt < attempts - 1:
                time.sleep(min(self.backoff_base * (2**attempt), 8.0))
        detail = f"{last_error}" if last_error else "unknown error"
        raise KnowledgeUnavailable(f"knowledge request failed after {attempts} attempts: {detail}")


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class KnowledgeProvider(ABC):
    """Abstract knowledge source."""

    name: str = "abstract"
    source_label: str = "unknown"
    supports_search: bool = True

    def __init__(self, *, cache: DiskCache | None = None) -> None:
        self.cache = cache

    @abstractmethod
    def search_entities(
        self, query: str, *, limit: int = 5, context: str | None = None
    ) -> list[EntityCandidate]:
        """Return candidate entities for a mention (may raise KnowledgeUnavailable)."""

    @abstractmethod
    def get_facts(self, entity_id: str, *, max_facts: int = 10) -> list[Fact]:
        """Return structured facts about ``entity_id`` (may raise KnowledgeUnavailable)."""

    def known_entities(self) -> list[str]:  # pragma: no cover - overridden by subclasses
        """Optional lexicon for the NER backend (used by offline providers)."""
        return []

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


# ---------------------------------------------------------------------------
# Offline providers
# ---------------------------------------------------------------------------


@dataclass
class _OfflineEntity:
    """Internal record of an offline knowledge entry."""

    entity_id: str
    label: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    facts: tuple[dict[str, Any], ...] = ()


class MockKnowledgeProvider(KnowledgeProvider):
    """In-memory provider for tests and for ``--provider mock`` demos.

    Data must be supplied explicitly through ``entities``; nothing is invented.
    Passing an empty mapping makes every lookup return "no result", which is the
    documented behaviour used to test the unlinked/empty-fact paths.
    """

    name = "mock"
    source_label = "MockKB"

    def __init__(
        self,
        entities: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
        *,
        cache: DiskCache | None = None,
        available: bool = True,
        source_label: str = "MockKB",
    ) -> None:
        super().__init__(cache=cache)
        self.available = available
        self.source_label = source_label
        self._entities: list[_OfflineEntity] = []
        records: Sequence[dict[str, Any]]
        if entities is None:
            records = []
        elif isinstance(entities, dict):
            records = [{"entity_id": key, **value} for key, value in entities.items()]
        else:
            records = list(entities)
        for record in records:
            self._entities.append(self._parse_record(record))

    @staticmethod
    def _parse_record(record: dict[str, Any]) -> _OfflineEntity:
        missing = [k for k in ("entity_id", "label") if not record.get(k)]
        if missing:
            raise ValueError(f"mock knowledge entry missing required field(s): {missing}")
        facts = record.get("facts") or []
        if not isinstance(facts, list):
            raise ValueError("mock entity 'facts' must be a list")
        aliases = record.get("aliases") or []
        return _OfflineEntity(
            entity_id=str(record["entity_id"]),
            label=str(record["label"]),
            description=str(record.get("description") or ""),
            aliases=tuple(str(a) for a in aliases),
            facts=tuple(facts),
        )

    def _check_available(self) -> None:
        if not self.available:
            raise KnowledgeUnavailable("mock knowledge provider marked unavailable", provider=self.name)

    def search_entities(
        self, query: str, *, limit: int = 5, context: str | None = None
    ) -> list[EntityCandidate]:
        self._check_available()
        needle = (query or "").strip().lower()
        if not needle:
            return []
        matches: list[EntityCandidate] = []
        for entity in self._entities:
            names = {entity.label.lower(), *(a.lower() for a in entity.aliases), entity.entity_id.lower()}
            if needle in names:
                score = 1.0 if entity.label.lower() == needle else 0.9
                matches.append(build_candidate(entity.entity_id, entity.label, description=entity.description, aliases=entity.aliases, score=score, source=self.source_label))
            elif any(needle in name for name in names):
                matches.append(build_candidate(entity.entity_id, entity.label, description=entity.description, aliases=entity.aliases, score=0.5, source=self.source_label))
        matches.sort(key=lambda c: -c.score)
        return matches[:limit]

    def get_facts(self, entity_id: str, *, max_facts: int = 10) -> list[Fact]:
        self._check_available()
        for entity in self._entities:
            if entity.entity_id != entity_id:
                continue
            return [self._to_fact(entity, raw) for raw in entity.facts[:max_facts]]
        logger.debug("mock provider has no entity %s", entity_id)
        return []

    def _to_fact(self, entity: _OfflineEntity, raw: dict[str, Any]) -> Fact:
        qualifiers_raw = raw.get("qualifiers") or []
        qualifiers = tuple(
            Qualifier(
                predicate_id=str(q.get("predicate_id", "")),
                predicate_label=q.get("predicate_label"),
                value=str(q.get("value", "")),
                value_type=q.get("value_type", "literal"),
                value_id=q.get("value_id"),
            )
            for q in qualifiers_raw
        )
        return Fact(
            subject_id=entity.entity_id,
            subject_label=entity.label,
            predicate_id=str(raw.get("predicate_id", "")),
            predicate_label=raw.get("predicate_label"),
            object_id=raw.get("object_id"),
            object_label=str(raw.get("object_label", "")),
            # An object without an id cannot be an entity reference: treat it as a
            # literal unless the fixture says otherwise (this keeps offline
            # fixtures valid instead of letting them look malformed).
            object_type=raw.get("object_type") or ("entity" if raw.get("object_id") else "literal"),
            object_datatype=raw.get("object_datatype"),
            qualifiers=qualifiers,
            source=raw.get("source") or self.source_label,
            source_url=raw.get("source_url"),
            confidence=float(raw.get("confidence", 1.0)),
            retrieved_at=utc_now_iso(),
        )

    def known_entities(self) -> list[str]:
        out: list[str] = []
        for entity in self._entities:
            out.append(entity.label)
            out.extend(entity.aliases)
        return out


class JsonFileKnowledgeProvider(MockKnowledgeProvider):
    """Offline provider reading the same record format from a JSONL/JSON file.

    File format (one JSON object per line)::

        {"entity_id": "Q7186", "label": "Marie Curie",
         "aliases": ["Madame Curie"], "description": "...",
         "facts": [{"predicate_id": "P106", "predicate_label": "occupation",
                    "object_label": "physicist", "object_id": "Q169470"}]}

    The file is read once at construction; malformed records raise with the line
    number so a broken knowledge file cannot silently degrade the experiment.
    """

    name = "json_file"

    def __init__(
        self,
        path: Path | str,
        *,
        cache: DiskCache | None = None,
        source_label: str = "LocalJSON",
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"offline knowledge file {self.path} not found. "
                "Provide retrieval.local_json_path or use provider='mock'."
            )
        records = self._load(self.path)
        super().__init__(records, cache=cache, source_label=source_label)

    @staticmethod
    def _load(path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        stripped = text.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if not isinstance(parsed, list):
                raise ValueError(f"{path} must contain a JSON array or JSONL records")
            return [dict(record) for record in parsed]
        records: list[dict[str, Any]] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{lineno}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno} must be a JSON object")
            records.append(record)
        return records
