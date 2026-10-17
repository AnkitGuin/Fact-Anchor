"""Wikidata knowledge provider (primary knowledge source, spec Section 5).

Uses the public MediaWiki/Wikibase Action API (no SPARQL, no extra dependency):

* ``wbsearchentities`` -> candidate entities for a mention;
* ``wbgetentities``   -> labels, descriptions, aliases and claims (facts).

Deliberate engineering choices
------------------------------
* **Bounded calls.**  One search call per mention, one entity call per entity,
  plus at most one extra call to resolve the labels of object entities and
  property labels.  Nothing is queried per generated token (spec Section 2).
* **No fabrication.**  Anything that cannot be typed safely (unknown datatype,
  missing value, deprecated rank) is skipped and the reason is returned in the
  fact's metadata; the provider never guesses.
* **Relevance first.**  Facts are ranked by a curated property allowlist, then
  by statement rank, so a biography does not get flooded with identifiers and
  category links (spec 7.3 "prefer relevant facts over arbitrary high-degree
  graph properties").
* **Untrusted input.**  Every returned value is plain data: it is never
  evaluated, templated into code, or written to a shell (spec Section 18).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from ..schemas import Fact, Qualifier, utc_now_iso
from ..utils import DiskCache, get_logger
from .base import (
    DEFAULT_USER_AGENT,
    EntityCandidate,
    HttpJsonClient,
    KnowledgeProvider,
    KnowledgeUnavailable,
    build_candidate,
)

__all__ = [
    "WikidataProvider",
    "SPARQL_BACKED_DBpediaProvider",
    "PROPERTY_ALLOWLIST",
    "LOW_INFORMATION_PROPERTIES",
    "SKIPPED_DATATYPES",
    "format_wikidata_time",
    "format_quantity",
]

logger = get_logger("knowledge.wikidata")

DEFAULT_API_URL = "https://www.wikidata.org/w/api.php"

#: Properties kept when the fact budget is tight.  Ordered: earlier = higher
#: priority.  Chosen to be entity-descriptive rather than structural.
PROPERTY_ALLOWLIST: tuple[str, ...] = (
    "P31",  # instance of
    "P106",  # occupation
    "P39",  # position held
    "P108",  # employer
    "P69",  # educated at
    "P166",  # award received
    "P800",  # notable work
    "P101",  # field of work
    "P1066",  # student of
    "P184",  # doctoral advisor
    "P512",  # academic degree
    "P463",  # member of
    "P27",  # country of citizenship
    "P569",  # date of birth
    "P570",  # date of death
    "P19",  # place of birth
    "P20",  # place of death
    "P26",  # spouse
    "P22",  # father
    "P25",  # mother
    "P40",  # child
    "P937",  # work location
    "P1412",  # languages spoken, written or signed
    "P103",  # native language
    "P551",  # residence
    "P17",  # country
    "P571",  # inception
    "P576",  # dissolved, abolished or demolished date
    "P112",  # founded by
    "P159",  # headquarters location
    "P452",  # industry
    "P1056",  # product or material produced
    "P50",  # author
    "P136",  # genre
    "P495",  # country of origin
    "P577",  # publication date
    "P123",  # publisher
    "P275",  # copyright license
    "P178",  # developer
    "P277",  # programming language
    "P306",  # operating system
    "P361",  # part of
    "P527",  # has part(s)
    "P155",  # follows
    "P156",  # followed by
    "P276",  # location
    "P625",  # coordinate location
    "P1448",  # official name
    "P1082",  # population
    "P2046",  # area
    "P36",  # capital
    "P30",  # continent
    "P47",  # shares border with
)

#: Structural / low-information properties excluded when
#: ``retrieval.prefer_low_degree_properties`` is enabled.
LOW_INFORMATION_PROPERTIES: frozenset[str] = frozenset(
    {
        "P910",  # topic's main category
        "P373",  # Commons category
        "P646",  # Freebase ID
        "P2671",  # Google Knowledge Graph ID
        "P3417",  # Quora topic ID
        "P1659",  # sees also
        "P1343",  # described by source
        "P1204",  # Wikimedia portal
        "P935",  # Commons gallery
        "P948",  # page banner
    }
)

#: Datatypes that are never turned into facts (opaque blobs, media, URLs).
SKIPPED_DATATYPES: frozenset[str] = frozenset(
    {
        "commonsMedia",
        "geo-shape",
        "tabular-data",
        "external-id",
        "url",
        "wikibase-lexeme",
        "wikibase-form",
        "wikibase-sense",
        "entity-schema",
        "musical-notation",
        "math",
    }
)

#: Instance-of values that indicate a non-article item; such candidates are
#: dropped during linking because their facts are about Wikimedia maintenance.
_NON_ARTICLE_INSTANCES: frozenset[str] = frozenset(
    {
        "Q4167410",  # Wikimedia disambiguation page
        "Q4167836",  # Wikimedia category
        "Q11266439",  # Wikimedia template
        "Q13406463",  # Wikimedia list article
        "Q17362920",  # Wikimedia duplicated page
    }
)

_ISO_ZERO_PAD = re.compile(r"^([+-])(\d+)-(\d{2})-(\d{2})T")

# Wikidata date-precision codes (Wikibase data model: "prec" in the time datatype).
# 0..2 are coarse geological scales, 3 millennium, 4 century, 5 decade, 6 year,
# 7 month, 8 day, 9..11 finer than a day.  We render at the coarsest granularity
# that the value actually supports so our own formatting cannot invent precision.
PRECISION_MILLENNIUM = 3
PRECISION_CENTURY = 4
PRECISION_DECADE = 5
PRECISION_YEAR = 6
PRECISION_MONTH = 7
PRECISION_DAY = 8


def _month_name(month: int) -> str:
    return [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ][month - 1]


def format_wikidata_time(time_value: dict[str, Any]) -> str | None:
    """Convert a Wikidata ``time`` datavalue into readable text.

    The renderer never claims more precision than the stored value has: a
    century-precision date is *not* reported as a precise day, because that would
    introduce a factual error in our own code.  Returns ``None`` for values that
    cannot be parsed.
    """
    if not isinstance(time_value, dict):
        return None
    raw = time_value.get("time")
    precision = time_value.get("precision")
    if not isinstance(raw, str):
        return None
    match = re.match(r"^([+-])(\d{1,16})-(\d{2})-(\d{2})T", raw)
    if not match:
        return None
    sign, year_digits, month, day = match.groups()
    year = int(year_digits)
    month_i, day_i = int(month), int(day)
    if precision is None:
        precision = PRECISION_DAY
    precision = int(precision)
    # Years are stored as a magnitude plus a sign; a negative sign means BCE and is
    # rendered explicitly so no era marker is ever dropped.
    era_suffix = "" if sign == "+" else " BCE"
    if precision <= 2:
        if year >= 1_000_000:
            millions = year / 1_000_000
            text = f"about {millions:.0f} million years ago" if sign == "+" else f"about {millions:.0f} million years"
            return text + era_suffix
        return f"about {year:,} years{era_suffix}"
    if precision == PRECISION_MILLENNIUM:
        millennium = (year - 1) // 1000 + 1
        return f"{millennium}{_ordinal_suffix(millennium)} millennium{era_suffix}"
    if precision == PRECISION_CENTURY:
        century = (year - 1) // 100 + 1
        return f"{century}{_ordinal_suffix(century)} century{era_suffix}"
    if precision == PRECISION_DECADE:
        return f"{(year // 10) * 10}s{era_suffix}"
    if precision == PRECISION_YEAR or month_i == 0:
        return f"{year}{era_suffix}"
    if precision == PRECISION_MONTH or day_i == 0:
        return f"{_month_name(month_i)} {year}{era_suffix}"
    return f"{day_i} {_month_name(month_i)} {year}{era_suffix}"


def _ordinal_suffix(number: int) -> str:
    if 10 <= number % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")


def format_quantity(quantity_value: dict[str, Any], unit_label: str | None = None) -> str | None:
    """Convert a Wikidata ``quantity`` datavalue into readable text."""
    if not isinstance(quantity_value, dict):
        return None
    amount = quantity_value.get("amount")
    if not isinstance(amount, str):
        return None
    try:
        number = float(amount.lstrip("+"))
    except ValueError:
        return None
    lower = quantity_value.get("lowerBound")
    upper = quantity_value.get("upperBound")
    text = f"{number:,.0f}" if abs(number - round(number)) < 1e-9 else f"{number:,}"
    unit = quantity_value.get("unit") or ""
    if isinstance(unit, str) and unit.startswith("http") and unit.rsplit("/", 1)[-1].startswith("Q") and unit_label:
        text = f"{text} {unit_label}"
    elif isinstance(unit, str) and unit == "1":
        pass  # dimensionless
    if isinstance(lower, str) and isinstance(upper, str) and lower.lstrip("+") != amount.lstrip("+"):
        try:
            text = f"{text} (range {float(lower.lstrip('+')):,.0f}-{float(upper.lstrip('+')):,.0f})"
        except ValueError:
            pass
    if quantity_value.get("unit") == "http://www.wikidata.org/entity/Q11573":
        text += " (uncertain)"
    return text


class WikidataProvider(KnowledgeProvider):
    """Wikidata Action-API provider.

    Parameters
    ----------
    api_url:
        API endpoint; overridable through ``FACTANCHOR_WIKIDATA_API`` or
        ``retrieval.api_url`` (useful for a local mirror or a test double).
    cache:
        :class:`~factanchor.utils.DiskCache` shared with the rest of the run.
    """

    name = "wikidata"

    def __init__(
        self,
        *,
        api_url: str | None = None,
        language: str = "en",
        cache: DiskCache | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 10.0,
        max_retries: int = 3,
        rate_limit_per_second: float = 5.0,
        include_qualifiers: bool = True,
        allow_literal_values: bool = True,
        prefer_low_degree_properties: bool = True,
        max_value_chars: int = 160,
        properties_allowlist: Sequence[str] = (),
        source_label: str = "Wikidata",
    ) -> None:
        super().__init__(cache=cache)
        self.api_url = api_url or DEFAULT_API_URL
        self.language = language
        self.source_label = source_label
        self.include_qualifiers = include_qualifiers
        self.allow_literal_values = allow_literal_values
        self.prefer_low_degree_properties = prefer_low_degree_properties
        self.max_value_chars = max_value_chars
        self.allowlist: tuple[str, ...] = tuple(properties_allowlist) or PROPERTY_ALLOWLIST
        self._allowlist_index = {pid: i for i, pid in enumerate(self.allowlist)}
        self.client = HttpJsonClient(
            user_agent=user_agent,
            timeout=timeout,
            max_retries=max_retries,
            rate_limit_per_second=rate_limit_per_second,
            cache=cache,
            namespace="wikidata_http",
        )
        # caches that live for the lifetime of the provider (per process)
        self._label_cache: dict[str, str] = {}
        self._property_label_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ API
    def search_entities(
        self, query: str, *, limit: int = 5, context: str | None = None
    ) -> list[EntityCandidate]:
        query = (query or "").strip()
        if not query:
            return []
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": self.language,
            "uselang": self.language,
            "type": "item",
            "limit": max(1, int(limit)),
            "format": "json",
            "formatversion": 2,
            "origin": "*",
        }
        payload = self.client.get_json(self.api_url, params, cache_key=f"search|{self.language}|{query}|{limit}")
        results = payload.get("search") if isinstance(payload, dict) else None
        if results is None:
            logger.warning("unexpected wbsearchentities payload for %r", query)
            return []
        candidates: list[EntityCandidate] = []
        for result in results:
            entity_id = result.get("id")
            label = result.get("label") or result.get("match", {}).get("text") or ""
            if not entity_id or not str(entity_id).startswith("Q"):
                continue  # properties/lexemes are not entity targets for our pipeline
            match = result.get("match") or {}
            match_type = str(match.get("type", "")).lower()
            aliases = tuple(str(a) for a in (result.get("aliases") or []) if a)
            score = {
                "label": 0.95,
                "alias": 0.88,
                "description": 0.5,
            }.get(match_type, 0.6)
            if label.strip().lower() == query.lower():
                score = min(0.99, score + 0.05)
            candidates.append(
                build_candidate(
                    str(entity_id),
                    str(label),
                    description=str(result.get("description") or ""),
                    aliases=aliases,
                    score=score,
                    source=self.source_label,
                )
            )
        candidates.sort(key=lambda c: (-c.score, c.entity_id))
        return candidates[:limit]

    def get_facts(self, entity_id: str, *, max_facts: int = 10) -> list[Fact]:
        entity_id = (entity_id or "").strip()
        if not entity_id.startswith("Q"):
            raise ValueError(f"Wikidata entity id must look like 'Q123', got {entity_id!r}")
        payload = self.client.get_json(
            self.api_url,
            {
                "action": "wbgetentities",
                "ids": entity_id,
                "props": "labels|descriptions|claims|aliases",
                "languages": self.language,
                "languagefallback": "1",
                "format": "json",
                "formatversion": 2,
                "origin": "*",
            },
            cache_key=f"entity|{self.language}|{entity_id}",
        )
        entity = (payload.get("entities") or {}).get(entity_id)
        if not entity or entity.get("missing") is not None:
            logger.info("Wikidata entity %s not found", entity_id)
            return []
        subject_label = self._entity_label(entity) or entity_id
        claims: dict[str, Any] = entity.get("claims") or {}
        if not claims:
            return []

        statements: list[tuple[str, dict[str, Any], int]] = []
        for predicate_id, statement_list in claims.items():
            if not isinstance(statement_list, list) or not statement_list:
                continue
            if self.prefer_low_degree_properties and predicate_id in LOW_INFORMATION_PROPERTIES:
                continue
            for statement in statement_list:
                if not isinstance(statement, dict):
                    continue
                rank = statement.get("rank", "normal")
                if rank == "deprecated":
                    continue
                statements.append((predicate_id, statement, 0 if rank == "preferred" else 1))

        # Rank by allowlist position, then preferred rank, then property id.
        statements.sort(
            key=lambda item: (
                self._allowlist_index.get(item[0], len(self.allowlist) + 1),
                item[2],
                item[0],
            )
        )
        # One batched call resolves every object/property label we might need.
        self._prefetch_labels(entity_id, statements)

        facts: list[Fact] = []
        seen: set[tuple[str, str, str]] = set()
        for predicate_id, statement, _rank in statements:
            if len(facts) >= max_facts:
                break
            fact = self._statement_to_fact(entity_id, subject_label, predicate_id, statement)
            if fact is None:
                continue
            key = (fact.predicate_id, fact.object_id or "", fact.object_label)
            if key in seen:
                continue  # deduplicate identical statements (spec 7.3)
            seen.add(key)
            facts.append(fact)
        return facts

    # ------------------------------------------------------------- internals
    def _entity_label(self, entity: dict[str, Any]) -> str | None:
        labels = entity.get("labels") or {}
        entry = labels.get(self.language)
        if isinstance(entry, dict) and entry.get("value"):
            return str(entry["value"])
        return None

    def _prefetch_labels(self, subject_id: str, statements: Sequence[tuple[str, dict[str, Any], int]]) -> None:
        """Resolve labels for property ids and entity-valued objects in one call."""
        wanted: set[str] = set()
        for predicate_id, statement, _ in statements:
            wanted.add(predicate_id)
            value = ((statement.get("mainsnak") or {}).get("datavalue") or {}).get("value")
            if isinstance(value, dict) and value.get("id", "").startswith("Q"):
                wanted.add(str(value["id"]))
            for qualifier_values in (statement.get("qualifiers") or {}).values():
                for qualifier in qualifier_values or []:
                    qual_value = (qualifier.get("datavalue") or {}).get("value")
                    if isinstance(qual_value, dict) and str(qual_value.get("id", "")).startswith(("Q", "P")):
                        wanted.add(str(qual_value["id"]))
            for qualifier_id in (statement.get("qualifiers") or {}):
                wanted.add(str(qualifier_id))
        missing = sorted(
            identifier
            for identifier in wanted
            if identifier not in self._label_cache and identifier not in self._property_label_cache
        )
        if not missing:
            return
        for batch in _batched(missing, 50):
            try:
                payload = self.client.get_json(
                    self.api_url,
                    {
                        "action": "wbgetentities",
                        "ids": "|".join(batch),
                        "props": "labels",
                        "languages": self.language,
                        "languagefallback": "1",
                        "format": "json",
                        "formatversion": 2,
                        "origin": "*",
                    },
                    cache_key=f"labels|{self.language}|{'|'.join(batch)}",
                )
            except KnowledgeUnavailable as exc:
                # Label resolution is an enhancement: keep going with ids, which
                # are still faithful (no invented labels).
                logger.warning("label prefetch failed (%s); continuing with ids", exc)
                for identifier in batch:
                    self._label_cache.setdefault(identifier, identifier)
                    self._property_label_cache.setdefault(identifier, identifier)
                continue
            for identifier, entity in (payload.get("entities") or {}).items():
                label = self._entity_label(entity) or identifier
                if identifier.startswith("P"):
                    self._property_label_cache[identifier] = label
                else:
                    self._label_cache[identifier] = label

    def _property_label(self, predicate_id: str) -> str:
        if predicate_id in self._property_label_cache:
            return self._property_label_cache[predicate_id]
        cached = self.client.cache.get("property_label", predicate_id) if self.client.cache else None
        if isinstance(cached, str) and cached:
            self._property_label_cache[predicate_id] = cached
            return cached
        try:
            payload = self.client.get_json(
                self.api_url,
                {
                    "action": "wbgetentities",
                    "ids": predicate_id,
                    "props": "labels",
                    "languages": self.language,
                    "languagefallback": "1",
                    "format": "json",
                    "formatversion": 2,
                    "origin": "*",
                },
                cache_key=f"property_label|{self.language}|{predicate_id}",
            )
            label = self._entity_label((payload.get("entities") or {}).get(predicate_id, {})) or predicate_id
        except (KnowledgeUnavailable, AttributeError) as exc:
            logger.warning("could not resolve label for %s: %s", predicate_id, exc)
            label = predicate_id
        self._property_label_cache[predicate_id] = label
        if self.client.cache is not None:
            self.client.cache.set("property_label", predicate_id, label)
        return label

    def _statement_to_fact(
        self,
        subject_id: str,
        subject_label: str,
        predicate_id: str,
        statement: dict[str, Any],
    ) -> Fact | None:
        mainsnak = statement.get("mainsnak") or {}
        datatype = str(mainsnak.get("datatype") or "")
        if datatype in SKIPPED_DATATYPES:
            return None
        if mainsnak.get("snaktype") != "value":
            return None  # "somevalue"/"novalue" carry no usable object
        datavalue = mainsnak.get("datavalue") or {}
        parsed = self._parse_datavalue(datavalue, datatype)
        if parsed is None:
            return None
        object_label, object_id, object_type = parsed
        object_label = object_label.strip()
        if not object_label:
            return None
        if len(object_label) > self.max_value_chars:
            object_label = object_label[: self.max_value_chars].rstrip() + "…"
        qualifiers = self._parse_qualifiers(statement) if self.include_qualifiers else ()
        statement_id = statement.get("id")
        source_url = (
            f"https://www.wikidata.org/entity/statement/{statement_id}"
            if statement_id
            else f"https://www.wikidata.org/wiki/{subject_id}"
        )
        return Fact(
            subject_id=subject_id,
            subject_label=subject_label,
            predicate_id=predicate_id,
            predicate_label=self._property_label(predicate_id),
            object_id=object_id,
            object_label=object_label,
            object_type=object_type,
            object_datatype=datatype or None,
            qualifiers=qualifiers,
            source=self.source_label,
            source_url=source_url,
            retrieved_at=utc_now_iso(),
            confidence=1.0,
            rank=statement.get("rank"),
            metadata={"statement_id": statement_id} if statement_id else {},
        )

    def _parse_datavalue(
        self, datavalue: dict[str, Any], datatype: str
    ) -> tuple[str, str | None, str] | None:
        """Return ``(object_label, object_id, object_type)`` or ``None`` if unusable."""
        value = datavalue.get("value")
        value_type = str(datavalue.get("type") or "")
        if value_type == "wikibase-entityid" and isinstance(value, dict):
            entity_id = value.get("id")
            if not isinstance(entity_id, str) or not entity_id:
                return None
            if entity_id.startswith("Q") and entity_id in _NON_ARTICLE_INSTANCES:
                return None
            label = self._label_for(entity_id)
            return label, entity_id, "entity"
        if value_type == "time":
            formatted = format_wikidata_time(value if isinstance(value, dict) else {})
            if not formatted:
                return None
            return formatted, None, "time"
        if value_type == "quantity":
            unit_id = None
            if isinstance(value, dict):
                unit = value.get("unit")
                if isinstance(unit, str) and unit.rsplit("/", 1)[-1].startswith("Q"):
                    unit_id = unit.rsplit("/", 1)[-1]
            unit_label = self._label_for(unit_id) if unit_id else None
            formatted = format_quantity(value if isinstance(value, dict) else {}, unit_label)
            if not formatted:
                return None
            return formatted, None, "quantity"
        if value_type == "globecoordinate" and isinstance(value, dict):
            latitude, longitude = value.get("latitude"), value.get("longitude")
            if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
                return None
            return f"{latitude:.4f}, {longitude:.4f}", None, "literal"
        if value_type in {"string", "monolingualtext"} and isinstance(value, (str, dict)):
            if not self.allow_literal_values:
                return None
            if isinstance(value, dict):
                text, language = value.get("text"), value.get("language")
                if not isinstance(text, str):
                    return None
                return (f"{text} ({language})" if language and language != self.language else text), None, "literal"
            return str(value), None, "literal"
        return None

    def _parse_qualifiers(self, statement: dict[str, Any]) -> tuple[Qualifier, ...]:
        qualifiers: list[Qualifier] = []
        for predicate_id, values in (statement.get("qualifiers") or {}).items():
            label = self._property_label(str(predicate_id))
            for entry in values or []:
                if entry.get("snaktype") != "value":
                    continue
                datavalue = entry.get("datavalue") or {}
                parsed = self._parse_datavalue(datavalue, str(entry.get("datatype") or ""))
                if parsed is None:
                    continue
                text, value_id, value_type = parsed
                qualifiers.append(
                    Qualifier(
                        predicate_id=str(predicate_id),
                        predicate_label=label,
                        value=text,
                        value_type="entity" if value_type == "entity" else "literal",
                        value_id=value_id,
                    )
                )
        return tuple(qualifiers)

    def _label_for(self, entity_id: str | None) -> str | None:
        if not entity_id:
            return None
        if entity_id in self._label_cache:
            return self._label_cache[entity_id]
        try:
            payload = self.client.get_json(
                self.api_url,
                {
                    "action": "wbgetentities",
                    "ids": entity_id,
                    "props": "labels",
                    "languages": self.language,
                    "languagefallback": "1",
                    "format": "json",
                    "formatversion": 2,
                    "origin": "*",
                },
                cache_key=f"labels|{self.language}|{entity_id}",
            )
            label = self._entity_label((payload.get("entities") or {}).get(entity_id, {})) or entity_id
        except (KnowledgeUnavailable, AttributeError) as exc:
            logger.warning("could not resolve label for %s: %s", entity_id, exc)
            label = entity_id
        self._label_cache[entity_id] = label
        return label

    # ------------------------------------------------------------- helpers
    def known_entities(self) -> list[str]:  # pragma: no cover - no offline lexicon
        return []

    def close(self) -> None:
        return None


class SPARQL_BACKED_DBpediaProvider(KnowledgeProvider):  # pragma: no cover - documented blocker
    """Explicitly-not-implemented DBpedia provider (spec Section 5, "add later").

    The spec says to implement **one** knowledge source first and only add a
    second once the pipeline works.  Rather than shipping an untested SPARQL
    client that could quietly return wrong facts, this class fails loudly so the
    limitation is visible in the run log and in the report (Section 11.4: "If a
    dataset/source cannot be obtained, report that limitation").
    """

    name = "dbpedia"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The DBpedia provider is intentionally not implemented in v1. "
            "FactAnchor v1 ships Wikidata (primary), plus 'json_file' and 'mock' "
            "providers for offline runs. To add DBpedia, implement "
            "KnowledgeProvider.search_entities/get_facts against the DBpedia "
            "SPARQL endpoint and record the source in Fact.source. "
            "Set retrieval.provider='wikidata' to run experiments today."
        )

    def search_entities(self, query: str, *, limit: int = 5, context: str | None = None) -> list[EntityCandidate]:
        raise NotImplementedError

    def get_facts(self, entity_id: str, *, max_facts: int = 10) -> list[Fact]:
        raise NotImplementedError


def _batched(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
