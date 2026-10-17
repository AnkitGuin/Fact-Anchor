"""Named-entity extraction (spec 7.1) and mention-level helpers.

Three interchangeable backends implement :class:`EntityExtractor`:

``rule``
    Dependency-free, deterministic, capitalisation + heuristics based.  This is
    the default so the whole pipeline runs offline and in CI.
``simple``
    Lexicon based: matches a supplied list of known surface forms (used with the
    offline JSON knowledge provider and in tests).
``spacy``
    Optional spaCy NER (``pip install -e .[ner]`` + ``python -m spacy download``).
    It is *not* installed by default and raises a readable error if missing.

Design notes
------------
* Extraction never assumes every capitalised token is a real-world entity: the
  rule backend requires either a multi-token span, a mid-sentence position, or a
  title/pattern cue, and it filters a stoplist of sentence-initial function words.
* Overlapping mentions are resolved by keeping the longest span (a "Marie Curie"
  span wins over "Marie"), then ties are broken by confidence and position.
* Empty input returns ``[]`` -- never an exception (spec 7.1 and unit tests).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from .config import Config, NerConfig
from .schemas import EntityMention
from .utils import get_logger

__all__ = [
    "EntityExtractor",
    "RuleBasedEntityExtractor",
    "LexiconEntityExtractor",
    "SpacyEntityExtractor",
    "build_extractor",
    "extract_entities",
    "deduplicate_mentions",
    "MENTION_TOKEN_RE",
]

logger = get_logger("entities")

# Tokens that can appear inside a name: letters, apostrophes, hyphens, periods
# (initials), and digits (e.g. "Apollo 11", "Louis XIV").
MENTION_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’\-\.]*|\d+(?:\.\d+)?")

# Sentence-initial / general function words that are capitalised but are not
# entities on their own.  Kept deliberately small and documented.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the this that these those it its he she they we you i his her their our your
    in on at by for with about from into over under during after before between of to
    and or but nor so yet if then than as because while when where which who whom whose
    what how why is are was were be been being do does did can could may might must shall
    should will would have has had there here also however therefore thus hence moreover
    furthermore meanwhile although though since until unless within without across among
    story stories biography bio answer question tell explain describe give name named
    please thanks thank hi hello yes no not true false both each every all any some
    many much more most few less least other another such same new old first second third
    according based regarding despite besides instead rather quite very just only even
    """.split()
)

# Cues that make a single capitalised token much more likely to be an entity.
_TITLE_CUES = frozenset(
    """
    mr mrs ms miss dr professor prof sir dame lord lady saint st king queen prince princess
    president emperor empress pope captain general colonel major admiral senator governor
    """.split()
)

_QUESTION_CUES = re.compile(
    r"\b(who|whom|whose|which|where|what)\s+(?:is|was|are|were|did|does|do)\b",
    re.IGNORECASE,
)


@runtime_checkable
class EntityExtractor(Protocol):
    """Interface every NER backend must satisfy (spec 7.1)."""

    name: str

    def extract(self, text: str) -> list[EntityMention]:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Rule-based backend
# ---------------------------------------------------------------------------


@dataclass
class RuleBasedEntityExtractor:
    """Deterministic capitalisation/pattern based extractor.

    Scores each candidate span with transparent, additive evidence so the
    behaviour of the extractor can be explained in a report:

    ========================================  ======
    evidence                                  score
    ========================================  ======
    base (capitalised candidate)              0.30
    multi-token span (+0.25)                  0.55
    span length >= 3 tokens (+0.10)           0.65
    special particle ("van", "de", "al")      +0.05
    title cue ("Dr", "President")             +0.15
    question cue before the span              +0.10
    quoted mention                            +0.20
    ========================================  ======
    """

    config: NerConfig = field(default_factory=NerConfig)
    known_entities: Sequence[str] = ()
    name: str = "rule"

    _known_lookup: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._known_lookup = {e.strip().lower(): e.strip() for e in self.known_entities if e.strip()}
        self._stopwords = _STOPWORDS | {t.lower() for t in self.config.stopword_tokens}

    # -- public API ---------------------------------------------------------
    def extract(self, text: str) -> list[EntityMention]:
        if not text or not text.strip():
            return []
        mentions: list[EntityMention] = []
        mentions.extend(self._known_mentions(text))
        mentions.extend(self._capitalised_mentions(text))
        mentions.extend(self._quoted_mentions(text))
        mentions = deduplicate_mentions(mentions, text=text)
        mentions.sort(key=lambda m: (-m.confidence, m.start))
        return mentions[: self.config.max_entities]

    # -- individual strategies ---------------------------------------------
    def _known_mentions(self, text: str) -> list[EntityMention]:
        """Longest-match lexicon hits (highest confidence: an exact known name)."""
        found: list[EntityMention] = []
        if self._known_lookup:
            pattern = "|".join(sorted((re.escape(k) for k in self._known_lookup), key=len, reverse=True))
            for match in re.finditer(rf"(?<!\w)(?:{pattern})(?!\w)", text, flags=re.IGNORECASE):
                surface = match.group(0)
                canonical = self._known_lookup.get(surface.lower(), surface)
                found.append(
                    EntityMention(
                        mention=canonical,
                        start=match.start(),
                        end=match.end(),
                        label=None,
                        confidence=0.95,
                        source="lexicon",
                        context=None,
                    )
                )
        return found

    def _capitalised_mentions(self, text: str) -> list[EntityMention]:
        tokens = [(m.group(0), m.start(), m.end()) for m in MENTION_TOKEN_RE.finditer(text)]
        found: list[EntityMention] = []
        index = 0
        question_cue_end = self._question_cue_end(text)
        while index < len(tokens):
            token, start, _ = tokens[index]
            if not self._is_capitalised_candidate(token):
                index += 1
                continue
            # grow the span while subsequent tokens continue the name
            end_index = index
            while end_index + 1 < len(tokens):
                nxt, nxt_start, nxt_end = tokens[end_index + 1]
                if not self._continues_span(text, nxt, nxt_start):
                    break
                end_index += 1
            surface_tokens = [t[0] for t in tokens[index : end_index + 1]]
            mention_text = text[tokens[index][1] : tokens[end_index][2]]
            if not any(character.isalpha() for character in mention_text):
                # pure numbers / dates are not entities (they are handled as claim values)
                index = end_index + 1
                continue
            confidence, reason = self._score_span(
                surface_tokens=surface_tokens,
                span_start=tokens[index][1],
                text=text,
                question_cue_end=question_cue_end,
            )
            if confidence > 0.0:
                found.append(
                    EntityMention(
                        mention=mention_text,
                        start=tokens[index][1],
                        end=tokens[end_index][2],
                        label=None,
                        confidence=round(confidence, 3),
                        source=f"rule:{reason}",
                        context=self._context_window(text, tokens[index][1], tokens[end_index][2]),
                    )
                )
            index = end_index + 1
        return found

    def _quoted_mentions(self, text: str) -> list[EntityMention]:
        found: list[EntityMention] = []
        for match in re.finditer(r"[\"“'‘]([A-Z][^\"”'’]{1,60})[\"”'’]", text):
            inner = match.group(1).strip()
            if len(inner) < self.config.min_mention_chars or inner.lower() in self._stopwords:
                continue
            if len(inner.split()) > 6:
                continue
            found.append(
                EntityMention(
                    mention=inner,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.6,
                    source="rule:quoted",
                    context=self._context_window(text, match.start(1), match.end(1)),
                )
            )
        return found

    # -- scoring helpers ----------------------------------------------------
    def _is_capitalised_candidate(self, token: str) -> bool:
        core = token.strip(".")
        if not core:
            return False
        if len(core) < self.config.min_mention_chars:
            return False
        if not core[0].isupper() and not core[0].isdigit():
            # allow particles that are commonly part of names (van, de, von...)
            return core.lower() in {"van", "von", "de", "del", "della", "di", "da", "al", "bin", "ibn", "mac", "mc"}
        return True

    def _continues_span(self, text: str, next_token: str, next_start: int) -> bool:
        """Is ``next_token`` likely part of the same name?"""
        gap = text[max(0, next_start - 6) : next_start]
        # sentence boundary breaks a span (unless the previous char is an initial)
        if re.search(r"[.!?]\s+$", gap) and not re.search(r"\b[A-Z]\.\s+$", gap):
            return False
        if next_token[0].isupper() or next_token[0].isdigit():
            return True
        return next_token.lower() in {"van", "von", "de", "del", "della", "di", "da", "al", "bin", "ibn", "and"}

    def _score_span(
        self,
        *,
        surface_tokens: list[str],
        span_start: int,
        text: str,
        question_cue_end: int | None,
    ) -> tuple[float, str]:
        score = 0.30
        reasons = ["capitalised"]
        if len(surface_tokens) >= 2:
            score += 0.25
            reasons.append("multi_token")
        if len(surface_tokens) >= 3:
            score += 0.10
            reasons.append("long_span")
        lowered = [t.lower().strip(".") for t in surface_tokens]
        if any(t in _TITLE_CUES for t in lowered[:-1]) or any(t in {"van", "von", "de", "al"} for t in lowered[1:]):
            score += 0.15
            reasons.append("title_or_particle")
        if question_cue_end is not None and span_start >= question_cue_end:
            score += 0.10
            reasons.append("question_cue")
        # A single, sentence-initial, ordinary word is almost never an entity.
        if len(surface_tokens) == 1:
            word = lowered[0]
            if word in self._stopwords:
                return 0.0, "stopword"
            preceding = text[max(0, span_start - 2) : span_start]
            is_sentence_start = span_start == 0 or bool(re.search(r"[.!?\n]\s*$", preceding))
            if is_sentence_start and not self.config.allow_single_token:
                return 0.0, "single_token_disallowed"
            if is_sentence_start and word not in _TITLE_CUES and len(word) <= 4:
                # e.g. "Rome was..." -> keep; "In 1903..." -> drop (already a stopword)
                return 0.0, "sentence_initial_short"
        return min(score, 0.99), "+".join(reasons)

    @staticmethod
    def _question_cue_end(text: str) -> int | None:
        match = _QUESTION_CUES.search(text)
        return match.end() if match else None

    @staticmethod
    def _context_window(text: str, start: int, end: int, window: int = 80) -> str:
        left = max(0, start - window)
        right = min(len(text), end + window)
        return text[left:right].strip()


# ---------------------------------------------------------------------------
# Lexicon backend
# ---------------------------------------------------------------------------


@dataclass
class LexiconEntityExtractor:
    """Exact/longest-match extraction from a supplied lexicon (offline workhorse)."""

    known_entities: Sequence[str] = ()
    config: NerConfig = field(default_factory=NerConfig)
    name: str = "simple"

    def extract(self, text: str) -> list[EntityMention]:
        if not text or not text.strip():
            return []
        lookup = {e.strip().lower(): e.strip() for e in self.known_entities if e.strip()}
        if not lookup:
            logger.debug("lexicon extractor has an empty lexicon; returning no mentions")
            return []
        pattern = "|".join(sorted((re.escape(k) for k in lookup), key=len, reverse=True))
        mentions: list[EntityMention] = []
        for match in re.finditer(rf"(?<!\w)(?:{pattern})(?!\w)", text, flags=re.IGNORECASE):
            surface = match.group(0)
            mentions.append(
                EntityMention(
                    mention=lookup[surface.lower()],
                    start=match.start(),
                    end=match.end(),
                    label=None,
                    confidence=0.9,
                    source="lexicon",
                    context=text[max(0, match.start() - 40) : match.end() + 40].strip(),
                )
            )
        mentions = deduplicate_mentions(mentions, text=text)
        return mentions[: self.config.max_entities]


# ---------------------------------------------------------------------------
# spaCy backend (optional)
# ---------------------------------------------------------------------------


@dataclass
class SpacyEntityExtractor:
    """spaCy NER backend; requires ``pip install -e .[ner]`` and a downloaded model."""

    spacy_model: str = "en_core_web_sm"
    config: NerConfig = field(default_factory=NerConfig)
    name: str = "spacy"

    _nlp: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import spacy  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "spaCy backend requested but spaCy is not installed. "
                "Install with `pip install -e .[ner]` and "
                f"`python -m spacy download {self.spacy_model}`, or use ner.backend='rule'."
            ) from exc
        try:
            self._nlp = spacy.load(self.spacy_model)
        except OSError as exc:  # pragma: no cover - model not downloaded
            raise RuntimeError(
                f"spaCy model {self.spacy_model!r} is not available. "
                f"Run `python -m spacy download {self.spacy_model}` or set ner.backend='rule'."
            ) from exc

    def extract(self, text: str) -> list[EntityMention]:
        if not text or not text.strip():
            return []
        doc = self._nlp(text)
        mentions = [
            EntityMention(
                mention=ent.text,
                start=ent.start_char,
                end=ent.end_char,
                label=ent.label_,
                confidence=0.8,
                source=f"spacy:{ent.label_}",
                context=text[max(0, ent.start_char - 60) : ent.end_char + 60].strip(),
            )
            for ent in doc.ents
            if len(ent.text.strip()) >= self.config.min_mention_chars
        ]
        mentions = deduplicate_mentions(mentions, text=text)
        mentions.sort(key=lambda m: (-m.confidence, m.start))
        return mentions[: self.config.max_entities]


# ---------------------------------------------------------------------------
# Factory / functional API
# ---------------------------------------------------------------------------


def build_extractor(
    config: NerConfig | Config,
    *,
    known_entities: Sequence[str] | None = None,
) -> EntityExtractor:
    """Instantiate the configured NER backend.

    ``known_entities`` (e.g. labels from the offline JSON knowledge base) is
    passed to backends that can use it.
    """
    ner_config = config.ner if isinstance(config, Config) else config
    backend = ner_config.backend
    if backend == "spacy":
        return SpacyEntityExtractor(spacy_model=ner_config.spacy_model, config=ner_config)
    if backend == "simple":
        return LexiconEntityExtractor(known_entities=known_entities or (), config=ner_config)
    return RuleBasedEntityExtractor(known_entities=known_entities or (), config=ner_config)


def extract_entities(
    text: str,
    *,
    backend: EntityExtractor | None = None,
    config: Config | NerConfig | None = None,
    known_entities: Sequence[str] | None = None,
) -> list[EntityMention]:
    """Convenience wrapper: ``extract_entities(text) -> list[EntityMention]``."""
    if backend is None:
        backend = build_extractor(config or NerConfig(), known_entities=known_entities)
    return backend.extract(text)


def deduplicate_mentions(mentions: Sequence[EntityMention], *, text: str | None = None) -> list[EntityMention]:
    """Remove duplicates and overlapping spans, keeping the strongest mention.

    Ranking: longer span first, then higher confidence, then longer surface text,
    then earliest position.  A weaker mention strictly inside a stronger one is
    dropped (the classic "Marie" vs "Marie Curie" case).
    """
    if not mentions:
        return []
    ranked = sorted(
        mentions,
        key=lambda m: (-(m.end - m.start), -m.confidence, -len(m.mention), m.start),
    )
    kept: list[EntityMention] = []
    for candidate in ranked:
        if any(_overlaps(candidate, existing) for existing in kept):
            continue
        kept.append(candidate)
    # exact duplicates by normalised surface form: keep the first (highest ranked)
    seen: set[str] = set()
    unique: list[EntityMention] = []
    for mention in sorted(kept, key=lambda m: m.start):
        key = mention.mention.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(mention)
    if text is not None:
        for mention in unique:
            if mention.end > len(text):
                raise ValueError(
                    f"mention {mention.mention!r} has offsets ({mention.start}, {mention.end}) "
                    f"outside text of length {len(text)}"
                )
    return unique


def _overlaps(a: EntityMention, b: EntityMention) -> bool:
    return a.start < b.end and b.start < a.end


def mentions_to_strings(mentions: Sequence[EntityMention]) -> list[str]:
    """Helper used by claim extraction (kept here to avoid a circular import)."""
    return [m.mention for m in mentions]
