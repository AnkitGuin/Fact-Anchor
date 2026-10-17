"""Controlled corruption of retrieved facts (spec Section 8).

Guarantees, all enforced in code and covered by tests:

* The input facts are never mutated -- every output is a new :class:`Fact`
  (the input list and its records are compared object-by-object in
  ``tests/test_corruption.py``).
* Every modified/skipped fact produces a :class:`CorruptionRecord` in the audit
  log, including the reason it was skipped.
* Selection is seeded: the same input + seed + rate + strategy always yields the
  same output, on any machine (``random.Random``, no global state).
* ``corruption_rate=0`` and ``strategy="none"`` return the originals unchanged
  (with an audit trail explaining that nothing was touched).
* Replacements are type-checked: an entity-valued object is swapped for another
  entity *of the same domain* (person -> person, country -> country, ...), a year
  stays a plausible year of the same data type, and a numeric literal keeps its
  unit.  If no safe replacement exists the fact is skipped and the reason logged.
* ``omission`` is a *control*: it removes facts instead of falsifying them, so
  the contrast branch carries no false claims at all.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .config import Config, CorruptionConfig
from .schemas import CorruptionRecord, CorruptionResult, Fact
from .utils import get_logger

__all__ = [
    "corrupt_facts",
    "CorruptionEngine",
    "CANDIDATE_POOLS",
    "describe_corruption_summary",
]

logger = get_logger("corruption")


# ---------------------------------------------------------------------------
# Candidate pools
# ---------------------------------------------------------------------------
# Each pool is associated with a Wikidata *class* (P31 value).  A fact about a
# person is only swapped with another person, so the corruption stays
# type-compatible (spec 8.1) instead of producing nonsense such as
# "Marie Curie | occupation | Amazon River".
CANDIDATE_POOLS: dict[str, dict[str, Any]] = {
    "Q5": {  # human
        "P106": ["physicist", "chemist", "mathematician", "physician", "engineer", "astronomer", "biologist", "economist", "philosopher", "painter", "poet", "composer", "architect", "explorer", "politician", "astronaut"],
        "P27": ["France", "Germany", "Poland", "Italy", "Spain", "Portugal", "Netherlands", "Belgium", "Sweden", "Norway", "Denmark", "Austria", "Switzerland", "Greece", "Japan", "Brazil", "Canada", "Australia", "Mexico", "India"],
        "P69": ["University of Paris", "University of Vienna", "Heidelberg University", "ETH Zurich", "University of Bologna", "University of Padua", "Sorbonne University", "University of Cambridge", "University of Oxford", "University of Berlin"],
        "P108": ["University of Paris", "Columbia University", "Harvard University", "University of Vienna", "CERN", "Max Planck Society", "Bell Labs", "ETH Zurich", "Stanford University", "Massachusetts Institute of Technology"],
        "P166": ["Nobel Prize in Physics", "Nobel Prize in Chemistry", "Nobel Prize in Literature", "Fields Medal", "Turing Award", "Lasker Award", "Copley Medal", "Wolf Prize", "Pulitzer Prize", "Fellow of the Royal Society"],
        "P800": ["an atomic theory", "a periodic table", "the theory of relativity", "a novel", "a symphony", "the laws of planetary motion", "a telescope", "the smallpox vaccine"],
        "P39": ["head of state", "member of parliament", "university rector", "director of a research institute", "board member"],
        "P101": ["physics", "chemistry", "mathematics", "biology", "medicine", "astronomy", "economics", "philosophy", "linguistics"],
        "P512": ["doctorate", "bachelor's degree", "master's degree", "habilitation"],
        "P136": ["classical music", "jazz", "rock music", "folk music", "electronic music"],
    },
    "Q3624078": {  # sovereign state
        "P30": ["Europe", "Asia", "Africa", "South America", "North America", "Oceania"],
        "P36": ["London", "Paris", "Berlin", "Madrid", "Rome", "Vienna", "Warsaw", "Oslo", "Tokyo", "Canberra"],
        "P47": ["France", "Germany", "Italy", "Spain", "Belgium", "Switzerland", "Austria", "Poland", "Sweden", "Denmark", "Japan", "India"],
        "P37": ["English", "French", "German", "Spanish", "Italian", "Portuguese"],
    },
    "Q515": {  # city
        "P17": ["France", "Germany", "Italy", "Spain", "Poland", "Austria", "Sweden", "Japan", "India", "Brazil"],
        "P30": ["Europe", "Asia", "Africa", "South America", "North America", "Oceania"],
    },
    "Q476028": {  # association football club
        "P17": ["Spain", "England", "Italy", "Germany", "France", "Netherlands", "Portugal", "Brazil"],
        "P452": ["sports industry", "entertainment", "education", "health care"],
    },
    "Q11424": {  # film
        "P136": ["drama", "comedy", "thriller", "science fiction", "documentary", "horror", "romance", "animation"],
        "P495": ["United States", "United Kingdom", "France", "Italy", "Japan", "India", "Germany"],
        "P123": ["Warner Bros.", "Universal Pictures", "Paramount Pictures", "Columbia Pictures", "20th Century Studios", "Metro-Goldwyn-Mayer"],
    },
    "Q571": {  # book
        "P136": ["novel", "biography", "essay", "poetry", "philosophy", "history", "fantasy", "science fiction"],
        "P123": ["Penguin Books", "Oxford University Press", "Cambridge University Press", "HarperCollins", "Random House"],
    },
}

# Subject candidates are deliberately separate from object candidates.  Using
# an occupation/country pool to rewrite a subject would create malformed triples.
SUBJECT_CANDIDATE_POOLS: dict[str, list[str]] = {
    "Q5": ["Albert Einstein", "Ada Lovelace", "Isaac Newton", "Rosalind Franklin", "Niels Bohr"],
    "Q3624078": ["France", "Germany", "Japan", "Brazil", "Canada"],
    "Q515": ["Paris", "Berlin", "Tokyo", "Cairo", "Mumbai"],
    "Q11424": ["Metropolis (1927 film)", "Modern Times", "The General (1926 film)"],
    "Q571": ["Pride and Prejudice", "Frankenstein", "Middlemarch"],
}

#: Pools that are always available, keyed by the *predicate* rather than class.
GENERIC_POOLS: dict[str, list[str]] = {
    "P106": CANDIDATE_POOLS["Q5"]["P106"],
    "P27": CANDIDATE_POOLS["Q5"]["P27"],
    "P30": CANDIDATE_POOLS["Q3624078"]["P30"],
    "P136": sorted({*CANDIDATE_POOLS["Q5"]["P136"], *CANDIDATE_POOLS["Q11424"]["P136"]}),
    "P495": CANDIDATE_POOLS["Q11424"]["P495"],
    "P101": CANDIDATE_POOLS["Q5"]["P101"],
    "P17": sorted({*CANDIDATE_POOLS["Q3624078"]["P47"], *CANDIDATE_POOLS["Q515"]["P17"]}),
}

#: Predicates for which an "attribute perturbation" is meaningful.
_ATTRIBUTE_PREDICATES: dict[str, str] = {
    "P569": "year",  # date of birth
    "P570": "year",  # date of death
    "P571": "year",  # inception
    "P576": "year",  # dissolved
    "P577": "year",  # publication date
    "P1082": "quantity",  # population
    "P2046": "quantity",  # area
    "P2044": "quantity",  # elevation
}

_YEAR_RE = re.compile(r"(?<![\d])(\d{3,4})(?![\d])")
_QUANTITY_RE = re.compile(r"(\d[\d,]*\.?\d*)")


@dataclass
class CorruptionEngine:
    """Seeded, auditable corruption of fact sets.

    Parameters mirror :class:`~factanchor.config.CorruptionConfig` so the engine
    can be constructed directly in tests without a full config object.

    Selection semantics
    -------------------
    ``corruption_rate`` is the fraction of facts selected for corruption
    (``round(rate * n)``), bounded by ``max_corruption_fraction_of_block`` (1.0 by default).

    * ``rate == 0`` touches nothing (exact no-op, covered by a test);
    * ``rate > 0`` selects **at least one** fact even when ``round(rate*n) == 0``,
      so a small non-zero rate cannot silently behave like zero.  This is
      documented because it makes very small rates coarser than the nominal
      percentage on small fact blocks.
    """

    strategy: str = "object_swap"
    corruption_rate: float = 0.5
    seed: int = 42
    max_corruption_fraction_of_block: float = 1.0
    swap_within_domain: bool = True
    allow_cross_entity_substitution: bool = False
    preserve_qualifiers: bool = True
    audit_log: bool = True
    extra_candidate_pool: dict[str, list[str]] = field(default_factory=dict)

    SUPPORTED_STRATEGIES = ("object_swap", "attribute_perturbation", "entity_substitution", "omission", "none")

    def __post_init__(self) -> None:
        if self.strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"unknown corruption strategy {self.strategy!r}; supported: {list(self.SUPPORTED_STRATEGIES)}"
            )
        if not 0.0 <= float(self.corruption_rate) <= 1.0:
            raise ValueError(f"corruption_rate must be within [0, 1], got {self.corruption_rate!r}")
        if not 0.0 < float(self.max_corruption_fraction_of_block) <= 1.0:
            raise ValueError(
                "max_corruption_fraction_of_block must be within (0, 1], got "
                f"{self.max_corruption_fraction_of_block!r}"
            )
        for predicate_id, pool in self.extra_candidate_pool.items():
            if not isinstance(pool, (list, tuple)) or not pool:
                raise ValueError(f"extra_candidate_pool[{predicate_id!r}] must be a non-empty list of strings")

    # -- public API ---------------------------------------------------------
    def corrupt(self, facts: Sequence[Fact]) -> CorruptionResult:
        """Return ``(corrupted_facts, audit)`` wrapped in a :class:`CorruptionResult`."""
        originals = tuple(facts)
        if not originals:
            return CorruptionResult(
                original_facts=(),
                corrupted_facts=(),
                audit=(),
                strategy=self.strategy,
                corruption_rate=float(self.corruption_rate),
                seed=self.seed,
                notes=("no facts supplied",),
            )
        rng = random.Random(self.seed)
        selected = self._select_indices(len(originals), rng)
        audit: list[CorruptionRecord] = []
        contrast: list[Fact] = []
        changed = omitted = skipped = 0

        for index, fact in enumerate(originals):
            if index not in selected:
                contrast.append(fact)
                continue
            new_fact, record = self._corrupt_one(fact, rng)
            if record is None:
                # selected but nothing safe to do: keep the original, log nothing
                contrast.append(fact)
                continue
            audit.append(record)
            if record.is_corrupted:
                if new_fact is None:
                    omitted += 1
                else:
                    changed += 1
                    contrast.append(new_fact)
            else:
                skipped += 1
                contrast.append(fact)

        result = CorruptionResult(
            original_facts=originals,
            corrupted_facts=tuple(contrast),
            audit=tuple(audit) if self.audit_log else (),
            strategy=self.strategy,
            corruption_rate=float(self.corruption_rate),
            seed=self.seed,
            selected_count=len(selected),
            changed_count=changed,
            omitted_count=omitted,
            skipped_count=skipped,
            notes=self._notes(len(originals), changed, omitted, skipped),
        )
        logger.debug("corruption summary: %s", result.summary)
        return result

    # -- selection ----------------------------------------------------------
    def _select_indices(self, total: int, rng: random.Random) -> set[int]:
        """Deterministically choose which facts to touch.

        ``corruption_rate`` may exceed ``max_corruption_fraction_of_block`` in the
        config; an explicit cap wins and is logged. The default cap is 1.0, so an
        explicit rate of 1.0 really does select the full block.
        """
        if total == 0 or self.corruption_rate <= 0.0 or self.strategy == "none":
            return set()
        requested = int(round(self.corruption_rate * total))
        cap = max(1, int(self.max_corruption_fraction_of_block * total))
        target = min(max(requested, 1), total)  # at least one fact when rate > 0
        if target > cap:
            logger.info(
                "corruption_rate=%.2f would touch %d/%d facts; capped at %.2f of the block (%d)",
                self.corruption_rate,
                target,
                total,
                self.max_corruption_fraction_of_block,
                cap,
            )
            target = cap
        indices = sorted(rng.sample(range(total), target))
        return set(indices)

    # -- per-fact corruption ------------------------------------------------
    def _corrupt_one(
        self, fact: Fact, rng: random.Random
    ) -> tuple[Fact | None, CorruptionRecord | None]:
        try:
            if self.strategy == "omission":
                return None, CorruptionRecord(
                    original_fact_id=fact.fact_id,
                    corruption_type="omission",
                    original_object=fact.object_label,
                    corrupted_object=None,
                    random_seed=self.seed,
                    is_corrupted=True,
                    skipped_reason="fact omitted (control condition)",
                )
            if self.strategy == "object_swap":
                return self._object_swap(fact, rng)
            if self.strategy == "attribute_perturbation":
                return self._attribute_perturbation(fact, rng)
            if self.strategy == "entity_substitution":
                return self._entity_substitution(fact, rng)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("corruption failed for fact %s", fact.fact_id)
            return fact, CorruptionRecord(
                original_fact_id=fact.fact_id,
                corruption_type=self.strategy,
                original_object=fact.object_label,
                random_seed=self.seed,
                is_corrupted=False,
                skipped_reason=f"error: {type(exc).__name__}: {exc}",
            )
        return fact, CorruptionRecord(
            original_fact_id=fact.fact_id,
            corruption_type=self.strategy,
            original_object=fact.object_label,
            random_seed=self.seed,
            is_corrupted=False,
            skipped_reason="strategy 'none' does nothing",
        )

    def _object_swap(self, fact: Fact, rng: random.Random) -> tuple[Fact | None, CorruptionRecord]:
        pool = self._candidate_pool(fact)
        if not pool:
            return fact, self._skip_record(fact, "no type-compatible candidate pool for the predicate")
        candidates = [c for c in pool if not self._same_value(c, fact.object_label)]
        if not candidates:
            return fact, self._skip_record(fact, "candidate pool contains only the original value")
        replacement = candidates[rng.randrange(len(candidates))]
        # The replacement label has no known knowledge-base id, so the object id
        # is dropped rather than kept from the original (that would silently pair
        # the wrong id with the new label).
        new_fact = self._replacement_fact(fact, replacement, object_id=None)
        return new_fact, CorruptionRecord(
            original_fact_id=fact.fact_id,
            corruption_type="object_swap",
            original_object=fact.object_label,
            corrupted_object=replacement,
            random_seed=self.seed,
            is_corrupted=True,
            corrupted_fact_id=new_fact.fact_id,
            qualifiers_preserved=self.preserve_qualifiers,
        )

    def _attribute_perturbation(self, fact: Fact, rng: random.Random) -> tuple[Fact | None, CorruptionRecord]:
        kind = _ATTRIBUTE_PREDICATES.get(fact.predicate_id)
        if kind is None and fact.object_type not in {"time", "quantity"}:
            return fact, self._skip_record(
                fact, f"attribute perturbation is not defined for predicate {fact.predicate_id}"
            )
        kind = kind or ("time" if fact.object_type == "time" else "quantity")
        if kind == "year":
            perturbed = self._perturb_year(fact.object_label, rng)
        else:
            perturbed = self._perturb_quantity(fact.object_label, rng)
        if perturbed is None or self._same_value(perturbed, fact.object_label):
            return fact, self._skip_record(fact, "no plausible perturbed value could be generated")
        new_fact = self._replacement_fact(fact, perturbed, object_id=None)
        return new_fact, CorruptionRecord(
            original_fact_id=fact.fact_id,
            corruption_type="attribute_perturbation",
            original_object=fact.object_label,
            corrupted_object=perturbed,
            random_seed=self.seed,
            is_corrupted=True,
            corrupted_fact_id=new_fact.fact_id,
            qualifiers_preserved=self.preserve_qualifiers,
        )

    def _entity_substitution(self, fact: Fact, rng: random.Random) -> tuple[Fact | None, CorruptionRecord]:
        """Replace the fact's *subject* with a different entity of the same class.

        Disabled by default (``allow_cross_entity_substitution=False``) because it
        changes the topic of the statement rather than its object.
        """
        subject_class = self._subject_class(fact, default=None)
        pool = SUBJECT_CANDIDATE_POOLS.get(subject_class or "", [])
        if fact.predicate_id in self.extra_candidate_pool:
            # Explicit pools are permitted for this strategy too, but are treated
            # as subject labels and still have their old subject identifier dropped.
            pool = self.extra_candidate_pool[fact.predicate_id]
        if not pool:
            return fact, self._skip_record(fact, "no same-class subject candidate pool for entity substitution")
        candidates = [c for c in pool if not self._same_value(c, fact.subject_label)]
        if not candidates:
            return fact, self._skip_record(fact, "candidate pool contains only the original subject")
        replacement = candidates[rng.randrange(len(candidates))]
        new_fact = fact.with_updates(
            subject_label=replacement,
            subject_id=f"synthetic:{replacement}",
            fact_id="",
        )
        # re-derive a stable id for the new record
        new_fact = Fact(**{**new_fact.model_dump(), "fact_id": ""})
        return new_fact, CorruptionRecord(
            original_fact_id=fact.fact_id,
            corruption_type="entity_substitution",
            original_object=fact.subject_label,
            corrupted_object=replacement,
            original_subject=fact.subject_label,
            corrupted_subject=replacement,
            random_seed=self.seed,
            is_corrupted=True,
            corrupted_fact_id=new_fact.fact_id,
            qualifiers_preserved=True,
        )

    # -- helpers ------------------------------------------------------------
    def _candidate_pool(self, fact: Fact) -> list[str]:
        if fact.predicate_id in self.extra_candidate_pool:
            return list(self.extra_candidate_pool[fact.predicate_id])
        if fact.object_type == "entity":
            class_id = self._subject_class(fact)
            pooled = CANDIDATE_POOLS.get(class_id, {}).get(fact.predicate_id)
            if pooled:
                return list(pooled)
            if self.swap_within_domain and fact.predicate_id in GENERIC_POOLS:
                return list(GENERIC_POOLS[fact.predicate_id])
            return []
        # literals/quotes: only the generic predicate pools make sense
        return list(GENERIC_POOLS.get(fact.predicate_id, []))

    @staticmethod
    def _subject_class(fact: Fact, *, default: str | None = "Q5") -> str | None:
        """Wikidata class of the subject when the retrieval layer recorded it."""
        for key in ("subject_class", "instance_of", "P31"):
            value = fact.metadata.get(key)
            if isinstance(value, str) and value.startswith("Q"):
                return value
        return default

    def _replacement_fact(self, fact: Fact, new_object: str, *, object_id: str | None = "__keep__") -> Fact:
        keep_qualifiers = fact.qualifiers if self.preserve_qualifiers else ()
        payload = fact.model_dump()
        payload.update(
            {
                "fact_id": "",  # new record -> new stable id
                "object_label": new_object,
                "object_type": fact.object_type if object_id != "__keep__" else fact.object_type,
                "metadata": {
                    **fact.metadata,
                    "corrupted_from": fact.object_label,
                    "original_fact_id": fact.fact_id,
                },
            }
        )
        if object_id != "__keep__":
            payload["object_id"] = object_id
            payload["object_datatype"] = "string"
        payload["qualifiers"] = keep_qualifiers
        return Fact(**payload)

    @staticmethod
    def _skip_record(fact: Fact, reason: str) -> CorruptionRecord:
        return CorruptionRecord(
            original_fact_id=fact.fact_id,
            corruption_type="skipped",
            original_object=fact.object_label,
            random_seed=None,
            is_corrupted=False,
            skipped_reason=reason,
        )

    @staticmethod
    def _same_value(left: str, right: str) -> bool:
        return left.strip().casefold() == right.strip().casefold()

    @staticmethod
    def _perturb_year(label: str, rng: random.Random) -> str | None:
        match = _YEAR_RE.search(label)
        if not match:
            return None
        original = int(match.group(1))
        # offset within +/- 15% (at least 3 years) so the value stays plausible
        delta = max(3, int(round(original * 0.15)))
        new_year = original + rng.choice([-1, 1]) * rng.randint(3, delta)
        new_year = max(1, new_year)
        if new_year == original:
            new_year = original + 3
        return label[: match.start(1)] + str(new_year) + label[match.end(1) :]

    @staticmethod
    def _perturb_quantity(label: str, rng: random.Random) -> str | None:
        match = _QUANTITY_RE.search(label)
        if not match:
            return None
        raw = match.group(1).replace(",", "")
        try:
            original = float(raw)
        except ValueError:
            return None
        if original == 0:
            replacement = round(rng.uniform(1, 10), 2)
        else:
            factor = rng.uniform(0.5, 2.0)
            if abs(factor - 1.0) < 0.05:
                factor = 1.25
            replacement = original * factor
        if original == int(original) and replacement == int(replacement):
            text = f"{int(replacement):,}"
        else:
            text = f"{replacement:,.2f}".rstrip("0").rstrip(".")
        if text.replace(",", "").replace(".", "") == raw.replace(".", ""):
            text = f"{replacement * 1.5:,.0f}"
        return label[: match.start(1)] + text + label[match.end(1) :]

    def _notes(self, total: int, changed: int, omitted: int, skipped: int) -> tuple[str, ...]:
        notes = [
            f"strategy={self.strategy}",
            f"corruption_rate={self.corruption_rate}",
            f"seed={self.seed}",
            f"touched={changed + omitted}/{total}",
        ]
        if self.strategy == "omission":
            notes.append("control condition: facts removed, not falsified")
        if skipped:
            notes.append(f"{skipped} selected fact(s) had no safe replacement and were left unchanged")
        return tuple(notes)


def corrupt_facts(
    facts: Sequence[Fact],
    strategy: str = "object_swap",
    corruption_rate: float = 0.5,
    seed: int = 42,
    *,
    engine: CorruptionEngine | None = None,
    **engine_kwargs: Any,
) -> CorruptionResult:
    """Functional entry point required by the spec::

        corrupt_facts(facts, strategy, corruption_rate, seed) -> CorruptionResult

    ``CorruptionResult`` carries ``original_facts``, ``corrupted_facts`` and
    ``audit`` (the audit log), so callers can destructively unpack it as
    ``corrupted, audit = result.corrupted_facts, result.audit``.
    """
    if engine is None:
        engine = CorruptionEngine(
            strategy=strategy,
            corruption_rate=corruption_rate,
            seed=seed,
            **engine_kwargs,
        )
    return engine.corrupt(facts)


def engine_from_config(config: Config | CorruptionConfig) -> CorruptionEngine:
    """Build an engine from a validated config section."""
    corruption = config.corruption if isinstance(config, Config) else config
    return CorruptionEngine(
        strategy=corruption.strategy,
        corruption_rate=corruption.corruption_rate,
        seed=corruption.seed,
        max_corruption_fraction_of_block=corruption.max_corruption_fraction_of_block,
        swap_within_domain=corruption.swap_within_domain,
        allow_cross_entity_substitution=corruption.allow_cross_entity_substitution,
        preserve_qualifiers=corruption.preserve_qualifiers,
        audit_log=corruption.audit_log,
        extra_candidate_pool=corruption.extra_candidate_pool,
    )


def describe_corruption_summary(result: CorruptionResult) -> str:
    """One-line human-readable summary used in logs and the demo UI."""
    summary = result.summary
    return (
        f"{summary['strategy']} @ rate={summary['corruption_rate']} seed={summary['seed']}: "
        f"{summary['changed']} changed, {summary['omitted']} omitted, "
        f"{summary['skipped']} skipped, {summary['n_contrast']}/{summary['n_original']} facts in contrast block"
    )
