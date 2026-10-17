"""Metrics: claim extraction, adjudication, semantic quality, latency, stats.

Design rules (spec Section 12):

* every metric is defined in :data:`METRIC_DEFINITIONS` and reported separately --
  no composite "factuality score" that hides which component moved;
* ``None`` means "not applicable / not measurable" (e.g. no claims extracted, no
  reference available) and is never reported as ``0.0``;
* adjudication records the judge used, and the lexical judge is explicit about
  being a *proxy*: it checks claim/evidence content overlap, it is not an
  entailment model.  A model judge is available but must be named in the config.
* latency metrics separate generation-only from end-to-end and include the
  overhead relative to a reference condition.

Claim extraction (rule based) is deliberately conservative: a claim is one
sentence (or clause) that contains at least one of {a number, a capitalised
entity, a copula + property}, because unsupported *factual* claims are what the
metric is about.  Numbers and entity tokens are recorded on the claim so the
lexical judge can check them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .schemas import (
    ClaimJudgement,
    ClaimRecord,
    Evidence,
    ExampleMetrics,
    GenerationRecord,
    JudgeVerdict,
    stable_id,
)
from .utils import get_logger, safe_ratio

__all__ = [
    "METRIC_DEFINITIONS",
    "extract_claims",
    "LexicalEvidenceJudge",
    "GoldReferenceJudge",
    "ModelJudge",
    "judge_claims",
    "score_record",
    "rouge_l",
    "token_f1",
    "exact_match",
    "bootstrap_ci",
    "summarise_latency",
    "latency_overhead",
    "aggregate_metrics",
]

logger = get_logger("metrics")

METRIC_DEFINITIONS: dict[str, str] = {
    "claim_support_rate": (
        "Fraction of evidence-adjudicable extracted factual claims whose evidence "
        "overlap meets the judge threshold ('supported'). Claims with no usable "
        "evidence are excluded; evidence_coverage reports that fraction separately. "
        "Unit of analysis: atomic claim."
    ),
    "hallucination_rate": (
        "Fraction of evidence-adjudicable extracted factual claims judged "
        "'unsupported' or 'contradicted'. Claims with no usable evidence are not "
        "classified as hallucinations; evidence_coverage reports missing evidence. "
        "Unit of analysis: atomic claim."
    ),
    "contradiction_rate": (
        "Fraction of evidence-adjudicable extracted claims whose evidence explicitly "
        "conflicts with the claim."
    ),
    "evidence_coverage": (
        "Fraction of extracted claims for which the evaluator found usable evidence."
    ),
    "factscore": (
        "Approximation of FActScore: mean over claims of the claim-support indicator "
        "(1.0 supported, 0.0 unsupported/contradicted, excluded when unjudged). "
        "Atomic-claim extraction and evidence source are recorded per run; this is "
        "NOT the official FActScore implementation (which uses retrieval + an LLM "
        "judge) and must be labelled as an approximation."
    ),
    "bertscore_f1": (
        "BERTScore F1 against the reference answer (semantic similarity, NOT factuality). "
        "Disabled by default because it downloads a model."
    ),
    "rouge_l": "ROUGE-L F1 against the reference (longest common subsequence).",
    "token_f1": "Token-level F1 against the reference (order-insensitive).",
    "exact_match": "Normalised exact match against the reference.",
    "generation_latency_s": "Wall-clock seconds for the token loop only.",
    "total_latency_s": "End-to-end wall-clock seconds (load + knowledge + decode).",
    "ms_per_output_token": "1000 * generation_latency_s / generated_tokens (median reported).",
    "latency_overhead_pct": (
        "(T_condition - T_reference) / T_reference * 100, computed per example and then "
        "aggregated (median), on the same device and model settings."
    ),
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_CAP_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z'’\-]{1,}\b")
_CLAIM_CUE_RE = re.compile(
    r"\b(is|was|were|are|has|had|have|became|born|died|founded|invented|discovered|"
    r"wrote|won|served|worked|studied|created|developed|led|located|known|considered)\b",
    re.IGNORECASE,
)

#: Words that make a "sentence" a non-claim (hedges, questions, meta text).
_NON_CLAIM_PREFIXES = (
    "i ",
    "i'm",
    "as an ai",
    "here",
    "sure",
    "note:",
    "however, in summary",
    "determining",
    "to answer",
    "based on",
)


def extract_claims(
    answer: str,
    *,
    extractor: str = "rule",
    min_chars: int = 12,
    max_claims: int = 40,
) -> list[ClaimRecord]:
    """Extract atomic-ish claims from a generated answer (spec 12.1).

    ``extractor="rule"`` splits on sentence boundaries and clause separators
    (``;``/`` — ``), then keeps units that look like factual assertions:
    containing a number, at least one capitalised token, or a claim cue verb.
    Purely procedural sentences ("I hope this helps") are dropped.  The output is
    deterministic, so metrics can be reproduced from saved generations.
    """
    if extractor not in {"rule", "model"}:
        raise ValueError(f"unknown claim extractor {extractor!r}")
    if extractor == "model":
        raise NotImplementedError(
            "claim_extraction='model' is not implemented in v1 because it requires a "
            "second model call per answer and a documented prompt; the rule extractor "
            "is used instead. Set evaluation.claim_extraction='rule' (the run records "
            "which extractor produced the claims)."
        )
    text = (answer or "").strip()
    if not text:
        return []
    claims: list[ClaimRecord] = []
    for sentence_index, sentence in enumerate(_split_sentences(text)):
        for clause_index, clause in enumerate(_split_clauses(sentence)):
            candidate = clause.strip(" -*•\t")
            if len(candidate) < min_chars:
                continue
            lowered = candidate.lower()
            if any(lowered.startswith(prefix) for prefix in _NON_CLAIM_PREFIXES):
                continue
            numbers = _NUMBER_RE.findall(candidate)
            entities = _CAP_TOKEN_RE.findall(candidate)
            if not numbers and not entities and not _CLAIM_CUE_RE.search(candidate):
                continue
            if not _CLAIM_CUE_RE.search(candidate) and not numbers and len(entities) < 2:
                # e.g. a heading or a name list: not an evaluable factual claim
                continue
            claims.append(
                ClaimRecord(
                    claim_id=stable_id("claim", sentence_index, clause_index, candidate),
                    text=candidate,
                    source_sentence=sentence,
                    sentence_index=sentence_index,
                    entities=entities,
                    numbers=numbers,
                    extractor="rule",
                )
            )
            if len(claims) >= max_claims:
                return claims
    return claims


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return parts or [text.strip()]


def _split_clauses(sentence: str) -> list[str]:
    """Split on strong clause separators, keeping ``and``-joined lists intact."""
    parts = re.split(r"(?<=[a-z0-9\)\"])\s*(?:;|—|--)\s*", sentence)
    return [part for part in (p.strip() for p in parts) if part]


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


@dataclass
class LexicalEvidenceJudge:
    """Evidence-overlap judge used when no LLM judge is configured.

    For each claim it computes the best content-word/overlap score against the
    available evidence sentences:

    * ``supported``     -- overlap >= ``support_threshold`` and no conflicting number;
    * ``contradicted``  -- a shared subject but a *different* number/year;
    * ``unsupported``   -- evidence exists but does not overlap enough;
    * ``no_evidence``   -- no evidence at all was supplied.

    It is a proxy: it can miss entailments and is fooled by paraphrases.  Reports
    must label it as ``heuristic_lexical`` and the metric tables distinguish it
    from ``model_judge`` results (spec 12.1, 11.4).
    """

    name: str = "heuristic_lexical"
    support_threshold: float = 0.35
    contradiction_threshold: float = 0.5
    ignore_labeled_nonfactual: bool = True

    def judge(self, claim: ClaimRecord, evidence: Sequence[Evidence]) -> JudgeVerdict:
        # Explicitly nonfactual/refuted spans are labels about the span, not
        # supporting evidence.  In particular, WikiBio includes both factual and
        # hallucinated sentences from the source model in one example.
        usable_evidence = [
            item for item in evidence
            if not self.ignore_labeled_nonfactual
            or (item.label or "").strip().lower() not in {"nonfactual", "refuted", "refutes"}
        ]
        if not usable_evidence:
            return JudgeVerdict(
                claim_id=claim.claim_id,
                label="no_evidence",
                score=0.0,
                rationale="no evidence available for this example",
                judge=self.name,
            )
        best: tuple[float, Evidence, list[str]] | None = None
        for item in usable_evidence:
            overlap, shared = self._overlap(claim, item.text)
            if best is None or overlap > best[0]:
                best = (overlap, item, shared)
        assert best is not None
        overlap, item, shared = best
        evidence_used = [item.text]
        claim_numbers = set(claim.numbers)
        evidence_numbers = set(_NUMBER_RE.findall(item.text))
        subject_overlap = len(shared) >= 2
        number_conflict = bool(claim_numbers) and bool(evidence_numbers) and not (claim_numbers & evidence_numbers)
        if number_conflict and subject_overlap and overlap >= self.contradiction_threshold:
            return JudgeVerdict(
                claim_id=claim.claim_id,
                label="contradicted",
                score=overlap,
                evidence_used=evidence_used,
                rationale=(
                    f"shared context but conflicting figures (claim {sorted(claim_numbers)} vs "
                    f"evidence {sorted(evidence_numbers)})"
                ),
                judge=self.name,
            )
        if overlap >= self.support_threshold:
            return JudgeVerdict(
                claim_id=claim.claim_id,
                label="supported",
                score=overlap,
                evidence_used=evidence_used,
                rationale=f"content overlap {overlap:.2f} >= {self.support_threshold}",
                judge=self.name,
            )
        return JudgeVerdict(
            claim_id=claim.claim_id,
            label="unsupported",
            score=overlap,
            evidence_used=evidence_used,
            rationale=f"content overlap {overlap:.2f} < {self.support_threshold}",
            judge=self.name,
        )

    @staticmethod
    def _overlap(claim: ClaimRecord, evidence_text: str) -> tuple[float, list[str]]:
        claim_tokens = _content_tokens(claim.text)
        evidence_tokens = _content_tokens(evidence_text)
        if not claim_tokens:
            return 0.0, []
        shared = sorted(claim_tokens & evidence_tokens)
        # containment (how much of the claim is covered) balanced with Jaccard
        containment = len(shared) / len(claim_tokens)
        jaccard = len(shared) / len(claim_tokens | evidence_tokens) if claim_tokens | evidence_tokens else 0.0
        return float(0.7 * containment + 0.3 * jaccard), shared


_STOPWORDS = frozenset(
    """
    a an the of and or in on at for with from to by is was were are be been being as that this these
    those it its he she they his her their has have had not no also more than then there here which who
    whom whose what when where why how was were will would can could may might must should shall do does
    did done about into over under during after before between among across within without his/her
    """.split()
)


def _content_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z0-9']+", (text or "").lower()))
    return {token for token in tokens if token not in _STOPWORDS and len(token) > 1}


@dataclass
class GoldReferenceJudge:
    """Compare claims against the dataset's *labelled* spans (WikiBio-style).

    ``evidence`` must carry ``label`` in {"factual", "nonfactual"} (as produced by
    the WikiBio loader).  A claim is "supported" when it overlaps a span the
    dataset labelled factual, "contradicted" when it overlaps a span labelled
    nonfactual.  This measures *dataset agreement*, not truth -- and the report
    must say so.
    """

    name: str = "gold_reference"
    min_overlap: float = 0.3

    def judge(self, claim: ClaimRecord, evidence: Sequence[Evidence]) -> JudgeVerdict:
        best_supported: JudgeVerdict | None = None
        best_nonfactual: JudgeVerdict | None = None
        for item in evidence:
            lexical = LexicalEvidenceJudge(
                support_threshold=self.min_overlap,
                ignore_labeled_nonfactual=False,
            )
            verdict = lexical.judge(claim, [item])
            label = (item.label or "").lower()
            if label in {"nonfactual", "refuted", "refutes"} and verdict.label in {"supported", "contradicted"}:
                if best_nonfactual is None or verdict.score > best_nonfactual.score:
                    best_nonfactual = verdict.model_copy(
                        update={
                            "label": "contradicted",
                            "judge": self.name,
                            "rationale": f"claim overlaps a span the dataset labelled {item.label}",
                        }
                    )
            elif verdict.label == "supported" and (best_supported is None or verdict.score > best_supported.score):
                best_supported = verdict.model_copy(
                    update={"judge": self.name, "rationale": "claim overlaps a span the dataset labelled factual"}
                )
        if best_nonfactual is not None and (best_supported is None or best_nonfactual.score >= best_supported.score):
            return best_nonfactual
        if best_supported is not None:
            return best_supported
        if not evidence:
            return JudgeVerdict(claim_id=claim.claim_id, label="no_evidence", score=0.0, judge=self.name, rationale="no labelled spans")
        return JudgeVerdict(
            claim_id=claim.claim_id,
            label="unsupported",
            score=0.0,
            judge=self.name,
            rationale="no labelled span agrees with the claim",
        )


@dataclass
class ModelJudge:
    """LLM-based claim adjudicator (optional; must be named in the config).

    The prompt is fixed and stored with the run so the judging procedure is
    reproducible.  Output is parsed strictly: anything that is not one of the four
    labels is recorded as ``no_evidence`` with the raw text in the rationale, so a
    broken judge cannot silently inflate the support rate.
    """

    adapter: Any
    name: str = "model_judge"
    max_new_tokens: int = 128
    temperature: float = 0.0
    prompt_template: str = (
        "You are a careful fact-checking assistant.\n"
        "Decide whether the CLAIM is supported by the EVIDENCE.\n"
        "Answer with exactly one word: supported, contradicted, unsupported, or no_evidence.\n\n"
        "EVIDENCE:\n{evidence}\n\nCLAIM:\n{claim}\n\nVERDICT:"
    )

    def judge(self, claim: ClaimRecord, evidence: Sequence[Evidence]) -> JudgeVerdict:
        if not evidence:
            return JudgeVerdict(claim_id=claim.claim_id, label="no_evidence", score=0.0, judge=self.name, rationale="no evidence")
        joined = "\n---\n".join(item.text for item in evidence[:8])
        prompt = self.prompt_template.format(evidence=joined, claim=claim.text)
        from .model_adapter import BranchCache  # local import: avoids a hard dependency

        logits, cache = self.adapter.advance(self.adapter.encode(self.adapter.render(prompt)), None)
        cache = cache or BranchCache()
        generated: list[int] = []
        rng = np.random.default_rng(0)
        for _ in range(self.max_new_tokens):
            from .decoding import select_token

            token_id, _handled = select_token(
                np.asarray(logits).ravel(),
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                top_p=1.0,
                top_k=0,
                rng=rng,
            )
            if token_id in self.adapter.eos_token_ids:
                break
            generated.append(token_id)
            logits, cache = self.adapter.advance([token_id], cache)
        raw = self.adapter.decode(generated, skip_special_tokens=True)
        label = _parse_verdict(raw)
        return JudgeVerdict(
            claim_id=claim.claim_id,
            label=label,
            score=1.0 if label == "supported" else 0.0,
            evidence_used=[joined],
            rationale=f"judge output: {raw.strip()[:200]!r}",
            judge=self.name,
        )


def _parse_verdict(raw: str) -> str:
    text = (raw or "").strip().lower()
    for label in ("no_evidence", "contradicted", "unsupported", "supported"):
        if label in text:
            # "unsupported" contains "supported"; check the longer label first
            if label == "supported" and "unsupported" in text:
                continue
            return "unsupported" if label == "unsupported" else label
    return "no_evidence"


def judge_claims(
    claims: Sequence[ClaimRecord],
    evidence: Sequence[Evidence],
    *,
    judge: str = "heuristic_lexical",
    adapter: Any = None,
    model_id: str | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> ClaimJudgement:
    """Adjudicate every claim with the configured judge."""
    if judge == "heuristic_lexical":
        engine: Any = LexicalEvidenceJudge()
    elif judge == "gold_reference":
        engine = GoldReferenceJudge()
    elif judge == "model_judge":
        if adapter is None:
            raise ValueError(
                "claim_judge='model_judge' requires a judge model. Set evaluation.judge_model_id "
                "and ensure the model can be loaded (the run never invents a judge)."
            )
        engine = ModelJudge(adapter=adapter, max_new_tokens=max_new_tokens, temperature=temperature)
    else:
        raise ValueError(f"unknown claim judge {judge!r}")

    verdicts = [engine.judge(claim, evidence) for claim in claims]
    return ClaimJudgement(claims=list(claims), verdicts=verdicts, judge=str(getattr(engine, "name", judge)))


# ---------------------------------------------------------------------------
# Semantic / surface similarity
# ---------------------------------------------------------------------------


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def token_f1(prediction: str, reference: str) -> float | None:
    """Order-insensitive token F1; ``None`` when either side is empty."""
    pred_tokens = _normalise_text(prediction).split()
    ref_tokens = _normalise_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return None
    pred_counts = {token: pred_tokens.count(token) for token in set(pred_tokens)}
    ref_counts = {token: ref_tokens.count(token) for token in set(ref_tokens)}
    shared = sum(min(pred_counts[token], ref_counts.get(token, 0)) for token in pred_counts)
    if shared == 0:
        return 0.0
    precision = shared / len(pred_tokens)
    recall = shared / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float | None:
    if not (prediction or "").strip() or not (reference or "").strip():
        return None
    return 1.0 if _normalise_text(prediction) == _normalise_text(reference) else 0.0


def rouge_l(prediction: str, reference: str) -> float | None:
    """ROUGE-L F1 (LCS-based), implemented locally to avoid an extra dependency."""
    pred_tokens = _normalise_text(prediction).split()
    ref_tokens = _normalise_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return None
    # standard dynamic-programming LCS
    previous = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for index, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str = "roberta-large",
    lang: str = "en",
) -> list[float | None]:
    """Optional BERTScore wrapper (requires ``pip install -e .[bertscore]``).

    Raises a readable error when the optional dependency is missing instead of
    silently skipping the metric (the report must state which metrics ran).
    """
    try:
        from bert_score import score as bert_score_score  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "evaluation.bertscore_enabled=true requires `pip install -e .[bertscore]` "
            "(and a model download). Metrics that did not run are reported as null."
        ) from exc
    if not predictions or not references:
        return []
    _p, _r, f1 = bert_score_score(
        list(predictions), list(references), model_type=model_type, lang=lang, verbose=False, rescale_with_baseline=False
    )
    return [float(value) for value in f1]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 12345,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> list[float] | None:
    """Percentile bootstrap CI for a statistic of ``values``.

    Returns ``None`` for fewer than two values (a CI from one example would be
    fake precision -- spec 12.4).
    """
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if array.size < 2 or samples <= 0:
        return None
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, array.size, size=array.size)
        estimates[index] = statistic(array[draw])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return [float(low), float(high)]


def summarise_latency(records: Sequence[GenerationRecord]) -> dict[str, float]:
    """Median/mean/spread latency summary for one condition (spec 12.3)."""
    if not records:
        return {}
    def collect(name: str, extractor: Callable[[GenerationRecord], float | None]) -> list[float]:
        values = [extractor(record) for record in records]
        return [float(v) for v in values if v is not None and math.isfinite(float(v))]

    generation = collect("generation", lambda r: r.latency.generation_s)
    total = collect("total", lambda r: r.latency.total_s)
    per_token = collect("ms_per_token", lambda r: r.ms_per_token)
    retrieval = collect("retrieval", lambda r: r.latency.fact_retrieval_s)
    corruption = collect("corruption", lambda r: r.latency.corruption_s)
    linking = collect("linking", lambda r: r.latency.entity_linking_s)
    tokens = collect("output_tokens", lambda r: float(r.output_tokens))

    summary: dict[str, float] = {"n": float(len(records))}
    for name, values in (
        ("generation_s", generation),
        ("total_s", total),
        ("ms_per_token", per_token),
        ("retrieval_s", retrieval),
        ("corruption_s", corruption),
        ("linking_s", linking),
        ("output_tokens", tokens),
    ):
        if values:
            array = np.asarray(values, dtype=np.float64)
            summary[f"{name}_median"] = float(np.median(array))
            summary[f"{name}_mean"] = float(np.mean(array))
            summary[f"{name}_p10"] = float(np.quantile(array, 0.10))
            summary[f"{name}_p90"] = float(np.quantile(array, 0.90))
    summary["records"] = float(len(records))
    return summary


def latency_overhead(
    condition: Sequence[GenerationRecord],
    reference: Sequence[GenerationRecord],
    *,
    key: Callable[[GenerationRecord], float] = lambda record: record.latency.generation_s,
) -> dict[str, float | None]:
    """Per-example paired latency overhead of a condition vs a reference.

    Paired by ``example_id`` so that different prompt lengths cannot masquerade as
    an effect.  Reports the median/mean of
    ``(T_condition - T_reference) / T_reference * 100``.
    """
    reference_by_id = {record.example_id: key(record) for record in reference}
    deltas: list[float] = []
    absolute: list[float] = []
    for record in condition:
        base = reference_by_id.get(record.example_id)
        value = key(record)
        if base is None or base <= 0:
            continue
        deltas.append((value - base) / base * 100.0)
        absolute.append(value - base)
    if not deltas:
        return {"n_pairs": 0, "overhead_pct_median": None, "overhead_pct_mean": None}
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "n_pairs": int(array.size),
        "overhead_pct_median": float(np.median(array)),
        "overhead_pct_mean": float(np.mean(array)),
        "overhead_pct_p10": float(np.quantile(array, 0.10)),
        "overhead_pct_p90": float(np.quantile(array, 0.90)),
        "absolute_delta_s_median": float(np.median(np.asarray(absolute, dtype=np.float64))),
    }


# ---------------------------------------------------------------------------
# Per-example scoring
# ---------------------------------------------------------------------------


def score_record(
    record: GenerationRecord,
    *,
    judge: str = "heuristic_lexical",
    evidence: Sequence[Evidence] | None = None,
    judge_adapter: Any = None,
    judge_max_new_tokens: int = 128,
    judge_temperature: float = 0.0,
    compute_semantic: bool = True,
) -> tuple[ExampleMetrics, ClaimJudgement]:
    """Score one generation record and return its metrics plus the claim detail."""
    evidence_list = list(evidence if evidence is not None else [])
    claims = extract_claims(record.answer) if record.status.value in {"ok", "degraded"} else []
    judgement = judge_claims(
        claims,
        evidence_list,
        judge=judge,
        adapter=judge_adapter,
        max_new_tokens=judge_max_new_tokens,
        temperature=judge_temperature,
    )
    counts = judgement.counts
    n_claims = len(claims)
    n_supported = counts.get("supported", 0)
    n_contradicted = counts.get("contradicted", 0)
    n_unsupported = counts.get("unsupported", 0)
    n_no_evidence = counts.get("no_evidence", 0)
    judged = n_supported + n_contradicted + n_unsupported
    evidence_coverage = safe_ratio(judged, n_claims)
    claim_support_rate = safe_ratio(n_supported, judged)
    hallucination_rate = safe_ratio(n_unsupported + n_contradicted, judged)
    contradiction_rate = safe_ratio(n_contradicted, judged)
    factscore = safe_ratio(n_supported, judged) if judged else None

    notes: list[str] = []
    if n_claims == 0:
        notes.append("no evaluable claims were extracted from the answer")
    if n_no_evidence:
        notes.append(f"{n_no_evidence} claim(s) had no usable evidence and were not adjudicated")
    if record.status.value != "ok":
        notes.append(f"record status={record.status.value} reason={record.degraded_reason}")
    if record.is_toy_model:
        notes.append("toy model: metrics are plumbing checks only, not scientific results")

    reference = record.reference
    metrics = ExampleMetrics(
        example_id=record.example_id,
        condition=record.condition or record.mode,
        mode=record.mode,
        n_claims=n_claims,
        n_supported=n_supported,
        n_contradicted=n_contradicted,
        n_unsupported=n_unsupported,
        n_no_evidence=n_no_evidence,
        claim_support_rate=claim_support_rate,
        hallucination_rate=hallucination_rate,
        contradiction_rate=contradiction_rate,
        evidence_coverage=evidence_coverage,
        factscore=factscore,
        token_f1=token_f1(record.answer, reference) if (compute_semantic and reference) else None,
        rouge_l=rouge_l(record.answer, reference) if (compute_semantic and reference) else None,
        exact_match=exact_match(record.answer, reference) if (compute_semantic and reference) else None,
        output_tokens=record.output_tokens,
        generation_s=record.latency.generation_s,
        total_s=record.latency.total_s,
        ms_per_token=record.ms_per_token,
        status=record.status.value,
        notes=notes,
    )
    return metrics, judgement


def aggregate_metrics(
    per_example: Sequence[ExampleMetrics],
    *,
    condition: str = "",
    mode: str = "",
    variant: str = "",
    judge: str = "",
    evidence_source: str = "",
    bootstrap_samples: int = 1000,
    bootstrap_confidence: float = 0.95,
    latency: dict[str, float] | None = None,
    latency_by_condition: dict[str, dict[str, float]] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Aggregate per-example metrics with explicit sample sizes and CIs."""
    if not per_example:
        return {
            "condition": condition,
            "mode": mode,
            "variant": variant,
            "n_examples": 0,
            "notes": ["no examples were scored"],
            "definitions": METRIC_DEFINITIONS,
        }
    def series(attribute: str, *, only_claims: bool = False) -> list[float]:
        values = []
        for item in per_example:
            value = getattr(item, attribute)
            if value is None:
                continue
            if only_claims and item.n_claims == 0:
                continue
            values.append(float(value))
        return values

    ci: dict[str, list[float]] = {}
    for attribute in ("claim_support_rate", "hallucination_rate", "contradiction_rate", "evidence_coverage", "factscore"):
        values = series(attribute, only_claims=(attribute != "evidence_coverage"))
        interval = bootstrap_ci(values, samples=bootstrap_samples, confidence=bootstrap_confidence)
        if interval:
            ci[attribute] = interval

    def mean_of(values: Sequence[float]) -> float | None:
        return float(np.mean(values)) if values else None

    summary: dict[str, Any] = {
        "condition": condition,
        "mode": mode,
        "variant": variant,
        "n_examples": len(per_example),
        "n_ok": sum(1 for item in per_example if item.status == "ok"),
        "n_degraded": sum(1 for item in per_example if item.status == "degraded"),
        "n_skipped": sum(1 for item in per_example if item.status == "skipped"),
        "n_errors": sum(1 for item in per_example if item.status == "error"),
        "n_with_claims": sum(1 for item in per_example if item.n_claims > 0),
        "n_total_claims": sum(item.n_claims for item in per_example),
        "mean_claim_support_rate": mean_of(series("claim_support_rate", only_claims=True)),
        "mean_hallucination_rate": mean_of(series("hallucination_rate", only_claims=True)),
        "mean_contradiction_rate": mean_of(series("contradiction_rate", only_claims=True)),
        "mean_evidence_coverage": mean_of(series("evidence_coverage")),
        "mean_factscore": mean_of(series("factscore", only_claims=True)),
        "mean_bertscore_f1": mean_of(series("bertscore_f1")),
        "mean_rouge_l": mean_of(series("rouge_l")),
        "mean_exact_match": mean_of(series("exact_match")),
        "mean_token_f1": mean_of(series("token_f1")),
        "ci": ci,
        "latency": latency or {},
        "latency_by_condition": latency_by_condition or {},
        "definitions": METRIC_DEFINITIONS,
        "judge": judge,
        "evidence_source": evidence_source,
        "notes": list(notes),
    }
    return summary
