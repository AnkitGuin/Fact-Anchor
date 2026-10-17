"""Contrastive decoding engine (spec 3.2, 3.3, 10.2).

The scoring rule implemented here is the proposal's own hypothesis:

.. math::

    s_t(v) = \\log p_b(v \\mid x, y_{<t}) - \\alpha \\log p_c(v \\mid x, y_{<t}, z_c)

Everything is computed in log space.  ``alpha == 0`` reproduces the baseline
scoring path *exactly* (same filters, same RNG use), which is asserted by
``tests/test_decoding.py``.

Documented order of operations (spec 3.2 item 4 requires this to be recorded):
1. base logits -> ``log_softmax`` -> ``log p_b``;
2. contrast logits -> ``log_softmax`` -> ``log p_c`` (skipped entirely when
   ``alpha == 0``, i.e. the baseline never pays for the second forward pass);
3. *base-side* constraints are applied to ``log p_b``: repetition penalty,
   no-repeat-ngram bans and ``min_new_tokens`` EOS suppression;
4. contrastive score ``s = log p_b - alpha * log p_c`` (or the logit-difference
   ablation, which subtracts raw logits instead and is labelled as a different
   variant everywhere);
5. adaptive plausibility mask (optional, off by default);
6. selection: greedy ``argmax`` or sampling from ``softmax(s / T)`` restricted by
   ``top_k`` / ``top_p`` using a seeded generator.

NaN/Inf handling: non-finite scores are replaced by ``-inf``; if that would make
every token impossible, the step is flagged and the *base* argmax is used, so a
numerical accident can never silently produce a random token.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from .config import DecodingConfig
from .model_adapter import log_softmax
from .schemas import TokenStep
from .utils import get_logger

__all__ = [
    "StepDiagnostics",
    "contrastive_scores",
    "apply_repetition_penalty",
    "apply_no_repeat_ngram",
    "apply_plausibility_mask",
    "select_token",
    "sample_from_scores",
    "ContrastiveDecoder",
]

logger = get_logger("decoding")

_NEG_INF = float("-inf")


@dataclass
class StepDiagnostics:
    """Per-step bookkeeping handed back to the generation loop (spec 3.2 item 8)."""

    step: int
    token_id: int
    token: str
    base_logprob: float
    contrast_logprob: float | None
    score: float
    entropy_base: float | None
    contrast_applied: bool
    top1_gap: float | None
    nonfinite_handled: bool = False
    notes: tuple[str, ...] = ()

    def to_record(self) -> TokenStep:
        return TokenStep(
            step=self.step,
            token_id=self.token_id,
            token=self.token,
            base_logprob=self.base_logprob,
            contrast_logprob=self.contrast_logprob,
            score=self.score,
            entropy_base=self.entropy_base,
            contrast_applied=self.contrast_applied,
            score_top1_gap=self.top1_gap,
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def contrastive_scores(
    base_logits: np.ndarray,
    contrast_logits: np.ndarray | None,
    alpha: float,
    *,
    variant: Literal["contrastive_logprob", "logit_difference"] = "contrastive_logprob",
    contrast_warmup_active: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the contrastive score in log space.

    Returns ``(scores, diagnostics)`` where ``scores`` is a 1-D float64 array.
    ``contrast_logits=None`` (or ``alpha == 0`` or warm-up active) returns the
    plain base log-probabilities.
    """
    base_logits = np.asarray(base_logits, dtype=np.float64).ravel()
    if base_logits.size == 0:
        raise ValueError("contrastive_scores received empty base logits")
    base_log_probs = log_softmax(base_logits)
    info: dict[str, Any] = {
        "variant": variant,
        "alpha": float(alpha),
        "contrast_used": False,
        "nonfinite_scores": 0,
    }
    if contrast_logits is None or alpha == 0.0 or not contrast_warmup_active:
        info["base_log_probs"] = base_log_probs
        return base_log_probs, info

    contrast_logits = np.asarray(contrast_logits, dtype=np.float64).ravel()
    if contrast_logits.shape != base_logits.shape:
        raise ValueError(
            f"base and contrast logits must have the same vocabulary size, got "
            f"{base_logits.shape} vs {contrast_logits.shape}"
        )
    contrast_log_probs = log_softmax(contrast_logits)
    if variant == "logit_difference":
        # Documented ablation: raw logits are compared, which is NOT equivalent to
        # log-probability differences because logits carry an arbitrary additive
        # constant per branch.  Kept for the ablation grid only.
        scores = base_logits - float(alpha) * contrast_logits
        scores = scores - np.max(scores)  # cosmetic: keep magnitudes comparable
        info["variant_note"] = (
            "logit-difference variant: raw logits include an arbitrary additive constant, "
            "so this is not a log-probability difference (see spec 3.3)"
        )
    else:
        scores = base_log_probs - float(alpha) * contrast_log_probs
    nonfinite = int(np.count_nonzero(~np.isfinite(scores)))
    if nonfinite:
        # Inf can only come from an Inf/NaN logit; make it explicit and safe.
        info["nonfinite_scores"] = nonfinite
        scores = np.where(np.isfinite(scores), scores, _NEG_INF)
        logger.warning("%d non-finite contrastive score(s) replaced with -inf", nonfinite)
    info.update(
        {
            "contrast_used": True,
            "base_log_probs": base_log_probs,
            "contrast_log_probs": contrast_log_probs,
        }
    )
    return scores, info


# ---------------------------------------------------------------------------
# Base-side constraints  (applied to log p_b, before contrastive scoring)
# ---------------------------------------------------------------------------


def apply_repetition_penalty(
    log_probs: np.ndarray,
    generated_ids: Sequence[int],
    penalty: float,
) -> np.ndarray:
    """Add a log-space penalty on already-generated tokens.

    The penalty is applied in log space (``log p - log(penalty)``) rather than on
    logits, because the rest of this module operates on log-probabilities.  With
    ``penalty == 1.0`` the array is returned unchanged.
    """
    if penalty == 1.0 or not generated_ids:
        return log_probs
    if penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")
    out = np.array(log_probs, copy=True)
    log_penalty = np.log(float(penalty))
    for token_id in set(int(t) for t in generated_ids):
        if 0 <= token_id < out.size:
            out[token_id] -= log_penalty
    return out


def apply_no_repeat_ngram(
    log_probs: np.ndarray,
    generated_ids: Sequence[int],
    ngram_size: int,
) -> np.ndarray:
    """Ban tokens that would complete a repeated n-gram (== HF semantics)."""
    if ngram_size <= 0 or len(generated_ids) < ngram_size:
        return log_probs
    ids = [int(t) for t in generated_ids]
    out = np.array(log_probs, copy=True)
    prefix = tuple(ids[-(ngram_size - 1) :]) if ngram_size > 1 else ()
    banned: set[int] = set()
    for index in range(len(ids) - ngram_size + 1):
        window = tuple(ids[index : index + ngram_size - 1])
        if window == prefix:
            banned.add(ids[index + ngram_size - 1])
    for token_id in banned:
        if 0 <= token_id < out.size:
            out[token_id] = _NEG_INF
    return out


def apply_plausibility_mask(
    scores: np.ndarray,
    base_log_probs: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Adaptive plausibility constraint (Li et al., 2023): keep plausible tokens only.

    A token survives when ``p_b(v) >= epsilon * max_w p_b(w)``.  This guards the
    documented failure mode where an unrelated contrast context boosts arbitrary
    tokens (spec 3.1 "Important").
    """
    if epsilon <= 0:
        return scores
    cutoff = np.log(epsilon) + float(np.max(base_log_probs))
    out = np.array(scores, copy=True)
    out[base_log_probs < cutoff] = _NEG_INF
    return out


def suppress_eos(scores: np.ndarray, eos_token_ids: Sequence[int]) -> np.ndarray:
    """Force continuation by banning every EOS id (``min_new_tokens`` support)."""
    out = np.array(scores, copy=True)
    for token_id in eos_token_ids:
        if 0 <= int(token_id) < out.size:
            out[int(token_id)] = _NEG_INF
    return out


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def sample_from_scores(
    scores: np.ndarray,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    rng: np.random.Generator,
) -> int:
    """Sample one token id from a score vector.

    The scores are treated as unnormalised log-weights (spec 3.2 item 6): they
    are divided by the temperature and passed through a softmax before top-k /
    top-p filtering and sampling.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 when sampling")
    candidate = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(candidate)
    if not np.any(finite):
        raise ValueError("all scores are -inf; nothing can be sampled")
    scaled = np.full_like(candidate, _NEG_INF)
    scaled[finite] = candidate[finite] / temperature
    probs = np.exp(scaled - np.max(scaled[finite]))
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0:  # pragma: no cover - numerical guard
        raise ValueError("probability mass collapsed to zero during sampling")
    probs = probs / total

    if top_k and top_k > 0 and top_k < int(np.count_nonzero(probs)):
        keep = np.argpartition(-probs, top_k - 1)[:top_k]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs = probs / float(np.sum(probs))
    if 0 < top_p < 1.0:
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        # keep the smallest prefix whose cumulative mass reaches top_p
        cutoff_index = int(np.searchsorted(cumulative, top_p, side="left"))
        cutoff_index = min(cutoff_index, len(sorted_probs) - 1)
        keep_ids = order[: cutoff_index + 1]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep_ids] = True
        probs = np.where(mask, probs, 0.0)
        # always keep the argmax so the distribution never degenerates to zero
        if not np.any(probs > 0):  # pragma: no cover - defensive
            probs[int(np.argmax(scores))] = 1.0
        probs = probs / float(np.sum(probs))
    return int(rng.choice(probs.shape[0], p=probs))


def select_token(
    scores: np.ndarray,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    rng: np.random.Generator,
    base_log_probs: np.ndarray | None = None,
) -> tuple[int, bool]:
    """Choose the next token id from ``scores``.

    Returns ``(token_id, handled_nonfinite)``.  ``handled_nonfinite`` is True when
    every score was non-finite and the base argmax had to be used as a fallback.
    """
    array = np.asarray(scores, dtype=np.float64)
    if array.size == 0:
        raise ValueError("select_token received an empty score vector")
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        if base_log_probs is None:
            raise ValueError("all scores are non-finite and no base distribution was supplied")
        fallback = int(np.argmax(np.asarray(base_log_probs, dtype=np.float64)))
        logger.warning("all contrastive scores non-finite; falling back to the base argmax token %d", fallback)
        return fallback, True
    if do_sample:
        return sample_from_scores(array, temperature=temperature, top_p=top_p, top_k=top_k, rng=rng), False
    return int(np.argmax(array)), False


# ---------------------------------------------------------------------------
# Facade used by the generation loop
# ---------------------------------------------------------------------------


@dataclass
class ContrastiveDecoder:
    """Applies contract configuration to a single decoding step."""

    config: DecodingConfig
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    steps: list[StepDiagnostics] = field(default_factory=list, repr=False)
    eos_suppression_events: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # -- helpers ------------------------------------------------------------
    @property
    def alpha(self) -> float:
        return float(self.config.alpha)

    @property
    def contrast_enabled(self) -> bool:
        return self.config.mode == "factanchor" and self.alpha > 0.0

    def reset(self) -> None:
        self.steps.clear()
        self.eos_suppression_events = 0
        self._rng = np.random.default_rng(self.seed)

    def contrast_active_at(self, step: int) -> bool:
        """Contrast is applied from ``contrast_warmup_tokens`` onwards."""
        return self.contrast_enabled and step >= int(self.config.contrast_warmup_tokens)

    # -- one step -----------------------------------------------------------
    def step(
        self,
        *,
        step_index: int,
        base_logits: np.ndarray,
        contrast_logits: np.ndarray | None,
        generated_ids: Sequence[int],
        eos_token_ids: Sequence[int],
    ) -> tuple[int, StepDiagnostics]:
        """Compute the next token for one step and return it with diagnostics."""
        decode_config = self.config
        # The contrast term can only be applied when a contrast distribution was
        # actually computed for this step (e.g. FactAnchor with no available facts
        # falls back to the baseline path and passes `None`).
        use_contrast = self.contrast_active_at(step_index) and contrast_logits is not None
        scores, info = contrastive_scores(
            base_logits,
            contrast_logits if use_contrast else None,
            self.alpha,
            variant=decode_config.variant,
            contrast_warmup_active=True,
        )
        raw_base_log_probs = info["base_log_probs"]
        base_log_probs = apply_repetition_penalty(
            raw_base_log_probs, generated_ids, decode_config.repetition_penalty
        )
        base_log_probs = apply_no_repeat_ngram(
            base_log_probs, generated_ids, decode_config.no_repeat_ngram_size
        )

        # Steps 3-4 of the documented order: the *base-side* constraints are folded
        # into the contrastive score.  For the log-probability variant the score is
        # simply recomputed from the constrained base log-probabilities, which is
        # exact; for the logit-difference ablation the constraint is added as a
        # delta on the raw-logit score (documented in the variant note).
        if not use_contrast:
            scores = base_log_probs
        elif "contrast_log_probs" not in info:  # pragma: no cover - defensive invariant
            raise RuntimeError(
                "contrast log-probabilities are missing although the contrast term was requested; "
                "refusing to score with an inconsistent state"
            )
        elif decode_config.variant == "logit_difference":
            finite = np.isfinite(raw_base_log_probs)
            delta = np.where(finite, base_log_probs - raw_base_log_probs, 0.0)
            scores = scores + delta
            scores = np.where(np.isfinite(base_log_probs), scores, _NEG_INF)
        else:
            scores = base_log_probs - self.alpha * info["contrast_log_probs"]

        if decode_config.adaptive_plausibility:
            scores = apply_plausibility_mask(
                scores, base_log_probs, decode_config.plausibility_epsilon
            )
        if step_index < int(decode_config.min_new_tokens):
            scores = suppress_eos(scores, eos_token_ids)
            self.eos_suppression_events += 1

        if use_contrast and self.alpha == 0.0:  # pragma: no cover - defensive invariant
            raise AssertionError("alpha=0 must never use the contrast branch")

        token_id, handled = select_token(
            scores,
            do_sample=decode_config.do_sample,
            temperature=decode_config.temperature,
            top_p=decode_config.top_p,
            top_k=decode_config.top_k,
            rng=self._rng,
            base_log_probs=base_log_probs,
        )
        top1_gap = None
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size >= 2:
            top_two = np.sort(finite_scores)[-2:]
            top1_gap = float(top_two[-1] - top_two[0])
        diagnostics = StepDiagnostics(
            step=step_index,
            token_id=token_id,
            token="",
            base_logprob=float(base_log_probs[token_id]),
            contrast_logprob=(
                float(info["contrast_log_probs"][token_id]) if info.get("contrast_used") else None
            ),
            score=float(scores[token_id]),
            entropy_base=None,
            contrast_applied=bool(info.get("contrast_used")),
            top1_gap=top1_gap,
            nonfinite_handled=handled,
            notes=(info["variant_note"],) if "variant_note" in info else (),
        )
        return token_id, diagnostics

    def base_entropy(self, scores: np.ndarray) -> float:
        """Shannon entropy (nats) of the softmax of ``scores`` (diagnostics only)."""
        array = np.asarray(scores, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return 0.0
        probs = np.exp(finite - np.max(finite))
        probs /= probs.sum()
        return float(-np.sum(probs * np.log(np.clip(probs, 1e-300, None))))
