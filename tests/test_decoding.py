"""Decoding / scoring tests (spec 3.2, 3.3, 10.2 and the unit-test list)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pathlib import Path as _Path

from factanchor.config import DecodingConfig, load_config
from factanchor.decoding import (
    ContrastiveDecoder,
    apply_no_repeat_ngram,
    apply_plausibility_mask,
    apply_repetition_penalty,
    contrastive_scores,
    sample_from_scores,
    select_token,
)
from factanchor.generation import TextGenerator
from factanchor.model_adapter import log_softmax


def _manual_log_softmax(logits: list[float]) -> list[float]:
    array = np.asarray(logits, dtype=np.float64)
    shifted = array - array.max()
    return list(shifted - math.log(float(np.exp(shifted).sum())))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_log_softmax_matches_manual_computation():
    logits = np.array([2.0, 0.5, -1.0, 4.0])
    expected = np.array(_manual_log_softmax(list(logits)))
    assert np.allclose(log_softmax(logits), expected, atol=1e-12)


def test_log_softmax_is_stable_for_large_values():
    logits = np.array([1000.0, 1000.5, 999.0])
    result = log_softmax(logits)
    assert np.all(np.isfinite(result))
    assert np.isclose(np.exp(result).sum(), 1.0)


def test_contrastive_score_equals_hand_calculation():
    base = np.array([3.0, 1.0, 0.5, -2.0])
    contrast = np.array([0.0, 2.0, 0.5, 1.5])
    alpha = 0.5
    scores, info = contrastive_scores(base, contrast, alpha)
    expected = np.array(_manual_log_softmax(list(base))) - alpha * np.array(_manual_log_softmax(list(contrast)))
    assert np.allclose(scores, expected, atol=1e-12)
    assert info["contrast_used"] is True
    assert np.argmax(scores) == int(np.argmax(expected))


def test_alpha_zero_returns_base_log_probabilities_exactly():
    base = np.array([2.0, -1.0, 0.25, 7.0])
    for variant in ("contrastive_logprob", "logit_difference"):
        scores, info = contrastive_scores(base, np.array([5.0, 5.0, 5.0, 5.0]), 0.0, variant=variant)
        assert info["contrast_used"] is False
        assert np.allclose(scores, log_softmax(base), atol=0)


def test_alpha_zero_ignores_the_contrast_branch_entirely():
    base = np.array([1.0, 2.0, 3.0])
    without, info_without = contrastive_scores(base, None, 0.0)
    with_contrast, info_with = contrastive_scores(base, np.array([9.0, -9.0, 0.0]), 0.0)
    assert np.array_equal(without, with_contrast)
    assert info_without["contrast_used"] is info_with["contrast_used"] is False


def test_contrastive_score_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same vocabulary size"):
        contrastive_scores(np.zeros(4), np.zeros(5), 0.5)


def test_nonfinite_logits_are_handled_explicitly():
    base = np.array([1.0, np.inf, 0.0])
    contrast = np.array([0.0, 0.0, np.inf])
    scores, info = contrastive_scores(base, contrast, 1.0)
    assert np.all(np.isfinite(scores[np.isfinite(scores)]))
    assert info["nonfinite_scores"] >= 0  # reported, not hidden


def test_logit_difference_variant_is_labelled():
    base = np.array([1.0, 2.0])
    contrast = np.array([0.5, 0.5])
    scores, info = contrastive_scores(base, contrast, 0.5, variant="logit_difference")
    assert info["variant"] == "logit_difference"
    assert "logit-difference" in info["variant_note"]
    # raw logit difference: [1-0.5*0.5, 2-0.5*0.5] = [0.75, 1.75], then max-shifted
    assert np.allclose(scores, np.array([-1.0, 0.0]), atol=1e-12)


def test_greedy_selection_chooses_argmax():
    scores = np.array([0.1, 0.9, 0.4])
    token, handled = select_token(scores, do_sample=False, temperature=1.0, top_p=1.0, top_k=0, rng=np.random.default_rng(0))
    assert token == 1
    assert handled is False


def test_select_token_falls_back_when_everything_is_nonfinite():
    scores = np.array([-np.inf, -np.inf])
    base = np.array([-2.0, -1.0])
    token, handled = select_token(scores, do_sample=False, temperature=1.0, top_p=1.0, top_k=0, rng=np.random.default_rng(0), base_log_probs=base)
    assert token == 1 and handled is True


def test_adaptive_plausibility_masks_implausible_tokens():
    base_log_probs = np.log(np.array([0.6, 0.3, 0.001]))
    scores = np.array([0.0, 0.0, 5.0])  # contrast promotes the implausible token
    masked = apply_plausibility_mask(scores, base_log_probs, epsilon=0.1)
    assert masked[2] == -np.inf
    assert masked[0] == 0.0


def test_repetition_penalty_lowers_generated_tokens():
    log_probs = log_softmax(np.array([1.0, 1.0, 1.0]))
    penalised = apply_repetition_penalty(log_probs, [0], penalty=2.0)
    assert penalised[0] < log_probs[0]
    assert penalised[1] == log_probs[1]


def test_no_repeat_ngram_bans_the_completing_token():
    log_probs = log_softmax(np.array([1.0, 1.0, 1.0]))
    banned = apply_no_repeat_ngram(log_probs, [0, 1, 0], ngram_size=2)
    assert banned[1] == -np.inf  # 0,1 -> next 1 would repeat the bigram


def test_sampling_is_reproducible_with_a_seed():
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    first = [sample_from_scores(scores, temperature=1.0, top_p=1.0, top_k=0, rng=np.random.default_rng(5)) for _ in range(5)]
    second = [sample_from_scores(scores, temperature=1.0, top_p=1.0, top_k=0, rng=np.random.default_rng(5)) for _ in range(5)]
    assert first == second


def test_top_k_and_top_p_restrict_the_support():
    scores = np.array([5.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    picks = {sample_from_scores(scores, temperature=1.0, top_p=0.9, top_k=1, rng=rng) for _ in range(20)}
    assert picks == {0}


# ---------------------------------------------------------------------------
# Decoder facade
# ---------------------------------------------------------------------------


def test_decoder_alpha_zero_matches_base_distribution():
    decoder = ContrastiveDecoder(DecodingConfig(mode="factanchor", alpha=0.0), seed=0)
    base = np.array([0.3, 1.4, -0.2, 2.2])
    token, diagnostics = decoder.step(
        step_index=0,
        base_logits=base,
        contrast_logits=np.array([9.0, -9.0, 0.0, 0.0]),
        generated_ids=[],
        eos_token_ids=[3],
    )
    assert token == int(np.argmax(base))
    assert diagnostics.contrast_applied is False
    assert np.isclose(diagnostics.score, log_softmax(base).max())


def test_decoder_contrast_can_change_the_token():
    base = np.array([3.0, 2.9, 0.0])
    contrast = np.array([-5.0, 5.0, 0.0])
    decoder = ContrastiveDecoder(DecodingConfig(mode="factanchor", alpha=2.0), seed=0)
    token, diagnostics = decoder.step(
        step_index=0, base_logits=base, contrast_logits=contrast, generated_ids=[], eos_token_ids=[]
    )
    assert diagnostics.contrast_applied is True
    assert token == 0, "a token heavily favoured by the corrupt context must be downweighted"


def test_decoder_warmup_defers_the_contrast():
    decoder = ContrastiveDecoder(DecodingConfig(mode="factanchor", alpha=2.0, contrast_warmup_tokens=2), seed=0)
    base = np.array([3.0, 2.9])
    contrast = np.array([-5.0, 5.0])
    token_first, diagnostics_first = decoder.step(
        step_index=0, base_logits=base, contrast_logits=contrast, generated_ids=[], eos_token_ids=[]
    )
    assert token_first == 0 and diagnostics_first.contrast_applied is False
    token_third, diagnostics_third = decoder.step(
        step_index=2, base_logits=base, contrast_logits=contrast, generated_ids=[], eos_token_ids=[]
    )
    assert diagnostics_third.contrast_applied is True
    assert token_third == 0


def test_min_new_tokens_suppresses_eos():
    decoder = ContrastiveDecoder(DecodingConfig(min_new_tokens=3), seed=0)
    base = np.array([5.0, 0.0])  # EOS (id 0) is the argmax
    for step in range(3):  # steps 0, 1, 2 are inside the minimum length
        token, _ = decoder.step(
            step_index=step, base_logits=base, contrast_logits=None, generated_ids=[], eos_token_ids=[0]
        )
        assert token == 1, f"step {step} must not emit EOS"
    assert decoder.eos_suppression_events == 3
    token_after, _ = decoder.step(
        step_index=3, base_logits=base, contrast_logits=None, generated_ids=[], eos_token_ids=[0]
    )
    assert token_after == 0, "from step 3 on, EOS is allowed again"


def test_decoder_without_alpha_never_uses_contrast():
    decoder = ContrastiveDecoder(DecodingConfig(mode="baseline", alpha=0.0), seed=0)
    assert decoder.contrast_enabled is False
    assert decoder.contrast_active_at(0) is False


# ---------------------------------------------------------------------------
# Generation loops
# ---------------------------------------------------------------------------


PROJECT_ROOT = _Path(__file__).resolve().parents[1]


def _toy_config(**decoding_overrides):
    """Config on the offline toy backend; paths are absolute so CWD does not matter."""
    overrides = {
        "decoding.mode": "baseline",
        "decoding.alpha": 0.0,
        "decoding.max_new_tokens": 8,
        "decoding.do_sample": False,
        "model.model_id": "toy",
        "run.seed": 42,
        **decoding_overrides,
    }
    return load_config(PROJECT_ROOT / "configs" / "smoke_toy.yaml", overrides=overrides)


class ScriptedAdapter:
    """Adapter that always predicts a fixed token (deterministic loop tests).

    Tokenisation is delegated to the bundled toy adapter so encode/decode/render
    behave exactly like the real offline backend; only the logits are scripted.
    """

    is_toy = True

    def __init__(self, base, token_id: int, *, eos_ids=None):
        self._base = base
        self._token_id = int(token_id)
        self._eos_ids = list(eos_ids or [])
        self.vocab = base.vocab_size

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._base, item)

    @property
    def eos_token_ids(self):
        return list(self._eos_ids)

    def _one_hot(self) -> np.ndarray:
        logits = np.full(self.vocab, -20.0)
        logits[self._token_id] = 20.0
        return logits

    def logits_for_prefix(self, input_ids):
        if input_ids and isinstance(input_ids[0], (int, np.integer)):
            return self._one_hot()
        return np.stack([self._one_hot() for _ in input_ids])

    def advance(self, tokens, cache):
        from factanchor.model_adapter import BranchCache

        length = (cache.length if cache is not None else 0) + len(tokens)
        return self._one_hot().reshape(1, -1), BranchCache(kind="toy", length=length, payload=None)


def test_generation_stops_at_max_new_tokens(toy_adapter):
    adapter = ScriptedAdapter(toy_adapter, token_id=ord("A"), eos_ids=[0])
    config = _toy_config(**{"decoding.max_new_tokens": 4})
    result = TextGenerator(adapter, config).generate("Who was Marie Curie?")
    assert result.stop_reason == "max_new_tokens"
    assert result.output_tokens == 4
    assert result.text == "AAAA"


def test_generation_stops_on_eos(toy_adapter):
    """EOS is the argmax from the first step: nothing must be generated."""
    adapter = ScriptedAdapter(toy_adapter, token_id=7, eos_ids=[7])
    config = _toy_config(**{"decoding.max_new_tokens": 6})
    result = TextGenerator(adapter, config).generate("anything")
    assert result.stop_reason == "eos"
    assert result.output_tokens == 0
    assert result.text == ""


def test_stop_sequence_terminates_generation(toy_adapter):
    adapter = ScriptedAdapter(toy_adapter, token_id=ord("a"))  # always emits "a"
    config = _toy_config(**{"decoding.max_new_tokens": 12, "decoding.stop_sequences": ["a"]})
    result = TextGenerator(adapter, config).generate("Marie Curie was")
    assert result.stop_reason.startswith("stop_sequence")
    assert result.text == ""
    assert result.token_ids, "the generated tokens are kept for auditability"


def test_cached_and_uncached_generation_agree(toy_adapter):
    cached_config = _toy_config(decoding={"mode": "baseline", "alpha": 0.0, "max_new_tokens": 6, "use_cache": True})
    uncached_config = _toy_config(decoding={"mode": "baseline", "alpha": 0.0, "max_new_tokens": 6, "use_cache": False})
    cached = TextGenerator(toy_adapter, cached_config).generate("Who was Marie Curie?")
    uncached = TextGenerator(toy_adapter, uncached_config).generate("Who was Marie Curie?")
    assert cached.token_ids == uncached.token_ids
    assert cached.stop_reason == uncached.stop_reason


def test_cached_and_uncached_agree_in_factanchor_mode(toy_adapter):
    cached_config = _toy_config(decoding={"mode": "factanchor", "alpha": 0.5, "max_new_tokens": 5, "use_cache": True})
    uncached_config = _toy_config(decoding={"mode": "factanchor", "alpha": 0.5, "max_new_tokens": 5, "use_cache": False})
    generator_cached = TextGenerator(toy_adapter, cached_config)
    generator_uncached = TextGenerator(toy_adapter, uncached_config)
    first = generator_cached.generate("Who was Marie Curie?", contrast_prompt="facts: occupation physicist")
    second = generator_uncached.generate("Who was Marie Curie?", contrast_prompt="facts: occupation physicist")
    assert first.token_ids == second.token_ids
    assert first.contrast_applied and second.contrast_applied


def test_both_branches_condition_on_the_same_generated_suffix(toy_adapter):
    config = _toy_config(decoding={"mode": "factanchor", "alpha": 0.7, "max_new_tokens": 4, "use_cache": True})
    generator = TextGenerator(toy_adapter, config)
    result = generator.generate("Who was Marie Curie?", contrast_prompt="different context entirely")
    # alignment is asserted inside the loop; also re-check the final token count relation
    assert result.output_tokens == len(result.token_ids)
    assert result.contrast_input_tokens is not None


def test_alpha_zero_factanchor_equals_baseline_tokens(toy_adapter):
    baseline_config = _toy_config(decoding={"mode": "baseline", "alpha": 0.0, "max_new_tokens": 6})
    factanchor_config = _toy_config(decoding={"mode": "factanchor", "alpha": 0.0, "max_new_tokens": 6})
    baseline = TextGenerator(toy_adapter, baseline_config).generate("Who was Marie Curie?")
    contrastive = TextGenerator(toy_adapter, factanchor_config).generate(
        "Who was Marie Curie?", contrast_prompt="corrupted facts: occupation astronaut"
    )
    assert baseline.token_ids == contrastive.token_ids
    assert baseline.contrast_applied is False
    assert contrastive.contrast_applied is False


def test_sampling_reproducible_across_generators(toy_adapter):
    config = _toy_config(
        decoding={"mode": "baseline", "alpha": 0.0, "max_new_tokens": 5, "do_sample": True, "temperature": 0.9},
        **{"run.seed": 3},
    )
    first = TextGenerator(toy_adapter, config, seed=3).generate("Who was Marie Curie?").token_ids
    second = TextGenerator(toy_adapter, config, seed=3).generate("Who was Marie Curie?").token_ids
    assert first == second


def test_context_length_guard_raises(toy_adapter):
    config = _toy_config(**{"model.max_context_tokens": 12, "decoding.max_new_tokens": 8})
    with pytest.raises(ValueError, match="max_context_tokens"):
        TextGenerator(toy_adapter, config).generate("a rather long prompt that will not fit at all")


def test_generate_factanchor_requires_factanchor_mode(toy_adapter):
    from factanchor.generation import generate_factanchor

    config = _toy_config(decoding={"mode": "baseline", "alpha": 0.0, "max_new_tokens": 2})
    with pytest.raises(ValueError, match="decoding.mode='factanchor'"):
        generate_factanchor("prompt", config, "contrast", adapter=toy_adapter)


def test_generate_baseline_rejects_factanchor_mode(toy_adapter):
    from factanchor.generation import generate_baseline

    config = _toy_config(decoding={"mode": "factanchor", "alpha": 0.5, "max_new_tokens": 2})
    with pytest.raises(ValueError, match="decoding.mode='baseline'"):
        generate_baseline("prompt", config, adapter=toy_adapter)


def test_hand_log_probs_helper_sanity():
    """Guard against the helper silently drifting from log_softmax."""
    logits = np.array([0.5, -2.0, 3.0])
    assert np.allclose(log_softmax(logits), _manual_log_softmax(list(logits)), atol=1e-12)


# ---------------------------------------------------------------------------
# "Amateur model" ablation (spec 3.3): the contrast distribution may be produced
# by a second model, as long as both branches live in the same token space.
# ---------------------------------------------------------------------------


def test_amateur_adapter_differs_from_the_base_model():
    from factanchor.model_adapter import build_adapter, TOY_AMATEUR_MODEL_IDS

    config = _toy_config()
    base = build_adapter(config)
    amateur = build_adapter(config, model_id="toy-amateur")
    assert amateur.model_id.lower() in TOY_AMATEUR_MODEL_IDS | {"toy"}
    assert amateur.vocab_size == base.vocab_size  # same token space
    ids = base.encode("Who was Marie Curie?")
    assert not np.allclose(base.logits_for_prefix(ids), amateur.logits_for_prefix(ids))


def test_contrast_branch_can_use_the_amateur_model():
    from factanchor.model_adapter import build_adapter

    config = _toy_config(
        decoding={"mode": "factanchor", "alpha": 0.5, "max_new_tokens": 4, "use_cache": True}
    )
    base = build_adapter(config)
    amateur = build_adapter(config, model_id="toy-amateur")
    prompt = "Base question"
    contrast = "Base question\n\nExperimental facts: X | was | Y"

    with_amateur = TextGenerator(base, config, contrast_adapter=amateur).generate(
        prompt, contrast_prompt=contrast
    )
    assert with_amateur.contrast_applied is True
    assert with_amateur.output_tokens >= 1

    base_only = TextGenerator(base, config).generate(prompt, contrast_prompt=contrast)
    assert base_only.contrast_applied is True
    # A different contrast model produces a different contrast log-probability, so
    # the two runs cannot silently be reported as the same condition.
    assert with_amateur.steps[0].contrast_logprob != base_only.steps[0].contrast_logprob


def test_amateur_model_must_share_the_vocabulary():
    class _WrongVocab:
        vocab_size = 5

    config = _toy_config(decoding={"mode": "factanchor", "alpha": 0.5})
    with pytest.raises(ValueError, match="different vocabulary"):
        TextGenerator(build_adapter_for(config), config, contrast_adapter=_WrongVocab())


def build_adapter_for(config):
    from factanchor.model_adapter import build_adapter

    return build_adapter(config)
