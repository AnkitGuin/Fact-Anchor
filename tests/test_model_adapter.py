"""Model-adapter tests: toy transformer correctness, determinism, KV cache."""

from __future__ import annotations

import numpy as np
import pytest

from factanchor.model_adapter import (
    TOY_BOS_TOKEN_ID,
    TOY_EOS_TOKEN_ID,
    BranchCache,
    ToyCausalLMAdapter,
    ToyTransformer,
    ToyTransformerConfig,
    build_adapter,
    log_softmax,
)


# ---------------------------------------------------------------------------
# Toy transformer internals
# ---------------------------------------------------------------------------


def test_toy_weights_are_deterministic_for_a_seed():
    first = ToyTransformer(ToyTransformerConfig(seed=5))
    second = ToyTransformer(ToyTransformerConfig(seed=5))
    third = ToyTransformer(ToyTransformerConfig(seed=6))
    assert np.array_equal(first.tok_emb, second.tok_emb)
    assert not np.array_equal(first.tok_emb, third.tok_emb)


def test_toy_forward_shapes_and_finiteness():
    model = ToyTransformer()
    logits, cache = model.forward([1, 2, 3])
    assert logits.shape == (model.config.vocab_size,)
    assert np.all(np.isfinite(logits))
    assert cache["length"] == 3
    assert len(cache["keys"]) == model.config.n_layers


def test_toy_transformer_rejects_bad_input():
    model = ToyTransformer()
    with pytest.raises(ValueError, match="empty token sequence"):
        model.forward([])
    with pytest.raises(ValueError, match="within"):
        model.forward([model.config.vocab_size + 1])
    with pytest.raises(ValueError, match="position"):
        model.forward([1] * 10, None, start_position=model.config.max_position)


def test_toy_transformer_is_causal():
    """Changing a later token must not change an earlier position's logits."""
    model = ToyTransformer()
    tokens = [10, 11, 12, 13]
    full, _ = model.forward(tokens)
    prefix_logits, _ = model.forward(tokens[:2])
    # recompute the logits at position 1 with an extra token appended
    shifted, _ = model.forward([10, 11, 99])
    assert full.shape == shifted.shape
    assert np.allclose(prefix_logits, model.forward([10, 11], None)[0])
    assert not np.allclose(prefix_logits, shifted), "later tokens must not affect earlier positions"


def test_toy_transformer_kv_cache_matches_uncached_forward():
    model = ToyTransformer()
    tokens = [4, 5, 6, 7, 8]
    uncached, _ = model.forward(tokens)
    _logits_prefix, cache = model.forward(tokens[:3])
    cached, _ = model.forward(tokens[3:], cache, start_position=3)
    assert np.allclose(uncached, cached, atol=1e-9)


# ---------------------------------------------------------------------------
# Adapter surface
# ---------------------------------------------------------------------------


def test_adapter_identity_and_tokens(toy_adapter):
    assert toy_adapter.model_id == "toy"
    assert toy_adapter.is_toy is True
    assert toy_adapter.vocab_size == 258
    assert toy_adapter.eos_token_ids == [TOY_EOS_TOKEN_ID]
    assert toy_adapter.context_length >= 256


def test_adapter_encode_decode_roundtrip(toy_adapter):
    assert toy_adapter.encode("abc") == [97, 98, 99]
    assert toy_adapter.decode([97, 98, 99]) == "abc"
    assert toy_adapter.encode("x", add_special_tokens=True)[0] == TOY_BOS_TOKEN_ID
    assert toy_adapter.decode([TOY_EOS_TOKEN_ID], skip_special_tokens=True) == ""
    assert toy_adapter.render("hello") == "hello"
    assert toy_adapter.render("hello", system_prompt="sys") == "sys\n\nhello"


def test_adapter_handles_multibyte_characters(toy_adapter):
    text = "café — 1967"
    ids = toy_adapter.encode(text)
    assert all(0 <= token < 256 for token in ids)
    assert toy_adapter.decode(ids) == text


def test_adapter_rejects_chat_template_for_the_toy_model():
    with pytest.raises(RuntimeError, match="chat template"):
        ToyCausalLMAdapter(use_chat_template=True)


def test_adapter_logits_for_prefix_single_and_batch(toy_adapter):
    single = toy_adapter.logits_for_prefix([1, 2, 3])
    batched = toy_adapter.logits_for_prefix([[1, 2, 3], [4, 5, 6, 7]])
    single_direct = toy_adapter.logits_for_prefix([[1, 2, 3]])
    assert single.shape == (toy_adapter.vocab_size,)
    assert batched.shape == (2, toy_adapter.vocab_size)
    assert np.allclose(single, single_direct[0])


def test_adapter_advance_increments_cache_length(toy_adapter):
    logits, cache = toy_adapter.advance([1, 2, 3], None)
    assert logits.shape == (1, toy_adapter.vocab_size)
    assert cache.length == 3
    logits_two, cache_two = toy_adapter.advance([4], cache)
    assert cache_two.length == 4
    assert np.allclose(toy_adapter.advance([1, 2, 3, 4], None)[0][0], logits_two[0], atol=1e-9)


def test_adapter_detects_a_drifted_cache(toy_adapter):
    _logits, cache = toy_adapter.advance([1, 2], None)
    cache.length = 5  # simulate drift between branches
    with pytest.raises(RuntimeError, match="drifted|disagrees"):
        toy_adapter.advance([3], cache)


def test_adapter_advance_rejects_empty_tokens(toy_adapter):
    with pytest.raises(ValueError, match="empty token list"):
        toy_adapter.advance([], None)


def test_build_adapter_selects_the_toy_backend():
    from factanchor.config import ModelConfig

    adapter = build_adapter(ModelConfig(model_id="toy"), device="cpu", dtype="float32")
    assert adapter.is_toy
    adapter.close()


def test_log_softmax_rejects_empty_input():
    with pytest.raises(ValueError, match="empty array"):
        log_softmax(np.array([]))


def test_log_softmax_handles_all_negative_infinity():
    result = log_softmax(np.array([-np.inf, -np.inf]))
    assert np.all(np.isneginf(result))


def test_branch_cache_clear():
    cache = BranchCache(kind="toy", length=3, payload={"a": 1})
    cache.clear()
    assert cache.length == 0 and cache.payload is None and cache.kind == "none"
