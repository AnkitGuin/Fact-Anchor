"""Model adapters: next-token logits for one or two contexts (spec Section 10.1).

Two interchangeable backends implement the same interface:

``HFCausalLMAdapter``
    Any Hugging Face causal LM (``AutoModelForCausalLM``), with left padding,
    attention masks, the tokenizer's chat template and a KV cache.
``ToyCausalLMAdapter``
    A small, randomly initialised transformer implemented in NumPy with a
    byte-level tokenizer and a real KV cache.  It needs no torch/no download, so
    the entire pipeline (including cache-vs-uncached equivalence tests) runs
    offline.  Its outputs are **not** meaningful language -- every record it
    produces is flagged ``is_toy_model=True``.

Interface contract used by :mod:`factanchor.decoding`:

``logits_for_prefix(input_ids)``
    Recompute logits for the last position of one sequence ``list[int]``
    (returns shape ``(V,)``) or of a batch ``list[list[int]]`` (returns ``(B, V)``).
    Padding is handled internally via an attention mask.
``advance(tokens, cache)``
    Incremental step: feed the tokens not yet in ``cache`` and return the logits
    of the last one plus the updated cache.  ``cache=None`` means "start from
    scratch with these tokens".
``render(text)`` / ``encode(text)`` / ``decode(ids)``
    Tokenizer access, including the model's required chat template.

Both branches of FactAnchor always use the *same* adapter instance, dtype and
device, so the contrast is never confounded by a second model (spec 10.4).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from .config import Config, ModelConfig
from .utils import get_logger

__all__ = [
    "BranchCache",
    "ModelAdapter",
    "HFCausalLMAdapter",
    "ToyCausalLMAdapter",
    "build_adapter",
    "resolve_device",
    "resolve_dtype",
    "log_softmax",
    "TOY_MODEL_ID",
    "TOY_BOS_TOKEN_ID",
    "TOY_EOS_TOKEN_ID",
    "TOY_VOCAB_SIZE",
]

logger = get_logger("model_adapter")

TOY_MODEL_ID = "toy"
# "Amateur model" ablation (spec 3.3): a second toy instance with different weights
# but the same byte vocabulary, so p_c is a different distribution in the same token
# space as p_b.
TOY_AMATEUR_MODEL_IDS = {"toy-amateur", "toy_amateur"}
TOY_AMATEUR_SEED = 7
TOY_VOCAB_SIZE = 258  # 256 byte values + BOS + EOS
TOY_BOS_TOKEN_ID = 256
TOY_EOS_TOKEN_ID = 257


# ---------------------------------------------------------------------------
# Numerics shared by both backends
# ---------------------------------------------------------------------------


def log_softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax.

    Subtracts the max before exponentiating (spec 3.2) and returns ``-inf`` for
    positions that are ``-inf`` on input instead of producing NaNs.
    """
    array = np.asarray(logits, dtype=np.float64)
    if array.size == 0:
        raise ValueError("log_softmax received an empty array")
    # +inf logits mean "certain": substitute the largest finite float so the
    # subtraction below cannot produce inf - inf = NaN.
    array = np.where(np.isposinf(array), np.finfo(np.float64).max, array)
    finite = np.isfinite(array)
    any_finite = finite.any(axis=axis, keepdims=True)
    safe = np.where(finite, array, -np.inf)
    row_max = np.max(safe, axis=axis, keepdims=True)
    row_max = np.where(any_finite, row_max, 0.0)
    shifted = np.where(any_finite, array - row_max, 0.0)
    exp = np.exp(np.where(any_finite, shifted, 0.0))
    total = np.sum(exp, axis=axis, keepdims=True)
    out = shifted - np.log(np.clip(total, 1e-300, None))
    # rows with no finite logit at all keep -inf (they are genuinely impossible),
    # minus the offset introduced by the machine epsilon above
    out = np.where(finite, out, -np.inf)
    return np.where(any_finite, out, -np.inf)


def _softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass
class BranchCache:
    """Opaque per-branch cache (KV cache for real models; empty for toy/uncached).

    ``length`` is always the number of tokens the cache already represents, which
    lets callers detect misalignment between the two branches (a bug that would
    silently break the contrast -- asserted in tests).
    """

    kind: Literal["none", "torch", "toy"] = "none"
    length: int = 0
    payload: Any = None

    def clear(self) -> None:
        self.kind = "none"
        self.length = 0
        self.payload = None


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class ModelAdapter(ABC):
    """Common interface for next-token logits (spec 10.1)."""

    is_toy: bool = False

    def __init__(self) -> None:
        self.load_time_s: float = 0.0

    # -- identity -----------------------------------------------------------
    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def revision(self) -> str | None: ...

    @property
    @abstractmethod
    def tokenizer_id(self) -> str: ...

    @property
    @abstractmethod
    def tokenizer_revision(self) -> str | None: ...

    @property
    @abstractmethod
    def device(self) -> str: ...

    @property
    @abstractmethod
    def dtype(self) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def eos_token_ids(self) -> list[int]: ...

    @property
    def context_length(self) -> int:
        """Maximum number of tokens the adapter can accept."""
        return 1_000_000

    # -- text ---------------------------------------------------------------
    @abstractmethod
    def render(self, text: str, *, system_prompt: str | None = None) -> str:
        """Apply the model's chat template (when configured) to a raw prompt."""

    @abstractmethod
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    @abstractmethod
    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str: ...

    # -- inference ----------------------------------------------------------
    @abstractmethod
    def logits_for_prefix(self, input_ids: Sequence[int] | Sequence[Sequence[int]]) -> np.ndarray:
        """Uncached logits for the last position of each sequence (float64 numpy)."""

    @abstractmethod
    def advance(self, tokens: Sequence[int], cache: BranchCache | None) -> tuple[np.ndarray, BranchCache]:
        """Incremental forward pass; returns ``(logits (1, V), new_cache)``."""

    def new_cache(self) -> BranchCache:
        """Create an empty cache appropriate for this adapter."""
        return BranchCache(kind="torch" if not self.is_toy else "toy", length=0, payload=None)

    # -- housekeeping -------------------------------------------------------
    def peak_memory_mb(self) -> float | None:
        return None

    def close(self) -> None:  # pragma: no cover - trivial for both backends
        return None


# ---------------------------------------------------------------------------
# Device / dtype resolution
# ---------------------------------------------------------------------------


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to the best available device **without** silently changing it.

    ``auto`` -> cuda:0 if CUDA is available, else mps, else cpu.  An explicit
    request that is unavailable raises, so a run never quietly ends up on a
    different device than the one recorded in its manifest (spec 10.4).
    """
    import torch  # local import: keeps the toy backend torch-free

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={requested!r} requested but CUDA is not available. "
                "Use run.device='cpu' (or 'auto') -- FactAnchor will not silently downgrade the device."
            )
        return requested
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("device='mps' requested but the MPS backend is unavailable")
        return requested
    return "cpu"


def resolve_dtype(requested: str, device: str) -> str:
    """Resolve ``auto`` to float32 on CPU and bfloat16/float16 on accelerators."""
    import torch

    if requested != "auto":
        return requested
    if device.startswith("cuda"):
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if device == "mps":
        return "float16"
    return "float32"  # CPU: float32 keeps greedy decoding reproducible


# ---------------------------------------------------------------------------
# Hugging Face backend
# ---------------------------------------------------------------------------


class HFCausalLMAdapter(ModelAdapter):
    """Adapter around ``transformers.AutoModelForCausalLM``."""

    is_toy = False

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        tokenizer_id: str | None = None,
        tokenizer_revision: str | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
        add_generation_prompt: bool = True,
        load_in_8bit: bool = False,
        attn_implementation: str | None = None,
        low_cpu_mem_usage: bool = True,
        local_files_only: bool = False,
        eager_eval: bool = True,
    ) -> None:
        super().__init__()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "torch/transformers are not installed. Install them with "
                "`pip install -e .[hf]` (or use model.model_id='toy' for the offline backend)."
            ) from exc

        self._model_id = model_id
        self._revision = revision
        self._tokenizer_id = tokenizer_id or model_id
        self._tokenizer_revision = tokenizer_revision or revision
        self._device = device
        self._dtype = dtype
        self.use_chat_template = use_chat_template
        self.add_generation_prompt = add_generation_prompt
        self._torch = torch

        if load_in_8bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "model.load_in_8bit=true requires bitsandbytes, which is not installed. "
                    "Install it or set load_in_8bit=false (the run will not silently change precision)."
                ) from exc

        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }[dtype]

        logger.info(
            "loading model %s (revision=%s) on %s in %s", model_id, revision, device, dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_id,
            revision=self._tokenizer_revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "torch_dtype": torch_dtype,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if load_in_8bit:
            # Quantisation is an explicit, recorded choice: it is never applied
            # silently, and both branches use the same quantised weights.
            from transformers import BitsAndBytesConfig

            model_kwargs.pop("torch_dtype", None)
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if not load_in_8bit:
            self.model.to(device)
        self._dtype = "int8" if load_in_8bit else dtype
        self.model.eval()
        if eager_eval:
            self._forbid_grad()

        # padding / special tokens
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:  # pragma: no cover - exotic tokenizer
                raise RuntimeError(
                    f"tokenizer {self._tokenizer_id} has neither a pad nor an eos token; "
                    "cannot batch safely"
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._pad_token_id = int(self.tokenizer.pad_token_id)
        self._eos_ids = self._collect_eos_ids()
        self._context_length = int(getattr(self.model.config, "max_position_embeddings", 4096) or 4096)
        self._warned_no_template = False
        logger.info(
            "model ready: %s params, vocab=%d, eos=%s, context=%d",
            _human_int(sum(p.numel() for p in self.model.parameters())),
            self.vocab_size,
            self._eos_ids,
            self._context_length,
        )

    def _forbid_grad(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _collect_eos_ids(self) -> list[int]:
        ids: list[int] = []
        eos = self.tokenizer.eos_token_id
        if isinstance(eos, int):
            ids.append(eos)
        elif isinstance(eos, (list, tuple)):
            ids.extend(int(i) for i in eos)
        generation_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(generation_eos, int):
            ids.append(generation_eos)
        elif isinstance(generation_eos, (list, tuple)):
            ids.extend(int(i) for i in generation_eos)
        # keep order, drop duplicates
        return list(dict.fromkeys(ids))

    # -- identity -----------------------------------------------------------
    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str | None:
        return self._revision

    @property
    def tokenizer_id(self) -> str:
        return self._tokenizer_id

    @property
    def tokenizer_revision(self) -> str | None:
        return self._tokenizer_revision

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    @property
    def eos_token_ids(self) -> list[int]:
        return list(self._eos_ids)

    @property
    def context_length(self) -> int:
        return self._context_length

    # -- text ---------------------------------------------------------------
    def render(self, text: str, *, system_prompt: str | None = None) -> str:
        if not self.use_chat_template:
            return text
        template = getattr(self.tokenizer, "chat_template", None)
        if not template:
            if not self._warned_no_template:
                logger.warning(
                    "model.use_chat_template=true but tokenizer %s has no chat template; "
                    "falling back to the raw prompt",
                    self._tokenizer_id,
                )
                self._warned_no_template = True
            return text
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
        )

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    # -- inference ----------------------------------------------------------
    def logits_for_prefix(self, input_ids: Sequence[int] | Sequence[Sequence[int]]) -> np.ndarray:
        """Uncached forward pass; supports single sequences and batches."""
        torch = self._torch
        single = bool(input_ids) and isinstance(input_ids[0], (int, np.integer))
        if not input_ids:
            raise ValueError("logits_for_prefix received an empty sequence")
        if single:
            batch = [list(int(i) for i in input_ids)]  # type: ignore[arg-type]
        else:
            batch = [list(int(i) for i in seq) for seq in input_ids]  # type: ignore[union-attr]
        if not batch or any(len(seq) == 0 for seq in batch):
            raise ValueError("logits_for_prefix received an empty sequence")
        longest = max(len(seq) for seq in batch)
        if longest > self._context_length:
            raise ValueError(
                f"input of {longest} tokens exceeds the model context length {self._context_length}"
            )
        padded = [[self._pad_token_id] * (longest - len(seq)) + seq for seq in batch]
        attention = [[0] * (longest - len(seq)) + [1] * len(seq) for seq in batch]
        input_tensor = torch.tensor(padded, dtype=torch.long, device=self._device)
        mask_tensor = torch.tensor(attention, dtype=torch.long, device=self._device)
        position_ids = mask_tensor.cumsum(dim=-1) - 1
        position_ids.masked_fill_(mask_tensor == 0, 0)
        with torch.inference_mode():
            output = self.model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = output.logits  # (B, T, V)
            # Inputs are left-padded, so every row's last real token is at the
            # final tensor position.  ``sum(mask)-1`` is a length, not an index
            # into a left-padded row, and selects the wrong token for short rows.
            last_index = torch.full(
                (mask_tensor.shape[0],), mask_tensor.shape[1] - 1,
                dtype=torch.long, device=self._device,
            )
            gathered = logits[torch.arange(logits.shape[0], device=self._device), last_index]
            array = gathered.float().cpu().numpy()
        if not np.all(np.isfinite(array)):
            # NaN/Inf can appear in low precision; make it visible immediately.
            logger.warning("non-finite logits in logits_for_prefix (dtype=%s)", self._dtype)
        return array[0] if single else array

    def advance(self, tokens: Sequence[int], cache: BranchCache | None) -> tuple[np.ndarray, BranchCache]:
        """Incremental decode step using ``past_key_values`` (real KV cache)."""
        torch = self._torch
        tokens = [int(t) for t in tokens]
        if not tokens:
            raise ValueError("advance received an empty token list")
        cache = cache or BranchCache(kind="torch", length=0, payload=None)
        if cache.payload is None:
            history: list[int] = []
            past = None
        else:
            history = list(cache.payload["input_ids"])
            past = cache.payload.get("past")
        positions_in_cache = len(history)
        if positions_in_cache != cache.length:
            raise RuntimeError(
                f"cache length {cache.length} disagrees with stored history {positions_in_cache}; "
                "branches have drifted out of alignment"
            )
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self._device)
        history.extend(tokens)
        mask = torch.ones((1, len(history)), dtype=torch.long, device=self._device)
        if len(history) > self._context_length:
            raise ValueError(
                f"sequence of {len(history)} tokens exceeds the model context length {self._context_length}"
            )
        with torch.inference_mode():
            output = self.model(
                input_ids=input_tensor,
                attention_mask=mask,
                past_key_values=past,
                use_cache=True,
            )
        array = output.logits[:, -1, :].float().cpu().numpy()
        if not np.all(np.isfinite(array)):
            logger.warning("non-finite logits in advance (dtype=%s)", self._dtype)
        new_cache = BranchCache(
            kind="torch",
            length=len(history),
            payload={"input_ids": history, "past": output.past_key_values},
        )
        return array, new_cache

    def peak_memory_mb(self) -> float | None:
        torch = self._torch
        if self._device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self._device) / (1024 * 1024)
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return usage / 1024.0  # ru_maxrss is in KB on Linux
        except Exception:  # pragma: no cover
            return None

    def close(self) -> None:
        torch = self._torch
        del self.model
        if self._device.startswith("cuda") and torch.cuda.is_available():  # pragma: no cover
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Offline NumPy backend
# ---------------------------------------------------------------------------


@dataclass
class ToyTransformerConfig:
    """Configuration of the bundled toy transformer."""

    vocab_size: int = TOY_VOCAB_SIZE
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    d_ff: int = 64
    max_position: int = 1024
    seed: int = 1234


class ToyTransformer:
    """A minimal causal transformer in NumPy with a real KV cache.

    It exists so that the *decoding algorithm* can be tested end-to-end without
    torch, GPUs or model downloads.  Weights are random but fixed by ``seed``, so
    identical inputs always give identical logits -- which is exactly what the
    reproducibility and cache-equivalence tests need.
    """

    def __init__(self, config: ToyTransformerConfig | None = None) -> None:
        self.config = config or ToyTransformerConfig()
        if self.config.d_model % self.config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self._init_weights()

    def _init_weights(self) -> None:
        rng = np.random.RandomState(self.config.seed)
        d, ff, vocab, heads = (
            self.config.d_model,
            self.config.d_ff,
            self.config.vocab_size,
            self.config.n_heads,
        )
        scale = 1.0 / math.sqrt(d)
        self.tok_emb = rng.normal(0.0, 0.02, size=(vocab, d))
        self.pos_emb = rng.normal(0.0, 0.01, size=(self.config.max_position, d))
        self.layers: list[dict[str, np.ndarray]] = []
        for _ in range(self.config.n_layers):
            self.layers.append(
                {
                    "ln1_g": np.ones(d),
                    "ln1_b": np.zeros(d),
                    "wq": rng.normal(0.0, scale, size=(d, d)),
                    "wk": rng.normal(0.0, scale, size=(d, d)),
                    "wv": rng.normal(0.0, scale, size=(d, d)),
                    "wo": rng.normal(0.0, scale, size=(d, d)),
                    "ln2_g": np.ones(d),
                    "ln2_b": np.zeros(d),
                    "w1": rng.normal(0.0, scale, size=(d, ff)),
                    "b1": np.zeros(ff),
                    "w2": rng.normal(0.0, scale, size=(ff, d)),
                    "b2": np.zeros(d),
                }
            )
        self.ln_f_g = np.ones(d)
        self.ln_f_b = np.zeros(d)
        self._head_scale = 1.0 / math.sqrt(heads)

    # -- numerics -----------------------------------------------------------
    @staticmethod
    def _layer_norm(x: np.ndarray, gain: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(variance + eps) * gain + bias

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        tokens: Sequence[int],
        cache: dict[str, Any] | None = None,
        *,
        start_position: int = 0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return ``(logits for the last token, new_cache)``.

        ``cache`` carries ``keys``/``values`` per layer and the number of tokens
        already processed.  Only tokens not yet in the cache are computed, which
        is what makes the cached and uncached paths comparable.
        """
        tokens = [int(t) for t in tokens]
        if not tokens:
            raise ValueError("toy transformer received an empty token sequence")
        if any(t < 0 or t >= self.config.vocab_size for t in tokens):
            raise ValueError(f"token ids must be within [0, {self.config.vocab_size})")
        if start_position + len(tokens) > self.config.max_position:
            raise ValueError(
                f"sequence length {start_position + len(tokens)} exceeds toy max_position="
                f"{self.config.max_position}"
            )
        index = np.asarray(tokens, dtype=np.int64)
        x = self.tok_emb[index] + self.pos_emb[start_position : start_position + len(tokens)]
        new_cache: dict[str, Any] = {
            "keys": [],
            "values": [],
            "length": start_position + len(tokens),
        }
        n_heads = self.config.n_heads
        d_head = self.config.d_model // n_heads
        for layer_index, layer in enumerate(self.layers):
            normed = self._layer_norm(x, layer["ln1_g"], layer["ln1_b"])
            query = normed @ layer["wq"]
            key = normed @ layer["wk"]
            value = normed @ layer["wv"]

            def split_heads(array: np.ndarray) -> np.ndarray:
                # (T, d) -> (heads, T, d_head)
                return array.reshape(array.shape[0], n_heads, d_head).transpose(1, 0, 2)

            query_h, key_h, value_h = split_heads(query), split_heads(key), split_heads(value)
            if cache is not None:
                past_keys = cache["keys"][layer_index]
                past_values = cache["values"][layer_index]
                if past_keys is not None and past_keys.shape[1] > 0:
                    key_all = np.concatenate([past_keys, key_h], axis=1)
                    value_all = np.concatenate([past_values, value_h], axis=1)
                else:
                    key_all, value_all = key_h, value_h
            else:
                key_all, value_all = key_h, value_h
            new_cache["keys"].append(key_all)
            new_cache["values"].append(value_all)

            scores = (query_h @ key_all.transpose(0, 2, 1)) * self._head_scale
            # causal mask: new query positions follow the cached ones
            total_keys = key_all.shape[1]
            new_len = query_h.shape[1]
            allowed = np.arange(total_keys)[None, :] <= (start_position + np.arange(new_len))[:, None]
            scores = np.where(allowed[None, :, :], scores, -np.inf)
            weights = _softmax(scores, axis=-1)
            weights = np.where(np.isfinite(weights), weights, 0.0)
            context = weights @ value_all  # (heads, T, d_head)
            merged = context.transpose(1, 0, 2).reshape(new_len, self.config.d_model)
            x = x + merged @ layer["wo"]
            normed2 = self._layer_norm(x, layer["ln2_g"], layer["ln2_b"])
            hidden = self._gelu(normed2 @ layer["w1"] + layer["b1"])
            x = x + hidden @ layer["w2"] + layer["b2"]
        x = self._layer_norm(x, self.ln_f_g, self.ln_f_b)
        logits = x[-1] @ self.tok_emb.T  # tied head
        return logits, new_cache

    def num_parameters(self) -> int:
        total = self.tok_emb.size + self.pos_emb.size
        for layer in self.layers:
            total += sum(parameter.size for parameter in layer.values())
        total += self.ln_f_g.size + self.ln_f_b.size
        return int(total)


class ToyCausalLMAdapter(ModelAdapter):
    """Offline adapter around :class:`ToyTransformer` with a byte tokenizer."""

    is_toy = True

    def __init__(
        self,
        *,
        use_chat_template: bool = False,
        seed: int = 1234,
        eager_eval: bool = True,  # kept for interface parity
    ) -> None:
        super().__init__()
        if use_chat_template:
            raise RuntimeError(
                "model.model_id='toy' has no chat template. Set model.use_chat_template=false "
                "(or pick a real model); the toy backend will not pretend to have one."
            )
        self.transformer = ToyTransformer(ToyTransformerConfig(seed=seed))
        logger.info(
            "toy model ready: %d parameters, vocab=%d (randomly initialised -- outputs are not meaningful text)",
            self.transformer.num_parameters(),
            self.transformer.config.vocab_size,
        )

    # -- identity -----------------------------------------------------------
    @property
    def model_id(self) -> str:
        return TOY_MODEL_ID

    @property
    def revision(self) -> str | None:
        return "bundled-numpy-seed-1234"

    @property
    def tokenizer_id(self) -> str:
        return "factanchor-byte-tokenizer"

    @property
    def tokenizer_revision(self) -> str | None:
        return "1"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def dtype(self) -> str:
        return "float64"

    @property
    def vocab_size(self) -> int:
        return self.transformer.config.vocab_size

    @property
    def eos_token_ids(self) -> list[int]:
        return [TOY_EOS_TOKEN_ID]

    @property
    def context_length(self) -> int:
        return self.transformer.config.max_position

    # -- text ---------------------------------------------------------------
    def render(self, text: str, *, system_prompt: str | None = None) -> str:
        if system_prompt:
            return f"{system_prompt}\n\n{text}"
        return text

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        payload = list(text.encode("utf-8"))
        required = len(payload) + int(add_special_tokens)
        if required > self.context_length:
            raise ValueError(
                f"toy prompt has {required} tokens, exceeding context length "
                f"{self.context_length}; refusing to truncate the prompt"
            )
        if add_special_tokens:
            return [TOY_BOS_TOKEN_ID, *payload]
        return payload

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        raw = bytes(int(t) for t in token_ids if 0 <= int(t) < 256)
        return raw.decode("utf-8", errors="replace")

    # -- inference ----------------------------------------------------------
    def logits_for_prefix(self, input_ids: Sequence[int] | Sequence[Sequence[int]]) -> np.ndarray:
        if not input_ids:
            raise ValueError("logits_for_prefix received an empty sequence")
        single = isinstance(input_ids[0], (int, np.integer))
        sequences = [list(int(t) for t in input_ids)] if single else [list(map(int, seq)) for seq in input_ids]  # type: ignore[arg-type]
        rows: list[np.ndarray] = []
        for sequence in sequences:
            if not sequence:
                raise ValueError("logits_for_prefix received an empty sequence")
            if len(sequence) > self.context_length:
                raise ValueError(
                    f"input of {len(sequence)} tokens exceeds toy context length "
                    f"{self.context_length}; refusing to truncate the prefix"
                )
            logits, _ = self.transformer.forward(sequence, None, start_position=0)
            rows.append(logits)
        stacked = np.stack(rows, axis=0)
        return stacked[0] if single else stacked

    def advance(self, tokens: Sequence[int], cache: BranchCache | None) -> tuple[np.ndarray, BranchCache]:
        tokens = [int(t) for t in tokens]
        if not tokens:
            raise ValueError("advance received an empty token list")
        cache = cache or BranchCache(kind="toy", length=0, payload=None)
        past = cache.payload
        expected = int(past["length"]) if past else 0
        if expected != cache.length:
            raise RuntimeError(
                f"toy cache length {cache.length} disagrees with its payload ({expected}); branches drifted"
            )
        logits, new_payload = self.transformer.forward(tokens, past, start_position=expected)
        return logits[None, :], BranchCache(kind="toy", length=new_payload["length"], payload=new_payload)

    def peak_memory_mb(self) -> float | None:
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:  # pragma: no cover
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_adapter(
    config: Config | ModelConfig,
    *,
    device: str | None = None,
    dtype: str | None = None,
    model_id: str | None = None,
    revision: str | None = None,
) -> ModelAdapter:
    """Instantiate the configured model backend.

    ``model_id='toy'`` selects the bundled NumPy transformer (offline).  Anything
    else is loaded through transformers; a missing dependency raises a readable
    error instead of silently falling back to another model (spec "Do not
    silently substitute ... unavailable model checkpoints").
    """
    model_config = config.model if isinstance(config, Config) else config
    requested_device = device if device is not None else (config.run.device if isinstance(config, Config) else "auto")
    requested_dtype = dtype if dtype is not None else (config.run.dtype if isinstance(config, Config) else "auto")

    requested_id = (model_id if model_id is not None else model_config.model_id).strip()
    if requested_id.lower() == TOY_MODEL_ID:
        return ToyCausalLMAdapter(use_chat_template=False)
    if requested_id.lower() in TOY_AMATEUR_MODEL_IDS:
        # Same byte vocabulary, different random weights: a genuinely different
        # distribution in the same token space, which is what the amateur-model
        # ablation needs.
        return ToyCausalLMAdapter(seed=TOY_AMATEUR_SEED, use_chat_template=False)

    resolved_device = resolve_device(requested_device)
    resolved_dtype = resolve_dtype(requested_dtype, resolved_device)
    if resolved_dtype == "float64" and not resolved_device.startswith("cuda"):
        logger.warning("float64 on CPU is supported but slow; consider run.dtype='float32'")
    return HFCausalLMAdapter(
        requested_id,
        revision=revision if revision is not None else model_config.revision,
        tokenizer_id=model_config.tokenizer_id,
        tokenizer_revision=model_config.tokenizer_revision,
        device=resolved_device,
        dtype=resolved_dtype,
        trust_remote_code=model_config.trust_remote_code,
        use_chat_template=model_config.use_chat_template,
        add_generation_prompt=model_config.add_generation_prompt,
        load_in_8bit=model_config.load_in_8bit,
        attn_implementation=model_config.attn_implementation,
        low_cpu_mem_usage=model_config.low_cpu_mem_usage,
        local_files_only=model_config.local_files_only,
    )


def _human_int(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)
