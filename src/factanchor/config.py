"""Typed, validated configuration for FactAnchor.

Design rules (spec Sections 10.3, 16 and "Required engineering principles"):

* Every parameter is validated at startup; **unknown keys raise an error**
  (``extra="forbid"``) so a typo can never be silently ignored.
* Ranges are enforced (``alpha >= 0``, ``0 <= corruption_rate <= 1``, ...).
* Configs may inherit from another config file through a top-level ``defaults``
  key; the child file wins.  ``defaults`` is resolved relative to the file that
  declares it, and cycles are detected.
* Paths are resolved relative to the *project root* so that scripts can be run
  from anywhere.

Nothing in this module imports torch/transformers, so config validation works
in a bare environment and inside the test suite.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "Config",
    "RunConfig",
    "ModelConfig",
    "DecodingConfig",
    "RetrievalConfig",
    "NerConfig",
    "CorruptionConfig",
    "ContextConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "ConfigError",
    "load_config",
    "deep_merge",
    "set_by_path",
    "get_by_path",
    "find_project_root",
    "CONFIG_SEARCH_PATHS",
]

CONFIG_SEARCH_PATHS: tuple[str, ...] = ("configs", ".")


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded or is invalid."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_project_root(start: Path | str | None = None) -> Path:
    """Return the project root (the directory containing ``pyproject.toml``).

    Falls back to the current working directory when no marker file is found,
    so the package is still usable when installed into site-packages.
    """
    here = Path(start) if start is not None else Path(__file__).resolve()
    here = here.parent if here.is_file() else here
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / "configs").is_dir():
            return candidate
    return Path.cwd()


def get_by_path(data: dict[str, Any], dotted: str) -> Any:
    """Fetch ``a.b.c`` from a nested dict; raises :class:`ConfigError` if absent."""
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"unknown configuration path: {dotted!r} (missing {part!r})")
        node = node[part]
    return node


def set_by_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``a.b.c = value`` in a nested dict, creating missing parents.

    Unknown *leaf* keys are allowed here on purpose: the value is validated
    afterwards by pydantic, which rejects genuinely unknown fields.
    """
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        nxt = node.get(part)
        if nxt is None:
            nxt = {}
            node[part] = nxt
        if not isinstance(nxt, dict):
            raise ConfigError(f"cannot set {dotted!r}: {part!r} is not a mapping")
        node = nxt
    node[parts[-1]] = value


def coerce_scalar(text: str) -> Any:
    """Best-effort conversion of a CLI string to bool/int/float/None/list/dict.

    Used by ``--set key=value`` overrides.  JSON is attempted first so that
    ``[1,2]`` and ``{"a": 1}`` work; plain words stay strings.

    Null is spelled ``null``/``~`` or an empty value -- *not* ``none``:
    configuration files use the literal string ``none`` (for example
    ``corruption.strategy: none``), and ``--set corruption.strategy=none`` must
    mean the same thing as the YAML.  Booleans accept ``true/yes/on`` (and the
    negations), matching YAML 1.1.
    """
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in {"null", "~", ""}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass
    for cast in (int, float):
        try:
            return cast(stripped)
        except ValueError:
            continue
    return stripped


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict).

    Lists and scalars are replaced, not concatenated -- except for
    ``stop_sequences``/``datasets``-style lists where replacement is the only
    behaviour that keeps an experiment's intent explicit.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - depends on file content
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"configuration root must be a mapping, got {type(raw).__name__} in {path}")
    return raw


def _resolve_config_path(name: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a config name/path against CWD, the given base dir, then configs/."""
    candidate = Path(name)
    if candidate.exists():
        return candidate.resolve()
    if base_dir is not None and (base_dir / candidate).exists():
        return (base_dir / candidate).resolve()
    root = find_project_root()
    for sub in CONFIG_SEARCH_PATHS:
        probe = root / sub / candidate
        if probe.exists():
            return probe.resolve()
    raise ConfigError(
        f"configuration file not found: {name!r} (tried {candidate}"
        + (f", {Path(base_dir) / str(name)}" if base_dir is not None else "")
        + f", {find_project_root() / 'configs' / str(name)})"
    )


def _load_raw(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML config, resolving an optional ``defaults`` inheritance chain."""
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(p) for p in (*seen, resolved))
        raise ConfigError(f"circular config inheritance: {chain}")
    raw = _read_yaml(resolved)
    parent_name = raw.pop("defaults", None)
    if parent_name is None:
        return raw
    if not isinstance(parent_name, (str, Path)):
        raise ConfigError(f"`defaults` must be a path string, got {type(parent_name).__name__}")
    parent_path = _resolve_config_path(parent_name, base_dir=resolved.parent)
    parent = _load_raw(parent_path, seen=(*seen, resolved))
    return deep_merge(parent, raw)


# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=False, frozen=False)


class RunConfig(_Section):
    """Run-level settings: seeds, output locations, hardware envelope."""

    name: str = "default"
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    output_dir: str = "artifacts"
    run_id: str | None = None
    overwrite: bool = False
    allow_run_dir_reuse: bool = False   # explicit opt-in to append into an existing run dir
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    device: str = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16", "float64"] = "auto"
    log_prompts: bool = True
    write_jsonl_per_example: bool = True
    deterministic_algorithms: bool = True
    save_debug_scores: bool = False
    git_hash: bool = True
    is_toy_model: bool = False  # set when using the bundled random-weight toy model
    tags: list[str] = Field(default_factory=list)

    @field_validator("device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        if not re.fullmatch(r"(auto|cpu|mps|cuda(:\d+)?)", value):
            raise ValueError(
                f"device must be one of auto|cpu|cuda|cuda:N|mps, got {value!r} "
                "(model and both contrast branches always use the same device)"
            )
        return value


class ModelConfig(_Section):
    """Model / tokenizer identity and loading envelope."""

    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    revision: str | None = "main"
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    use_chat_template: bool = True
    add_generation_prompt: bool = True
    max_context_tokens: int = Field(default=1024, ge=8, le=131072)
    input_side_padding: Literal["left", "right"] = "left"
    load_in_8bit: bool = False
    attn_implementation: str | None = None
    low_cpu_mem_usage: bool = True
    local_files_only: bool = False

    @field_validator("model_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("model.model_id must be a non-empty string")
        return value.strip()


class DecodingConfig(_Section):
    """Decoding policy.  ``alpha=0`` reproduces the baseline path exactly."""

    mode: Literal["baseline", "factanchor"] = "factanchor"
    variant: Literal["contrastive_logprob", "logit_difference"] = "contrastive_logprob"
    alpha: float = Field(default=0.5, ge=0.0, le=5.0)
    max_new_tokens: int = Field(default=64, ge=1, le=4096)
    min_new_tokens: int = Field(default=0, ge=0, le=4096)
    do_sample: bool = False
    temperature: float = Field(default=0.7, gt=0.0, le=10.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0, le=200000)
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=10.0)
    no_repeat_ngram_size: int = Field(default=0, ge=0, le=20)
    use_cache: bool = True
    stop_sequences: list[str] = Field(default_factory=list)
    include_prompt_in_output: bool = False
    adaptive_plausibility: bool = False
    plausibility_epsilon: float = Field(default=0.1, gt=0.0, le=1.0)
    normalize_scores: bool = False
    contrast_warmup_tokens: int = Field(default=0, ge=0, le=4096)
    on_missing_contrast: Literal["skip_contrast", "empty_block"] = "skip_contrast"
    # "Amateur model" ablation (spec 3.3): when set, the *contrast* distribution is
    # produced by this model instead of the base model.  Both models must share a
    # token vocabulary -- cross-tokenizer contrast is not implemented and raises a
    # readable error rather than silently comparing incomparable logits.
    amateur_model_id: str | None = None

    @model_validator(mode="after")
    def _cross_checks(self) -> DecodingConfig:
        if self.mode == "baseline" and self.alpha != 0.0:
            raise ValueError(
                f"decoding.mode='baseline' implies alpha=0 (got alpha={self.alpha}); "
                "a baseline run with a non-zero contrast weight would be ambiguous in the "
                "records. Set decoding.alpha=0.0 or use decoding.mode='factanchor'."
            )
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                f"decoding.min_new_tokens ({self.min_new_tokens}) must be <= "
                f"decoding.max_new_tokens ({self.max_new_tokens})"
            )
        if self.do_sample and self.temperature <= 0:
            raise ValueError("decoding.temperature must be > 0 when do_sample is true")
        if not self.do_sample and self.contrast_warmup_tokens > 0 and self.mode == "baseline":
            # harmless but almost certainly a mistake in the config
            raise ValueError("contrast_warmup_tokens is meaningless in baseline mode")
        return self

    @property
    def is_contrastive(self) -> bool:
        """True when the contrast branch must actually be evaluated."""
        return self.mode == "factanchor" and self.alpha > 0.0


class RetrievalConfig(_Section):
    """Entity linking + fact retrieval bounds, caching and failure behaviour."""

    enabled: bool = True
    provider: Literal["wikidata", "dbpedia", "json_file", "mock"] = "wikidata"
    #: Human-readable label attached to every retrieved fact.  ``None`` derives it
    #: from the provider so an offline fixture can never be labelled as Wikidata.
    source_name: str | None = None
    max_entities: int = Field(default=3, ge=1, le=50)
    max_facts_per_entity: int = Field(default=6, ge=1, le=200)
    min_linking_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    min_fact_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    allow_literal_values: bool = True
    include_qualifiers: bool = True
    max_value_chars: int = Field(default=160, ge=8, le=4000)
    prefer_low_degree_properties: bool = True
    entity_context_hint_chars: int = Field(default=240, ge=0, le=4000)
    local_json_path: str | None = None
    request_timeout_seconds: float = Field(default=10.0, gt=0.0, le=300.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    rate_limit_per_second: float = Field(default=5.0, gt=0.0, le=100.0)
    cache_dir: str = ".cache/factanchor"
    cache_ttl_seconds: int = Field(default=2592000, ge=0)
    fallback_on_unavailable: Literal["cache_only", "no_facts", "error"] = "cache_only"
    user_agent: str | None = None
    api_url: str | None = None
    properties_allowlist: list[str] = Field(default_factory=list)

    @field_validator("source_name")
    @classmethod
    def _non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("retrieval.source_name must be non-empty when set")
        return value.strip()

    @model_validator(mode="after")
    def _cross_checks(self) -> RetrievalConfig:
        if self.provider == "json_file" and not self.local_json_path:
            raise ValueError("retrieval.local_json_path is required when provider='json_file'")
        if self.provider in {"json_file", "mock"} and (self.source_name or "") in {"Wikidata", "DBpedia"}:
            raise ValueError(
                f"retrieval.provider={self.provider!r} cannot be labelled source_name="
                f"{self.source_name!r}: offline fixtures are not that knowledge source, and "
                "mislabelling them would falsify the provenance of every fact. Leave "
                "source_name unset to use the provider's own label."
            )
        if self.include_qualifiers and not self.allow_literal_values:
            raise ValueError(
                "retrieval.include_qualifiers=true requires allow_literal_values=true "
                "(qualifier values are literals by definition)"
            )
        return self


class NerConfig(_Section):
    """Named-entity extraction backend."""

    backend: Literal["rule", "simple", "spacy"] = "rule"
    spacy_model: str = "en_core_web_sm"
    max_entities: int = Field(default=3, ge=1, le=50)
    min_mention_chars: int = Field(default=2, ge=1, le=64)
    allow_single_token: bool = True
    stopword_tokens: list[str] = Field(default_factory=list)


class CorruptionConfig(_Section):
    """Controlled corruption of retrieved facts (never in place)."""

    strategy: Literal[
        "object_swap",
        "attribute_perturbation",
        "entity_substitution",
        "omission",
        "none",
    ] = "object_swap"
    corruption_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    max_corruption_fraction_of_block: float = Field(default=1.0, gt=0.0, le=1.0)
    swap_within_domain: bool = True
    allow_cross_entity_substitution: bool = False
    preserve_qualifiers: bool = True
    audit_log: bool = True
    extra_candidate_pool: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _cross_checks(self) -> CorruptionConfig:
        if self.strategy == "entity_substitution" and not self.allow_cross_entity_substitution:
            raise ValueError(
                "corruption.strategy='entity_substitution' requires "
                "corruption.allow_cross_entity_substitution=true (it rewrites the "
                "subject of the statement, which changes the topic)"
            )
        if self.strategy == "none" and self.corruption_rate > 0.0:
            raise ValueError(
                "corruption.strategy='none' is a no-op control; set "
                "corruption.corruption_rate=0.0 as well (refusing to accept a config "
                "that looks like it corrupts but does not)"
            )
        return self


class ContextConfig(_Section):
    """Prompt construction for both branches (Section 9)."""

    template_version: str = "v1"
    base_template: str = "{question}"
    contrast_template: str = (
        "User question:\n{question}\n\n"
        "Experimental entity-fact context (may contain intentionally corrupted facts):\n{fact_block}"
    )
    fact_line_format: str = "- {subject} | {predicate} | {object}"
    trusted_facts_in_base: bool = False
    include_empty_fact_block: bool = False
    max_fact_block_chars: int = Field(default=1200, ge=32, le=20000)
    label_as_experimental: bool = True
    system_prompt: str | None = None

    @model_validator(mode="after")
    def _check_placeholders(self) -> ContextConfig:
        for field_name in ("base_template", "contrast_template"):
            template = getattr(self, field_name)
            if "{question}" not in template:
                raise ValueError(f"context.{field_name} must contain a '{{question}}' placeholder")
        if "{fact_block}" not in self.contrast_template:
            raise ValueError("context.contrast_template must contain a '{fact_block}' placeholder")
        for placeholder in ("subject", "predicate", "object"):
            if "{" + placeholder + "}" not in self.fact_line_format:
                raise ValueError(
                    f"context.fact_line_format must contain '{{{placeholder}}}'"
                )
        return self


class EvaluationConfig(_Section):
    """Evaluation protocol and metric settings."""

    datasets: list[str] = Field(default_factory=lambda: ["wikibio_hallucination"])
    split: str = "test"
    max_examples: int = Field(default=50, ge=1, le=100000)
    development_subset: bool = True
    claim_extraction: Literal["rule", "model"] = "rule"
    claim_judge: Literal["heuristic_lexical", "model_judge", "gold_reference"] = "heuristic_lexical"
    judge_model_id: str | None = None
    judge_max_new_tokens: int = Field(default=128, ge=8, le=1024)
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    bertscore_enabled: bool = False
    bertscore_model: str = "roberta-large"
    bertscore_lang: str = "en"
    bootstrap_samples: int = Field(default=1000, ge=0, le=100000)
    bootstrap_confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    manual_review_sample: int = Field(default=30, ge=0, le=1000)

    @model_validator(mode="after")
    def _cross_checks(self) -> EvaluationConfig:
        if self.claim_judge == "model_judge" and not self.judge_model_id:
            raise ValueError("evaluation.judge_model_id is required for claim_judge='model_judge'")
        if self.claim_judge == "gold_reference" and self.claim_extraction == "model":
            raise ValueError(
                "claim_judge='gold_reference' is only supported with claim_extraction='rule' "
                "(gold spans are sentence-level)"
            )
        return self


class LoggingConfig(_Section):
    """Log/artifact hygiene."""

    jsonl_buffer_size: int = Field(default=0, ge=0, le=10000)
    include_timestamps: bool = True
    redact_prompts: bool = False
    max_snippet_chars: int = Field(default=2000, ge=16, le=100000)


class Config(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(extra="forbid")

    run: RunConfig = Field(default_factory=RunConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ner: NerConfig = Field(default_factory=NerConfig)
    corruption: CorruptionConfig = Field(default_factory=CorruptionConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging_: LoggingConfig = Field(default_factory=LoggingConfig)
    #: Which environment variables were applied while loading (audit trail; the
    #: run manifest records this so a result can be traced to its environment).
    env_applied: dict[str, Any] = Field(default_factory=dict, exclude=True)

    # -- convenience --------------------------------------------------------
    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.run.output_dir)

    @property
    def cache_dir(self) -> Path:
        return self.resolve_path(self.retrieval.cache_dir)

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a possibly-relative path against the project root."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.project_root / candidate)

    project_root: Path = Field(default_factory=find_project_root, exclude=True)

    # -- (de)serialisation --------------------------------------------------
    def to_dict(self, *, redact_secrets: bool = True) -> dict[str, Any]:
        data = self.model_dump(mode="json", exclude={"project_root"})
        if redact_secrets:
            data = _redact(data)
        return data

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, width=100)

    def apply_overrides(self, overrides: dict[str, Any] | list[str] | None) -> Config:
        """Return a new validated Config with dotted-path overrides applied.

        ``overrides`` may be a mapping of ``{"decoding.alpha": 0.25}`` or a list
        of ``"decoding.alpha=0.25"`` strings.  Unknown paths raise
        :class:`ConfigError`; values are type-checked by pydantic.
        """
        if not overrides:
            return self
        items: list[tuple[str, Any]] = []
        if isinstance(overrides, dict):
            items = list(overrides.items())
        else:
            for entry in overrides:
                if "=" not in entry:
                    raise ConfigError(f"override must look like key.path=value, got {entry!r}")
                key, _, raw = entry.partition("=")
                items.append((key.strip(), coerce_scalar(raw)))
        data = self.to_dict(redact_secrets=False)
        for key, value in items:
            get_by_path(data, key)  # raises for unknown paths
            set_by_path(data, key, value)
        data["project_root"] = str(self.project_root)
        try:
            return Config.model_validate(data)
        except Exception as exc:  # pydantic ValidationError -> readable ConfigError
            raise ConfigError(f"invalid configuration after overrides: {exc}") from exc

    def with_section(self, section: str, values: dict[str, Any]) -> Config:
        """Return a copy with ``values`` merged into a named section."""
        data = self.to_dict(redact_secrets=False)
        if section not in data:
            raise ConfigError(f"unknown config section {section!r}")
        data[section] = deep_merge(data[section], values)
        data["project_root"] = str(self.project_root)
        return Config.model_validate(data)


_SECRET_PATTERN = re.compile(r"(token|api_?key|secret|password|authorization)", re.IGNORECASE)


def _redact(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, key) for v in value]
    if isinstance(value, str) and key and _SECRET_PATTERN.search(key):
        return "***redacted***"
    return value


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | list[str] | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    """Load, merge, validate and return a :class:`Config`.

    Parameters
    ----------
    path:
        YAML file (or name resolvable inside ``configs/``).  ``None`` falls back
        to ``configs/default.yaml``; if that does not exist either, the built-in
        defaults are used (used by unit tests).
    overrides:
        Dotted-path overrides applied after file inheritance.
    env:
        Environment mapping (defaults to ``os.environ``).  Selected
        ``FACTANCHOR_*`` variables act as **defaults only**, so the effective
        precedence is: built-in defaults < environment < YAML file <
        ``overrides``.  Rationale: a run must be reproducible from its config
        file alone, so a stray exported variable can never silently change a
        recorded experiment.  Pass ``--set key.path=value`` for CLI overrides.
    """
    env = dict(os.environ if env is None else env)
    raw: dict[str, Any] = {}
    if path is not None:
        raw = _load_raw(_resolve_config_path(path))
    else:
        try:
            raw = _load_raw(_resolve_config_path(Path("default.yaml")))
        except ConfigError:
            raw = {}

    root = find_project_root()
    env_defaults = _read_env(env, _ENV_LOW_PRIORITY)
    env_overrides = _read_env(env, _ENV_HIGH_PRIORITY)
    merged = deep_merge(env_defaults, raw)          # YAML wins over defaults
    merged = deep_merge(merged, env_overrides)      # operational env vars win
    merged.setdefault("project_root", str(root))
    merged["env_applied"] = {
        **{key: f"default -> {path}" for key, (path, _c) in _ENV_LOW_PRIORITY.items() if env.get(key)},
        **{key: f"override -> {path}" for key, (path, _c) in _ENV_HIGH_PRIORITY.items() if env.get(key)},
    }

    try:
        config = Config.model_validate(merged)
    except Exception as exc:
        raise ConfigError(f"invalid configuration ({path or 'built-in defaults'}): {exc}") from exc

    config = config.model_copy(update={"project_root": root})
    if overrides:
        config = config.apply_overrides(overrides)
    _ensure_directories(config)
    return config


#: Environment variables that override the YAML file.  These are *operational*
#: settings (endpoints, timeouts, cache locations) that legitimately differ
#: between machines and do not change what a recorded experiment means.
_ENV_HIGH_PRIORITY: dict[str, tuple[str, Any]] = {
    "FACTANCHOR_WIKIDATA_API": ("retrieval.api_url", str),
    "FACTANCHOR_USER_AGENT": ("retrieval.user_agent", str),
    "FACTANCHOR_REQUEST_TIMEOUT": ("retrieval.request_timeout_seconds", float),
    "FACTANCHOR_MAX_RETRIES": ("retrieval.max_retries", int),
    "FACTANCHOR_RATE_LIMIT_PER_SECOND": ("retrieval.rate_limit_per_second", float),
    "FACTANCHOR_CACHE_DIR": ("retrieval.cache_dir", str),
    "FACTANCHOR_CACHE_TTL": ("retrieval.cache_ttl_seconds", int),
    "FACTANCHOR_RETRIEVAL_FALLBACK": ("retrieval.fallback_on_unavailable", str),
}

#: Environment variables that act as *defaults only*.  These decide what the
#: experiment is (model identity, where artifacts go), so the config file and CLI
#: always win: a stray exported variable must never silently change a result.
_ENV_LOW_PRIORITY: dict[str, tuple[str, Any]] = {
    "FACTANCHOR_MODEL_ID": ("model.model_id", str),
    "FACTANCHOR_MODEL_REVISION": ("model.revision", str),
    "FACTANCHOR_OUTPUT_DIR": ("run.output_dir", str),
}


def _read_env(env: dict[str, str], mapping: dict[str, tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for env_key, (path, caster) in mapping.items():
        raw = env.get(env_key)
        if raw is None or raw == "":
            continue
        try:
            value = caster(raw)
        except ValueError as exc:
            raise ConfigError(
                f"environment variable {env_key}={raw!r} is not a valid {caster.__name__}"
            ) from exc
        set_by_path(out, path, value)
    return out


def _ensure_directories(config: Config) -> None:
    """Create output/cache directories.  Failures are surfaced, never ignored."""
    for directory in (config.output_dir, config.cache_dir):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"cannot create directory {directory}: {exc}") from exc


# ---------------------------------------------------------------------------
# Lightweight view used by the ablation runner
# ---------------------------------------------------------------------------


@dataclass
class Condition:
    """One named experimental condition (config + provenance note)."""

    name: str
    config: Config
    note: str = ""
    overrides: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.overrides is None:
            self.overrides = {}

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "note": self.note, "overrides": self.overrides}
