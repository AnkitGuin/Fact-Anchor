"""Configuration validation tests (spec 10.3: validate at startup, ignore nothing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from factanchor.config import (
    Config,
    ConfigError,
    CorruptionConfig,
    DecodingConfig,
    EvaluationConfig,
    RetrievalConfig,
    coerce_scalar,
    deep_merge,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT_ROOT / "configs"


def test_shipped_configs_all_load_and_validate():
    for name in ("default.yaml", "baseline.yaml", "smoke_toy.yaml"):
        config = load_config(CONFIGS / name)
        assert isinstance(config, Config)
        assert config.model.model_id


def test_experiments_config_conditions_are_valid():
    from factanchor.experiment import build_conditions

    conditions = build_conditions(CONFIGS / "experiments.yaml")
    names = {condition.name for condition in conditions}
    assert {"base_model_no_context", "factanchor_object_swap", "factanchor_omission_control"} <= names
    assert any("-" in name for name in names), "the grid should produce cartesian-product conditions"
    for condition in conditions:
        assert isinstance(condition.config, Config)


def test_unknown_keys_are_rejected():
    with pytest.raises(Exception, match="Extra inputs are not permitted|extra_forbidden"):
        Config.model_validate({"decoding": {"alphaa": 0.5}})


def test_unknown_path_in_override_raises_config_error():
    config = load_config(CONFIGS / "default.yaml")
    with pytest.raises(ConfigError, match="unknown configuration path"):
        config.apply_overrides({"decoding.not_a_field": 1})


def test_alpha_and_rate_ranges_are_enforced():
    with pytest.raises(ValueError, match="alpha"):
        DecodingConfig(alpha=-0.1)
    with pytest.raises(ValueError, match="corruption_rate"):
        CorruptionConfig(corruption_rate=1.5)


def test_device_values_are_validated():
    config = load_config(CONFIGS / "default.yaml")
    assert config.apply_overrides({"run.device": "cpu"}).run.device == "cpu"
    with pytest.raises(ConfigError):
        config.apply_overrides({"run.device": "quantum"})


def test_decoding_cross_checks():
    with pytest.raises(ValueError, match="min_new_tokens"):
        DecodingConfig(min_new_tokens=10, max_new_tokens=5)
    with pytest.raises(ValueError, match="contrast_warmup_tokens"):
        DecodingConfig(mode="baseline", contrast_warmup_tokens=2)


def test_evaluation_requires_judge_model_when_using_model_judge():
    with pytest.raises(ValueError, match="judge_model_id"):
        EvaluationConfig(claim_judge="model_judge", judge_model_id=None)
    assert EvaluationConfig(claim_judge="model_judge", judge_model_id="toy").judge_model_id == "toy"


def test_config_inheritance_merges_parent_and_child(tmp_path):
    parent = tmp_path / "parent.yaml"
    parent.write_text("decoding:\n  alpha: 0.25\n  max_new_tokens: 10\nrun:\n  seed: 5\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("defaults: parent.yaml\ndecoding:\n  max_new_tokens: 32\n", encoding="utf-8")
    config = load_config(child)
    assert config.decoding.alpha == 0.25  # inherited
    assert config.decoding.max_new_tokens == 32  # overridden
    assert config.run.seed == 5


def test_circular_inheritance_is_detected(tmp_path):
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    first.write_text("defaults: b.yaml\n", encoding="utf-8")
    second.write_text("defaults: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="circular config inheritance"):
        load_config(first)


def test_missing_config_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        load_config("configs/definitely_missing.yaml")


def test_invalid_yaml_reports_the_file(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("decoding: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_experiment_defining_env_vars_are_only_defaults():
    """A stray exported model id must not silently change a recorded experiment."""
    env = {"FACTANCHOR_MODEL_ID": "from-env-model", "FACTANCHOR_OUTPUT_DIR": "/tmp/elsewhere"}
    config = load_config(CONFIGS / "default.yaml", env=env)
    assert config.model.model_id == "HuggingFaceTB/SmolLM2-135M-Instruct"  # file wins
    assert config.output_dir == config.project_root / "artifacts"
    assert config.env_applied["FACTANCHOR_MODEL_ID"].startswith("default")


def test_operational_env_vars_override_the_file():
    env = {"FACTANCHOR_REQUEST_TIMEOUT": "3.5", "FACTANCHOR_MAX_RETRIES": "1"}
    config = load_config(CONFIGS / "default.yaml", env=env)
    assert config.retrieval.request_timeout_seconds == 3.5
    assert config.retrieval.max_retries == 1
    assert config.env_applied["FACTANCHOR_REQUEST_TIMEOUT"].startswith("override")


def test_operational_env_defaults_apply_when_the_file_is_silent(tmp_path):
    path = tmp_path / "minimal.yaml"
    path.write_text("decoding:\n  alpha: 0.5\n", encoding="utf-8")
    config = load_config(path, env={"FACTANCHOR_REQUEST_TIMEOUT": "7.5"})
    assert config.retrieval.request_timeout_seconds == 7.5


def test_user_agent_resolution_prefers_config_then_env():
    from factanchor.knowledge.base import DEFAULT_USER_AGENT, resolve_user_agent

    assert resolve_user_agent("Custom/1.0", env={}) == "Custom/1.0"
    assert resolve_user_agent(None, env={"FACTANCHOR_USER_AGENT": "Env/2.0"}) == "Env/2.0"
    assert resolve_user_agent(None, env={"FACTANCHOR_CONTACT": "a@b.c"}) == (
        "FactAnchor/0.1 (research prototype; contact: a@b.c)"
    )
    assert resolve_user_agent(None, env={}) == DEFAULT_USER_AGENT


def test_environment_variable_type_errors_are_reported():
    with pytest.raises(ConfigError, match="FACTANCHOR_REQUEST_TIMEOUT"):
        load_config(CONFIGS / "default.yaml", env={"FACTANCHOR_REQUEST_TIMEOUT": "soon"})


def test_override_strings_are_coerced():
    assert coerce_scalar("true") is True
    assert coerce_scalar("0.5") == 0.5
    assert coerce_scalar("7") == 7
    assert coerce_scalar("[1, 2]") == [1, 2]
    assert coerce_scalar("null") is None
    assert coerce_scalar("~") is None
    assert coerce_scalar("") is None
    # YAML uses the literal string "none" for `strategy: none`, and the CLI must
    # agree with the file rather than inventing a null.
    assert coerce_scalar("none") == "none"
    assert coerce_scalar("hello world") == "hello world"


def test_override_list_form_and_validation(project_root):
    config = load_config(CONFIGS / "default.yaml", overrides=["decoding.alpha=1.25", "run.seed=7"])
    assert config.decoding.alpha == 1.25
    assert config.run.seed == 7
    with pytest.raises(ConfigError, match="key.path=value"):
        config.apply_overrides(["nonsense"])


def test_deep_merge_replaces_lists_and_merges_dicts():
    merged = deep_merge({"a": {"b": 1, "c": [1, 2]}}, {"a": {"c": [3], "d": 4}})
    assert merged == {"a": {"b": 1, "c": [3], "d": 4}}


def test_config_serialisation_redacts_secrets():
    config = load_config(CONFIGS / "default.yaml")
    payload = config.to_dict(redact_secrets=True)
    if payload["retrieval"].get("user_agent"):
        assert payload["retrieval"]["user_agent"] != "***redacted***"  # not a secret
    assert "project_root" not in payload


def test_output_directories_are_created(tmp_path):
    config = load_config(
        CONFIGS / "smoke_toy.yaml",
        overrides={"run.output_dir": str(tmp_path / "out"), "retrieval.cache_dir": str(tmp_path / "cache")},
    )
    assert (tmp_path / "out").is_dir()
    assert (tmp_path / "cache").is_dir()
    assert config.output_dir == tmp_path / "out"


def test_with_section_returns_a_validated_copy():
    config = load_config(CONFIGS / "default.yaml")
    updated = config.with_section("decoding", {"alpha": 0.75, "mode": "factanchor"})
    assert updated.decoding.alpha == 0.75
    assert config.decoding.alpha != 0.75  # original untouched
    with pytest.raises(ConfigError):
        config.with_section("nope", {})


def test_baseline_config_disables_retrieval_and_contrast():
    config = load_config(CONFIGS / "baseline.yaml")
    assert config.decoding.mode == "baseline"
    assert config.decoding.alpha == 0.0
    assert config.retrieval.enabled is False
    assert config.corruption.strategy == "none"
    assert config.corruption.corruption_rate == 0.0


def test_retrieval_config_rejects_inconsistent_qualifier_settings():
    with pytest.raises(ValueError, match="include_qualifiers"):
        RetrievalConfig(include_qualifiers=True, allow_literal_values=False)
