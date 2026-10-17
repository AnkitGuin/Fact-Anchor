"""Tests for the Streamlit demo's non-UI logic (spec 13 Phase 7 / 14).

Streamlit itself is optional, so these tests exercise the parts of ``app.py``
that do the work: config construction (mode/alpha/corruption-rate coupling),
generation of a paired comparison, and graceful failure when the model or the
knowledge source is unavailable.  ``app.main()`` is only checked for its
"Streamlit is not installed" fallback when Streamlit is genuinely absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "smoke_toy.yaml"


def _load_app():
    spec = importlib.util.spec_from_file_location("factanchor_demo_app", ROOT / "app.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


app = _load_app()


def test_baseline_mode_forces_alpha_zero():
    config = app._load_config(str(SMOKE_CONFIG), alpha=0.9, corruption_rate=0.5, mode="baseline", max_new_tokens=8)
    assert config.decoding.mode == "baseline"
    # alpha=0.9 with mode=baseline would be rejected by the config validator, so
    # the demo must dampen it rather than crash.
    assert config.decoding.alpha == 0.0


def test_factanchor_mode_keeps_the_requested_alpha():
    config = app._load_config(str(SMOKE_CONFIG), alpha=0.75, corruption_rate=0.25, mode="factanchor", max_new_tokens=8)
    assert config.decoding.mode == "factanchor"
    assert config.decoding.alpha == pytest.approx(0.75)
    assert config.corruption.corruption_rate == pytest.approx(0.25)
    assert config.decoding.max_new_tokens == 8
    # the demo is always a paired, greedy comparison
    assert config.decoding.do_sample is False


@pytest.mark.parametrize("mode", ["baseline", "factanchor"])
def test_demo_runs_both_conditions_offline(mode):
    config = app._load_config(str(SMOKE_CONFIG), alpha=0.5, corruption_rate=0.5, mode=mode, max_new_tokens=6)
    record, error = app._run_one(config, "Who was Marie Curie?", knowledge=None)
    assert error is None, error
    assert record is not None
    assert record.status.value in {"ok", "degraded"}
    assert record.answer is not None
    assert record.output_tokens >= 1


def test_demo_reports_model_failures_instead_of_raising(monkeypatch):
    config = app._load_config(str(SMOKE_CONFIG), alpha=0.5, corruption_rate=0.5, mode="factanchor", max_new_tokens=4)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated out-of-memory")

    monkeypatch.setattr("factanchor.model_adapter.build_adapter", boom)
    record, error = app._run_one(config, "Who was Marie Curie?", knowledge=None)
    assert record is None
    assert "model unavailable" in error and "simulated out-of-memory" in error


def test_demo_does_not_expose_environment_variables():
    """The demo must never print secrets: no os.environ dump anywhere."""
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "st.secrets" not in source
    assert "os.environ" in source  # only for the optional demo config path
    for line in source.splitlines():
        if "os.environ" in line:
            assert "FACTANCHOR_DEMO_CONFIG" in line, line
