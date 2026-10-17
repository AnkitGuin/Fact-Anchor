"""CLI tests: exit codes, JSON output, no-network paths, artifact writing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from factanchor.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = str(PROJECT_ROOT / "configs" / "smoke_toy.yaml")
DEV_SMOKE = str(PROJECT_ROOT / "data" / "processed" / "dev_smoke.jsonl")


def _toy_set(extra: list[str] | None = None) -> list[str]:
    args = [
        "--config",
        SMOKE_CONFIG,
        "--set",
        f"retrieval.local_json_path={PROJECT_ROOT / 'data' / 'processed' / 'offline_knowledge.jsonl'}",
        "--set",
        "decoding.max_new_tokens=4",
    ]
    return [*args, *(extra or [])]


def test_generate_json_output(capsys):
    code = main(["generate", *_toy_set(["--prompt", "Who was Marie Curie?", "--json"])])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "factanchor"
    assert payload["status"] in {"ok", "degraded"}
    assert payload["is_toy_model"] is True
    assert "answer" in payload
    assert payload["warning"]


def test_generate_prompt_from_stdin(monkeypatch, capsys):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("Who was Ada Lovelace?"))
    monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": lambda self: False, "read": lambda self: "Who was Ada Lovelace?"})())
    code = main(["generate", *_toy_set(["--json"])])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "Ada Lovelace" in payload["prompt"]


def test_generate_prompt_file(tmp_path, capsys):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Who was Douglas Adams?", encoding="utf-8")
    code = main(["generate", *_toy_set(["--prompt-file", str(prompt_file), "--json"])])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["prompt"] == "Who was Douglas Adams?"


def test_generate_without_prompt_fails_clearly(monkeypatch):
    monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": lambda self: True})())
    code = main(["generate", *_toy_set([])])
    assert code == 2  # configuration error


def test_baseline_mode_from_cli(capsys):
    code = main(["generate", *_toy_set(["--prompt", "Who was Marie Curie?", "--mode", "baseline", "--json"])])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "baseline"
    assert payload["alpha"] == 0.0
    assert payload["contrast_applied"] is False


def test_generate_can_save_a_run(tmp_path, capsys):
    out_dir = tmp_path / "artifacts"
    code = main(
        [
            "generate",
            *_toy_set(
                [
                    "--prompt",
                    "Who was Marie Curie?",
                    "--json",
                    "--save-run",
                    "--set",
                    f"run.output_dir={out_dir}",
                    "--set",
                    f"retrieval.cache_dir={tmp_path / 'cache'}",
                ]
            ),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_dir"]
    run_dir = Path(payload["run_dir"])
    assert (run_dir / "generations.jsonl").exists()
    assert (run_dir / "config.yaml").exists()


def test_run_command_with_a_dataset(tmp_path, capsys):
    code = main(
        [
            "run",
            *_toy_set(
                [
                    "--dataset",
                    DEV_SMOKE,
                    "--limit",
                    "2",
                    "--condition",
                    "cli-run",
                    "--json",
                    "--no-warmup",
                    "--set",
                    f"run.output_dir={tmp_path / 'artifacts'}",
                    "--set",
                    f"retrieval.cache_dir={tmp_path / 'cache'}",
                ]
            ),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_examples"] == 2
    assert payload["n_ok"] + payload["n_degraded"] == 2
    assert Path(payload["run_dir"], "metrics.json").exists()


def test_doctor_reports_environment(capsys):
    code = main(["doctor", "--config", SMOKE_CONFIG, "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["environment"]["python_version"]
    checks = {check["check"] for check in payload["checks"]}
    assert {"torch", "transformers", "datasets", "matplotlib"} <= checks
    assert payload["model"]["is_toy"] is True


def test_datasets_listing_is_offline(capsys):
    code = main(["datasets", "--config", SMOKE_CONFIG, "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    keys = {row["key"] for row in payload}
    assert {"wikibio_hallucination", "halueval", "fever", "truthfulqa"} <= keys
    assert all(row["licence"] for row in payload)


def test_evaluate_command_on_a_run(tmp_path, capsys):
    out_dir = tmp_path / "artifacts"
    common = _toy_set(
        [
            "--set",
            f"run.output_dir={out_dir}",
            "--set",
            f"retrieval.cache_dir={tmp_path / 'cache'}",
        ]
    )
    assert main(["run", *common, "--dataset", DEV_SMOKE, "--limit", "2", "--no-warmup", "--quiet"]) == 0
    capsys.readouterr()  # discard the human-readable summary of the run
    runs = sorted((out_dir / "runs").iterdir())
    assert runs, "expected a run directory"
    code = main(["evaluate", "--input", str(runs[-1]), "--config", SMOKE_CONFIG, "--examples", DEV_SMOKE, "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_records"] == 2
    assert payload["conditions"]


def test_cache_command_stats_and_clear(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    code = main(["cache", "--config", SMOKE_CONFIG, "--set", f"retrieval.cache_dir={cache_dir}", "--stats"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "stats" in payload
    code = main(["cache", "--config", SMOKE_CONFIG, "--set", f"retrieval.cache_dir={cache_dir}", "--clear"])
    assert code == 0


def test_unknown_flag_exits_with_argparse_error():
    with pytest.raises(SystemExit) as excinfo:
        main(["generate", "--definitely-not-a-flag"])
    assert excinfo.value.code == 2


def test_ablate_command_runs_a_small_grid(tmp_path, capsys):
    ablation = tmp_path / "ablation.yaml"
    ablation.write_text(
        f"""
base: {SMOKE_CONFIG}
knobs:
  model.model_id: toy
  decoding.max_new_tokens: 3
  evaluation.max_examples: 2
  evaluation.manual_review_sample: 0
  retrieval.local_json_path: {PROJECT_ROOT / 'data' / 'processed' / 'offline_knowledge.jsonl'}
grid:
  decoding.alpha: [0.0, 0.5]
conditions:
  - name: small_control
    overrides:
      decoding.mode: baseline
      decoding.alpha: 0.0
      retrieval.enabled: false
      corruption.strategy: none
      corruption.corruption_rate: 0.0
""",
        encoding="utf-8",
    )
    code = main(
        [
            "ablate",
            "--config",
            SMOKE_CONFIG,
            "--ablation",
            str(ablation),
            "--dataset",
            DEV_SMOKE,
            "--limit",
            "2",
            "--no-warmup",
            "--json",
            "--output",
            str(tmp_path / "ablation_results.json"),
            "--set",
            f"run.output_dir={tmp_path / 'artifacts'}",
            "--set",
            f"retrieval.cache_dir={tmp_path / 'cache'}",
        ]
    )
    assert code == 0, "ablation should run end-to-end"
    payload = json.loads(capsys.readouterr().out)
    assert payload["reference_condition"] == "small_control"
    assert len(payload["conditions"]) == 3  # two grid conditions + one named control
    results = json.loads((tmp_path / "ablation_results.json").read_text())
    assert results["warmup_policy"]
    assert results["comparison_note"]
    assert results["conditions"]["small_control"]["summary"]["n_examples"] == 2
