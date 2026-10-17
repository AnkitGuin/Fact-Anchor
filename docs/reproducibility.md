# Reproducibility

## 1. Environment of record

Recorded by `factanchor doctor` and stored in every run as `environment.json`:

| Item | Value |
|---|---|
| Python | 3.13.16 (`/usr/local/bin/python3.13`) |
| Platform | Linux 6.1.158+ x86_64, glibc 2.41 |
| CPU / RAM | 2 vCPU, ≈1 GB RAM |
| torch | 2.14.1+cpu, CUDA unavailable, MPS unavailable |
| transformers / tokenizers | 5.19.0 / 0.23.3 |
| numpy / pandas / scipy / matplotlib | 2.3.5 / 2.2.3 / 1.17.1 / 3.10.9 |
| pydantic / pyyaml / requests | 2.13.5 / 6.0.3 / 2.33.0 |
| pytest / spaCy | 9.0.3 / 3.8.14 |
| datasets / streamlit / bert-score | not installed |
| Network | none |

## 2. What makes a run reproducible here

* **Everything is in the config file.** Env vars act as *defaults*; the YAML file wins; `--set`
  wins over both. Precedence: built-in defaults < environment < YAML < `--set`.
  `factanchor doctor --json` prints the effective config, and each run stores a copy as
  `config.yaml` plus its SHA-256 in `manifest.json`.
* **Unknown keys fail loudly** (`extra="forbid"` on every config section), so a typo cannot
  silently fall back to a default.
* **Seeds** — `run.seed` seeds Python, NumPy and torch (`utils.set_global_seed`, which reports
  exactly what it touched); `corruption.seed` seeds the corruption RNG separately; the seed is
  in the manifest, the record and the metrics file.
* **Determinism** — greedy decoding by default; `run.deterministic_algorithms` enables strict
  torch algorithms where available. Two runs with the same config produce identical
  `generations.jsonl` content (verified in `tests/test_reproducibility.py`).
* **No overwrites** — a run directory is written once. If it exists, the new run goes to
  `<run_id>-1`, `-2`, … unless `run.overwrite: true` or `run.allow_run_dir_reuse: true` is set
  explicitly. An existing `config.yaml` is never replaced silently: a timestamped sibling is
  written instead.
* **Warm-up policy** — one one-token warm-up generation runs before the measured loop and is
  excluded from the latency statistics; the policy string is stored in `manifest.json` and in
  `metrics.json.warmup_policy`. On CUDA every timed section is wrapped in
  `torch.cuda.synchronize()`.
* **Latency reporting** — median, mean, p10 and p90 per run and per condition, plus *paired*
  per-example overhead against a named reference condition (`overhead_vs_reference`), so
  hardware noise does not masquerade as an effect.
* **Provenance** — each fact carries `source`, `source_url`, `retrieved_at` and `confidence`;
  each answer records prompt hash, model id + revision, tokenizer id + revision, device,
  dtype, decoding parameters, entity records, fact ids and the corruption audit.

## 3. Exact commands for the offline reproduction

```bash
cd factanchor
pip install -e . --no-deps            # core deps must be present: numpy, pyyaml, requests, pydantic
python -m pytest tests -q             # 230 passed in ~12 s (recorded before the current review fixes)

factanchor doctor --config configs/smoke_toy.yaml
factanchor demo --config configs/smoke_toy.yaml --prompt "Who was Marie Curie?"

factanchor run --config configs/smoke_toy.yaml --dataset data/processed/dev_smoke.jsonl \
    --limit 8 --condition base_model_no_context --run-id smoke_toy_baseline_alpha0 \
    --set decoding.mode=baseline --set decoding.alpha=0.0 --set retrieval.enabled=false \
    --set corruption.strategy=none --set corruption.corruption_rate=0.0

factanchor run --config configs/smoke_toy.yaml --dataset data/processed/dev_smoke.jsonl \
    --limit 8 --condition factanchor_alpha0p5_rate0p5 --run-id smoke_toy_factanchor_alpha0p5

python scripts/evaluate.py --input artifacts/runs/smoke_toy_baseline_alpha0 \
    --output artifacts/metrics/evaluation_baseline_smoke.json

python scripts/run_ablation.py --config configs/experiments_smoke_toy.yaml \
    --dataset data/processed/dev_smoke.jsonl --limit 8 --no-warmup \
    --output artifacts/metrics/ablation_smoke_toy.json
```

Expected artifacts afterwards:

```
artifacts/
├── metrics/{ablation_smoke_toy.json, evaluation_baseline_smoke.json, evaluation_factanchor_smoke.json}
├── plots/ablation_experiments_smoke_toy.png
└── runs/{smoke_toy_baseline_alpha0, smoke_toy_factanchor_alpha0p5, 2026...-<condition>}/…
```

## 4. Reproducing with a real model (requires network + GPU)

```bash
pip install -e ".[hf,datasets,analysis,demo]"
export HUGGING_FACE_HUB_TOKEN=...            # gated repos; never put tokens in a config file
factanchor doctor --config configs/default.yaml --check-model --check-knowledge
python scripts/download_data.py --dataset wikibio_hallucination --max-examples 100
factanchor run --config configs/default.yaml --dataset wikibio_hallucination --split test \
    --limit 50 --set model.model_id=<model> --set model.revision=<sha> \
    --set run.device=cuda --set run.dtype=bfloat16 --condition factanchor_baseline_model
python scripts/run_ablation.py --config configs/experiments.yaml --dataset wikibio_hallucination \
    --limit 50 --output artifacts/metrics/ablation_<model>.json
```

Pin `model.revision` and `tokenizer_revision` to commit SHAs for any number you intend to
publish; `main` is a moving target and the manifest records whatever was used.

## 5. Artifact inventory per run

| File | Contents |
|---|---|
| `config.yaml` | the fully resolved configuration (secrets redacted) used for this run |
| `environment.json` | interpreter, libraries, CPU/GPU, thread counts, peak RSS |
| `manifest.json` | run id, git hash/dirty, model + tokenizer ids/revisions, device, dtype, seed, dataset + split, knowledge provider/source label, corruption strategy/rate, config hash, warnings, notes (warm-up policy) |
| `generations.jsonl` | one `GenerationRecord` per example: prompt, contexts, entities, facts, corruption audit, token ids, timings, status, error |
| `errors.jsonl` | one line per failed example (empty file = no failures) |
| `metrics.json` | aggregate + per-example metrics, judgement evidence, bootstrap CIs, paired overhead, metric definitions, judge, warm-up policy |
| `manual_review.jsonl` | stratified sample for human inspection (lowest/highest support, errors, degraded records) |
| `logs.txt` | full log for the run, including warnings about degraded conditions |
| `plots/` | claim-metric bars, per-example latency, tokens-vs-time |

## 6. Known reproducibility gaps

1. No git repository exists in this sandbox, so `manifest.json.git_hash` is `null`; the config
   hash still identifies the exact configuration.
2. No real weights/datasets were available: the offline numbers reproduce the *pipeline*, not
   a scientific result. Any claim about factuality requires re-running on real weights and a
   real split, with at least three seeds.
3. Latency numbers are CPU-only and machine-specific; they are reported with median/p10/p90
   and as paired per-example deltas to keep them interpretable.
