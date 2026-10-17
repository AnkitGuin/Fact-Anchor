# FactAnchor

**Entity-grounded contrastive decoding for hallucination mitigation — research prototype.**

FactAnchor is a controlled implementation of one idea: at every decoding step, contrast the
base model's distribution with the distribution obtained from the *same* prompt plus an
**intentionally corrupted entity-fact context**. Tokens that only look likely because the
corrupted context suggests them are penalised:

```
s_t(v) = log p_b(v | x, y_<t)  −  α · log p_c(v | x, y_<t, z_c)
```

* `x` — the user prompt, `y_<t` — the generated suffix (identical in both branches),
* `z_c` — the retrieved entity facts after controlled, seeded, audited corruption,
* `α` — the contrast weight (`α = 0` must reproduce baseline **exactly**, and does — see
  [Results](#results-measured)).

The repository is a *complete, runnable* implementation of the specification in
`FactAnchor.md`: the pipeline, both conditions, the corruption engine, three judges, the
experiment runner, ablations, CLI, Streamlit demo and a 230-test suite — not a sketch.

> **Honest status.** Everything below was executed in a CPU-only, network-isolated sandbox.
> Model weights (Llama-3-8B / SmolLM2) and the named datasets (WikiBio Hallucination,
> HaluEval, FEVER) could **not** be downloaded, so the end-to-end pipeline is verified with a
> bundled NumPy toy model plus a local JSON knowledge base. Toy-model outputs are random
> characters: they prove the plumbing, **not** any scientific claim. See
> [What is blocked](#what-is-blocked) and [docs/limitations.md](docs/limitations.md).

---

## Contents

| | |
|---|---|
| [Install](#install) | [Phase status](#phase-status) |
| [Quick start (offline)](#quick-start-offline-60-seconds) | [Results](#results-measured) |
| [Using a real model](#using-a-real-model) | [What is blocked](#what-is-blocked) |
| [CLI reference](#cli-reference) | [Testing](#testing) |
| [Repository layout](#repository-layout) | [Responsible use](#responsible-use) |
| [Method and design decisions](#method-and-design-decisions) | [Deviation from spec §21](#deviation-from-spec-21) |

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Python >= 3.10 (3.10–3.13 tested)
pip install -e .                                      # core only: numpy, pyyaml, requests, pydantic
pip install -e ".[all]"                               # + torch, transformers, datasets, pandas, matplotlib, streamlit
cp .env.example .env                                  # optional: operational settings (never commit secrets)
```

Optional extras: `hf` (torch+transformers), `analysis` (pandas/matplotlib), `datasets`,
`ner` (spaCy), `bertscore`, `demo` (Streamlit), `dev` (pytest).

Nothing is downloaded at install time and no dataset is committed.

---

## Quick start (offline, ~60 seconds)

The `smoke_toy` profile runs the *entire* pipeline with a deterministic NumPy toy causal LM
and a local JSON knowledge base — no network, no model weights, no GPU:

```bash
# 1. feasibility report (environment, config validity, optional dependencies)
factanchor doctor --config configs/smoke_toy.yaml

# 2. baseline vs FactAnchor on one prompt, side by side
factanchor demo --config configs/smoke_toy.yaml --prompt "Who was Marie Curie?"

# 3. one full run with artifacts + metrics (8 dev examples)
factanchor run --config configs/smoke_toy.yaml \
    --dataset data/processed/dev_smoke.jsonl --limit 8 \
    --condition factanchor_alpha0p5_rate0p5 --run-id smoke_toy_factanchor_alpha0p5

# 4. alpha = 0 must reproduce the baseline exactly (spec §3.2)
factanchor generate --config configs/smoke_toy.yaml --prompt "Who was Marie Curie?" \
    --mode baseline --json | python -c "import json,sys; print(json.load(sys.stdin)['answer'])"
factanchor generate --config configs/smoke_toy.yaml --prompt "Who was Marie Curie?" \
    --mode factanchor --alpha 0 --json | python -c "import json,sys; print(json.load(sys.stdin)['answer'])"
```

The toy model prints a warning in every output (`model.model_id='toy' uses randomly
initialised weights: the text is not meaningful`), and run manifests record
`is_toy_model: true`, so toy results can never be mistaken for model results.

## Using a real model

```bash
pip install -e ".[hf]"
# configs/default.yaml ships with HuggingFaceTB/SmolLM2-135M-Instruct; override with --set
factanchor run --config configs/default.yaml \
    --dataset data/processed/dev_smoke.jsonl --limit 20 \
    --set model.model_id=meta-llama/Meta-Llama-3-8B-Instruct \
    --set model.revision=<commit-sha> \
    --set run.device=cuda --set run.dtype=bfloat16

# gated repositories need a token *in the environment*, never in the config file
export HUGGING_FACE_HUB_TOKEN=hf_...
```

`factanchor doctor --check-model` reports whether the configured model can actually be
loaded, and `run.overwrite` is the *only* way an existing run directory is ever reused
(otherwise a `-1`, `-2`, … suffix is appended — previous results are never destroyed).

## Datasets

```bash
factanchor datasets --list                                   # available loaders + licence notes
python scripts/download_data.py --dataset wikibio_hallucination --max-examples 50
python scripts/download_data.py --inspect --dataset halueval  # print the real field names
```

`download_data.py` refuses to substitute a different dataset when the requested one is
unavailable; it prints the blocker and exits non-zero. See [docs/datasets.md](docs/datasets.md).

---

## Phase status

Spec §13 defines phases 0–7 with exit criteria. All phases are implemented; the phases whose
exit criteria require network weights or datasets are *implemented and unit-verified but not
executed at scale*, and that is stated explicitly.

| Phase | Scope | State | Evidence |
|---|---|---|---|
| 0 | Environment & feasibility | ✅ done | `factanchor doctor`; [docs/reproducibility.md](docs/reproducibility.md) |
| 1 | Baseline generation | ✅ done | baseline run `artifacts/runs/smoke_toy_baseline_alpha0/`; `tests/test_decoding.py`, `tests/test_integration.py` |
| 2 | Entity & knowledge pipeline | ✅ done (offline KB) | `tests/test_entity_pipeline.py` (38 tests); Wikidata provider implemented, unexercised offline |
| 3 | Corruption pipeline | ✅ done | `tests/test_corruption.py`; audits in every `generations.jsonl` |
| 4 | Contrastive decoding | ✅ done | `tests/test_decoding.py` (32 tests); `α=0` equivalence demonstrated |
| 5 | Evaluation | ✅ done (heuristic + model judges) | `metrics.json` for both runs; `tests/test_metrics.py` |
| 6 | Ablations | ✅ done offline (16 conditions) | `artifacts/metrics/ablation_smoke_toy.json`, `artifacts/plots/ablation_experiments_smoke_toy.png` |
| 7 | Demo & documentation | ✅ done | `app.py`, `factanchor demo`, this README, `reports/final_report.md` |

## Results (measured)

Everything below was produced by the commands shown, in this repository, on the sandbox
(2 vCPU, CPU-only torch 2.14.1). **Toy-model numbers are plumbing checks, not findings.**
These run artifacts predate the review fixes described in the current source; rerun evaluation
before treating their metric values as current.

| Experiment | Command | Result |
|---|---|---|
| Test suite | `python -m pytest tests -q` | **230 passed** in 11.73 s (recorded before the current review fixes; not rerun here) |
| `α=0` ≡ baseline | `factanchor generate --mode baseline` vs `--mode factanchor --alpha 0` | identical answer and token count (verified) |
| Baseline run | `factanchor run … --condition base_model_no_context` | 8/8 ok, 0.58 ms/token median (toy) |
| FactAnchor run | `factanchor run … --condition factanchor_alpha0p5_rate0p5` | 6/8 ok, 2 degraded (`no_linked_entities`), 1.84 ms/token median (toy) |
| Latency overhead | `artifacts/metrics/ablation_smoke_toy.json` | **+235 % to +293 % generation time** vs baseline for α > 0 (consistent with two forward passes per token); α = 0 grid conditions +5 % … +41 % |
| Ablation grid | `python scripts/run_ablation.py --config configs/experiments_smoke_toy.yaml --dataset data/processed/dev_smoke.jsonl --limit 8` | 16 conditions ran, 0 errors, plots written |
| Factuality | same runs | **No scientific result.** The toy model emits uninterpretable text; claims without usable evidence are now reported as unjudged, not hallucinated. Recompute old run metrics with the current evaluator before comparing them. |

The one *scientific* statement these numbers support: the contrastive loop costs roughly
2–4× generation time on the same device, and the harness records that overhead paired per
example. No factuality claim is made, because none is measurable without real weights.

## What is blocked

| Blocker | Impact | Consequence in this repo |
|---|---|---|
| No network access | `datasets` loaders, Wikidata/DBpedia API, HF Hub downloads cannot run here | providers and loaders are implemented with real request code, caching, timeouts, retries and error paths; offline verification uses `JsonFileKnowledgeProvider` and `MockKnowledgeProvider` |
| 2 vCPU / ~1 GB RAM, CPU-only torch | Llama-3-8B (and any ≥1B model in fp32) is infeasible | `ToyCausalLMAdapter` (NumPy, deterministic) stands in for development; model id is a config value, never hard-coded |
| `datasets` and `streamlit` not installed in the sandbox | dataset/Streamlit paths unexecutable here | guarded imports with actionable messages; `tests/test_demo_app.py` covers the demo's logic without Streamlit |
| No git repository in the sandbox | manifest `git_hash` is `null` | `utils.git_info()` returns `None` rather than inventing a hash |

## Testing

```bash
python -m pytest tests -q                 # 230 tests, all offline, ~12 s
python -m pytest tests -m "not slow" -q   # fast subset
python -m compileall -q src scripts app.py
```

| File | Focus |
|---|---|
| `test_config.py` | precedence, validation errors, secrets redaction, condition grids |
| `test_corruption.py` | strategy coverage, rates, seeds, audit trail, "no safe replacement" path |
| `test_decoding.py` | score formula, constraints, EOS suppression, cache/uncached agreement, branch alignment |
| `test_entity_pipeline.py` | extraction, linking ambiguity, empty results, Wikidata time formatting |
| `test_metrics.py` | claim extraction, judges, aggregation, CIs, paired latency overhead |
| `test_prompts.py` | context separation (corrupted facts must never reach the base prompt), templates |
| `test_model_adapter.py` | log-softmax stability, BranchCache semantics, toy model determinism |
| `test_integration.py` | end-to-end baseline & FactAnchor runs, retrieval fallback, evaluation |
| `test_reproducibility.py` | identical seeds → identical corruptions; run directories never overwritten |
| `test_cli.py` | every CLI subcommand, exit codes, `--set` overrides |
| `test_demo_app.py` | Streamlit demo logic without the Streamlit UI |

---

## Repository layout

```
factanchor/
├── README.md, instructions.md, pyproject.toml, .env.example, app.py
├── configs/          default.yaml, baseline.yaml, experiments.yaml, experiments_smoke_toy.yaml, smoke_toy.yaml
├── data/             README.md (licences), raw/ (git-ignored), processed/{offline_knowledge,dev_smoke}.jsonl
├── artifacts/        runs/<run_id>/{config,environment,manifest,generations,errors,metrics,logs,plots}
├── examples/         prompts.jsonl (12 hand-written demo prompts)
├── docs/             architecture, method, datasets, reproducibility, limitations
├── reports/          final_report.md (spec §19 structure)
├── scripts/          download_data, run_baseline, run_factanchor, evaluate, run_ablation, doctor, demo
├── src/factanchor/   config, schemas, entities, knowledge/{base,pipeline,wikidata}, corruption, prompts,
│                     model_adapter, decoding, generation, metrics, evaluation, experiment, cli, utils
└── tests/            11 test modules (230 tests) + conftest.py
```

## CLI reference

```bash
factanchor --help
factanchor doctor      [--config …] [--check-model] [--check-knowledge]
factanchor generate    --prompt "…" | --prompt-file FILE | --prompt -   # stdin
                       [--mode baseline|factanchor] [--alpha A] [--max-new-tokens N]
                       [--show-contexts] [--save-run] [--json] [--set key=value …]
factanchor run         --dataset FILE_OR_KEY [--split S] [--limit N] [--condition NAME]
                       [--run-id ID] [--no-warmup] [--quiet] [--json]
factanchor evaluate    --input RUN_DIR_OR_JSONL [--judge heuristic_lexical|model|gold_reference]
                       [--dataset …] [--output PATH]
factanchor ablate      [--ablation configs/experiments.yaml] [--dataset …] [--limit N] [--json]
factanchor datasets    --list | --inspect | --download NAME [--max-examples N]
factanchor demo        --prompt "…" [--alpha A] [--corruption-rate R]
factanchor cache       stats | clear [--all]
```

Every subcommand accepts `--config PATH` and repeatable `--set dotted.path=value` overrides;
unknown keys are rejected (`extra="forbid"`), so a typo fails loudly instead of being ignored.
`python scripts/<name>.py` wrappers accept the same flags (see spec §14 interface).

## Method and design decisions

* **Strict branch separation** (spec §4.1). The base branch never receives corrupted facts.
  Trusted facts in the base prompt exist only as the separately named condition
  `trusted_facts_baseline` (retrieval-augmented baseline), exactly so a retrieval-only gain
  cannot be mistaken for a contrastive-decoding gain. `tests/test_prompts.py` asserts the
  corrupted objects never appear in `base_context`.
* **Which contrast context?** `z_c` is built from **corrupted** facts, i.e. the "amateur"
  distribution of spec §3.1 — not the trusted facts. Retrieval-augmented decoding with
  trusted context in the contrast branch would push *toward* the truth rather than away from
  a plausible falsehood; that variant is available as a separate, named ablation rather than
  being conflated with the main method.
* **Numerically stable scoring.** Both branches are `log_softmax`-normalised in float64
  before subtraction; non-finite scores are replaced by `-1e30` (never NaN); `α`,
  `β`-style constraints, repetition penalty, no-repeat-ngram and adaptive plausibility masks
  are applied to the base branch, then re-applied to the contrastive score so the two cannot
  drift apart.
* **Alignment invariant.** The contrast branch conditions on the *same* generated suffix as
  the base branch; the KV caches are asserted to stay offset by exactly
  `len(contrast_ids) − len(base_ids)` every step (`_check_suffix_alignment`).
* **No silent fallback.** No facts ⇒ the run is *degraded* and recorded
  (`degraded_reason="no_contrast_facts"`) and the decoder follows the configured
  `on_missing_contrast` policy; a retrieval outage raises or falls back to the configured
  provider, never to fabricated facts. OOM raises.
* **Corruption is auditable and conservative.** Seeded RNG per call, rate-limited selection,
  type-compatible candidate pools, `strategy: none` disallowed at rate > 0, facts without a
  safe replacement are left unchanged *and recorded as skipped with a reason*, and the
  original records are never mutated (deep-copy test).
* **Two judges plus gold labels.** `heuristic_lexical` (default, offline, deterministic),
  `model` (reuses `advance()` from the model adapter — the code path that tested KV caching,
  so a cache bug cannot hide), `gold_reference` (FEVER-style label matching). Metrics never
  merge human labels with judge labels, and FActScore is labelled an approximation.

## Deviation from spec §21

Spec §21 tells the coding LLM to implement **Phase 0 only** and stop for review. The user
instruction for this repository was to implement the whole project in one pass while keeping
the phase structure, so that is what was done: every phase is implemented and testable, and
`instructions.md` reports per phase **implemented / tests run / results / limitations / next
phase**, with the phase exit criteria listed explicitly. No phase is skipped, and no file
claims a result that was not produced by a command recorded in this repository.

## Responsible use

* Generated text requires independent verification. This prototype has not established that
  FactAnchor reduces hallucinations; it is **not** a truth guarantee and does not eliminate
  hallucination.
* Corrupted facts are synthetic experiment inputs. They are labelled as such in every
  prompt, record and UI, and must never be quoted as evidence.
* Retrieved content (Wikipedia/Wikidata/DBpedia/datasets) is treated as untrusted input: it
  is never executed, only formatted into text with length and count caps.
* Secrets live in environment variables only; `.env.example` contains placeholders, logs are
  redacted, and the demo never prints the environment.
* Respect API terms/rate limits (built-in rate limiter + disk cache) and dataset licences
  ([data/README.md](data/README.md)).

## Citation and licence

Research prototype, no licence is asserted for the datasets or model weights it consumes
(see [data/README.md](data/README.md)); the code in this repository is provided for
evaluation of the method described in `FactAnchor.md`.
