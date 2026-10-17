# instructions.md — how this repository was built, phase by phase

This file answers spec §21 ("Instructions to the coding LLM") literally, including the part
where it was deliberately not followed, and reports, per phase: **implemented / tests run /
results / known limitations / next phase**. Every claim here is reproducible with the commands
shown; nothing was measured on a machine other than the one described in
[docs/reproducibility.md](docs/reproducibility.md).

## 0. The deviation, stated up front

Spec §21 says: *"Start with Phase 0 only … Do not begin the full decoding algorithm until
Phase 0 has been reviewed."*

The user instruction for this task was: *"Implement the entire project (each and every code
file)"*, plus *"ensure that there are no logical as well as implementation mistakes"*.

**Resolution.** The request from the user (owner of the repository) superseded the
phase-gating instruction, so the whole project was implemented in one reviewable tree.
What was preserved from §21 is everything that protects quality:

1. the phase structure itself — every phase below has its exit criteria, evidence and status;
2. no fabricated data, no placeholder that silently returns invented values;
3. exact commands and outcomes are reported, including failures and blocked items;
4. baseline and experimental paths stay separate and separately named;
5. artifacts are never overwritten without an explicit flag;
6. run artifacts, tests and logs are committed alongside the code that produced them.

What was *not* preserved: the "stop and wait for review after Phase 0" gate. The
consequence is real and worth stating: phases 1–7 were built before phase 0's hardware
findings were reviewed, and the hardware findings (CPU-only, 1 GB RAM, no network) are exactly
why the executed end-to-end results use the toy model. Had the review gate been honoured, the
reviewer would have been told "no usable model weights are available offline" before the
decoding loop was written.

## 1. Environment inspected before writing code (Phase 0)

```bash
python3 -V; python3 - <<'PY'
import importlib; [print(m, getattr(importlib.import_module(m), "__version__", "?")) for m in ("numpy","yaml","torch","transformers","pydantic","pytest")]
PY
nproc; free -m
python3 -c "import torch; print(torch.cuda.is_available())"
```

| Finding | Value |
|---|---|
| Python | 3.13.16 |
| CPU / RAM | 2 vCPU, ≈1 GB RAM |
| torch | 2.14.1+cpu, `cuda_available=False`, no MPS |
| transformers / tokenizers | 5.19.0 / 0.23.3 (installed, but unusable without weights) |
| datasets / streamlit / bert-score | not installed |
| numpy / pyyaml / pydantic / matplotlib / pytest | 2.3.5 / 6.0.3 / 2.13.5 / 3.10.9 / 9.0.3 |
| network | isolated (no dataset, model, or Wikidata access) |

**Consequence for model choice:** Llama-3-8B is out of reach (≈16 GB fp16, no GPU) and
SmolLM2-135M would need a download that is not possible offline. A deterministic NumPy toy
causal LM (`model.model_id: toy`) was implemented *as a first-class adapter* — not as a stub —
so that the decoding loop, cache alignment and latency accounting are executed for real.
`model.model_id` stays a configuration value, and `configs/default.yaml` names a small real
model so the switch to real weights is a one-line change.

## 2. Phase-by-phase report

### Phase 0 — Environment and feasibility · ✅ complete

* **Implemented:** `src/factanchor/utils.py` (`collect_environment`, `git_info`,
  `Timer`, `set_global_seed`), `src/factanchor/cli.py::doctor`, `pyproject.toml` with extras,
  `.env.example`, `configs/default.yaml`, `scripts/doctor.py`, `docs/reproducibility.md`.
* **Tests run:** `python -m pytest tests/test_config.py tests/test_reproducibility.py -q` →
  35 passed. `factanchor doctor --config configs/smoke_toy.yaml --json` → config valid,
  `torch ok`, `datasets missing`, `streamlit missing`.
* **Results:** feasibility established for CPU/offline development; blockers recorded in the
  README table.
* **Limitations:** the machine used has no GPU, so all timings are CPU numbers and all
  latency overheads are CPU overheads.
* **Next phase:** baseline generation — blocked on weights, hence the toy adapter.

### Phase 1 — Baseline generation · ✅ complete

* **Implemented:** `model_adapter.py` (`ModelAdapter` ABC, `HFCausalLMAdapter`,
  `ToyCausalLMAdapter`, `BranchCache`), `decoding.py` (constraints, selection, `ContrastiveDecoder`),
  `generation.py` (`TextGenerator`, `generate_baseline`), `configs/baseline.yaml`,
  `scripts/run_baseline.py`.
* **Tests run:** `python -m pytest tests/test_decoding.py tests/test_model_adapter.py -q` →
  49 passed (cache/uncached agreement, EOS and stop-sequence handling, deterministic seeds,
  `min_new_tokens`, NaN-free log-softmax over ±1e30 logits).
* **Command + result:**

  ```bash
  factanchor run --config configs/smoke_toy.yaml --dataset data/processed/dev_smoke.jsonl \
      --limit 8 --condition base_model_no_context --run-id smoke_toy_baseline_alpha0 \
      --set decoding.mode=baseline --set decoding.alpha=0.0 --set retrieval.enabled=false
  # -> 8 example(s), 8 ok, 0 degraded, 0 errors; median 0.58 ms/token; artifacts in
  #    artifacts/runs/smoke_toy_baseline_alpha0/
  ```
* **Limitations:** no real-weights baseline has been run here; the CLI/config for it is
  complete but unexercised.
* **Next phase:** knowledge pipeline.

### Phase 2 — Entity and knowledge pipeline · ✅ complete (offline KB); ⚠️ Wikidata unexercised

* **Implemented:** `entities.py` (rule-based extractor with offsets/confidence/unlinked
  state, lexicon backend, optional spaCy backend, linking with ambiguity candidates),
  `knowledge/base.py` (provider ABC, bounded/cached HTTP client, mock and JSON-file
  providers, `KnowledgeUnavailable`), `knowledge/pipeline.py` (extract → link → retrieve →
  normalise → build `FactRecord`s with provenance and `source_label`), `knowledge/wikidata.py`
  (Wikibase time precision formatting, SPARQL/DBpedia as explicitly unimplemented).
* **Tests run:** `python -m pytest tests/test_entity_pipeline.py -q` → 38 passed (empty
  results, wrong entity, confidence threshold, low-confidence unlinked state, precision
  codes 0–11, `None` for garbage timestamps, provider outage → configured fallback).
* **Command + result:**

  ```bash
  factanchor generate --config configs/smoke_toy.yaml --prompt "Who was Marie Curie?" --json
  # -> 1 linked entity (Q7186, confidence 0.948, source LocalJSON), 6 facts, 1/6 corrupted
  ```
* **Limitations:** the Wikidata provider is written against the real API (SPARQL + entity
  JSON, rate limiting, retries, disk cache, TTL) but could not be executed offline; its
  parsing/formatting logic is unit-tested with recorded payload shapes instead.
* **Next phase:** corruption.

### Phase 3 — Corruption pipeline · ✅ complete

* **Implemented:** `corruption.py` — `corrupt_facts(facts, strategy, corruption_rate, seed)`,
  `CorruptionEngine`, strategies `object_swap`, `attribute_perturbation`,
  `entity_substitution`, `omission`, `none`; type-compatible candidate pools; per-fact
  `CorruptionRecord` audit (original fact id, original/corrupted object, seed, strategy,
  `skipped_reason`, `is_corrupted`).
* **Tests run:** `python -m pytest tests/test_corruption.py -q` → 28 passed (zero/one/all
  rates, determinism across identical seeds, invalid rate → readable error, strategy `none`
  with a non-zero rate rejected, originals never mutated, "no safe replacement" recorded
  rather than faked).
* **Results:** the smoke run reports `selected=3, changed=1, skipped=2` — i.e. two selected
  facts had no type-compatible replacement and were left untouched **and logged**, exactly as
  spec §8.3 requires.
* **Limitations:** candidate pools are curated Wikidata-property lists
  (`CANDIDATE_POOLS`), not a full type hierarchy; unknown predicates are skipped rather than
  guessed.
* **Next phase:** contrastive decoding.

### Phase 4 — Contrastive decoding · ✅ complete

* **Implemented:** `decoding.contrastive_scores` (`log_pb − α·log_pc`, log-softmax in
  float64, non-finite → `-1e30`), `ContrastiveDecoder.step` (base-side constraints re-applied
  to the contrastive score, EOS suppression for `min_new_tokens`, plausibility masking,
  logit-difference and likelihood-ratio variants), `generation.TextGenerator` (cached and
  uncached paths, per-step CUDA synchronisation, monotonic timer, branch-alignment checks),
  `prompts.py` (`ContextBuilder`, strict base/contrast separation).
* **Tests run:** `python -m pytest tests/test_decoding.py tests/test_prompts.py -q` → 48
  passed, including `test_alpha_zero_equals_baseline` and
  `test_corrupted_facts_never_reach_the_base_context`.
* **Command + result (α = 0 equivalence, spec §3.2 / DoD):**

  ```bash
  factanchor generate --config configs/smoke_toy.yaml --prompt "Who was Marie Curie?" --mode baseline --json
  factanchor generate --config configs/smoke_toy.yaml --prompt "Who was Marie Curie?" --mode factanchor --alpha 0 --json
  # both -> answer '…' identical, 24 tokens, stop_reason=max_new_tokens, contrast_applied=False
  ```
* **Limitations:** the sampled (non-greedy) path is tested for reproducibility with fixed
  seeds but not benchmarked for variance across seeds on a real model.
* **Next phase:** evaluation.

### Phase 5 — Evaluation · ✅ complete

* **Implemented:** `metrics.py` (claim extraction, three judges, per-claim evidence
  preservation, `claim_support_rate`, `hallucination_rate`, `contradiction_rate`,
  `evidence_coverage`, approximate FActScore, token-F1/ROUGE-L/EM, optional BERTScore,
  bootstrap CIs, paired latency overhead, metric definitions embedded in the output),
  `evaluation.py` (dataset loaders, `evaluate_records`, `evaluate_run_dir`,
  `write_manual_review_sample`), `scripts/evaluate.py`.
* **Tests run:** `python -m pytest tests/test_metrics.py -q` → 29 passed (empty answers,
  judgement without evidence, CI bounds, median-of-two overhead, judge edge cases).
* **Command + result:**

  ```bash
  python scripts/evaluate.py --input artifacts/runs/smoke_toy_baseline_alpha0 \
      --output artifacts/metrics/evaluation_baseline_smoke.json
  # -> 8 records, 8 examples; judge=heuristic_lexical; metrics.json written next to the run
  ```
* **Limitations:** with a toy model every answer is characters, so every claim is
  `no_evidence`: support 0.0 / hallucination 1.0 for *both* conditions. This is reported as a
  null result, not as a FactAnchor failure. The model judge is implemented but needs real
  weights; BERTScore needs `bert-score` plus a model download.
* **Next phase:** ablations.

### Phase 6 — Ablations and analysis · ✅ complete (offline grid)

* **Implemented:** `experiment.py` (`create_run`, manifests, environments, per-condition run
  directories, `run_ablation`, paired overhead vs a named reference condition, plots),
  `configs/experiments.yaml` (α × corruption-rate grid plus named controls),
  `configs/experiments_smoke_toy.yaml` (same structure, offline), `scripts/run_ablation.py`.
* **Command + result:**

  ```bash
  python scripts/run_ablation.py --config configs/experiments_smoke_toy.yaml \
      --dataset data/processed/dev_smoke.jsonl --limit 8 --no-warmup \
      --output artifacts/metrics/ablation_smoke_toy.json
  # -> 16 conditions, 0 errors; plot at artifacts/plots/ablation_experiments_smoke_toy.png
  ```
* **Measured (toy, CPU, 8 examples, 24 tokens/example):** generation-time overhead versus
  `base_model_no_context`: `+169 %` (omission control) … `+337 %` (α=1.0, rate=0). Grid
  conditions with α=0 stay within noise of baseline (`+4.8 %…+41.4 %`), i.e. the cost appears
  exactly where the second forward pass appears; the retrieval-augmented baseline shows
  `−0.3 %` (retrieval only, and its cost is inside the noise of a 2 vCPU box). Factuality is flat (0.0 support / 1.0 hallucination)
  everywhere, as expected for random weights.
* **Limitations:** 8 examples, one seed, no significance testing — the JSON contains an
  explicit `comparison_note` saying so. Statistics must be recomputed on a real model and a
  real split.
* **Next phase:** demo and documentation.

### Phase 7 — Demo and documentation · ✅ complete

* **Implemented:** `app.py` (Streamlit: prompt box, α / corruption-rate / token / mode
  controls, side-by-side outputs, entity-fact-corruption audit expanders, latency and token
  counts, responsible-use banner, graceful API/model failure handling),
  `factanchor demo` (console), `scripts/demo.py`, this file, `README.md`, `docs/*`,
  `reports/final_report.md`.
* **Tests run:** `python -m pytest tests/test_demo_app.py tests/test_cli.py -q` → 19 passed
  (demo config coupling, paired generation, model-failure path, and a check that the demo
  never dumps the environment).
* **Command + result:** `factanchor demo --config configs/smoke_toy.yaml --prompt "Who was
  Marie Curie?"` prints both answers, the linked entity, the fact count, the corruption audit
  summary and the "not evidence" warning.
* **Limitations:** Streamlit is not installed in the sandbox, so the UI was verified through
  its logic layer (`tests/test_demo_app.py`) and `python -m py_compile`; the rendering itself
  has not been seen running.
* **Next phase:** none — spec phases are all implemented; next work is scientific (real
  weights, real splits, multiple seeds, human review sample).

## 3. Verification summary (exact commands and outcomes)

```bash
python -m pytest tests -q                       # 230 passed in 11.73s
python -m compileall -q src scripts app.py      # clean
factanchor doctor --config configs/smoke_toy.yaml
factanchor demo --config configs/smoke_toy.yaml --prompt "Who was Marie Curie?"
factanchor run --config configs/smoke_toy.yaml --dataset data/processed/dev_smoke.jsonl --limit 8
python scripts/run_ablation.py --config configs/experiments_smoke_toy.yaml --dataset data/processed/dev_smoke.jsonl --limit 8
python scripts/evaluate.py --input artifacts/runs/smoke_toy_baseline_alpha0 --output artifacts/metrics/evaluation_baseline_smoke.json
```

## 4. Mistakes found and fixed during this build (kept as a record)

| # | Mistake | Symptom | Fix |
|---|---|---|---|
| 1 | `--json` defined twice (parent + demo subparser) | `argparse.ArgumentError` on every command | flags moved to the shared parent parser; no duplicate definitions |
| 2 | `--run-id` accepted but ignored by `factanchor run` | run directories ignored the requested id | `run_condition(..., run_id=...)` threaded through to `create_run` |
| 3 | `--set corruption.strategy=none` coerced to `None` | config validation error, YAML and CLI disagreed | null is only `null`/`~`/empty; `none` stays the literal string the YAML uses |
| 4 | `metrics.json` top-level `condition` was `null` | consumers could not tell which condition was scored | `evaluate_run_dir` now records the condition (or `"multiple"`) |
| 5 | `fallback_on_unavailable: mock` in an ablation profile | invalid literal, whole grid aborted | corrected to `no_facts`; invalid conditions now abort loudly instead of silently skipping |
| 6 | A `baseline`-mode condition carried `alpha=0.5` | ambiguous record (baseline with contrast weight) | config validator rejects `mode=baseline` with `alpha≠0`; grid configs and the demo set α=0 |
| 7 | CLI read metrics at the wrong nesting level | console printed `support=None` although metrics existed | `summary.metrics["by_condition"][condition]` |
| 8 | `errors.jsonl` missing from run directories with no errors | documented layout incomplete | created empty at run creation |
| 9 | `write_ablation_plot` existed but was never called | no ablation figure | wired into `factanchor ablate` |
| 10 | `fallback_on_unavailable: mock` was not a valid literal; the whole grid aborted | invalid literal | corrected to `no_facts` |

## 5. Known limitations (summary)

1. No real model: all executed numbers come from a randomly initialised toy LM.
2. No real dataset: `dev_smoke.jsonl` is 8 hand-written examples, not a benchmark split.
3. Wikidata/DBpedia/`datasets`/Streamlit/`bert-score` code paths are implemented but
   unexercised in this environment.
4. Ablations are 8 examples × 1 seed; no significance testing was performed, so no
   comparison in this repo is a statistical result.
5. The heuristic judge measures lexical overlap with supplied evidence; it will call correct
   paraphrases unsupported and can be fooled by copying evidence text. It is a proxy, and the
   report says so.
6. Corruption pools are curated, not derived from a full type hierarchy.
