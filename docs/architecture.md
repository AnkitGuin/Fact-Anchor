# Architecture

Spec §4 lists 13 components. Each one maps to a module, and the mapping is one-directional:
a component never reaches into another component's internals except through the names below.

| # | Spec component | Module / symbol |
|---|---|---|
| 1 | Input adapter | `schemas.Example`, `evaluation.load_examples` (`load_jsonl_examples`, `load_hf_examples`) |
| 2 | Entity extractor | `entities.build_extractor`, `RuleBasedEntityExtractor`, `LexiconEntityExtractor`, `SpacyEntityExtractor` |
| 3 | Entity linker | `entities.link_entities` / `KnowledgePipeline.link` |
| 4 | Fact retriever | `knowledge.base.*Provider`, `knowledge.pipeline.KnowledgePipeline.retrieve` |
| 5 | Fact normaliser | `knowledge.pipeline` (qualified/labelled `Fact` records, `knowledge.wikidata.format_wikidata_time`) |
| 6 | Corruption engine | `corruption.CorruptionEngine`, `corrupt_facts`, `engine_from_config` |
| 7 | Context builder | `prompts.ContextBuilder`, `build_contexts`, `ContextPair` |
| 8 | Model adapter | `model_adapter.ModelAdapter`, `HFCausalLMAdapter`, `ToyCausalLMAdapter`, `BranchCache` |
| 9 | Decoding engine | `decoding.ContrastiveDecoder`, `contrastive_scores`, `generation.TextGenerator` |
| 10 | Output postprocessor | `generation` (stop-sequence trimming and record assembly in `generate_example`) |
| 11 | Evaluator | `metrics` (judges, claim scoring), `evaluation` (datasets, run-level evaluation) |
| 12 | Experiment runner | `experiment.create_run`, `run_condition`, `run_ablation`, `write_run_plots` |
| 13 | Demo | `cli.cmd_generate/cmd_demo`, `app.py` (Streamlit) |

## Data flow (one example)

```
Example (prompt, reference, evidence)
   │
   ├─ entities.extract_entities ──► EntityMention[]
   │        └─ linker ──► EntityRecord (unlinked records are kept, not dropped)
   │
   ├─ knowledge.retrieve ──► Fact[]  (trusted branch input, provenance + confidence)
   │
   ├─ corruption.corrupt_facts(facts, strategy, rate, seed)
   │        ├─► corrupted Fact[]  + per-fact CorruptionRecord audit
   │        └─► original Fact[]   (never mutated)
   │
   ├─ prompts.ContextBuilder.build            ┌─ base context: prompt only
   │        └─ ContextPair                    └─ contrast context: prompt + EXPERIMENTAL
   │                                             header + corrupted fact block
   ├─ generation.generate_example
   │        ├─ base branch: p_b(v | x, y_<t)              ─┐
   │        ├─ contrast branch: p_c(v | x, y_<t, z_c)     ─┤ same suffix, same weights
   │        ├─ scores = log p_b − α · log p_c             ─┘
   │        └─ constraints → select → append token → check alignment
   │
   └─ GenerationRecord (prompt, contexts, facts, audit, timings, status, error)
              │
              └─ metrics.score_record ──► ExampleMetrics + ClaimJudgement[]
                        └─ aggregate_metrics ──► metrics.json (+ per-example rows)
```

## Invariants enforced in code

| Invariant | Where | Test |
|---|---|---|
| The base context never contains corrupted facts | `prompts.ContextBuilder.build` raises `ReferenceLeakError` if a corrupted object appears in the base prompt | `tests/test_prompts.py` |
| Trusted facts in the base prompt only as a named condition | `context.trusted_facts_in_base` + `trusted_facts_baseline` condition | `tests/test_prompts.py`, `tests/test_integration.py` |
| Both branches condition on the same suffix | `generation._check_alignment`, `_check_suffix_alignment` (offset = `len(contrast_ids) − len(base_ids)`) | `tests/test_decoding.py` |
| `α = 0` ⇒ baseline, exactly | `decoding.step` skips the contrast branch; `DecodingConfig` forbids `mode=baseline` with `α≠0` | `tests/test_decoding.py`, verified end-to-end |
| Corrupted facts are never written back into retrieved facts | `corruption` deep-copies before mutation; `CorruptionResult.original_facts` is the untouched input | `tests/test_corruption.py` |
| No silent fallback | `retrieval_fallback` policy, `on_missing_contrast` policy, `Status.DEGRADED` + `degraded_reason`, OOM raises | `tests/test_integration.py`, `tests/test_decoding.py` |
| Artifacts are never overwritten implicitly | `utils.resolve_run_dir` suffixes `-1`, `-2`, … unless `run.overwrite`/`run.allow_run_dir_reuse` | `tests/test_reproducibility.py` |
| One run directory carries everything needed to reproduce it | `experiment.create_run` writes config, environment, manifest, logs; `run_condition` writes generations/errors/metrics/plots | `tests/test_reproducibility.py` |
| No secrets in configs, logs or artifacts | `config.redact_secrets`, `.env.example` placeholders | `tests/test_config.py` |

## Directory responsibilities

* `configs/` — every experiment is a file. `default.yaml` (full settings),
  `baseline.yaml` (baseline mode, retrieval off), `experiments.yaml` (α × corruption grid +
  named controls), `smoke_toy.yaml` (offline stack), `experiments_smoke_toy.yaml`
  (offline grid).
* `data/` — `raw/` is git-ignored; `processed/` holds the offline knowledge base and
  smoke/dev example files; `data/README.md` documents licences and the JSONL envelope.
* `artifacts/runs/<run_id>/` — `config.yaml`, `environment.json`, `manifest.json`,
  `generations.jsonl`, `errors.jsonl`, `metrics.json`, `logs.txt`, `plots/`,
  `manual_review.jsonl`.
* `src/factanchor/knowledge/` — a subpackage, because the provider, the HTTP/cache layer and
  the orchestration have genuinely different reasons to change.
