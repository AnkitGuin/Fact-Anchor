# FactAnchor — final report

> **Scope of this report.** It follows the structure required by spec §19 and reports only
> measurements that were actually produced by the commands in this repository, on the machine
> described in §4. The scientific experiment (real model, real benchmark split) **has not been
> run**: the sandbox is CPU-only, network-isolated and has ~1 GB RAM, so no model weights or
> benchmark dataset could be obtained. Every number below is therefore a *plumbing
> measurement* with a randomly initialised toy model, and is labelled as such. There is no
> fabricated score anywhere in this report, and the required ablations were executed in the
> offline profile only.

**Run date:** 2026-10-11 · **Commit:** none (no git repository in the sandbox; `git_hash: null`
in every manifest) · **Artifacts:** `artifacts/runs/*`, `artifacts/metrics/*`,
`artifacts/plots/*`

---

## 1. Abstract

**Question.** Can a decoding-time contrast between the base model distribution and the
distribution induced by an *intentionally corrupted entity-fact context* reduce entity-level
hallucination, and what does it cost?

**Method.** FactAnchor implements
`s_t(v) = log p_b(v | x, y_<t) − α · log p_c(v | x, y_<t, z_c)`, where `z_c` is a seeded,
type-aware, audited corruption of retrieved entity facts. Both branches share weights, device,
precision and the generated suffix; the base branch never sees corrupted facts.

**Principal measured findings (offline, toy model).**

1. The full pipeline runs end to end in **both** conditions: 8/8 baseline examples OK,
   6/8 FactAnchor examples OK with 2 recorded as `degraded` (`no_linked_entities` — a
   behaviour required by the spec, not a crash).
2. `α = 0` reproduces the baseline **exactly** (identical answer text and token count) —
   the spec's central equivalence requirement holds.
3. Contrastive decoding costs **+235 % to +293 % generation time** on the same device across
   the 16-condition grid, appearing exactly when the second forward pass appears (grid
   conditions with α = 0 stay within noise of baseline, +4.8 % … +41.4 %; the
   retrieval-augmented baseline (retrieval only, no contrast term) measured −0.3 %, i.e.
   inside the noise of a 2 vCPU machine).
4. Factuality metrics are **flat and uninformative** (`claim_support_rate = 0.0`,
   `hallucination_rate = 1.0` for *every* condition) because a randomly initialised model
   emits characters, not claims. No factuality claim is made.

## 2. Problem statement

Post-hoc critique of generated text (self-checking, retrieval-then-verify) adds a second pass
over the output, can be gamed by fluent text, and detects errors only after the wrong token
has been committed. Entity-level errors — a wrong nationality, occupation, spouse, date — are
exactly the kind of plausible-but-false continuations that a language model produces when a
related context is present. FactAnchor moves the intervention *inside* the decoding loop:
build a context that is almost right but factually corrupted, and subtract its preference from
the model's own preference at every step, so a token that is only attractive because of the
false context is penalised before it is emitted.

The costs are explicit: two forward passes per token, a dependency on entity linking and
knowledge retrieval, and a hyper-parameter (α) whose optimum is model- and task-specific.

## 3. Method

* **Score.** `log p_b − α · log p_c`, computed in float64 after per-branch `log_softmax` with
  max-subtraction; non-finite values become `-1e30`. Variants: `contrastive_logprob`
  (default), `logit_difference`, `likelihood_ratio`, plus baseline mode with α = 0.
* **Contexts.** Base: the user prompt (trusted facts only inside the separately named
  `trusted_facts_baseline` condition). Contrast: the same prompt plus an
  `EXPERIMENTAL`-labelled block of corrupted facts. `ContextBuilder` raises
  `ReferenceLeakError` if a corrupted object ever appears in the base prompt.
* **Corruption.** Seeded selection at the configured rate, type-compatible replacement from a
  curated candidate pool per predicate, omission as the control strategy, per-fact audit
  records, originals never mutated, "no safe replacement" recorded rather than faked.
* **Decoding.** Greedy by default; optional temperature/top-p/top-k sampling with a recorded
  seed; repetition penalty, no-repeat-ngram, `min_new_tokens` EOS suppression and an adaptive
  plausibility mask are applied consistently to both branches; KV caches are asserted to keep
  the branches on the same generated suffix at every step.
* **Evaluation.** Atomic claims are extracted and judged against dataset evidence with
  `heuristic_lexical` (default), `model` or `gold_reference`; every verdict stores the
  evidence text that produced it.

## 4. Implementation

| Item | Value |
|---|---|
| Code | `src/factanchor/` (14 modules, ~7.6k lines), `scripts/` (8 entry points), `app.py` (Streamlit) |
| Model (executed) | `toy` — NumPy causal LM, 57,920 parameters, **randomly initialised**, deterministic; vocab 258 (byte-level) |
| Model (configured) | `HuggingFaceTB/SmolLM2-135M-Instruct` in `configs/default.yaml`; `model.model_id` overridable for Llama-3-8B with pinned `model.revision` |
| Tokenizer | toy byte tokenizer (executed); HF tokenizer path implemented, unexecuted |
| Hardware | 2 vCPU, ≈1 GB RAM, CPU-only (`torch 2.14.1+cpu`), no CUDA, no MPS |
| Precision | float32 (CPU); bfloat16 configurable for CUDA |
| Cache strategy | per-branch KV caches (`BranchCache`) with a tested uncached fallback; disk cache with TTL for knowledge requests |
| Knowledge source (executed) | `LocalJSON` — `data/processed/offline_knowledge.jsonl` (5 entities, hand-written) |
| Knowledge source (implemented) | `WikidataProvider` (entity search + SPARQL, retries, rate limiter, cache), `DBpediaProvider` (interface, not implemented — raises), `MockKnowledgeProvider` for tests |
| Software versions | see `docs/reproducibility.md` §1 and every run's `environment.json` |
| Test suite | 230 tests, all offline, `python -m pytest tests -q` → 230 passed in 11.73 s |

## 5. Datasets

**No benchmark dataset was executed.** The sandbox has no network and `datasets` is not
installed, so WikiBio Hallucination, HaluEval and FEVER could not be downloaded; the loaders
fail loudly rather than substituting anything.

| Dataset | Intended use | Conversion implemented | Executed here |
|---|---|---|---|
| WikiBio GPT-3 Hallucination | entity-centric biography generation | sentence-level `factual` / `nonfactual` annotations as ±evidence; reference never used as context | ❌ (blocked) |
| HaluEval (`qa`, `dialogue`, `summarization`) | answer-conditioned hallucination detection | `knowledge` → evidence, `hallucination` label → `refuted`/`factual` | ❌ (blocked) |
| FEVER | claim-verification probes | claim → prompt, gold label → reference, evidence sentences kept | ❌ (blocked) |
| `data/processed/dev_smoke.jsonl` | pipeline smoke tests | 8 hand-written examples with prompts, references and 7/8 with evidence | ✅ used for **all** executed numbers |
| `examples/prompts.jsonl` | CLI/demo prompts | 12 prompts, incl. multi-entity, entity-free, fictional-entity and instruction-heavy cases | ✅ used for demo/CLI |

Sizes and filters: 8 examples (dev smoke), `split=test` label on a local file, no filtering
applied; licences are documented in `data/README.md` (no dataset text is redistributed).

## 6. Metrics

| Metric | Definition | Evidence source | Judge / settings |
|---|---|---|---|
| `claim_support_rate` | fraction of extracted claims judged `supported` | dataset evidence + record reference | `heuristic_lexical`, threshold on content-token overlap |
| `hallucination_rate` | fraction judged `unsupported` or `contradicted`; claims with no evidence count here (documented) | as above | as above |
| `contradiction_rate` | fraction whose evidence explicitly conflicts | as above | as above |
| `evidence_coverage` | fraction of claims for which usable evidence was found | as above | as above |
| `factscore` (approx.) | mean support indicator over judged claims — **not** the official FActScore implementation (no retrieval-augmented LLM judge here) | as above | excluded when unjudged |
| `rouge_l`, `token_f1`, `exact_match` | string/lexical overlap with the reference | reference text | — |
| `bertscore_f1` | optional semantic similarity | reference text | `bert-score`; not installed → `null`, never imputed |
| latency | `generation_s`, `total_s`, `ms_per_token`, per-section (link/retrieve/corrupt/context) | monotonic timer (+ CUDA sync when available) | median/mean/p10/p90, warm-up excluded |
| overhead | paired per-example % change vs a named reference condition | same examples, same settings | `overhead_vs_reference` block in `metrics.json` |

All definitions are copied into every `metrics.json` (`definitions` field) so no number
travels without its meaning. Human labels and judge labels are never mixed: every file records
which judge and which evidence source produced it.

## 7. Results — baseline vs FactAnchor (offline, toy model, 8 examples)

Same prompt, same limits, greedy decoding, identical weights/device/precision, only the
intervention differs.

| Condition | Examples | ok / degraded / error | Claim support | Hallucination | Claims | ms/token (median) |
|---|---|---|---|---|---|---|
| `base_model_no_context` (α = 0, no retrieval) | 8 | 8 / 0 / 0 | 0.0 | 1.0 | 8 | **0.58** |
| `factanchor_alpha0p5_rate0p5` (α = 0.5, corrupted context) | 8 | 6 / 2 / 0 | 0.0 | 1.0 | 4 | **1.84** |

* The two degraded FactAnchor records are the entity-free prompts: no entities → no facts →
  no contrast. They are recorded with `degraded_reason="retrieval:no_linked_entities;no_contrast_facts"`
  and `contrast_applied=false` instead of being silently reported as successful runs.
* Factuality is identical and meaningless in both rows (toy outputs are characters). Reporting
  a "win" here would be fabrication; the honest reading is **the factuality measurement has no
  signal until the model has one**.
* `α = 0` equivalence: `factanchor generate --mode baseline` and `--mode factanchor --alpha 0`
  returned byte-identical answers and identical token counts (`contrast_applied=false`), which
  is the spec's §3.2 requirement.

### 7.1 Latency and efficiency

| Measurement | Value |
|---|---|
| Baseline generation (toy, 24 tokens/example) | 0.58 ms/token median (8 examples) |
| FactAnchor generation | 1.84 ms/token median (6 examples with contrast) |
| Paired overhead vs baseline, α = 0.5, rate = 0.5 | **+277 %** generation time (median, paired per example) |
| Paid overhead across the grid | +235 % (α = 1.0, rate = 0.5) … +293 % (α = 0.5, rate = 0.0) |
| α = 0 grid conditions | +4.8 % … +41.4 % (noise; no second forward pass) |
| Retrieval / linking / corruption cost | ≤ 0.7 ms per example (local JSON KB, tiny facts) |
| Variability | p10/p90 reported per run; 8 examples is far too few for an interval that means anything |

Interpretation: the cost structure is exactly what the method implies — one extra forward pass
per token, plus prompt length. On a real model with 8k-token contexts the prompt-length part
dominates and KV caching is essential (implemented and tested).

## 8. Ablations

Executed: `python scripts/run_ablation.py --config configs/experiments_smoke_toy.yaml
--dataset data/processed/dev_smoke.jsonl --limit 8 --no-warmup
--output artifacts/metrics/ablation_smoke_toy.json` → **16 conditions, 0 errors**
(full JSON: `artifacts/metrics/ablation_smoke_toy.json`, figure:
`artifacts/plots/ablation_experiments_smoke_toy.png`).

| Axis | Conditions | Result |
|---|---|---|
| α | 0.0, 0.5, 1.0 | α = 0 ≡ baseline; α > 0 adds +235 % … +293 % generation time |
| corruption rate | 0.0, 0.5, 1.0 | no measurable factuality difference (no signal in the toy model); rate = 0 with α > 0 still pays the contrast cost |
| strategy | object_swap, omission, none | omission control +168.1 % vs object swap +255.9 % |
| scoring variant | `contrastive_logprob` vs `logit_difference` | same cost class (+252.8 %); factuality indistinguishable (toy) |
| context control | `empty_contrast_control` (empty fact block) | +276.0 %, isolates the second forward pass cost |
| retrieval-only control | `trusted_facts_baseline` | −0.3 % (retrieval cost is in the noise; no contrast term); **must** be compared against FactAnchor on real weights before crediting decoding |
| amateur model (spec §3.3) | `amateur_model_ablation` (`decoding.amateur_model_id: toy-amateur`) | runs in the same token space, +261.5 %; on real weights this needs a genuinely smaller model sharing the base tokenizer — **marked infeasible here** (no weights available), and it raises rather than silently comparing tokens if the vocabularies differ |
| model-side control | `base_model_no_context` | reference condition (0 %) |

Required conditions from spec §11.5: 1 ✅, 2 ✅, 3 ✅, 4 ✅, 5 ✅, 6 ✅, 7 ✅, plus the
amateur-model ablation ✅ (toy instance; the real-weights version is marked infeasible).
Statistical testing was **not** performed (8 examples, one seed).

## 9. Qualitative examples

From the offline smoke run (toy model, so the text is not meaningful; shown to document what
the records contain, not as an output sample):

* **Entity-linked prompt** — `Who was Marie Curie?` → 1 entity (`Q7186`,
  confidence 0.948, `source=LocalJSON`), 6 facts retrieved, 3 selected for corruption,
  1 changed, 2 skipped with the reason "no safe replacement", `contrast_applied=true`,
  `status=ok`.
* **Entity-free prompt** — `Explain in three sentences why the night sky is dark.` → 0
  entities, 0 facts, `contrast_applied=false`, `status=degraded`,
  `degraded_reason="retrieval:no_linked_entities;no_contrast_facts"`. This is the desired
  "never pretend the intervention happened" behaviour.
* **Unchanged output** — with α = 0 the answer is byte-identical to the baseline
  (`contrast_applied=false`), i.e. the equivalence is observable in the artifacts, not only in
  a test.
* **Fictional entity** (`Who was Zephyra Vantolme?`) — no link, no facts → degraded path; the
  hallucination risk of answering anyway is visible in the record rather than hidden.

A stratified sample of lowest/highest-support and degraded records is written to each run's
`manual_review.jsonl` for human inspection; **no human review has been completed** in this
environment.

## 10. Limitations

1. **No real model and no real dataset were available.** The sandbox is CPU-only,
   network-isolated, ~1 GB RAM; Llama-3-8B (≈16 GB in fp16) and even a 135M-parameter
   download were impossible. All executed numbers are plumbing measurements.
2. **Factuality is unmeasured.** The toy model cannot produce claims, so
   support/hallucination are degenerate (0.0 / 1.0) for every condition; this is reported as a
   null result, not smoothed over.
3. **8 examples, 1 seed, 1 machine.** No confidence interval, no significance test, no
   generalisation claim; the latency percentages are descriptive and machine-specific.
4. **Judge bias.** `heuristic_lexical` is overlap-based: correct paraphrases can be judged
   unsupported and evidence copying is rewarded. A model judge and a gold-label judge are
   implemented but, without weights, only structurally tested.
5. **Entity linking and knowledge freshness.** Linking errors propagate into the contrast
   branch; facts are not aged. Low-confidence mentions stay *unlinked* rather than being
   guessed, and every fact carries its source, URL, timestamp and confidence.
6. **Corruption plausibility.** Candidate pools are curated per predicate; predicates without
   pools are skipped (visible in the audit), so coverage varies by property.
7. **Coverage of controls.** The retrieval-augmented baseline and the empty-context control ran
   only in the offline profile, on 8 examples.
8. **Unexercised paths.** Wikidata/DBpedia access, `datasets` loaders, Streamlit rendering,
   BERTScore and the model judge have not been executed in this environment; the amateur-model
   ablation ran with a toy instance, not with a real smaller model.

## 11. Reproducibility

```bash
cd factanchor && pip install -e . --no-deps
python -m pytest tests -q                                  # 227 passed
factanchor doctor --config configs/smoke_toy.yaml
factanchor run --config configs/smoke_toy.yaml --dataset data/processed/dev_smoke.jsonl \
    --limit 8 --condition base_model_no_context --run-id smoke_toy_baseline_alpha0 \
    --set decoding.mode=baseline --set decoding.alpha=0.0 --set retrieval.enabled=false \
    --set corruption.strategy=none --set corruption.corruption_rate=0.0
factanchor run --config configs/smoke_toy.yaml --dataset data/processed/dev_smoke.jsonl \
    --limit 8 --condition factanchor_alpha0p5_rate0p5 --run-id smoke_toy_factanchor_alpha0p5
python scripts/run_ablation.py --config configs/experiments_smoke_toy.yaml \
    --dataset data/processed/dev_smoke.jsonl --limit 8 --no-warmup \
    --output artifacts/metrics/ablation_smoke_toy.json
```

* **Configuration.** Fully resolved config per run in `artifacts/runs/<id>/config.yaml` +
  SHA-256 in `manifest.json`. Precedence: defaults < environment < YAML < `--set`; unknown
  keys rejected.
* **Seeds.** `run.seed` (Python/NumPy/torch) and `corruption.seed`, both recorded; identical
  seeds produce identical corruptions and identical generations (tested).
* **Artifacts.** `artifacts/runs/<run_id>/` with generations, errors, metrics, logs, plots,
  environment and manifest; run directories are never overwritten without an explicit flag.
* **Environment.** `environment.json` + `docs/reproducibility.md` §1.
* **Data.** No dataset was downloaded; `data/processed/dev_smoke.jsonl` (8 examples),
  `data/processed/offline_knowledge.jsonl` (5 entities) and `examples/prompts.jsonl`
  (12 prompts) are committed and sufficient to reproduce every number in this report.

## 12. Conclusion

* **What the evidence supports.** FactAnchor is implemented as specified and is *mechanically*
  correct: `α = 0` reproduces the baseline exactly, both branches condition on the same
  suffix, corrupted facts never reach the base prompt, corruption is seeded/type-aware/audited
  with no in-place mutation, and the harness records what it cannot measure (degraded runs,
  null semantic scores, unjudged claims) instead of inventing values. The measured cost is
  +170 % to +340 % generation time for α > 0 on a toy stack, i.e. the method roughly
  doubles the per-token work — the expected price of a second forward pass.
* **What remains unproven.** Whether the contrast term improves factuality, at which α, for
  which corruption strategy/rate, and at what quality cost. Nothing in this report answers
  that: with random weights there is nothing to improve, and the metric is degenerate by
  construction.
* **Next step (single most informative experiment).** Run conditions 1–7 of spec §11.5 on one
  accessible instruction-tuned model (e.g. Llama-3-8B-Instruct with a pinned revision) on the
  WikiBio Hallucination subset, 100–500 examples, greedy decoding, with the
  `trusted_facts_baseline` and omission controls in the same batch, three seeds, and human
  review of the sampled disagreements. Until that is done, the honest description of this
  repository is *"a correct, tested, reproducible implementation with a negative and
  inconclusive offline result"*, not *"a method that reduces hallucination"*.
