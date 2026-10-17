# Method

## 1. The score

At step `t`, with prompt `x`, generated suffix `y_<t`, and the corrupted entity-fact context
`z_c`:

```
s_t(v) = log p_b(v | x, y_<t)  −  α · log p_c(v | x, y_<t, z_c)
```

`p_b` and `p_c` are produced by **the same weights, precision and device**. Only the context
differs, and only the contrast branch sees `z_c`.

Implementation (`decoding.contrastive_scores`):

1. both logit vectors → `log_softmax` in **float64** (subtract max, so ±1e30 logits do not
   overflow);
2. `scores = log_pb − α · log_pc`;
3. non-finite entries are replaced by a large negative constant (`-1e30`) rather than
   propagated as `NaN`;
4. optional normalisation (`normalize_scores`) and the plausibility mask
   (`adaptive_plausibility`, keeping tokens with `p_b ≥ ε · max p_b`) are applied afterwards,
   so they can never reorder a token that the contrast term already eliminated.

## 2. What each variant means

| `decoding.variant` | Score | Notes |
|---|---|---|
| `contrastive_logprob` (default) | `log p_b − α · log p_c` | the method the spec defines; both terms are proper log-probabilities |
| `logit_difference` | `z_b − α · z_c` (raw logits) | ablation: no normalisation, so it is scale-sensitive to temperature and to differences in logit magnitude |
| `likelihood_ratio` | `log p_b − α · (log p_c − log p_b)` | an alternative parameterisation kept for comparison |
| `baseline` mode (`α = 0`) | `log p_b` | must reproduce baseline **exactly**; `mode=baseline` with `α ≠ 0` is rejected by the config validator |

Setting `decoding.mode: baseline` never builds the contrast context at all (no wasted forward
pass), and `decoding.on_missing_contrast: skip_contrast` (default) makes a FactAnchor run with
no usable facts follow the baseline path while being recorded as `degraded`
(`degraded_reason="no_contrast_facts"`), so it cannot be reported as a successful FactAnchor
run.

## 3. Which context goes where

| Branch | Context | Rationale |
|---|---|---|
| base | user prompt (optionally + trusted facts **only** in the `trusted_facts_baseline` condition) | the model the user would get without the intervention |
| contrast `z_c` | user prompt + experimental header + **corrupted** facts | the "amateur/incorrect" distribution whose plausible-but-wrong tokens must be penalised |

Two consequences are deliberate:

* the contrast branch uses the *corrupted* context rather than the trusted one. Contrasting
  against trusted facts would push the model **away** from the truth; the spec's construction
  is contrast against a plausible falsehood;
* `decoding.amateur_model_id` optionally decodes the contrast branch with a **second, weaker
  model** (the "amateur model" ablation of spec §3.3). Both models must share a token
  vocabulary, because the score compares log-probabilities token by token; a mismatched
  vocabulary raises a readable error instead of silently subtracting incomparable numbers.
  The second model's load time is `latency.contrast_model_load_s` (kept out of generation
  timing) and the id is stored in the record;
* adding trusted facts to the base branch is a **separately named condition**
  (`trusted_facts_baseline`), because retrieval augmentation alone improves factuality and
  would otherwise be credited to contrastive decoding.

`ContextBuilder.build` raises `ReferenceLeakError` if a corrupted object string appears in the
base prompt, and `tests/test_prompts.py` asserts this for every corruption strategy.

## 4. Constraints and how they stay consistent

Decoding constraints (repetition penalty, no-repeat n-gram, `min_new_tokens`, stop
sequences, EOS handling) are applied in one place. When the contrast term is active the
constraint delta computed on the base distribution is re-applied to the contrastive score, so
a token banned by n-gram blocking cannot reappear because the contrast branch liked it.
`min_new_tokens` suppresses EOS for the first *N* steps, and
`ContrastiveDecoder.eos_suppression_events` records how many times that happened (visible in
the diagnostics, testable).

## 5. Corruption

`corrupt_facts(facts, strategy, corruption_rate, seed)`:

1. selects `round(rate · |facts|)` facts with a seeded `random.Random(seed)` (a rate of 1.0
   must be requested explicitly; the default is 0.5);
2. for each selected fact, draws a replacement from a **type-compatible** candidate pool
   (`CANDIDATE_POOLS`, keyed by object type and predicate), never from the fact itself;
3. if no safe replacement exists — unknown predicate, empty pool, or a pool containing only
   the original value — the fact is **left unchanged** and a `CorruptionRecord` with
   `is_corrupted=False` and an explicit `skipped_reason` is written;
4. `strategy: omission` removes the fact instead of falsifying it (the spec's control
   condition); `strategy: none` is only legal at `corruption_rate: 0`;
5. nothing is mutated in place — corrupted facts are new `Fact` records, originals are
   returned untouched, and both are stored in the `CorruptionResult`.

Every corruption therefore satisfies: seeded, reproducible, type-aware, auditable
(original fact id, original/corrupted object, strategy, seed, skip reason), and labelled as
synthetic in the prompt text, the records and the UI.

## 6. Generation loop

```
encode(prompt) → ids
for step in 0..max_new_tokens−1:
    base_logits      = advance(tokens_fed_to_base)
    contrast_logits  = advance(tokens_fed_to_contrast)      # only if α > 0 and facts exist
    scores           = contrastive_scores(base_logits, contrast_logits, α, variant)
    scores           = constraints(scores)                  # penalty / n-gram / EOS / plausibility
    if do_sample: token = sample(scores, temperature, top_p, top_k) else: token = argmax(scores)
    append(token); check alignment; break on EOS / stop sequence
```

Guarantees:

* **same suffix in both branches** — the contrast branch receives exactly the tokens the base
  branch produced, and the caches are asserted to keep the constant offset
  `len(contrast_ids) − len(base_ids)` at every step (`_check_suffix_alignment`);
* **cached and uncached agreement** — `decoding.use_cache: false` recomputes full prefixes and
  must select identical tokens under deterministic decoding (tested);
* **timing** — `time.monotonic()` per step plus `torch.cuda.synchronize()` on CUDA; model load
  time, entity/link/retrieval/corruption/context-build times and generation time are recorded
  separately, with a warm-up generation excluded from the statistics (the policy string is
  stored in the manifest and in `metrics.json`);
* **no silent substitution** — an out-of-memory error raises; a retrieval outage follows the
  configured `retrieval.fallback_on_unavailable` policy (`cache_only`, `no_facts`, `error`)
  and is recorded in the record status.

## 7. Evaluation

Claim-level, evidence-backed:

1. extract atomic claims from the answer (rule-based; dedicated extractors per task);
2. for each claim, look for evidence from the example (dataset evidence) or the record's
   reference;
3. judge with one of `heuristic_lexical` (default: content-token overlap against the best
   matching evidence sentence), `model` (budgeted, deterministic LLM judge reusing the model
   adapter's `advance()` path), or `gold_reference` (FEVER-style label mapping);
4. keep the verdict **and the evidence text** in `ClaimJudgement`, so any number can be
   audited back to the sentence that produced it;
5. aggregate with explicit denominators and bootstrap confidence intervals.

Definitions live in `metrics.METRIC_DEFINITIONS` and are copied into every `metrics.json`, so
no number travels without its definition.
