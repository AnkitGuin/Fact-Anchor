# Limitations, failure modes and what to do next

This file is deliberately conservative: it lists what the prototype *cannot* support today,
what could go wrong when it is run at scale, and what each mitigation in the code actually
does (and does not) buy.

## 1. Limitations of the executed evidence

| # | Limitation | Effect on claims |
|---|---|---|
| 1 | **No real model weights** were available (CPU-only, 1 GB RAM, no network). All end-to-end runs use `ToyCausalLMAdapter`, a randomly initialised NumPy transformer. | No factuality claim of any kind is supported. The toy runs verify plumbing, determinism, artifact completeness and cost accounting only. |
| 2 | **No benchmark dataset** was downloaded (no network, `datasets` not installed). Executed runs use 8 hand-written examples. | No generalisation, no split integrity, no comparison with published numbers. |
| 3 | **8 examples, 1 seed, CPU.** | No significance testing. Anything said about latency is a paired description, not an estimate. |
| 4 | `bert-score`, `streamlit`, spaCy NER models and Wikidata/DBpedia access were unavailable. | BERTScore/semantic-similarity and the Streamlit UI are implemented but unexecuted; the UI logic is unit-tested without rendering. |
| 5 | Model judge (`claim_judge: model`) reuses the toy/real adapter, but no real weights were present. | The model-judge path is exercised only structurally. |

## 2. Method-level limitations

* **Contrast against synthetic falsehoods is not truth.** The method penalises tokens that the
  corrupted context makes likely; it does not verify anything. A model that ignores the
  contrast context at some steps simply decodes as the baseline there (the smoke run shows
  `contrast_applied = true` but outputs that are pure noise — the mechanism ran; it had
  nothing true to protect).
* **Corruption quality bounds the method.** Pools are curated per predicate (`CANDIDATE_POOLS`)
  rather than derived from a full type hierarchy, so predicates without pools are skipped. If
  corruptions are implausible (e.g. swapping in a nonsense value), the contrast branch is easy
  to distinguish and the method should help less, not more; conversely, near-miss corruptions
  (same country, adjacent date) are where the effect should appear — that hypothesis is
  untested here.
* **α is a hyper-parameter with a cost.** Larger α means more deviation from the base model in
  general (fluency, coverage, diversity), not only more factuality. The grid exists precisely
  because the trade-off must be measured, not assumed.
* **Entity linking errors propagate.** A wrong link corrupts the contrast branch. The pipeline
  keeps low-confidence mentions as *unlinked* rather than guessing, exposes
  `linking_confidence`, and records degraded reasons, but a confident wrong link is invisible
  to it.
* **Knowledge freshness.** Facts come from the configured provider with `retrieved_at`; the
  prototype neither ages facts nor detects stale ones.
* **Evaluation bias.** The heuristic judge measures lexical overlap with the supplied
  evidence: correct paraphrases can be judged unsupported, and copying evidence text is
  rewarded. It is a proxy, and every metrics file says so next to the numbers.

## 3. Engineering failure modes and the mitigation that is actually implemented

| Failure | Mitigation in this codebase | Residual risk |
|---|---|---|
| Corrupted branch unrelated to the answer | facts are restricted to the prompt's linked entities, capped by `retrieval.max_facts_per_entity`, with no-fact and omission controls | entity-free prompts always degrade to baseline (recorded as `degraded`) |
| Contrastive score unstable / overconfident | float64 log-softmax, non-finite → `-1e30`, α tuned only on development data, `logit_difference` and `likelihood_ratio` variants for comparison | a bad α still changes outputs; the ablation is the only honest answer |
| Wrong entity linked | confidence threshold (`retrieval.min_linking_confidence`), explicit unlinked state, candidate list stored in the record | high-confidence errors remain |
| Knowledge source slow/unavailable | timeouts, bounded retries, exponential back-off, disk cache with TTL, rate limiter, `fallback_on_unavailable` policy (`cache_only` / `no_facts` / `error`) | cache staleness is bounded by TTL, not eliminated |
| Model ignores fact context | the record keeps both contexts and the contrast flag; failures are reported as negative results | an invisible null result cannot be distinguished from "no effect" without inspecting prompts (run with `--show-contexts`) |
| Judges disagree | per-claim verdicts with evidence text are stored; `manual_review.jsonl` samples disagreements for human inspection | human review has not been performed here |
| Apparent gains come from added context | `trusted_facts_baseline` condition separates retrieval benefit from decoding benefit | requires the condition to actually be run |
| Latency overhead excessive | KV caching, capped facts, per-section timing, paired overhead reporting, warm-up excluded | two forward passes per token remain the dominant cost (+170 % … +340 % measured on the toy stack) |
| Cache/uncached divergence | both paths are tested for identical token selection under greedy decoding; step caches carry a length and a kind | sampling variance is separate from cache correctness and is not covered by that test |

## 4. What should happen next (in order)

1. **Run phase 1–6 again with real weights** on one GPU, 100–500 examples, greedy decoding,
   two or three seeds, with `trusted_facts_baseline` and the omission control in the same
   batch. Without this, the project has no scientific result.
2. **Validate the wiki/HaluEval/FEVER conversions** with `factanchor datasets --inspect`
   before trusting any field mapping; adjust and re-run.
3. **Human-label a stratified sample** (the `manual_review.jsonl` files are already written)
   and report judge–human agreement next to the judge numbers.
4. **Tune α and the corruption rate on development data only**, and report the test-split
   curve afterwards (never the other way round).
5. **Report negative results.** If contrastive decoding does not improve supported-claim rate
   beyond the retrieval-augmented baseline, that is the finding, and it is as publishable as a
   positive result.
