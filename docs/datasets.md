# Datasets

The spec ( §11) names three datasets and insists they are **not interchangeable**. This file
records what each one is used for here, how it is converted, and what must be checked before
publishing numbers.

| Key | Dataset | Task used for | Evidence material | Licence / access |
|---|---|---|---|---|
| `wikibio_hallucination` | WikiBio GPT-3 Hallucination (`potsawee/wiki_bio_gpt3_hallucination`) | entity-centric biography generation | `wiki_bio_text` reference + per-sentence `annotation` labels | public research dataset; cite the original paper; do not redistribute raw text |
| `halueval` | HaluEval (`pminervini/HaluEval`, configs `qa`, `dialogue`, `summarization`) | knowledge-grounded answer-generation probe derived from QA records | `knowledge` field; `hallucination` labels the source candidate answer | research use; follow the dataset card |
| `fever` | FEVER (`fever` / `fever_v1.0`) | claim verification probes (`SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`) | `evidence` sentences with page titles | CC BY-SA 3.0 (wikipedia text) + dataset terms; do not redistribute |
| `truthfulqa` | TruthfulQA generation config | misuse/truthfulness probe | `correct_answers` / `incorrect_answers` | research use |
| `dev_smoke` (local) | `data/processed/dev_smoke.jsonl` | offline pipeline tests | hand-written sentences | written for this repo; explicitly **not** a benchmark |
| `examples` (local) | `examples/prompts.jsonl` | demo/CLI prompts | short hand-written notes | written for this repo |

## Conversion rules implemented

* **WikiBio Hallucination** — prompt is the entity question (`Who is <entity>?`), generated
  answer is evaluated against `wiki_bio_text` and the sentence annotations. The lexical judge
  excludes sentences annotated `nonfactual`; the `gold_reference` judge can use these labels to
  measure agreement with the dataset's annotations. Reference text is **never** put in the
  generation context.
* **HaluEval** — the `knowledge` field is evidence and `question` is the prompt. The original
  `hallucination` label applies to the dataset's candidate answer, not to the knowledge passage;
  it is preserved as a gold label and is not transferred to evidence or used to judge a newly
  generated answer. This conversion is not the original HaluEval detection benchmark. Records
  whose fields do not match the documented layout are dropped **with a logged reason and a
  count**; they are never silently reinterpreted.
* **FEVER** — a claim-verification probe: the claim becomes the prompt, the gold label is the
  reference, evidence sentences are the evidence. Because this is not free-form generation,
  results are reported separately from the generation benchmarks and the conversion is
  described in the run's manifest.
* **TruthfulQA** — used only as a misuse probe; incorrect answers are never used as evidence.

## Rules honoured by the loaders

1. Licences/access are checked and documented before download (`factanchor datasets --list`
   prints them).
2. Raw data goes to `data/raw/` which is git-ignored; only derived, small files are committed.
3. Version/split metadata and the loader version are stored in each example's `provenance`
   and in the run manifest.
4. Train/dev/test separation is preserved; tuning uses the development subset
   (`evaluation.development_subset`), never the test split.
5. Subset size and filtering rules are written into the run's metrics (`n_examples_loaded`,
   `dataset`, `split`).
6. Human labels and judge labels are never mixed without stating which is which
   (`judge`, `evidence_source` fields in every metrics file).
7. **If a dataset cannot be obtained, the loader fails loudly** (`DatasetUnavailable` or the
   `datasets` import error) and the report says the benchmark was not run. A substitute is
   never presented as the original.

## Current status in this environment

`datasets` is not installed and the sandbox has no network, so no named benchmark was
downloaded. Consequently:

* every executed number in `reports/final_report.md` comes from `data/processed/dev_smoke.jsonl`
  (8 hand-written examples) with the toy model, and is labelled as a plumbing check;
* the loaders, conversions and field-inspection helper (`factanchor datasets --inspect`) are
  implemented and unit-tested against recorded record shapes;
* running the real benchmarks is the first scientific next step — see
  [limitations.md](limitations.md).
