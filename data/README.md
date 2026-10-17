# `data/` layout

```
data/
├── raw/          # downloads; NOT versioned (git-ignored)
├── processed/    # normalised JSONL the pipeline consumes; NOT versioned
└── README.md
```

## What is versioned

| Path | Versioned | Why |
| --- | --- | --- |
| `processed/offline_knowledge.jsonl` | yes | tiny synthetic knowledge base used by the **offline** quick-start (`configs/smoke_toy.yaml`) and by the tests. It is clearly synthetic data, not retrieved evidence. |
| `processed/dev_smoke.jsonl` | yes | 8 hand-written prompt/reference examples for smoke runs. Not a benchmark. |
| everything else | no | dataset licenses, size, and redistribution terms. |

## Dataset access (Section 11.4)

Raw data is **not** committed.  `scripts/download_data.py` downloads and
normalises the supported datasets; it checks availability and licensing first
and reports failures explicitly instead of silently substituting something else.

| Name | Source | License / terms | Filename |
| --- | --- | --- | --- |
| WikiBio "Hallucination" (a.k.a. WikiBio / GPT-3 biography generations) | HF `potsawee/wiki_bio_gpt3_hallucination` (data derived from Wikipedia, CC BY-SA 3.0) | research use; cite Manakul et al. 2023 | `processed/wikibio_hallucination.jsonl` |
| HaluEval (QA / dialogue / summarization subsets) | HF `pminervini/HaluEval` (a.k.a. `HaluEval`) | research use; cite Li et al. 2023 | `processed/halueval_<subset>.jsonl` |
| FEVER (claim verification) | HF `fever` / `fever_v1.0` | CC BY-SA 3.0 for Wikipedia text; cite Thorne et al. 2018 | `processed/fever.jsonl` |
| Open-domain PQA (fallback) | HF `truthful_qa` (`generation`) | Apache-2.0; cite Lin et al. 2021 | `processed/truthfulqa.jsonl` |

Why an extra dataset: the three named datasets do **not** all support
free-form generation evaluation directly.

* **WikiBio Hallucination** is the correct fit for entity-centric biography
  generation: each record is a model-written biography plus annotated
  `factual`/`nonfactual` sentence spans and the entity name.
* **HaluEval** records are *labelled pairs* (question/answer, knowledge/answer)
  and are designed for detecting hallucination in *given* text, not for scoring
  fresh generations.  We therefore expose a **claim-detection** conversion
  (`task=halueval_detection`) and report it separately from generation metrics.
* **FEVER** is claim verification with `SUPPORTS` / `REFUTES` /
  `NOT ENOUGH INFO` labels.  We expose a **claim-probe** conversion
  (`task=fever_claims`: generate a verdict answer for a claim) and document the
  mapping.  It is *not* a generation benchmark.
* **TruthfulQA** is included because it is an open-domain generation benchmark
  with reference answers and works even when the three above are unavailable.

Every processed record uses this envelope (see `factanchor.schemas.Example`):

```json
{
  "example_id": "wikibio::0",
  "dataset": "wikibio_hallucination",
  "split": "test",
  "task": "generation",
  "prompt": "Who is Marie Curie?",
  "reference": "…gold biography or reference answer (may be null)…",
  "evidence": [{"text": "…supporting passage…", "source": "wikibio", "url": null}],
  "metadata": {"entity": "Marie Curie", "licence": "…"},
  "provenance": {"loader_version": "0.1.0", "source_url": "...", "downloaded_at": "ISO-8601"}
}
```

## Offline / air-gapped use

```bash
python scripts/download_data.py --list                 # show availability, do nothing
python scripts/download_data.py --dataset wikibio_hallucination --max-examples 50
python scripts/download_data.py --dataset all --dry-run
```

If a download fails (no network, licence not accepted, dataset renamed
upstream) the script exits non-zero with the exact error and writes nothing.
It never writes a "similar" dataset under the expected filename.
