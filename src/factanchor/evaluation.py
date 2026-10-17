"""Dataset loaders and evaluation of saved generations (spec Section 11, Phase 5).

Principles
----------
* Datasets are used for the task each converter actually constructs (spec 11.1-11.3):
  WikiBio -> entity biography generation with labelled factual/nonfactual spans;
  HaluEval QA -> a knowledge-grounded answer-generation probe; its original
  candidate-answer hallucination label is retained but is not a generated-answer score;
  FEVER -> claim verification (a separate task); TruthfulQA -> open-domain
  generation fallback.
* A loader never substitutes a different dataset for a missing one: it raises and
  reports the limitation.
* Evaluation reads **saved** generation artifacts, so every reported metric can be
  traced back to a specific generation record.
* ``datasets`` is imported lazily; offline environments can still evaluate JSONL.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .config import Config
from .metrics import (
    aggregate_metrics,
    bertscore,
    latency_overhead,
    score_record,
    summarise_latency,
)
from .schemas import Example, Evidence, ExampleMetrics, GenerationRecord
from .utils import get_logger, iter_jsonl, read_json, write_json

__all__ = [
    "DatasetSpec",
    "DATASET_SPECS",
    "load_examples",
    "load_jsonl_examples",
    "load_hf_examples",
    "available_datasets",
    "describe_dataset",
    "EvaluationOutput",
    "evaluate_records",
    "evaluate_run_dir",
    "DATASET_REQUIREMENTS_NOTE",
]

logger = get_logger("evaluation")

DATASET_REQUIREMENTS_NOTE = (
    "License/access notes: WikiBio GPT-3 hallucination data is derived from Wikipedia "
    "(CC BY-SA 3.0) and is provided for research use; HaluEval and FEVER are research "
    "datasets with their own terms; TruthfulQA is Apache-2.0. Raw files are never "
    "committed to the repository (data/raw is git-ignored). See data/README.md."
)


@dataclass(frozen=True)
class DatasetSpec:
    """How a source dataset maps onto the pipeline's :class:`Example` schema."""

    name: str
    hf_path: str | None
    hf_config: str | None
    task: str
    default_split: str
    description: str
    prompt_style: str
    evidence_source: str
    licence: str
    citation: str
    notes: str = ""


DATASET_SPECS: dict[str, DatasetSpec] = {
    "wikibio_hallucination": DatasetSpec(
        name="wikibio_hallucination",
        # The canonical public mirror of the WikiBio GPT-3 hallucination data.
        hf_path="potsawee/wiki_bio_gpt3_hallucination",
        hf_config=None,
        task="generation",
        default_split="eval",
        description=(
            "Entity-centric biography generation. Each record contains a model-written "
            "biography plus per-sentence factual/nonfactual annotation and the "
            "corresponding Wikipedia paragraph."
        ),
        prompt_style="Who is {entity}?",
        evidence_source="dataset annotation (factual spans) + Wikipedia paragraph",
        licence="research use; Wikipedia text CC BY-SA 3.0",
        citation="Manakul et al. 2023, 'SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection' (WikiBio data)",
        notes=(
            "The loader maps 'wiki_bio_text' to the reference, 'gpt3_text' sentences with "
            "annotation=='factual'/'nonfactual' to labelled evidence, and the entity name to "
            "the prompt. If upstream field names change, run `--inspect` and update the "
            "mapping; the loader raises instead of guessing."
        ),
    ),
    "halueval": DatasetSpec(
        name="halueval",
        hf_path="pminervini/HaluEval",
        hf_config="qa",
        task="generation",
        default_split="data",
        description=(
            "Knowledge-grounded answer-generation probe built from HaluEval QA records. "
            "The source task's candidate answer and hallucination label are retained for "
            "reference, but the converted prompt does not evaluate the original detection task."
        ),
        prompt_style=(
            "Given this knowledge:\n{knowledge}\n\nQuestion: {question}\n\n"
            "Answer with RESPONSE: <a factual answer>."
        ),
        evidence_source="the record's own 'knowledge' field",
        licence="research use",
        citation="Li et al. 2023, 'HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs'",
        notes=(
            "Conversion uses the question and knowledge to prompt a new grounded answer; the "
            "candidate answer is a semantic reference, and the knowledge is evidence. The "
            "original 'hallucination' flag is retained as a source label only; it does not "
            "label the knowledge or score the newly generated answer. This is not the original "
            "HaluEval detection benchmark."
        ),
    ),
    "fever": DatasetSpec(
        name="fever",
        hf_path="fever",
        hf_config="v1.0",
        task="fever_claims",
        default_split="labelled_dev",
        description=(
            "Claim verification with SUPPORTS / REFUTES / NOT ENOUGH INFO labels. Used as a "
            "controlled factuality probe, not as a generation benchmark."
        ),
        prompt_style=(
            "Claim:\n{claim}\n\nDoes the claim follow from the evidence? Answer with exactly one "
            "of: SUPPORTS, REFUTES, NOT ENOUGH INFO."
        ),
        evidence_source="FEVER evidence sentences (when present in the split)",
        licence="research use; Wikipedia text CC BY-SA 3.0",
        citation="Thorne et al. 2018, 'FEVER: a Large-scale Dataset for Fact Extraction and VERification'",
        notes=(
            "Label mapping: SUPPORTS -> factual evidence, REFUTES -> refuted evidence, "
            "NOT ENOUGH INFO -> no evidence (claims then remain unadjudicated by evidence-based metrics). "
            "The conversion is documented so the numbers are not mistaken for generation metrics."
        ),
    ),
    "truthfulqa": DatasetSpec(
        name="truthfulqa",
        hf_path="truthful_qa",
        hf_config="generation",
        task="generation",
        default_split="validation",
        description="Open-domain generation fallback with reference answers.",
        prompt_style="{question}",
        evidence_source="correct_answers (reference answers, not passages)",
        licence="Apache-2.0",
        citation="Lin et al. 2021, 'TruthfulQA: Measuring How Models Mimic Human Falsehoods'",
        notes=(
            "Evidence here is a list of correct answers rather than passages, so the lexical "
            "judge is a weak signal; the metric tables record evidence_source accordingly."
        ),
    ),
    "jsonl": DatasetSpec(
        name="jsonl",
        hf_path=None,
        hf_config=None,
        task="generation",
        default_split="test",
        description="Local JSONL of Example records (offline development and smoke tests).",
        prompt_style="{prompt}",
        evidence_source="per-record 'evidence' field",
        licence="user provided",
        citation="n/a",
    ),
}


def available_datasets() -> list[str]:
    return list(DATASET_SPECS)


def describe_dataset(name: str) -> dict[str, Any]:
    spec = _spec(name)
    return {
        "name": spec.name,
        "hf_path": spec.hf_path,
        "hf_config": spec.hf_config,
        "task": spec.task,
        "default_split": spec.default_split,
        "description": spec.description,
        "evidence_source": spec.evidence_source,
        "licence": spec.licence,
        "citation": spec.citation,
        "notes": spec.notes,
    }


def _spec(name: str) -> DatasetSpec:
    key = name.strip().lower()
    aliases = {
        "wikibio": "wikibio_hallucination",
        "wiki_bio": "wikibio_hallucination",
        "wiki_bio_gpt3_hallucination": "wikibio_hallucination",
        "wikibio_hallucination": "wikibio_hallucination",
        "halueval": "halueval",
        "fever": "fever",
        "truthfulqa": "truthfulqa",
        "truthful_qa": "truthfulqa",
        "jsonl": "jsonl",
    }
    if key not in aliases:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(set(aliases))}")
    return DATASET_SPECS[aliases[key]]


# ---------------------------------------------------------------------------
# JSONL loading (offline path)
# ---------------------------------------------------------------------------


def load_jsonl_examples(
    path: Path | str,
    *,
    split: str | None = None,
    max_examples: int | None = None,
    dataset_name: str = "jsonl",
) -> list[Example]:
    """Load and validate :class:`Example` records from a JSONL/JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"example file {path} not found")
    if path.suffix == ".json":
        payload = read_json(path)
        records = payload if isinstance(payload, list) else payload.get("examples", [])
    else:
        records = list(iter_jsonl(path))
    examples: list[Example] = []
    for index, record in enumerate(records):
        record = dict(record)
        record.setdefault("example_id", f"{dataset_name}::{index}")
        record.setdefault("dataset", dataset_name)
        # accept a plain {"prompt": ..., "reference": ...} file as well
        if "evidence" in record and isinstance(record["evidence"], list):
            record["evidence"] = [
                item if isinstance(item, dict) else {"text": str(item)} for item in record["evidence"]
            ]
        try:
            example = Example.model_validate(record)
        except Exception as exc:
            raise ValueError(f"invalid example at {path}:{index + 1}: {exc}") from exc
        if split is not None and example.split != split:
            continue
        examples.append(example)
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


# ---------------------------------------------------------------------------
# Hugging Face loading
# ---------------------------------------------------------------------------


def _require_datasets():  # pragma: no cover - optional dependency
    try:
        import datasets  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Loading Hugging Face datasets requires the optional extra: `pip install -e .[datasets]`. "
            "Offline evaluation from JSONL (data/processed/dev_smoke.jsonl) needs no extra."
        ) from exc
    return datasets


def _split_candidates(datasets_module: Any, spec: DatasetSpec) -> list[str]:
    try:
        builder = datasets_module.load_dataset_builder(spec.hf_path, spec.hf_config)  # type: ignore[arg-type]
        splits = list(builder.info.splits.keys())  # type: ignore[union-attr]
    except Exception:  # pragma: no cover - network / metadata failure
        return [spec.default_split]
    if not splits:
        return [spec.default_split]
    ordered = [spec.default_split] if spec.default_split in splits else []
    ordered.extend(name for name in splits if name not in ordered)
    return ordered


def load_hf_examples(
    name: str,
    *,
    split: str | None = None,
    max_examples: int = 50,
    inspect: bool = False,
) -> list[Example]:
    """Download/convert examples from Hugging Face for the given dataset key."""
    spec = _spec(name)
    if spec.hf_path is None:
        raise ValueError(f"dataset {name!r} has no Hugging Face source; use load_jsonl_examples()")
    datasets_module = _require_datasets()

    candidates = [split] if split else _split_candidates(datasets_module, spec)
    last_error: Exception | None = None
    for candidate in candidates:
        try:  # pragma: no cover - requires network + data
            dataset = datasets_module.load_dataset(spec.hf_path, spec.hf_config, split=candidate)
        except Exception as exc:
            last_error = exc
            logger.warning("could not load split %r of %s: %s", candidate, spec.hf_path, exc)
            continue
        if inspect:
            print(json.dumps({"dataset": spec.hf_path, "config": spec.hf_config, "split": candidate,
                              "columns": list(dataset.column_names),
                              "first_record": _jsonable(dataset[0])}, indent=2)[:4000])
        return _convert(spec, dataset, split=candidate, max_examples=max_examples)
    raise RuntimeError(
        f"could not load {spec.hf_path!r} (config={spec.hf_config!r}) from any split among {candidates}. "
        f"Last error: {last_error}. Check network access, the licence/terms of the dataset, or run with a "
        "local JSONL instead (data/processed/dev_smoke.jsonl). No substitute dataset was used."
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _convert(spec: DatasetSpec, dataset: Any, *, split: str, max_examples: int) -> list[Example]:
    """Convert a Hugging Face dataset into :class:`Example` records."""
    converters = {
        "wikibio_hallucination": _convert_wikibio,
        "halueval": _convert_halueval,
        "fever": _convert_fever,
        "truthfulqa": _convert_truthfulqa,
    }
    converter = converters.get(spec.name)
    if converter is None:  # pragma: no cover - guarded by DATASET_SPECS
        raise ValueError(f"no converter registered for {spec.name!r}")
    examples: list[Example] = []
    for index, record in enumerate(dataset):
        if len(examples) >= max_examples:
            break
        converted = converter(spec, record, index=index, split=split)
        if converted is not None:
            examples.append(converted)
    if not examples:
        raise RuntimeError(
            f"converting {spec.hf_path} produced no usable examples; inspect the fields with "
            "`python scripts/download_data.py --dataset ... --inspect` and update the converter "
            "(the pipeline will not fabricate examples)."
        )
    return examples


def _convert_wikibio(spec: DatasetSpec, record: dict[str, Any], *, index: int, split: str) -> Example | None:
    biography = _first_str(record, ("gpt3_text", "text", "biography"))
    reference = _first_str(record, ("wiki_bio_text", "wikipedia_text", "reference"))
    entity = _first_str(record, ("wiki_bio_entity", "entity", "title", "name"))
    if entity is None and biography:
        # "X is a ..." / "X was a ..." pattern used by the original data generation
        head = biography.strip().split(",")[0]
        if " is " in head[:120]:
            entity = head.split(" is ")[0].strip()
        elif " was " in head[:120]:
            entity = head.split(" was ")[0].strip()
    if entity is None:
        entity = f"record {index}"
    prompt = spec.prompt_style.format(entity=entity)

    sentences = record.get("gpt3_sentences")
    annotation = record.get("annotation")
    evidence: list[Evidence] = []
    if isinstance(sentences, (list, tuple)) and isinstance(annotation, (list, tuple)):
        if len(sentences) != len(annotation):
            logger.warning("wikibio record %d: %d sentences vs %d annotations", index, len(sentences), len(annotation))
        for sentence_index, (sentence, label) in enumerate(zip(sentences, annotation)):
            if not isinstance(sentence, str) or not sentence.strip():
                continue
            evidence.append(
                Evidence(
                    text=sentence.strip(),
                    source="wikibio_annotation",
                    sentence_index=sentence_index,
                    label=str(label).strip().lower(),
                )
            )
    if reference:
        evidence.append(Evidence(text=reference.strip(), source="wikibio_reference", label=None))
    return Example(
        example_id=f"wikibio::{index}",
        prompt=prompt,
        dataset=spec.name,
        split=split,
        task="generation",
        reference=reference,
        evidence=evidence,
        gold_labels={
            "n_factual_sentences": _count_label(evidence, "factual"),
            "n_nonfactual_sentences": _count_label(evidence, "nonfactual"),
        },
        metadata={"entity": entity, "licence": spec.licence},
        provenance={"source": spec.hf_path, "citation": spec.citation, "loader_version": "0.1.0"},
    )


def _convert_halueval(spec: DatasetSpec, record: dict[str, Any], *, index: int, split: str) -> Example | None:
    knowledge = _first_str(record, ("knowledge", "context"))
    question = _first_str(record, ("question", "query"))
    answer = _first_str(record, ("answer", "response"))
    hallucination = record.get("hallucination")
    if question is None or answer is None:
        return None
    hallucination_label = str(hallucination).strip().lower()
    if hallucination_label not in {"yes", "no", "true", "false", "1", "0"}:
        logger.warning("halueval record %d has an unrecognized hallucination label: %r", index, hallucination)
        return None
    is_hallucinated = hallucination_label in {"yes", "true", "1"}
    prompt = spec.prompt_style.format(knowledge=knowledge or "(none provided)", question=question)
    # The hallucination label applies to this dataset's candidate answer, not to
    # the knowledge passage.  Keep the passage as evidence without transferring
    # the candidate's label to it (GoldReferenceJudge would otherwise invert it).
    evidence = [Evidence(text=knowledge.strip(), source="halueval_knowledge")] if knowledge else []
    return Example(
        example_id=f"halueval::{index}",
        prompt=prompt,
        dataset=spec.name,
        split=split,
        task="generation",
        reference=answer,
        evidence=evidence,
        gold_labels={"hallucination": bool(is_hallucinated)},
        metadata={
            "licence": spec.licence,
            "note": "grounded-generation probe derived from HaluEval QA; not the original detection task",
        },
        provenance={"source": spec.hf_path, "citation": spec.citation, "loader_version": "0.1.0"},
    )


def _convert_fever(spec: DatasetSpec, record: dict[str, Any], *, index: int, split: str) -> Example | None:
    claim = _first_str(record, ("claim", "text"))
    label = record.get("label")
    if claim is None:
        return None
    label_text = str(label).strip().upper() if label is not None else "UNKNOWN"
    evidence: list[Evidence] = []
    raw_evidence = record.get("evidence")
    if isinstance(raw_evidence, list):
        for group in raw_evidence:
            for item in group if isinstance(group, list) else [group]:
                if isinstance(item, dict):
                    sentence = item.get("text") or item.get("sentence")
                    if isinstance(sentence, str) and sentence.strip():
                        evidence.append(
                            Evidence(
                                text=sentence.strip(),
                                source="fever_evidence",
                                label={"SUPPORTS": "factual", "REFUTES": "refuted"}.get(label_text),
                            )
                        )
                elif isinstance(item, str) and item.strip():
                    evidence.append(Evidence(text=item.strip(), source="fever_evidence"))
    return Example(
        example_id=f"fever::{record.get('id', index)}",
        prompt=spec.prompt_style.format(claim=claim),
        dataset=spec.name,
        split=split,
        task="fever_claims",
        reference=label_text,
        evidence=evidence,
        gold_labels={"fever_label": label_text},
        metadata={"licence": spec.licence, "note": "claim verification probe; not a generation benchmark"},
        provenance={"source": spec.hf_path, "citation": spec.citation, "loader_version": "0.1.0"},
    )


def _convert_truthfulqa(spec: DatasetSpec, record: dict[str, Any], *, index: int, split: str) -> Example | None:
    question = _first_str(record, ("question", "prompt"))
    if question is None:
        return None
    correct = record.get("correct_answers") or []
    best = _first_str(record, ("best_answer",)) or (correct[0] if correct else None)
    incorrect = record.get("incorrect_answers") or []
    evidence = [Evidence(text=str(answer), source="truthfulqa_correct_answer") for answer in correct]
    evidence.extend(
        Evidence(text=str(answer), source="truthfulqa_incorrect_answer", label="refuted") for answer in incorrect
    )
    return Example(
        example_id=f"truthfulqa::{index}",
        prompt=question,
        dataset=spec.name,
        split=split,
        task="generation",
        reference=best,
        evidence=evidence,
        gold_labels={"n_correct": len(correct), "n_incorrect": len(incorrect)},
        metadata={
            "licence": spec.licence,
            "note": "evidence is a list of correct/incorrect answers, not passages; the lexical judge is weak here",
            "category": record.get("category"),
        },
        provenance={"source": spec.hf_path, "citation": spec.citation, "loader_version": "0.1.0"},
    )


def _first_str(record: dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _count_label(evidence: Iterable[Evidence], label: str) -> int:
    return sum(1 for item in evidence if (item.label or "").lower() == label)


def load_examples(
    source: str | Path,
    *,
    split: str | None = None,
    max_examples: int = 50,
    inspect: bool = False,
) -> list[Example]:
    """Load examples from a local JSONL file or a supported dataset key."""
    path = Path(source)
    if path.exists() and path.is_file():
        return load_jsonl_examples(path, split=split, max_examples=max_examples)
    key = str(source)
    if key.lower() in {alias for alias in ("wikibio", "wiki_bio", "wikibio_hallucination", "halueval", "fever", "truthfulqa", "truthful_qa")}:
        return load_hf_examples(key, split=split, max_examples=max_examples, inspect=inspect)
    raise FileNotFoundError(
        f"{source!r} is neither an existing file nor a known dataset key "
        f"({sorted(DATASET_SPECS)}). Refusing to guess which data was meant."
    )


# ---------------------------------------------------------------------------
# Evaluation of saved generations
# ---------------------------------------------------------------------------


@dataclass
class EvaluationOutput:
    """Everything an evaluation produces (written verbatim to metrics.json)."""

    aggregate: dict[str, Any] = field(default_factory=dict)
    per_example: list[ExampleMetrics] = field(default_factory=list)
    judgements: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_condition: dict[str, dict[str, Any]] = field(default_factory=dict)
    overhead_vs_reference: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate": self.aggregate,
            "per_example": [item.model_dump(mode="json") for item in self.per_example],
            "judgements": self.judgements,
            "by_condition": self.by_condition,
            "overhead_vs_reference": self.overhead_vs_reference,
            "notes": self.notes,
        }


def evaluate_records(
    records: Sequence[GenerationRecord],
    *,
    config: Config,
    examples: Sequence[Example] = (),
    judge_adapter: Any = None,
    reference_condition: str | None = None,
) -> EvaluationOutput:
    """Score every record, aggregate per condition and compute latency overhead.

    ``examples`` supplies evidence/reference material keyed by ``example_id``;
    records without a matching example are still scored, using whatever reference
    is stored in the record itself (they are marked in the notes).
    """
    evaluation = config.evaluation
    examples_by_id = {example.example_id: example for example in examples}
    if not examples_by_id:
        logger.warning("no examples supplied: claims will be judged without external evidence")
    output = EvaluationOutput()
    missing_examples = 0
    bertscore_predictions: list[str] = []
    bertscore_references: list[str] = []
    bertscore_targets: list[ExampleMetrics] = []

    for record in records:
        example = examples_by_id.get(record.example_id)
        if example is None:
            missing_examples += 1
        evidence = list(example.evidence) if example else []
        reference = record.reference or (example.reference if example else None)
        if reference and not record.reference:
            record = record.model_copy(update={"reference": reference})
        metrics, judgement = score_record(
            record,
            judge=evaluation.claim_judge,
            evidence=evidence,
            judge_adapter=judge_adapter,
            judge_max_new_tokens=evaluation.judge_max_new_tokens,
            judge_temperature=evaluation.judge_temperature,
        )
        metrics.condition = record.condition or record.mode
        output.per_example.append(metrics)
        output.judgements[record.example_id] = judgement.model_dump(mode="json")
        if (
            evaluation.bertscore_enabled
            and reference
            and record.answer.strip()
        ):
            bertscore_predictions.append(record.answer)
            bertscore_references.append(reference)
            bertscore_targets.append(metrics)

    if evaluation.bertscore_enabled and bertscore_predictions:
        scores = bertscore(
            bertscore_predictions,
            bertscore_references,
            model_type=evaluation.bertscore_model,
            lang=evaluation.bertscore_lang,
        )
        for target, value in zip(bertscore_targets, scores):
            target.bertscore_f1 = value
    elif evaluation.bertscore_enabled:
        output.notes.append("bertscore was enabled but no record had both an answer and a reference")

    if missing_examples:
        output.notes.append(
            f"{missing_examples}/{len(records)} record(s) had no matching example: judged without "
            "external evidence (their claims count as unevidenced)"
        )
    if not evaluation.bertscore_enabled:
        output.notes.append("bertscore was disabled in the configuration (no model download); bertscore_f1 is null")

    conditions = sorted({item.condition for item in output.per_example})
    if not conditions:
        return output

    # group records per condition for latency aggregation
    records_by_condition: dict[str, list[GenerationRecord]] = {}
    for record in records:
        records_by_condition.setdefault(record.condition or record.mode, []).append(record)

    latency_by_condition = {name: summarise_latency(items) for name, items in records_by_condition.items()}
    for condition in conditions:
        subset = [item for item in output.per_example if item.condition == condition]
        representative = records_by_condition[condition][0]
        output.by_condition[condition] = aggregate_metrics(
            subset,
            condition=condition,
            mode=representative.mode,
            variant=representative.variant,
            judge=evaluation.claim_judge,
            evidence_source="; ".join(
                sorted({item.source for example in examples for item in example.evidence})
            )
            or "none",
            bootstrap_samples=evaluation.bootstrap_samples,
            bootstrap_confidence=evaluation.bootstrap_confidence,
            latency=latency_by_condition.get(condition, {}),
            latency_by_condition=latency_by_condition,
        )

    # paired latency overhead against the reference condition
    reference_name = reference_condition or _guess_reference_condition(conditions)
    if reference_name and reference_name in records_by_condition:
        reference_records = records_by_condition[reference_name]
        for condition, items in records_by_condition.items():
            if condition == reference_name:
                continue
            output.overhead_vs_reference[condition] = {
                "reference": reference_name,
                "generation": latency_overhead(items, reference_records),
                "total": latency_overhead(items, reference_records, key=lambda r: r.latency.total_s),
            }
    if reference_name == "baseline":
        output.notes.append(
            "latency overhead is paired per example against the condition named 'baseline' "
            "(generation-only and end-to-end)"
        )
    output.aggregate = {
        "n_records": len(records),
        "conditions": conditions,
        "by_condition": output.by_condition,
        "overhead_vs_reference": output.overhead_vs_reference,
        "judge": evaluation.claim_judge,
        "definitions": output.by_condition.get(conditions[0], {}).get("definitions", {}),
        "dataset_note": DATASET_REQUIREMENTS_NOTE,
        "notes": output.notes,
    }
    return output


def _guess_reference_condition(conditions: Sequence[str]) -> str | None:
    for candidate in ("baseline", "base_model_no_context"):
        if candidate in conditions:
            return candidate
    return conditions[0] if conditions else None


def evaluate_run_dir(
    run_dir: Path | str,
    *,
    config: Config,
    examples_path: Path | str | None = None,
    dataset: str | None = None,
    split: str | None = None,
    max_examples: int | None = None,
    judge_adapter: Any = None,
    output_path: Path | str | None = None,
) -> EvaluationOutput:
    """Evaluate ``generations.jsonl`` inside a run directory (Phase 5 exit criteria)."""
    run_dir = Path(run_dir)
    generations_path = run_dir / "generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(f"{generations_path} not found; run a generation script first")
    records = [GenerationRecord.model_validate(record) for record in iter_jsonl(generations_path)]
    if not records:
        raise ValueError(f"{generations_path} contains no records")

    examples: list[Example] = []
    resolved_dataset = dataset or config.evaluation.datasets[0]
    resolved_split = split or config.evaluation.split
    resolved_max = max_examples if max_examples is not None else config.evaluation.max_examples
    source = examples_path
    if source is None:
        candidate = Path("data/processed") / f"{resolved_dataset}.jsonl"
        if candidate.exists():
            source = candidate
    if source is not None:
        examples = load_jsonl_examples(source, max_examples=resolved_max)
        if not examples:
            logger.warning("example file %s yielded no examples", source)
    else:
        try:
            examples = load_hf_examples(resolved_dataset, split=resolved_split, max_examples=resolved_max)
        except Exception as exc:
            logger.warning(
                "could not load dataset %s (%s); evaluating with in-record references only. "
                "Metrics that need evidence will be null/unevidenced -- report this limitation.",
                resolved_dataset,
                exc,
            )
            examples = []

    output = evaluate_records(records, config=config, examples=examples, judge_adapter=judge_adapter)
    payload = output.to_dict()
    conditions = sorted({record.condition or record.mode for record in records})
    payload["condition"] = conditions[0] if len(conditions) == 1 else "multiple"
    payload["run_dir"] = str(run_dir)
    payload["dataset"] = resolved_dataset
    payload["split"] = resolved_split
    payload["n_examples_loaded"] = len(examples)
    payload["judge"] = config.evaluation.claim_judge
    payload["definitions"] = output.aggregate.get("definitions", {})
    target = Path(output_path) if output_path else run_dir / "metrics.json"
    write_json(target, payload)
    logger.info("wrote metrics to %s (%d records, %d examples)", target, len(records), len(examples))
    return output


def write_manual_review_sample(
    output: EvaluationOutput,
    records: Sequence[GenerationRecord],
    path: Path | str,
    *,
    sample_size: int = 30,
    seed: int = 7,
) -> Path:
    """Write a stratified sample for human inspection (spec 12.1 exit criteria).

    Rows are chosen to over-represent disagreement cases: the lowest and highest
    support rates, plus error/degraded records.  The file is a JSONL of
    {example_id, condition, prompt, answer, claims, verdicts, metrics} so a human
    can check the judge without re-running anything.
    """
    import random

    records_by_key = {(record.example_id, record.condition): record for record in records}
    scored = [item for item in output.per_example if item.n_claims > 0]
    scored.sort(key=lambda item: (item.claim_support_rate if item.claim_support_rate is not None else -1))
    rng = random.Random(seed)
    half = max(1, sample_size // 3)
    picks = scored[:half] + scored[-half:]
    middle = scored[half : max(half, len(scored) - half)]
    if middle:
        picks.extend(rng.sample(middle, min(len(middle), sample_size - len(picks))))
    failures = [item for item in output.per_example if item.status != "ok"]
    picks.extend(failures[: max(0, sample_size - len(picks))])

    lines: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in picks:
        key = (item.example_id, item.condition)
        if key in seen:
            continue
        seen.add(key)
        record = records_by_key.get(key)
        judgement = output.judgements.get(item.example_id, {})
        lines.append(
            {
                "example_id": item.example_id,
                "condition": item.condition,
                "prompt": record.prompt if record else None,
                "answer": record.answer if record else None,
                "reference": record.reference if record else None,
                "claims": judgement.get("claims", []),
                "verdicts": judgement.get("verdicts", []),
                "metrics": item.model_dump(mode="json"),
            }
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    return target
