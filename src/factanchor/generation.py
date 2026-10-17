"""Autoregressive generation: baseline and FactAnchor loops (spec 10.2, 10.4).

``TextGenerator`` is the only place that runs the token loop.  It implements:

* identical handling of EOS, stop sequences and ``max_new_tokens`` in both modes;
* a cached (KV) and an uncached path that must agree under deterministic
  decoding (tested in ``tests/test_decoding.py``);
* explicit branch alignment: the contrast branch always conditions on the *same*
  generated suffix as the base branch, and the invariant is checked every step;
* wall-clock timing with CUDA synchronisation, one entry per generated token;
* no silent fallback: a numerical or context-length problem raises, or is
  recorded in the result with a reason.

``generate_baseline`` / ``generate_factanchor`` are the functional entry points
named by the spec.  ``generate_example`` runs the whole pipeline for one prompt
and returns a traceable :class:`~factanchor.schemas.GenerationRecord`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import Config, DecodingConfig
from .decoding import ContrastiveDecoder, StepDiagnostics
from .model_adapter import BranchCache, ModelAdapter, build_adapter
from .prompts import EXPERIMENTAL_HEADER, NEUTRAL_HEADER, ContextBuilder, ContextPair
from .schemas import (
    CorruptionResult,
    EntityRecord,
    Fact,
    GenerationParams,
    GenerationRecord,
    LatencyBreakdown,
    Status,
)
from .utils import Timer, get_logger, sha256_text, truncate

__all__ = [
    "GenerationResult",
    "TextGenerator",
    "generate_baseline",
    "generate_factanchor",
    "generate_example",
    "build_generation_params",
]

logger = get_logger("generation")


@dataclass
class GenerationResult:
    """Raw result of a token loop (before it becomes a ``GenerationRecord``)."""

    text: str
    token_ids: list[int] = field(default_factory=list)
    stop_reason: str = "max_new_tokens"
    input_tokens: int = 0
    contrast_input_tokens: int | None = None
    output_tokens: int = 0
    generation_s: float = 0.0
    per_step_seconds: list[float] = field(default_factory=list)
    steps: list[StepDiagnostics] = field(default_factory=list)
    contrast_applied: bool = False
    degraded_reason: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def ms_per_token(self) -> float | None:
        if self.output_tokens <= 0:
            return None
        return 1000.0 * self.generation_s / self.output_tokens

    @property
    def tokens_per_second(self) -> float | None:
        if self.generation_s <= 0:
            return None
        return self.output_tokens / self.generation_s


class TextGenerator:
    """Token-level generator shared by every mode."""

    def __init__(
        self,
        adapter: ModelAdapter,
        config: Config | DecodingConfig,
        *,
        seed: int | None = None,
        contrast_adapter: ModelAdapter | None = None,
    ) -> None:
        self.adapter = adapter
        # "Amateur model" ablation (spec 3.3): the contrast distribution may come
        # from a different model, provided it lives in the same token space.
        if contrast_adapter is not None and contrast_adapter.vocab_size != adapter.vocab_size:
            raise ValueError(
                "decoding.amateur_model_id selects a model with a different vocabulary "
                f"({contrast_adapter.vocab_size} vs {adapter.vocab_size} tokens). The contrast term "
                "subtracts log-probabilities token by token, so both branches must share a "
                "tokenizer; a cross-tokenizer contrast is not implemented. Use a model with the "
                "same vocabulary or unset decoding.amateur_model_id."
            )
        self.contrast_adapter = contrast_adapter or adapter
        self.config = config
        self.decoding = config.decoding if isinstance(config, Config) else config
        self.max_context_tokens = (
            config.model.max_context_tokens if isinstance(config, Config) else adapter.context_length
        )
        self.run_seed = seed if seed is not None else (config.run.seed if isinstance(config, Config) else 0)
        self.decoder = ContrastiveDecoder(self.decoding, seed=self.run_seed)

    # ------------------------------------------------------------------ API
    def generate(
        self,
        base_prompt: str,
        *,
        contrast_prompt: str | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Run the decoding loop.

        ``contrast_prompt`` is required for FactAnchor; when it is ``None`` (or
        ``alpha == 0``, or ``mode == 'baseline'``) the loop takes the baseline path
        and never evaluates a second forward pass.
        """
        decoding = self.decoding
        budget = int(max_new_tokens if max_new_tokens is not None else decoding.max_new_tokens)
        if budget < 1:
            raise ValueError("max_new_tokens must be >= 1")

        base_text = self.adapter.render(base_prompt, system_prompt=system_prompt)
        base_ids = self.adapter.encode(base_text)
        if not base_ids:
            raise ValueError("the base prompt tokenised to zero tokens; refusing to generate")
        contrast_ids: list[int] | None = None
        use_contrast = decoding.mode == "factanchor" and decoding.alpha > 0.0 and bool(contrast_prompt)
        if use_contrast and contrast_prompt is not None:
            contrast_text = self.contrast_adapter.render(contrast_prompt, system_prompt=system_prompt)
            contrast_ids = self.contrast_adapter.encode(contrast_text)
            if not contrast_ids:
                raise ValueError("the contrast prompt tokenised to zero tokens; refusing to generate")
        elif decoding.mode == "factanchor" and decoding.alpha > 0.0 and not contrast_prompt:
            logger.info(
                "FactAnchor requested with alpha=%.2f but no contrast context was supplied; "
                "running the baseline path (record it as degraded in the report)",
                decoding.alpha,
            )

        for label, ids in (("base", base_ids), ("contrast", contrast_ids)):
            if ids is not None and len(ids) + budget > self.max_context_tokens:
                raise ValueError(
                    f"{label} prompt has {len(ids)} tokens and max_new_tokens={budget} exceeds "
                    f"model.max_context_tokens={self.max_context_tokens}. Shorten the prompt, reduce "
                    "retrieval.max_facts_per_entity, or raise model.max_context_tokens (the run must "
                    "not silently truncate the prompt)."
                )

        use_cache = bool(decoding.use_cache)
        generated: list[int] = []
        per_step_seconds: list[float] = []
        stop_reason = "max_new_tokens"
        self.decoder.reset()
        base_cache: BranchCache | None = self.adapter.new_cache() if use_cache else None
        contrast_cache: BranchCache | None = self.contrast_adapter.new_cache() if use_cache else None
        eos_ids = self.adapter.eos_token_ids
        text = ""
        suppressed_logged = False
        notes: list[str] = []
        if not use_cache:
            notes.append("KV cache disabled: both branches recompute the whole prefix each step")

        wall_start = time.monotonic()
        for step in range(budget):
            step_start = time.monotonic()
            if use_cache:
                assert base_cache is not None and (contrast_ids is None or contrast_cache is not None)
                feed_base = base_ids if step == 0 else [generated[-1]]
                # After this call the base cache holds the prompt plus the `step`
                # tokens generated so far -- i.e. exactly what has been fed.
                base_logits, base_cache = self.adapter.advance(feed_base, base_cache)
                _check_alignment("base", base_cache, len(base_ids) + step)
                base_logits = base_logits[0]
                contrast_logits: np.ndarray | None = None
                if contrast_ids is not None:
                    feed_contrast = contrast_ids if step == 0 else [generated[-1]]
                    raw_contrast, contrast_cache = self.contrast_adapter.advance(feed_contrast, contrast_cache)
                    _check_alignment("contrast", contrast_cache, len(contrast_ids) + step)
                    _check_suffix_alignment(base_ids, base_cache, contrast_ids, contrast_cache)
                    contrast_logits = raw_contrast[0]
            else:
                base_logits = np.asarray(
                    self.adapter.logits_for_prefix(base_ids + generated), dtype=np.float64
                ).ravel()
                contrast_logits = (
                    np.asarray(
                        self.contrast_adapter.logits_for_prefix(contrast_ids + generated), dtype=np.float64
                    ).ravel()
                    if contrast_ids is not None
                    else None
                )

            token_id, diagnostics = self.decoder.step(
                step_index=step,
                base_logits=base_logits,
                contrast_logits=contrast_logits,
                generated_ids=generated,
                eos_token_ids=eos_ids,
            )
            diagnostics.token = self.adapter.decode([token_id], skip_special_tokens=False)
            self.decoder.steps.append(diagnostics)

            if not suppressed_logged and self.decoder.eos_suppression_events == 1:
                notes.append(
                    f"EOS suppressed for the first {min(int(decoding.min_new_tokens), budget)} step(s) "
                    "because decoding.min_new_tokens > 0"
                )
                suppressed_logged = True

            if token_id in eos_ids:
                stop_reason = "eos"
                break
            generated.append(token_id)

            text = self.adapter.decode(generated, skip_special_tokens=True)
            stop_hit = _match_stop_sequence(text, decoding.stop_sequences)
            per_step_seconds.append(time.monotonic() - step_start)
            if stop_hit is not None:
                # Text is cut at the stop string; the token list keeps what was
                # actually generated so the artifact remains auditable.
                text = text[:stop_hit]
                stop_reason = f"stop_sequence:{decoding.stop_sequences[stop_hit]!r}"
                break
        generation_s = time.monotonic() - wall_start

        if not generated and stop_reason == "max_new_tokens":  # pragma: no cover - defensive
            notes.append("no tokens were produced")

        return GenerationResult(
            text=text.strip(),
            token_ids=generated,
            stop_reason=stop_reason,
            input_tokens=len(base_ids),
            contrast_input_tokens=len(contrast_ids) if contrast_ids is not None else None,
            output_tokens=len(generated),
            generation_s=generation_s,
            per_step_seconds=per_step_seconds,
            steps=list(self.decoder.steps),
            contrast_applied=any(step.contrast_applied for step in self.decoder.steps),
            notes=tuple(notes + ([] if contrast_ids is not None else ["contrast branch not evaluated"])),
        )


def _run_generation(
    generator: TextGenerator,
    pair: ContextPair,
    contrast_context: str | None,
    config: Config,
) -> GenerationResult:
    """Run the token loop for one already-built context pair."""
    return generator.generate(
        pair.base_context,
        contrast_prompt=contrast_context,
        system_prompt=config.context.system_prompt,
    )


def _check_suffix_alignment(
    base_ids: Sequence[int],
    base_cache: BranchCache | None,
    contrast_ids: Sequence[int],
    contrast_cache: BranchCache | None,
) -> None:
    """The contrast branch must condition on the *same* generated suffix (spec 10.1).

    Both caches grow by exactly one token per step, so the difference in their
    lengths must stay equal to the difference in prompt lengths.  Any other value
    means one branch saw a different suffix, which would silently invalidate the
    contrast.
    """
    if base_cache is None or contrast_cache is None:
        return
    expected_offset = len(contrast_ids) - len(base_ids)
    actual_offset = contrast_cache.length - base_cache.length
    if actual_offset != expected_offset:
        raise RuntimeError(
            f"contrast branch is {actual_offset} tokens ahead of the base branch but "
            f"{expected_offset} was expected: the branches no longer condition on the same "
            "generated suffix (spec 10.1)"
        )


def _check_alignment(label: str, cache: BranchCache | None, expected_length: int) -> None:
    """Raise if a branch cache does not represent the expected number of tokens."""
    if cache is None:
        return
    if cache.length != expected_length:
        raise RuntimeError(
            f"{label} branch cache holds {cache.length} tokens but {expected_length} were expected: "
            "the two branches have drifted out of alignment (spec 10.1)"
        )


def _match_stop_sequence(text: str, stop_sequences: Sequence[str]) -> int | None:
    """Return the index of the earliest stop sequence start, or ``None``."""
    best: int | None = None
    for stop in stop_sequences:
        if not stop:
            continue
        index = text.find(stop)
        if index >= 0 and (best is None or index < best):
            best = index
    return best


# ---------------------------------------------------------------------------
# Functional entry points (spec 10.2)
# ---------------------------------------------------------------------------


def generate_baseline(
    prompt: str,
    config: Config,
    *,
    adapter: ModelAdapter | None = None,
    system_prompt: str | None = None,
    max_new_tokens: int | None = None,
) -> GenerationResult:
    """Baseline decoding: one forward pass per token, no knowledge, no contrast."""
    if config.decoding.alpha != 0.0 and config.decoding.mode == "factanchor":
        raise ValueError(
            "generate_baseline() requires decoding.mode='baseline' (or alpha=0). "
            "Use generate_factanchor() for the contrastive condition."
        )
    adapter = adapter or build_adapter(config)
    generator = TextGenerator(adapter, config)
    return generator.generate(prompt, contrast_prompt=None, system_prompt=system_prompt, max_new_tokens=max_new_tokens)


def generate_factanchor(
    prompt: str,
    config: Config,
    contrast_context: str | None,
    *,
    adapter: ModelAdapter | None = None,
    system_prompt: str | None = None,
    max_new_tokens: int | None = None,
) -> GenerationResult:
    """FactAnchor decoding: contrastive score against the corrupted-fact context."""
    if config.decoding.mode != "factanchor":
        raise ValueError(
            "generate_factanchor() requires decoding.mode='factanchor'; "
            f"got {config.decoding.mode!r}. Switch mode or use generate_baseline()."
        )
    adapter = adapter or build_adapter(config)
    generator = TextGenerator(adapter, config)
    return generator.generate(
        prompt, contrast_prompt=contrast_context, system_prompt=system_prompt, max_new_tokens=max_new_tokens
    )


def build_generation_params(config: Config) -> GenerationParams:
    decoding = config.decoding
    return GenerationParams(
        mode=decoding.mode,
        variant=decoding.variant,
        alpha=decoding.alpha,
        max_new_tokens=decoding.max_new_tokens,
        min_new_tokens=decoding.min_new_tokens,
        do_sample=decoding.do_sample,
        temperature=decoding.temperature,
        top_p=decoding.top_p,
        top_k=decoding.top_k,
        repetition_penalty=decoding.repetition_penalty,
        no_repeat_ngram_size=decoding.no_repeat_ngram_size,
        use_cache=decoding.use_cache,
        seed=config.run.seed,
        stop_sequences=list(decoding.stop_sequences),
        adaptive_plausibility=decoding.adaptive_plausibility,
        plausibility_epsilon=decoding.plausibility_epsilon,
        contrast_warmup_tokens=decoding.contrast_warmup_tokens,
        amateur_model_id=decoding.amateur_model_id,
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline for a single prompt
# ---------------------------------------------------------------------------


def generate_example(
    *,
    example_id: str,
    prompt: str,
    config: Config,
    adapter: ModelAdapter,
    run_id: str = "",
    condition: str = "",
    dataset: str = "unknown",
    split: str = "unknown",
    task: str = "generation",
    reference: str | None = None,
    entity_hint: str | None = None,
    knowledge=None,
    reporter=None,
    model_load_s: float = 0.0,
    contrast_adapter: ModelAdapter | None = None,
) -> GenerationRecord:
    """Run the full pipeline (entities -> facts -> corruption -> contexts -> decode).

    ``knowledge`` is a :class:`~factanchor.knowledge.pipeline.KnowledgePipeline`
    or ``None`` to disable retrieval.  ``reporter`` is an optional callback
    ``(stage: str, payload: dict) -> None`` used by the CLI/demo to stream
    progress without the library printing to stdout.
    """
    from .corruption import engine_from_config
    from .entities import build_extractor, extract_entities  # noqa: F401 (extract_entities re-exported for tests)

    latency = LatencyBreakdown(model_load_s=model_load_s, device=adapter.device)
    latency.synchronized = adapter.device.startswith("cuda")
    degraded_reasons: list[str] = []
    status = Status.OK
    error: str | None = None
    retrieval_failed = False

    def report(stage: str, payload: dict[str, Any]) -> None:
        if reporter is not None:
            try:
                reporter(stage, payload)
            except Exception:  # pragma: no cover - reporter must not break the run
                logger.debug("reporter raised for stage %s", stage, exc_info=True)

    retrieval_config = config.retrieval
    retrieval_enabled = bool(retrieval_config.enabled and knowledge is not None)
    entities: list[EntityRecord] = []
    facts: list[Fact] = []
    corruption_result: CorruptionResult | None = None
    contrast_context: str | None = None

    if retrieval_enabled:
        extractor = build_extractor(config.ner, known_entities=knowledge.known_entities())
        with Timer("extraction") as extraction_timer:
            mentions = extractor.extract(prompt)
        latency.entity_extraction_s = extraction_timer.elapsed
        report("mentions", {"mentions": [m.model_dump() for m in mentions]})

        with Timer("linking") as link_timer:
            link_result = knowledge.link_entities(mentions, prompt=prompt, entity_hint=entity_hint)
        latency.entity_linking_s = link_timer.elapsed
        entities = link_result.records
        if link_result.degraded_reason:
            degraded_reasons.append(f"linking:{link_result.degraded_reason}")
        report("entities", {"entities": [e.model_dump() for e in entities]})

        with Timer("retrieval") as retrieval_timer:
            try:
                retrieval_result = knowledge.retrieve_facts(entities)
            except Exception as exc:
                # retrieval.fallback_on_unavailable='error' propagates
                # KnowledgeUnavailable: the example is then recorded as an error
                # instead of quietly generating with a different condition.
                logger.error("fact retrieval failed for %s: %s", example_id, exc)
                retrieval_result = None
                error = f"{type(exc).__name__}: {exc}"
                degraded_reasons.append("retrieval_failed")
                status = Status.ERROR
                retrieval_failed = True
            latency.fact_retrieval_s = retrieval_timer.elapsed
        if retrieval_result is not None:
            facts = retrieval_result.facts
            if retrieval_result.degraded_reason:
                degraded_reasons.append(f"retrieval:{retrieval_result.degraded_reason}")
            report("facts", {"facts": [f.model_dump() for f in facts]})

        with Timer("corruption") as corruption_timer:
            engine = engine_from_config(config)
            corruption_result = engine.corrupt(facts)
        latency.corruption_s = corruption_timer.elapsed
        report("corruption", corruption_result.summary)

    with Timer("context") as context_timer:
        builder = ContextBuilder.from_config(config)
        pair = builder.build(
            prompt,
            retrieval=corruption_result,
            reference=reference,
            include_contrast=config.decoding.mode == "factanchor",
        )
    latency.context_build_s = context_timer.elapsed

    wants_contrast = config.decoding.mode == "factanchor" and config.decoding.alpha > 0.0
    contrast_context = pair.contrast_context if wants_contrast else None
    if wants_contrast and not pair.has_contrast_facts:
        if config.decoding.on_missing_contrast == "skip_contrast":
            degraded_reasons.append("no_contrast_facts")
            contrast_context = None
            logger.info(
                "FactAnchor is configured but no contrast facts are available for %s; running the "
                "baseline path and recording degraded_reason=no_contrast_facts",
                example_id,
            )
        else:  # empty_block: keep the prompt skeleton, drop the facts (control condition)
            header = EXPERIMENTAL_HEADER if config.context.label_as_experimental else NEUTRAL_HEADER
            contrast_context = config.context.contrast_template.format(
                question=prompt.strip(), fact_block=header
            )
            degraded_reasons.append("empty_contrast_block")

    # "Amateur model" ablation (spec 3.3): optionally decode the contrast branch
    # with a second, different model.
    amateur_id = config.decoding.amateur_model_id
    contrast_model_load_s = 0.0
    if (
        contrast_adapter is None
        and amateur_id
        and amateur_id.strip().lower() != adapter.model_id.strip().lower()
        and config.decoding.mode == "factanchor"
        and config.decoding.alpha > 0.0
    ):
        with Timer("contrast_model_load") as contrast_timer:
            contrast_adapter = build_adapter(config, model_id=amateur_id)
        contrast_model_load_s = contrast_timer.elapsed
        logger.info(
            "amateur-model ablation: contrast branch uses %s, base branch uses %s",
            contrast_adapter.model_id,
            adapter.model_id,
        )
    latency.contrast_model_load_s = contrast_model_load_s

    generator = TextGenerator(adapter, config, contrast_adapter=contrast_adapter)
    if retrieval_failed:
        # Knowledge retrieval failed hard (fallback_on_unavailable='error'): do not
        # generate a different, unlabelled condition -- record the failure.
        result = GenerationResult(text="", degraded_reason="retrieval_failed")
    else:
        try:
            result = _run_generation(generator, pair, contrast_context, config)
        except Exception as exc:
            logger.error("generation failed for %s: %s", example_id, exc)
            status = Status.ERROR
            error = f"{type(exc).__name__}: {exc}"
            result = GenerationResult(text="", degraded_reason="generation_failed")
    latency.generation_s = result.generation_s
    latency.peak_memory_mb = adapter.peak_memory_mb()
    latency.recompute_total()

    # In baseline mode the contrast branch simply does not exist; do not flag it.
    if status is Status.OK and degraded_reasons:
        status = Status.DEGRADED
    if error is not None and status is not Status.ERROR:
        status = Status.ERROR

    record = GenerationRecord(
        example_id=example_id,
        run_id=run_id,
        condition=condition,
        prompt=prompt if config.run.log_prompts else "[prompt logging disabled]",
        prompt_sha256=sha256_text(prompt),
        dataset=dataset,
        split=split,
        task=task,
        model_id=adapter.model_id,
        model_revision=adapter.revision,
        tokenizer_id=adapter.tokenizer_id,
        tokenizer_revision=adapter.tokenizer_revision,
        is_toy_model=adapter.is_toy,
        device=adapter.device,
        dtype=adapter.dtype,
        seed=config.run.seed,
        mode=config.decoding.mode,
        variant=config.decoding.variant,
        alpha=config.decoding.alpha,
        params=build_generation_params(config),
        contrast_applied=result.contrast_applied,
        degraded_reason=(";".join(degraded_reasons) if degraded_reasons else result.degraded_reason),
        entities=entities,
        facts=facts,
        corruption=corruption_result,
        base_context=pair.base_context if config.run.log_prompts else "[context logging disabled]",
        contrast_context=contrast_context if config.run.log_prompts else None,
        contrast_prompt_tokens=result.contrast_input_tokens,
        answer=result.text,
        raw_answer=result.text,
        generated_token_ids=result.token_ids,
        stop_reason=result.stop_reason,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency=latency,
        status=status,
        error=error,
        reference=reference,
        metadata={
            "condition": condition,
            "retrieval_enabled": retrieval_enabled,
            "knowledge_provider": retrieval_config.provider if retrieval_enabled else "none",
            "knowledge_source": retrieval_config.source_name,
            "template_version": pair.template_version,
            "context_notes": list(pair.notes),
            "generation_notes": list(result.notes),
            "stop_reason": result.stop_reason,
            "tokens_per_second": result.tokens_per_second,
            "is_toy_model": adapter.is_toy,
            "amateur_model_id": contrast_adapter.model_id if contrast_adapter is not None else None,
            "evidence_count": 0,
        },
    )
    if config.run.save_debug_scores:
        record.metadata["debug_steps"] = [step.to_record().model_dump() for step in result.steps]
    return record


def truncate_for_log(text: str, max_chars: int = 2000) -> str:
    """Truncate long text for logs without silently changing stored artifacts."""
    return truncate(text, max_chars)
