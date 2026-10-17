"""FactAnchor: entity-grounded contrastive decoding for hallucination suppression.

A research prototype -- **not** a truth guarantee.  The method contrasts the base
model's next-token distribution with a second distribution induced by a
deliberately corrupted entity-fact context, and measures the quality/latency
trade-off honestly (including negative results).

Quick start (offline, no downloads)::

    from factanchor import load_config, build_adapter, generate_example
    config = load_config("configs/smoke_toy.yaml")
    adapter = build_adapter(config)
    record = generate_example(
        example_id="demo::0", prompt="Who was Marie Curie?", config=config, adapter=adapter,
    )

The public surface mirrors the pipeline stages; see ``README.md`` for the CLI.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import (
    Config,
    ConfigError,
    Condition,
    ContextConfig,
    CorruptionConfig,
    DecodingConfig,
    EvaluationConfig,
    LoggingConfig,
    ModelConfig,
    NerConfig,
    RetrievalConfig,
    RunConfig,
    load_config,
)
from .corruption import CorruptionEngine, corrupt_facts, describe_corruption_summary, engine_from_config
from .decoding import (
    ContrastiveDecoder,
    apply_no_repeat_ngram,
    apply_plausibility_mask,
    apply_repetition_penalty,
    contrastive_scores,
    select_token,
)
from .entities import (
    EntityExtractor,
    LexiconEntityExtractor,
    RuleBasedEntityExtractor,
    build_extractor,
    extract_entities,
)
from .evaluation import (
    DATASET_SPECS,
    EvaluationOutput,
    available_datasets,
    evaluate_records,
    evaluate_run_dir,
    load_examples,
    load_jsonl_examples,
)
from .experiment import RunContext, RunSummary, build_conditions, create_run, run_ablation, run_condition
from .generation import (
    GenerationResult,
    TextGenerator,
    generate_baseline,
    generate_example,
    generate_factanchor,
)
from .knowledge import (
    EntityCandidate,
    JsonFileKnowledgeProvider,
    KnowledgePipeline,
    KnowledgeProvider,
    KnowledgeUnavailable,
    MockKnowledgeProvider,
    WikidataProvider,
)
from .metrics import (
    METRIC_DEFINITIONS,
    aggregate_metrics,
    bootstrap_ci,
    extract_claims,
    latency_overhead,
    score_record,
    summarise_latency,
)
from .model_adapter import (
    BranchCache,
    HFCausalLMAdapter,
    ModelAdapter,
    ToyCausalLMAdapter,
    build_adapter,
    log_softmax,
)
from .prompts import ContextBuilder, ContextPair
from .schemas import (
    ClaimJudgement,
    ClaimRecord,
    CorruptionRecord,
    CorruptionResult,
    EntityMention,
    EntityRecord,
    Evidence,
    Example,
    ExampleMetrics,
    Fact,
    GenerationParams,
    GenerationRecord,
    JudgeVerdict,
    LatencyBreakdown,
    Qualifier,
    RunManifest,
    Status,
)

__all__ = [
    "__version__",
    # config
    "Config",
    "ConfigError",
    "Condition",
    "RunConfig",
    "ModelConfig",
    "DecodingConfig",
    "RetrievalConfig",
    "NerConfig",
    "CorruptionConfig",
    "ContextConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "load_config",
    # entities
    "EntityExtractor",
    "RuleBasedEntityExtractor",
    "LexiconEntityExtractor",
    "build_extractor",
    "extract_entities",
    # knowledge
    "KnowledgeProvider",
    "KnowledgeUnavailable",
    "KnowledgePipeline",
    "WikidataProvider",
    "JsonFileKnowledgeProvider",
    "MockKnowledgeProvider",
    "EntityCandidate",
    # corruption
    "CorruptionEngine",
    "corrupt_facts",
    "engine_from_config",
    "describe_corruption_summary",
    # prompts / decoding / generation
    "ContextBuilder",
    "ContextPair",
    "ContrastiveDecoder",
    "contrastive_scores",
    "apply_repetition_penalty",
    "apply_no_repeat_ngram",
    "apply_plausibility_mask",
    "select_token",
    "ModelAdapter",
    "HFCausalLMAdapter",
    "ToyCausalLMAdapter",
    "BranchCache",
    "build_adapter",
    "log_softmax",
    "TextGenerator",
    "GenerationResult",
    "generate_baseline",
    "generate_factanchor",
    "generate_example",
    # evaluation
    "evaluate_records",
    "evaluate_run_dir",
    "load_examples",
    "load_jsonl_examples",
    "available_datasets",
    "DATASET_SPECS",
    "EvaluationOutput",
    "extract_claims",
    "score_record",
    "aggregate_metrics",
    "bootstrap_ci",
    "latency_overhead",
    "summarise_latency",
    "METRIC_DEFINITIONS",
    # experiments
    "create_run",
    "run_condition",
    "run_ablation",
    "build_conditions",
    "RunContext",
    "RunSummary",
    # schemas
    "EntityMention",
    "EntityRecord",
    "Fact",
    "Qualifier",
    "CorruptionRecord",
    "CorruptionResult",
    "Example",
    "Evidence",
    "GenerationRecord",
    "GenerationParams",
    "LatencyBreakdown",
    "ClaimRecord",
    "ClaimJudgement",
    "JudgeVerdict",
    "ExampleMetrics",
    "RunManifest",
    "Status",
]
