"""Streamlit demo: baseline vs FactAnchor on the same prompt (spec 4/14).

Run it with::

    pip install -e ".[demo]"
    streamlit run app.py

What it does
------------
* One prompt box, plus controls for decoding mode, alpha, corruption rate and
  ``max_new_tokens``.  Every control maps to a real configuration field; nothing
  is faked or hard-coded.
* Runs **both** conditions on the same prompt with the same model weights and
  decoding settings (greedy by default, so the comparison is paired).
* Shows answer, status, token counts and the latency breakdown for each side.
* Expandable panels for linked entities, retrieved facts and the corruption
  audit (original -> corrupted, with seed and strategy).
* A permanent warning that the corrupted branch is an *experimental internal
  condition*, that generated text needs independent verification, and that no
  API keys or environment variables are displayed.

The demo never prints environment variables, never writes credentials to disk,
and fails loudly (in the UI) when a model or knowledge source is unavailable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `streamlit run app.py` from a fresh clone without `pip install -e .`.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

TOY_WARNING = (
    "model.model_id='toy' uses randomly initialised weights: the text is not meaningful. "
    "Use it to check the plumbing only."
)
CORRUPTION_WARNING = (
    "The corrupted facts below are a synthetic experimental condition. They are **not** evidence, "
    "must never be quoted as facts, and are shown only so the intervention is auditable."
)


def _load_config(path: str, alpha: float, corruption_rate: float, mode: str, max_new_tokens: int):
    from factanchor.config import load_config

    overrides = {
        "decoding.mode": mode,
        "decoding.alpha": float(alpha),
        "corruption.corruption_rate": float(corruption_rate),
        "decoding.max_new_tokens": int(max_new_tokens),
        # The demo always compares greedily decoded, paired outputs.
        "decoding.do_sample": False,
        "run.log_prompts": True,
    }
    if mode == "baseline":
        # A baseline run with a non-zero contrast weight would be ambiguous.
        overrides["decoding.alpha"] = 0.0
    return load_config(path, overrides=overrides)


def _run_one(config, prompt: str, knowledge):
    """Generate one condition and return (record, error_message)."""
    from factanchor.generation import generate_example
    from factanchor.model_adapter import build_adapter

    try:
        adapter = build_adapter(config)
    except Exception as exc:  # model weights unavailable / gated / OOM
        return None, f"model unavailable: {type(exc).__name__}: {exc}"
    try:
        record = generate_example(
            example_id="demo::0",
            prompt=prompt,
            config=config,
            adapter=adapter,
            run_id="streamlit-demo",
            condition=f"demo_{config.decoding.mode}",
            dataset="demo",
            split="demo",
            knowledge=knowledge if config.retrieval.enabled else None,
        )
    except Exception as exc:
        return None, f"generation failed: {type(exc).__name__}: {exc}"
    return record, None


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError:  # pragma: no cover - depends on the environment
        print(
            "Streamlit is not installed. Install the optional extra first:\n"
            '    pip install -e ".[demo]"\n'
            "Or use the console demo instead:\n"
            '    python scripts/demo.py --config configs/smoke_toy.yaml --prompt "Who was Marie Curie?"'
        )
        raise SystemExit(2)

    from factanchor.knowledge.pipeline import build_provider, KnowledgePipeline
    from factanchor.utils import set_global_seed

    st.set_page_config(page_title="FactAnchor demo", page_icon="🧪", layout="wide")
    st.title("FactAnchor: entity-grounded contrastive decoding")
    st.caption(
        "Baseline vs FactAnchor on the same prompt. Research prototype -- generated text "
        "requires independent verification."
    )

    with st.sidebar:
        st.header("Run settings")
        config_path = st.text_input("Config file", value=os.environ.get("FACTANCHOR_DEMO_CONFIG", "configs/default.yaml"))
        prompt = st.text_area(
            "Prompt",
            value="Who was Marie Curie?",
            height=100,
            help="Use prompt files or stdin for very long prompts; the text box is for short questions.",
        )
        mode = st.radio("FactAnchor side", options=["factanchor", "baseline"], index=0)
        alpha = st.slider("alpha (contrast strength)", 0.0, 2.0, 0.5, 0.05)
        corruption_rate = st.slider("corruption rate", 0.0, 1.0, 0.5, 0.05)
        max_new_tokens = st.slider("max new tokens", 1, 256, 32, 1)
        seed = st.number_input("seed", min_value=0, value=42, step=1)
        run = st.button("Generate comparison", type="primary")

    st.warning(
        "**Responsible use.** This prototype does not guarantee truthfulness and does not eliminate "
        "hallucination. Corrupted facts shown in the audit panel are synthetic experiment inputs, not "
        "evidence. Nothing here is a substitute for independent verification."
    )

    st.info(
        "The toy model (`model.model_id: toy`) has randomly initialised weights: outputs are gibberish "
        "but every component runs offline. Set a real Hugging Face model id in the config for readable text."
    )

    if not run:
        st.stop()

    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()

    try:
        config = _load_config(config_path, alpha, corruption_rate, mode, max_new_tokens)
        config = config.model_copy(update={"run": config.run.model_copy(update={"seed": int(seed)})})
        set_global_seed(int(seed))
    except Exception as exc:
        st.error(f"configuration problem: {type(exc).__name__}: {exc}")
        st.stop()

    knowledge = None
    knowledge_error = None
    if config.retrieval.enabled:
        try:
            knowledge = KnowledgePipeline(config, provider=build_provider(config))
        except Exception as exc:  # API/auth/network problem -> keep going without retrieval
            knowledge_error = f"{type(exc).__name__}: {exc}"
    if knowledge_error:
        st.warning(
            f"knowledge source unavailable ({knowledge_error}). "
            "Generating without retrieved facts; FactAnchor will be recorded as degraded."
        )

    columns = st.columns(2)
    with st.spinner("Generating with both conditions..."):
        baseline_config = _load_config(config_path, 0.0, corruption_rate, "baseline", max_new_tokens)
        baseline_config = baseline_config.model_copy(
            update={"run": baseline_config.run.model_copy(update={"seed": int(seed)})}
        )
        baseline_record, baseline_error = _run_one(baseline_config, prompt, knowledge)
        config = config.model_copy(update={"run": config.run.model_copy(update={"seed": int(seed)})})
        factanchor_record, factanchor_error = _run_one(config, prompt, knowledge)

    for column, title, record, error in (
        (columns[0], "Baseline (no contrast)", baseline_record, baseline_error),
        (columns[1], f"FactAnchor (alpha={config.decoding.alpha:g})", factanchor_record, factanchor_error),
    ):
        with column:
            st.subheader(title)
            if error:
                st.error(error)
                continue
            status = record.status.value
            if status == "ok":
                st.success(f"status: {status}")
            elif status == "degraded":
                st.warning(f"status: {status} (reason: {record.degraded_reason})")
            else:
                st.error(f"status: {status} (reason: {record.error or record.degraded_reason})")
            st.write(record.answer or "_(empty output)_")
            latency = record.latency
            st.caption(
                f"{record.output_tokens} tokens | generation {latency.generation_s:.2f}s "
                f"| total {latency.total_s:.2f}s | stop={record.stop_reason}"
            )
            st.json(
                {
                    "mode": record.mode,
                    "variant": record.variant,
                    "alpha": record.alpha,
                    "corruption_rate": record.corruption.corruption_rate if record.corruption else 0.0,
                    "contrast_applied": record.contrast_applied,
                }
            )

    factanchor_record = factanchor_record or baseline_record
    if factanchor_record is not None:
        with st.expander("Linked entities and retrieved facts", expanded=False):
            if not factanchor_record.entities and not factanchor_record.facts:
                st.write("No entity was linked and no fact was retrieved for this prompt.")
            for entity in factanchor_record.entities:
                st.markdown(
                    f"- **{entity.mention}** -> `{entity.entity_id}` ({entity.label}) "
                    f"confidence={entity.linking_confidence:.2f}, method={entity.link_method}"
                )
            if factanchor_record.facts:
                st.markdown("**Retrieved facts (trusted branch input)**")
                for fact in factanchor_record.facts:
                    st.markdown(
                        f"- `{fact.subject_label}` — {fact.predicate_label} — `{fact.object_label}` "
                        f"(source: {fact.source})"
                    )

        with st.expander("Corruption audit (synthetic, NOT evidence)", expanded=False):
            st.markdown(CORRUPTION_WARNING)
            corruption = factanchor_record.corruption
            st.json(corruption.summary if corruption else {})
            if corruption and corruption.audit:
                st.table(
                    [
                        {
                            "original object": item.original_object,
                            "corrupted object": item.corrupted_object
                            or ("<removed>" if item.corruption_type == "omission" else "<unchanged>"),
                            "type": item.corruption_type,
                            "is_corrupted": item.is_corrupted,
                            "seed": item.random_seed,
                            "corruption source": item.corruption_source,
                        }
                        for item in corruption.audit
                    ]
                )

        with st.expander("Latency breakdown", expanded=False):
            latency = factanchor_record.latency
            st.json(
                {
                    "model_load_s": latency.model_load_s,
                    "entity_extraction_s": latency.entity_extraction_s,
                    "entity_linking_s": latency.entity_linking_s,
                    "fact_retrieval_s": latency.fact_retrieval_s,
                    "corruption_s": latency.corruption_s,
                    "context_build_s": latency.context_build_s,
                    "generation_s": latency.generation_s,
                    "total_s": latency.total_s,
                    "peak_memory_mb": latency.peak_memory_mb,
                    "device": latency.device,
                    "synchronized": latency.synchronized,
                }
            )

    if config.run.is_toy_model or config.model.model_id == "toy":
        st.warning(TOY_WARNING)


if __name__ == "__main__":
    main()
