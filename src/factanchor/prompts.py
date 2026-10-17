"""Context / prompt construction (spec Section 9).

A dedicated builder assembles both branches so that prompt wording cannot drift
between conditions:

* **base branch**: the user question, optionally preceded by the system prompt.
  Trusted facts are added *only* when ``context.trusted_facts_in_base`` is true,
  which is a separately named experiment (retrieval-augmented baseline).
* **contrast branch**: the same user question plus a clearly delimited fact
  block built from the corrupted facts.

Safety properties enforced here:

* corrupted facts are never placed in the base context (asserted in tests);
* a reference answer can never leak into either branch -- :meth:`ContextBuilder.build`
  accepts ``reference`` solely to raise if it appears in the generated text;
* the fact block is length-capped, and truncation is reported in ``notes`` instead
  of happening silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .config import Config, ContextConfig
from .schemas import CorruptionResult, Fact
from .utils import get_logger, truncate

__all__ = [
    "ContextBuilder",
    "ContextPair",
    "EXPERIMENTAL_HEADER",
    "NEUTRAL_HEADER",
    "ReferenceLeakError",
]

logger = get_logger("prompts")

#: Header used for the contrast fact block.  The wording is deliberately explicit
#: so the text can never be mistaken for real evidence in logs or in the UI
#: (spec 8.2 / Section 18).
EXPERIMENTAL_HEADER = "Experimental entity-fact context (may contain intentionally corrupted facts):"

#: Neutral header used by the ``label_as_experimental=False`` prompt ablation,
#: which tests whether the explicit warning changes model behaviour (Section 9).
NEUTRAL_HEADER = "Entity-fact context:"

TRUSTED_HEADER = "Retrieved entity facts (trusted):"


class ReferenceLeakError(RuntimeError):
    """Raised when a reference answer would leak into a generation context."""


@dataclass
class ContextPair:
    """The two contexts used for one example."""

    base_context: str
    contrast_context: str | None
    fact_block: str = ""
    fact_block_was_truncated: bool = False
    template_version: str = "v1"
    n_facts_rendered: int = 0
    notes: tuple[str, ...] = ()
    corrupted_fact_ids: tuple[str, ...] = ()

    @property
    def has_contrast_facts(self) -> bool:
        return bool(self.fact_block.strip())


@dataclass
class ContextBuilder:
    """Builds base/contrast contexts from a question and a fact set."""

    config: ContextConfig = field(default_factory=ContextConfig)
    system_prompt: str | None = None

    @classmethod
    def from_config(cls, config: Config | ContextConfig, *, system_prompt: str | None = None) -> ContextBuilder:
        context = config.context if isinstance(config, Config) else config
        prompt = system_prompt if system_prompt is not None else context.system_prompt
        return cls(config=context, system_prompt=prompt)

    # ------------------------------------------------------------------ build
    def build(
        self,
        question: str,
        *,
        retrieval: CorruptionResult | None = None,
        trusted_facts: Sequence[Fact] = (),
        reference: str | None = None,
        include_contrast: bool = True,
    ) -> ContextPair:
        """Build both contexts.

        Parameters
        ----------
        question:
            The user prompt (already trimmed of dataset artefacts by the loader).
        retrieval:
            Result of the corruption engine.  ``corrupted_facts`` feed the
            contrast branch; ``original_facts`` are only used when
            ``trusted_facts_in_base`` is enabled.
        trusted_facts:
            Facts to place in the *base* context.  Must be empty unless
            ``context.trusted_facts_in_base`` is true.
        reference:
            Optional reference answer; **never** inserted into a context.  It is
            used to verify that no prompt leaks ground truth.
        include_contrast:
            Set to false to build only the base context (baseline runs).
        """
        notes: list[str] = []
        facts_original = list(retrieval.original_facts) if retrieval else []
        facts_contrast = list(retrieval.corrupted_facts) if retrieval else []
        if trusted_facts and not self.config.trusted_facts_in_base:
            raise ValueError(
                "trusted_facts were supplied but context.trusted_facts_in_base is false. "
                "Retrieval augmentation of the base branch is a separate, named experiment "
                "(spec Section 4.1) and must be enabled explicitly."
            )

        base_context = self._render_base(question, trusted_facts if self.config.trusted_facts_in_base else ())
        contrast_context: str | None = None
        fact_block = ""
        truncated_flag = False
        n_rendered = 0
        if include_contrast:
            header = EXPERIMENTAL_HEADER if self.config.label_as_experimental else NEUTRAL_HEADER
            fact_block, truncated_flag, n_rendered = self._render_fact_block(facts_contrast, header=header)
            if not fact_block and self.config.include_empty_fact_block:
                # Empty-contrast control: keep the skeleton, drop the content.
                fact_block = header
            if self.config.include_empty_fact_block or fact_block.strip():
                contrast_context = self.config.contrast_template.format(
                    question=question.strip(),
                    fact_block=fact_block,
                )
            else:
                notes.append("no contrast facts available; contrast context not built")
        if truncated_flag:
            notes.append(
                f"fact block truncated to context.max_fact_block_chars="
                f"{self.config.max_fact_block_chars}; {n_rendered} facts rendered"
            )
        if self.system_prompt:
            notes.append("system prompt applied")

        pair = ContextPair(
            base_context=base_context,
            contrast_context=contrast_context,
            fact_block=fact_block,
            fact_block_was_truncated=truncated_flag,
            template_version=self.config.template_version,
            n_facts_rendered=n_rendered,
            notes=tuple(notes),
            corrupted_fact_ids=tuple(
                fact.fact_id for fact in facts_contrast if fact.fact_id not in {f.fact_id for f in facts_original}
            ),
        )
        if reference:
            self.assert_no_reference_leak(pair, reference)
        return pair

    # ----------------------------------------------------------------- render
    def _render_base(self, question: str, trusted_facts: Sequence[Fact]) -> str:
        base = self.config.base_template.format(question=question.strip())
        if not trusted_facts:
            return base
        block, _truncated, _n = self._render_fact_block(trusted_facts, header=TRUSTED_HEADER)
        # The trusted block goes *before* the question so that the contrast
        # template's structure ("question then facts") is not confused with it.
        return f"{block}\n\n{base}"

    def _render_fact_block(self, facts: Sequence[Fact], *, header: str) -> tuple[str, bool, int]:
        """Render facts into a delimited block, capped at ``max_fact_block_chars``."""
        if not facts:
            return "", False, 0
        lines: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        for fact in facts:
            key = fact.as_triple()
            if key in seen:
                continue
            seen.add(key)
            lines.append(fact.render(self.config.fact_line_format))
        body = "\n".join(lines)
        block = f"{header}\n{body}"
        limit = self.config.max_fact_block_chars
        if len(block) > limit:
            # Truncate on line boundaries so a fact is never half-stated.
            kept: list[str] = []
            running = len(header)
            for line in lines:
                if running + len(line) + 1 > limit:
                    break
                kept.append(line)
                running += len(line) + 1
            block = f"{header}\n" + "\n".join(kept)
            truncated = len(kept) != len(lines)
            if not truncated:  # header alone exceeds the cap: hard-truncate instead
                block = truncate(block, limit)
            return block, truncated, len(kept)
        return block, False, len(lines)

    # ------------------------------------------------------------ safeguards
    @staticmethod
    def assert_no_reference_leak(pair: ContextPair, reference: str, *, min_chars: int = 24) -> None:
        """Raise if (a meaningful slice of) the reference appears in a context.

        Short references are skipped because a few shared words are not a leak;
        a 24+ character verbatim overlap is.
        """
        text = reference.strip()
        if len(text) < min_chars:
            return
        probe = text[: min(len(text), 120)].strip()
        for label, context in (("base", pair.base_context), ("contrast", pair.contrast_context or "")):
            if probe and probe.lower() in context.lower():
                raise ReferenceLeakError(
                    f"reference answer leaked into the {label} context; "
                    "reference text must never be used as generation input"
                )

    # ------------------------------------------------------------- utilities
    def describe(self, pair: ContextPair) -> dict[str, object]:
        """Compact metadata for the generation record (no raw text by default)."""
        return {
            "template_version": pair.template_version,
            "base_context_chars": len(pair.base_context),
            "contrast_context_chars": len(pair.contrast_context or ""),
            "fact_block_chars": len(pair.fact_block),
            "n_facts_rendered": pair.n_facts_rendered,
            "fact_block_truncated": pair.fact_block_was_truncated,
            "notes": list(pair.notes),
        }


def build_contexts(
    question: str,
    *,
    config: Config,
    retrieval: CorruptionResult | None = None,
    include_contrast: bool = True,
    reference: str | None = None,
) -> ContextPair:
    """Module-level convenience wrapper around :class:`ContextBuilder`."""
    return ContextBuilder.from_config(config).build(
        question,
        retrieval=retrieval,
        reference=reference,
        include_contrast=include_contrast,
    )
