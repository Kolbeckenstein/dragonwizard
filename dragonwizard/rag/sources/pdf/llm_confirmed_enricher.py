"""
LLMHeadingEnricher — LLM-confirmed heading detection.

Extends StatisticalHeadingEnricher to send ambiguous heading candidates
(font_ratio in [low_ratio, high_ratio)) to an LLM for confirmation.

Two-tier decision:
    font_ratio >= high_ratio  → confirmed heading (no LLM call)
    font_ratio <  low_ratio   → rejected (no LLM call)
    low_ratio <= ratio < high_ratio → LLM batch classification (groups of 20)

On LLM failure: the entire batch is rejected (fail-safe behaviour — prefer
missing a heading over injecting a false one).
"""

from __future__ import annotations

import json
import logging

from litellm import acompletion

from dragonwizard.config.logging import get_logger
from dragonwizard.config.settings import LLMSettings
from dragonwizard.rag.base import Document
from dragonwizard.rag.sources.pdf.statistical_enricher import StatisticalHeadingEnricher, _HeadingCandidate

logger = get_logger(__name__)

_BATCH_SIZE = 20


class LLMHeadingEnricher(StatisticalHeadingEnricher):
    """
    Heading enricher that uses an LLM to confirm ambiguous font-size candidates.

    Candidates with font_ratio >= high_ratio are accepted without LLM.
    Candidates with font_ratio < low_ratio are rejected without LLM.
    Candidates in [low_ratio, high_ratio) are batched and sent to the LLM.

    Args:
        llm_settings: LiteLLM model settings (model, api_key, temperature, etc.)
        low_ratio: Lower bound of ambiguous zone (exclusive, default: 1.05)
        high_ratio: Upper bound of ambiguous zone (exclusive, default: 1.5)
        **kwargs: Passed to StatisticalHeadingEnricher (font_ratio_threshold, etc.)
    """

    def __init__(
        self,
        llm_settings: LLMSettings,
        low_ratio: float = 1.05,
        high_ratio: float = 1.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._llm_settings = llm_settings
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio

    def _detect_headings(self, document: Document) -> list[_HeadingCandidate]:
        """
        Synchronous wrapper — runs the async detection via a blocking call.

        For synchronous callers (e.g. unit tests calling _detect_headings directly),
        this runs the event loop. In the pipeline, enrich() calls _detect_headings_async.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in an async context — delegate to async version via task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._detect_headings_async(document))
                    return future.result()
            else:
                return loop.run_until_complete(self._detect_headings_async(document))
        except RuntimeError:
            return asyncio.run(self._detect_headings_async(document))

    async def _detect_headings_async(self, document: Document) -> list[_HeadingCandidate]:
        """
        Async heading detection with LLM confirmation for ambiguous candidates.
        """
        if document.metadata.source_type != "pdf":
            return []

        # Collect all raw candidates from the statistical base
        raw_candidates = self._detect_headings_from_file(document)

        confirmed: list[_HeadingCandidate] = []
        ambiguous: list[_HeadingCandidate] = []

        for candidate in raw_candidates:
            ratio = candidate.font_ratio
            if ratio >= self.high_ratio:
                confirmed.append(candidate)
            elif ratio >= self.low_ratio:
                ambiguous.append(candidate)
            # Below low_ratio → silently rejected

        # Process ambiguous candidates in batches
        llm_confirmed = await self._confirm_via_llm(ambiguous)
        return confirmed + llm_confirmed

    async def _confirm_via_llm(
        self,
        candidates: list[_HeadingCandidate],
    ) -> list[_HeadingCandidate]:
        """Send batches of ambiguous candidates to the LLM for classification."""
        if not candidates:
            return []

        confirmed: list[_HeadingCandidate] = []

        for batch_start in range(0, len(candidates), _BATCH_SIZE):
            batch = candidates[batch_start: batch_start + _BATCH_SIZE]
            try:
                batch_confirmed = await self._classify_batch(batch)
                confirmed.extend(batch_confirmed)
            except Exception as e:
                logger.warning(
                    f"LLM heading classification failed for batch of {len(batch)} candidates: {e}. "
                    "Rejecting entire batch (fail-safe)."
                )
                # Fail-safe: reject all candidates in this batch

        return confirmed

    async def _classify_batch(
        self,
        batch: list[_HeadingCandidate],
    ) -> list[_HeadingCandidate]:
        """
        Call the LLM with a batch of candidates and return confirmed ones.

        The LLM receives a numbered list of candidate texts and must return a
        JSON array of indices (0-based) that it considers to be headings.

        Example LLM response: "[0, 2, 5]" means candidates 0, 2, and 5 are headings.
        """
        numbered = "\n".join(f"{i}. {c.text}" for i, c in enumerate(batch))
        prompt = (
            "The following are text spans extracted from a D&D rulebook PDF. "
            "Each may or may not be a section heading. "
            "Reply with ONLY a JSON array of the indices (0-based) that are headings. "
            "Example: [0, 2] — do not include any other text.\n\n"
            f"{numbered}"
        )

        response = await acompletion(
            model=self._llm_settings.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self._llm_settings.api_key,
            max_tokens=256,
            temperature=0.0,
        )

        content = response.choices[0].message.content.strip()

        try:
            indices = json.loads(content)
            if not isinstance(indices, list):
                raise ValueError(f"Expected list, got {type(indices)}")
            return [batch[i] for i in indices if isinstance(i, int) and 0 <= i < len(batch)]
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.warning(f"Failed to parse LLM heading response {content!r}: {e}")
            return []

    async def enrich(self, chunks, document: Document):
        """Async enrich using LLM-confirmed headings."""
        headings = await self._detect_headings_async(document)
        if not headings:
            return chunks

        result = []
        for chunk in chunks:
            enriched = self._enrich_chunk(chunk, headings)
            result.append(enriched)
        return result
