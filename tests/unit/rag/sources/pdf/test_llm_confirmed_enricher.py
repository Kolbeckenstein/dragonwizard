"""
Unit tests for LLMHeadingEnricher.

LLMHeadingEnricher extends StatisticalHeadingEnricher to send ambiguous
heading candidates (font_ratio in [low_ratio, high_ratio)) to an LLM for
confirmation. Certain headings bypass the LLM (too large or too small).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dragonwizard.config.settings import LLMSettings
from dragonwizard.rag.base import Chunk, ChunkMetadata, Document, DocumentMetadata
from dragonwizard.rag.sources.pdf.llm_confirmed_enricher import LLMHeadingEnricher
from dragonwizard.rag.sources.pdf.statistical_enricher import StatisticalHeadingEnricher
from tests.unit.rag.sources.pdf.test_statistical_enricher import _make_fitz_page


def _make_llm_settings() -> LLMSettings:
    return LLMSettings(
        model="anthropic/claude-3-5-haiku-20241022",
        max_tokens=256,
        temperature=0.0,
        api_key="test-key",
    )


def _make_document(text: str = "Orc\nOrcs are fierce.", source_type: str = "pdf") -> Document:
    return Document(
        text=text,
        metadata=DocumentMetadata(
            source_file="test.pdf",
            source_type=source_type,
            title="Monster Manual",
        ),
    )


class TestLLMHeadingEnricherInit:
    """Constructor and configuration tests."""

    def test_default_ratios(self):
        """Default low/high ratios should define the ambiguous zone."""
        enricher = LLMHeadingEnricher(llm_settings=_make_llm_settings())
        assert enricher.low_ratio == pytest.approx(1.05)
        assert enricher.high_ratio == pytest.approx(1.5)

    def test_custom_ratios(self):
        """Constructor should accept custom low_ratio and high_ratio."""
        enricher = LLMHeadingEnricher(
            llm_settings=_make_llm_settings(),
            low_ratio=1.1,
            high_ratio=1.8,
        )
        assert enricher.low_ratio == pytest.approx(1.1)
        assert enricher.high_ratio == pytest.approx(1.8)

    def test_is_statistical_subclass(self):
        """LLMHeadingEnricher should extend StatisticalHeadingEnricher."""
        assert isinstance(LLMHeadingEnricher(llm_settings=_make_llm_settings()), StatisticalHeadingEnricher)

    def test_is_chunk_enricher(self):
        """Should implement ChunkEnricher ABC."""
        from dragonwizard.rag.base import ChunkEnricher
        assert isinstance(LLMHeadingEnricher(llm_settings=_make_llm_settings()), ChunkEnricher)


class TestLLMCandidateClassification:
    """Tests for _detect_headings() LLM branch logic."""

    def test_high_ratio_candidate_bypasses_llm(self, tmp_path):
        """Candidates with font_ratio >= high_ratio should be accepted without LLM."""
        enricher = LLMHeadingEnricher(
            llm_settings=_make_llm_settings(),
            low_ratio=1.05,
            high_ratio=1.5,
        )
        doc = _make_document("Orc\nOrcs are fierce creatures that dwell in dark places.")
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        # font_ratio = 18 / 10 = 1.8 >= high_ratio=1.5 → confirmed without LLM
        # (kept below 2.0 to avoid the filter_font_ratio_ceiling which rejects ornaments)
        spans = [
            {"size": 10.0, "flags": 0, "text": "Orcs are fierce creatures that dwell in dark places."},
            {"size": 18.0, "flags": 0, "text": "Orc"},
        ]
        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf), \
             patch("dragonwizard.rag.sources.pdf.llm_confirmed_enricher.acompletion") as mock_llm:
            headings = enricher._detect_headings(doc)

        mock_llm.assert_not_called()
        assert any(h.text == "Orc" for h in headings)

    def test_low_ratio_candidate_rejected_without_llm(self, tmp_path):
        """Candidates with font_ratio < low_ratio should be rejected without LLM."""
        enricher = LLMHeadingEnricher(
            llm_settings=_make_llm_settings(),
            low_ratio=1.05,
            high_ratio=1.5,
            font_ratio_threshold=1.0,
        )
        doc = _make_document("Barely bigger\nBody text here.")
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        # font_ratio = 10.2 / 10.0 = 1.02 < low_ratio=1.05 → rejected without LLM
        spans = [
            {"size": 10.0, "flags": 0, "text": "Body text here."},
            {"size": 10.2, "flags": 0, "text": "Barely bigger"},
        ]
        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf), \
             patch("dragonwizard.rag.sources.pdf.llm_confirmed_enricher.acompletion") as mock_llm:
            headings = enricher._detect_headings(doc)

        mock_llm.assert_not_called()
        assert not any(h.text == "Barely bigger" for h in headings)

    def test_ambiguous_candidate_calls_llm(self, tmp_path):
        """Candidates in [low_ratio, high_ratio) should be sent to the LLM."""
        enricher = LLMHeadingEnricher(
            llm_settings=_make_llm_settings(),
            low_ratio=1.05,
            high_ratio=1.5,
            font_ratio_threshold=1.0,
        )
        doc = _make_document("Maybe heading\nBody text here in the document.")
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        # font_ratio = 12 / 10 = 1.2 → in [1.05, 1.5) → ambiguous
        spans = [
            {"size": 10.0, "flags": 0, "text": "Body text here in the document."},
            {"size": 12.0, "flags": 0, "text": "Maybe heading"},
        ]
        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "[0]"

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf), \
             patch("dragonwizard.rag.sources.pdf.llm_confirmed_enricher.acompletion",
                   new=AsyncMock(return_value=mock_response)):
            import asyncio
            headings = asyncio.get_event_loop().run_until_complete(
                enricher._detect_headings_async(doc)
            )

        assert any(h.text == "Maybe heading" for h in headings)

    def test_llm_error_rejects_batch(self, tmp_path):
        """If the LLM call fails, the entire ambiguous batch should be rejected (fail-safe)."""
        enricher = LLMHeadingEnricher(
            llm_settings=_make_llm_settings(),
            low_ratio=1.05,
            high_ratio=1.5,
            font_ratio_threshold=1.0,
        )
        doc = _make_document("Maybe heading\nBody text here in the document.")
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        spans = [
            {"size": 10.0, "flags": 0, "text": "Body text here in the document."},
            {"size": 12.0, "flags": 0, "text": "Maybe heading"},
        ]
        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf), \
             patch("dragonwizard.rag.sources.pdf.llm_confirmed_enricher.acompletion",
                   new=AsyncMock(side_effect=Exception("API error"))):
            import asyncio
            headings = asyncio.get_event_loop().run_until_complete(
                enricher._detect_headings_async(doc)
            )

        assert not any(h.text == "Maybe heading" for h in headings)
