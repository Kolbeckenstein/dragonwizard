"""
Unit tests for WeightedHeadingEnricher.

WeightedHeadingEnricher extends StatisticalHeadingEnricher, differing only in
the heading prefix format: it includes the confidence score, e.g.
"[Section: Orc (0.87)]\n" vs "[Section: Orc]\n".
"""

from unittest.mock import MagicMock, patch

import pytest

from dragonwizard.rag.base import Document, DocumentMetadata
from dragonwizard.rag.sources.pdf.statistical_enricher import StatisticalHeadingEnricher
from dragonwizard.rag.sources.pdf.weighted_enricher import WeightedHeadingEnricher
from tests.unit.rag.sources.pdf.test_statistical_enricher import (
    _make_chunk,
    _make_document,
    _make_fitz_page,
)


class TestWeightedHeadingEnricher:
    """WeightedHeadingEnricher injects headings with confidence scores."""

    def test_is_statistical_subclass(self):
        """Should extend StatisticalHeadingEnricher."""
        assert isinstance(WeightedHeadingEnricher(), StatisticalHeadingEnricher)

    def test_is_chunk_enricher(self):
        """Should implement ChunkEnricher ABC."""
        from dragonwizard.rag.base import ChunkEnricher
        assert isinstance(WeightedHeadingEnricher(), ChunkEnricher)

    @pytest.mark.asyncio
    async def test_prefix_includes_confidence_score(self, tmp_path):
        """Injected heading prefix should include the confidence score."""
        enricher = WeightedHeadingEnricher(font_ratio_threshold=1.15)
        doc_text = "Orc\nOrcs are fierce creatures."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        chunks = [_make_chunk("Orcs are fierce creatures.", char_start=4, char_end=30)]

        spans = [
            {"size": 10.0, "flags": 0, "text": "Orcs are fierce creatures."},
            {"size": 14.0, "flags": 0, "text": "Orc"},
        ]
        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            result = await enricher.enrich(chunks, doc)

        assert result[0].text.startswith("[Section: Orc (")
        assert ")" in result[0].text.split("\n")[0]

    @pytest.mark.asyncio
    async def test_confidence_format_is_two_decimal_places(self, tmp_path):
        """Confidence score should be formatted to exactly 2 decimal places."""
        enricher = WeightedHeadingEnricher(font_ratio_threshold=1.15)
        doc_text = "Orc\nOrcs are fierce creatures."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        chunks = [_make_chunk("Orcs are fierce creatures.", char_start=4, char_end=30)]

        spans = [
            {"size": 10.0, "flags": 0, "text": "Orcs are fierce creatures."},
            {"size": 14.0, "flags": 0, "text": "Orc"},
        ]
        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            result = await enricher.enrich(chunks, doc)

        first_line = result[0].text.split("\n")[0]
        import re
        match = re.search(r"\((\d+\.\d{2})\)", first_line)
        assert match is not None, f"No 2-decimal confidence in: {first_line!r}"

    @pytest.mark.asyncio
    async def test_no_heading_prefix_unchanged(self, tmp_path):
        """Chunks without a nearby heading should remain unchanged."""
        enricher = WeightedHeadingEnricher(font_ratio_threshold=1.15, max_section_gap=5)
        doc_text = "Orc\n" + "x" * 100 + "Far away body text."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        far_start = 4 + 100
        chunks = [_make_chunk("Far away body text.", char_start=far_start, char_end=far_start + 19)]

        spans = [
            {"size": 10.0, "flags": 0, "text": "x" * 100 + "Far away body text."},
            {"size": 14.0, "flags": 0, "text": "Orc"},
        ]
        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            result = await enricher.enrich(chunks, doc)

        assert not result[0].text.startswith("[Section:")

    @pytest.mark.asyncio
    async def test_differs_from_statistical_prefix(self, tmp_path):
        """WeightedHeadingEnricher prefix should differ from StatisticalHeadingEnricher."""
        doc_text = "Orc\nOrcs are fierce creatures."
        doc_stat = _make_document(doc_text)
        doc_stat.metadata.source_file = str(tmp_path / "stat.pdf")
        doc_weight = _make_document(doc_text)
        doc_weight.metadata.source_file = str(tmp_path / "weight.pdf")

        spans = [
            {"size": 10.0, "flags": 0, "text": "Orcs are fierce creatures."},
            {"size": 14.0, "flags": 0, "text": "Orc"},
        ]
        mock_page1 = _make_fitz_page(spans)
        mock_page2 = _make_fitz_page(spans)
        mock_pdf1 = MagicMock()
        mock_pdf1.__iter__ = MagicMock(return_value=iter([mock_page1]))
        mock_pdf1.__enter__ = MagicMock(return_value=mock_pdf1)
        mock_pdf1.__exit__ = MagicMock(return_value=False)
        mock_pdf2 = MagicMock()
        mock_pdf2.__iter__ = MagicMock(return_value=iter([mock_page2]))
        mock_pdf2.__enter__ = MagicMock(return_value=mock_pdf2)
        mock_pdf2.__exit__ = MagicMock(return_value=False)

        stat = StatisticalHeadingEnricher(font_ratio_threshold=1.15)
        weighted = WeightedHeadingEnricher(font_ratio_threshold=1.15)

        chunk_s = _make_chunk("Orcs are fierce creatures.", char_start=4, char_end=30)
        chunk_w = _make_chunk("Orcs are fierce creatures.", char_start=4, char_end=30)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open",
                   side_effect=[mock_pdf1, mock_pdf2]):
            stat_result = await stat.enrich([chunk_s], doc_stat)
            weight_result = await weighted.enrich([chunk_w], doc_weight)

        assert stat_result[0].text.startswith("[Section: Orc]\n")
        assert weight_result[0].text.startswith("[Section: Orc (")
        assert stat_result[0].text != weight_result[0].text
