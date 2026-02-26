"""
Unit tests for StatisticalHeadingEnricher.

Tests verify:
- Heading detection from font-size ratios
- Heading injection into chunk text as [Section: ...] prefix
- Non-PDF documents return no headings (graceful degradation)
- Bold text within max_bold_length is treated as a heading candidate
- Chunks without a nearby heading are left unchanged
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dragonwizard.rag.base import Chunk, ChunkMetadata, Document, DocumentMetadata
from dragonwizard.rag.sources.pdf.statistical_enricher import (
    DEFAULT_HEADING_FILTERS,
    StatisticalHeadingEnricher,
    filter_font_ratio_ceiling,
    filter_max_length,
    filter_no_artifacts,
    filter_no_digit_artifacts,
    filter_no_trailing_punctuation,
    filter_not_all_lowercase,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    text: str,
    chunk_index: int = 0,
    char_start: int | None = None,
    char_end: int | None = None,
    source_type: str = "pdf",
) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            chunk_id=f"chunk-{chunk_index}",
            document_id="doc-001",
            source_file="test.pdf",
            source_type=source_type,
            title="Test",
            chunk_index=chunk_index,
            total_chunks=5,
            token_count=20,
            char_start=char_start,
            char_end=char_end,
        ),
        embedding=None,
    )


def _make_document(text: str = "Orc\nOrcs are fierce creatures.", source_type: str = "pdf") -> Document:
    return Document(
        text=text,
        metadata=DocumentMetadata(
            source_file="test.pdf",
            source_type=source_type,
            title="Monster Manual",
        ),
    )


def _make_fitz_page(spans: list[dict]) -> MagicMock:
    """
    Build a mock fitz page with a single block containing the given spans.
    Each span dict: {"size": float, "flags": int, "text": str}

    Uses actual Python dicts (not MagicMock) for blocks/lines/spans so that
    .get() and [] subscript work exactly as they do on real PyMuPDF output.
    """
    page_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [dict(s) for s in spans]
                    }
                ]
            }
        ]
    }
    mock_page = MagicMock()
    mock_page.get_text.return_value = page_dict
    mock_page.get_images.return_value = []
    return mock_page


# ---------------------------------------------------------------------------
# TestStatisticalHeadingEnricherInit
# ---------------------------------------------------------------------------

class TestStatisticalHeadingEnricherInit:
    """Constructor defaults and custom values."""

    def test_default_thresholds(self):
        """Default thresholds should be sensible values."""
        enricher = StatisticalHeadingEnricher()
        assert enricher.font_ratio_threshold == pytest.approx(1.15)
        assert enricher.max_bold_length == 60
        assert enricher.max_section_gap == 5000

    def test_custom_thresholds(self):
        """Constructor should accept custom threshold values."""
        enricher = StatisticalHeadingEnricher(
            font_ratio_threshold=1.3,
            max_bold_length=40,
            max_section_gap=2000,
        )
        assert enricher.font_ratio_threshold == pytest.approx(1.3)
        assert enricher.max_bold_length == 40
        assert enricher.max_section_gap == 2000

    def test_is_chunk_enricher(self):
        """Should implement the ChunkEnricher ABC."""
        from dragonwizard.rag.base import ChunkEnricher
        assert isinstance(StatisticalHeadingEnricher(), ChunkEnricher)

    def test_default_filters_set(self):
        """Default heading_filters should equal DEFAULT_HEADING_FILTERS."""
        enricher = StatisticalHeadingEnricher()
        assert enricher.heading_filters == DEFAULT_HEADING_FILTERS

    def test_custom_filters_accepted(self):
        """A custom heading_filters list should be stored as an independent copy."""
        custom = [filter_max_length]
        enricher = StatisticalHeadingEnricher(heading_filters=custom)
        assert enricher.heading_filters == custom
        # Mutating the original list must not affect the enricher's copy
        custom.append(filter_no_artifacts)
        assert len(enricher.heading_filters) == 1


# ---------------------------------------------------------------------------
# TestDefaultHeadingFilters
# ---------------------------------------------------------------------------

class TestDefaultHeadingFilters:
    """Unit tests for each built-in filter function."""

    # filter_no_artifacts

    def test_filter_no_artifacts_rejects_tilde(self):
        assert filter_no_artifacts("::~L", 1.2) is False

    def test_filter_no_artifacts_rejects_double_colon(self):
        assert filter_no_artifacts("::Section", 1.2) is False

    def test_filter_no_artifacts_accepts_clean_text(self):
        assert filter_no_artifacts("Orc", 1.4) is True

    # filter_not_all_lowercase

    def test_filter_not_all_lowercase_rejects_single_lowercase_word(self):
        assert filter_not_all_lowercase("ints", 1.2) is False

    def test_filter_not_all_lowercase_accepts_capitalised(self):
        assert filter_not_all_lowercase("Orc", 1.4) is True

    def test_filter_not_all_lowercase_accepts_multi_word_all_lower(self):
        # Multi-word is not filtered (only single-word all-lowercase is rejected)
        assert filter_not_all_lowercase("some text here", 1.4) is True

    # filter_no_trailing_punctuation

    def test_filter_no_trailing_punctuation_rejects_period(self):
        assert filter_no_trailing_punctuation("Heart.", 1.2) is False

    def test_filter_no_trailing_punctuation_rejects_comma(self):
        assert filter_no_trailing_punctuation("Baro,", 1.2) is False

    def test_filter_no_trailing_punctuation_accepts_clean(self):
        assert filter_no_trailing_punctuation("Vampire", 1.4) is True

    # filter_font_ratio_ceiling

    def test_filter_font_ratio_ceiling_rejects_ratio_gte_2(self):
        assert filter_font_ratio_ceiling("GENIUS", 2.0) is False

    def test_filter_font_ratio_ceiling_rejects_very_large(self):
        assert filter_font_ratio_ceiling("GENIUS", 3.5) is False

    def test_filter_font_ratio_ceiling_accepts_below_ceiling(self):
        assert filter_font_ratio_ceiling("Chapter Title", 1.8) is True

    # filter_no_digit_artifacts

    def test_filter_no_digit_artifacts_rejects_digit_in_caps(self):
        assert filter_no_digit_artifacts("0ROG", 1.3) is False

    def test_filter_no_digit_artifacts_accepts_normal_caps(self):
        assert filter_no_digit_artifacts("ORC", 1.3) is True

    def test_filter_no_digit_artifacts_accepts_multi_word(self):
        # Multi-word is not subject to the digit-in-caps check
        assert filter_no_digit_artifacts("0ROG Tribe", 1.3) is True

    # filter_max_length

    def test_filter_max_length_rejects_over_80(self):
        long_text = "A" * 81
        assert filter_max_length(long_text, 1.2) is False

    def test_filter_max_length_accepts_exactly_80(self):
        text = "A" * 80
        assert filter_max_length(text, 1.2) is True

    def test_filter_max_length_accepts_short(self):
        assert filter_max_length("Orc", 1.4) is True


# ---------------------------------------------------------------------------
# TestHeadingDetection
# ---------------------------------------------------------------------------

class TestHeadingDetection:
    """Tests for _detect_headings() internal logic."""

    def test_non_pdf_returns_empty(self):
        """Non-PDF documents should return an empty heading list without opening a file."""
        enricher = StatisticalHeadingEnricher()
        doc = _make_document(source_type="txt")
        doc.metadata.source_type = "txt"

        headings = enricher._detect_headings(doc)
        assert headings == []

    def test_large_font_span_detected_as_heading(self, tmp_path):
        """A span with font_ratio >= threshold should be detected as a heading."""
        enricher = StatisticalHeadingEnricher(font_ratio_threshold=1.15)
        doc_text = "Orc\nOrcs are fierce creatures that dwell in mountains."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        # body_size = 10, heading_size = 14 → ratio = 1.4 >= 1.15
        spans = [
            {"size": 10.0, "flags": 0, "text": "Orcs are fierce creatures that dwell in mountains."},
            {"size": 14.0, "flags": 0, "text": "Orc"},
        ]

        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            headings = enricher._detect_headings(doc)

        assert any(h.text == "Orc" for h in headings)

    def test_bold_short_span_detected_as_heading(self, tmp_path):
        """A bold span within max_bold_length should be detected as a heading candidate."""
        enricher = StatisticalHeadingEnricher(font_ratio_threshold=2.0, max_bold_length=60)
        doc_text = "Short Bold\nSome body text here that goes on a bit."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        # Same size but bold flag (bit 4 = 16)
        spans = [
            {"size": 10.0, "flags": 0, "text": "Some body text here that goes on a bit."},
            {"size": 10.0, "flags": 16, "text": "Short Bold"},  # bold flag
        ]

        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            headings = enricher._detect_headings(doc)

        assert any(h.text == "Short Bold" for h in headings)

    def test_long_span_not_detected_as_heading(self, tmp_path):
        """A span longer than max_bold_length should not be detected as heading even if bold."""
        enricher = StatisticalHeadingEnricher(font_ratio_threshold=2.0, max_bold_length=20)
        long_text = "This is a very long bold span that exceeds the max length limit."
        doc_text = f"{long_text}\nBody text."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        spans = [
            {"size": 10.0, "flags": 0, "text": "Body text."},
            {"size": 10.0, "flags": 16, "text": long_text},  # bold but too long
        ]

        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            headings = enricher._detect_headings(doc)

        assert not any(h.text == long_text for h in headings)

    def test_heading_confidence_between_zero_and_one(self, tmp_path):
        """All detected headings should have confidence in [0, 1]."""
        enricher = StatisticalHeadingEnricher(font_ratio_threshold=1.15)
        doc_text = "Orc\nOrcs are fierce."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        spans = [
            {"size": 10.0, "flags": 0, "text": "Orcs are fierce."},
            {"size": 14.0, "flags": 0, "text": "Orc"},
        ]

        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            headings = enricher._detect_headings(doc)

        for h in headings:
            assert 0.0 <= h.confidence <= 1.0

    def test_artifact_span_rejected_by_default_filters(self, tmp_path):
        """Spans like '::~L' should be rejected by the default filter pipeline."""
        enricher = StatisticalHeadingEnricher(font_ratio_threshold=1.15)
        doc_text = "::~L\nBody text here."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        spans = [
            {"size": 10.0, "flags": 0, "text": "Body text here."},
            {"size": 14.0, "flags": 0, "text": "::~L"},
        ]

        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            headings = enricher._detect_headings(doc)

        assert not any(h.text == "::~L" for h in headings)

    def test_oversized_span_rejected_by_ceiling_filter(self, tmp_path):
        """A span with font_ratio >= 2.0 should be rejected (chapter ornament)."""
        enricher = StatisticalHeadingEnricher(font_ratio_threshold=1.15)
        doc_text = "GENIUS\nBody text here."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        # body_size=10, heading_size=22 → ratio=2.2 ≥ 2.0 → rejected
        spans = [
            {"size": 10.0, "flags": 0, "text": "Body text here."},
            {"size": 22.0, "flags": 0, "text": "GENIUS"},
        ]

        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            headings = enricher._detect_headings(doc)

        assert not any(h.text == "GENIUS" for h in headings)

    def test_trailing_period_span_rejected(self, tmp_path):
        """A span ending with '.' should be rejected (sentence, not heading)."""
        enricher = StatisticalHeadingEnricher(font_ratio_threshold=1.15)
        doc_text = "Heart.\nBody text here."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        spans = [
            {"size": 10.0, "flags": 0, "text": "Body text here."},
            {"size": 14.0, "flags": 0, "text": "Heart."},
        ]

        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            headings = enricher._detect_headings(doc)

        assert not any(h.text == "Heart." for h in headings)

    def test_custom_empty_filters_accepts_all_candidates(self, tmp_path):
        """An empty heading_filters list should accept all candidates including artefacts."""
        enricher = StatisticalHeadingEnricher(
            font_ratio_threshold=1.15,
            heading_filters=[],
        )
        doc_text = "::~L\nBody text here."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        spans = [
            {"size": 10.0, "flags": 0, "text": "Body text here."},
            {"size": 14.0, "flags": 0, "text": "::~L"},
        ]

        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            headings = enricher._detect_headings(doc)

        assert any(h.text == "::~L" for h in headings)


# ---------------------------------------------------------------------------
# TestStatisticalEnricherInject
# ---------------------------------------------------------------------------

class TestStatisticalEnricherInject:
    """Tests for heading injection into chunks."""

    @pytest.mark.asyncio
    async def test_chunk_with_nearby_heading_gets_prefix(self, tmp_path):
        """A chunk preceded by a heading should have [Section: ...] prepended."""
        enricher = StatisticalHeadingEnricher(font_ratio_threshold=1.15, max_section_gap=5000)
        doc_text = "Orc\nOrcs are fierce creatures that live in dark caves."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        chunks = [_make_chunk("Orcs are fierce creatures that live in dark caves.", char_start=4, char_end=54)]

        spans = [
            {"size": 10.0, "flags": 0, "text": "Orcs are fierce creatures that live in dark caves."},
            {"size": 14.0, "flags": 0, "text": "Orc"},
        ]
        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            result = await enricher.enrich(chunks, doc)

        assert result[0].text.startswith("[Section: Orc]")

    @pytest.mark.asyncio
    async def test_chunk_without_char_start_is_unchanged(self, tmp_path):
        """Chunks with char_start=None should not be modified."""
        enricher = StatisticalHeadingEnricher()
        doc_text = "Orc\nOrcs are fierce."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        chunk = _make_chunk("Orcs are fierce.", char_start=None, char_end=None)

        spans = [
            {"size": 10.0, "flags": 0, "text": "Orcs are fierce."},
            {"size": 14.0, "flags": 0, "text": "Orc"},
        ]
        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            result = await enricher.enrich([chunk], doc)

        assert result[0].text == "Orcs are fierce."

    @pytest.mark.asyncio
    async def test_heading_too_far_away_does_not_inject(self, tmp_path):
        """A heading more than max_section_gap chars before the chunk is not used."""
        enricher = StatisticalHeadingEnricher(max_section_gap=10)
        doc_text = "Orc\n" + "x" * 100 + "Some body text far away."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        far_start = 4 + 100
        chunk = _make_chunk("Some body text far away.", char_start=far_start, char_end=far_start + 24)

        spans = [
            {"size": 10.0, "flags": 0, "text": "x" * 100 + "Some body text far away."},
            {"size": 14.0, "flags": 0, "text": "Orc"},
        ]
        mock_page = _make_fitz_page(spans)
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open", return_value=mock_pdf):
            result = await enricher.enrich([chunk], doc)

        assert not result[0].text.startswith("[Section:")

    @pytest.mark.asyncio
    async def test_non_pdf_chunks_unchanged(self):
        """Non-PDF chunks should be returned unchanged (no fitz.open called)."""
        enricher = StatisticalHeadingEnricher()
        chunk = _make_chunk("Some text.", char_start=0, char_end=10, source_type="txt")
        doc = _make_document(source_type="txt")

        with patch("dragonwizard.rag.sources.pdf.statistical_enricher.fitz.open") as mock_open:
            result = await enricher.enrich([chunk], doc)

        mock_open.assert_not_called()
        assert result[0].text == "Some text."

    @pytest.mark.asyncio
    async def test_enriched_chunks_are_new_objects(self, tmp_path):
        """Enricher must return new Chunk objects, not mutate the originals."""
        enricher = StatisticalHeadingEnricher(font_ratio_threshold=1.15)
        doc_text = "Orc\nOrcs are fierce creatures."
        doc = _make_document(doc_text)
        doc.metadata.source_file = str(tmp_path / "test.pdf")

        original_chunk = _make_chunk("Orcs are fierce creatures.", char_start=4, char_end=30)
        original_text = original_chunk.text

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
            result = await enricher.enrich([original_chunk], doc)

        assert original_chunk.text == original_text
        if result[0].text != original_text:
            assert result[0] is not original_chunk
