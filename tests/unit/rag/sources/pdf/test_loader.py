"""
Unit tests for PDFLoader (sources/pdf/loader.py).

Covers standard text extraction, OCR fallback, and column-aware mode.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

from dragonwizard.rag.sources.pdf.loader import ExtractionMode, PDFLoadError, PDFLoader


class TestPDFLoader:
    """Test PDFLoader functionality."""

    def test_supports_pdf_format(self):
        """Should support .pdf files."""
        loader = PDFLoader()

        assert loader.supports_format(Path("document.pdf")) is True
        assert loader.supports_format(Path("document.PDF")) is True

    def test_does_not_support_other_formats(self):
        """Should not support non-PDF files."""
        loader = PDFLoader()

        assert loader.supports_format(Path("document.txt")) is False
        assert loader.supports_format(Path("document.md")) is False

    @pytest.mark.asyncio
    async def test_load_pdf_file_successfully(self, tmp_path):
        """Should load PDF file with page data."""
        loader = PDFLoader()

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 2

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 content: Fireball"

        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 content: Magic Missile"

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf):
            test_file = tmp_path / "srd.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            assert doc.text == "Page 1 content: Fireball\n\nPage 2 content: Magic Missile"
            assert doc.metadata.source_file == str(test_file)
            assert doc.metadata.source_type == "pdf"
            assert doc.metadata.title == "srd"
            assert doc.metadata.page_count == 2

            assert len(doc.pages) == 2
            assert doc.pages[0]["page_num"] == 1
            assert doc.pages[0]["text"] == "Page 1 content: Fireball"
            assert doc.pages[1]["page_num"] == 2

            mock_pdf.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_encrypted_pdf_raises(self, tmp_path):
        """Should raise PDFLoadError for encrypted PDFs."""
        loader = PDFLoader()

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = True

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf):
            test_file = tmp_path / "encrypted.pdf"
            test_file.write_text("dummy")

            with pytest.raises(PDFLoadError, match="encrypted"):
                await loader.load(test_file)

            mock_pdf.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_pdf_with_empty_pages_skips_them(self, tmp_path):
        """Should skip pages with no text."""
        loader = PDFLoader()

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 3

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 text"

        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "   \n  "

        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = "Page 3 text"

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2, mock_page3]

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf):
            test_file = tmp_path / "test.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            assert len(doc.pages) == 2
            assert doc.pages[0]["page_num"] == 1
            assert doc.pages[1]["page_num"] == 3
            assert doc.metadata.page_count == 2

    @pytest.mark.asyncio
    async def test_load_pdf_with_failed_pages_continues(self, tmp_path, caplog):
        """Should continue processing if some pages fail."""
        import logging
        caplog.set_level(logging.WARNING)

        loader = PDFLoader()

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 3

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 text"

        mock_page2 = MagicMock()
        mock_page2.get_text.side_effect = Exception("Corrupted page")

        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = "Page 3 text"

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2, mock_page3]

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf):
            test_file = tmp_path / "partial.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            assert len(doc.pages) == 2
            assert "Failed to extract text from page 2" in caplog.text

    @pytest.mark.asyncio
    async def test_load_pdf_with_no_extractable_text_raises(self, tmp_path):
        """Should raise ValueError if PDF has no extractable text."""
        loader = PDFLoader()

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 2

        mock_page = MagicMock()
        mock_page.get_text.return_value = ""
        mock_pdf.__getitem__.return_value = mock_page

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf):
            test_file = tmp_path / "empty.pdf"
            test_file.write_text("dummy")

            with pytest.raises(ValueError, match="no extractable text"):
                await loader.load(test_file)

    @pytest.mark.asyncio
    async def test_load_corrupted_pdf_raises(self, tmp_path):
        """Should raise PDFLoadError for corrupted PDFs."""
        loader = PDFLoader()

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open") as mock_open:
            import fitz
            mock_open.side_effect = fitz.FileDataError("Invalid PDF")

            test_file = tmp_path / "corrupted.pdf"
            test_file.write_text("dummy")

            with pytest.raises(PDFLoadError, match="Invalid or corrupted"):
                await loader.load(test_file)

    @pytest.mark.asyncio
    async def test_load_file_not_found_raises(self):
        """Should raise FileNotFoundError for missing files."""
        loader = PDFLoader()

        with pytest.raises(FileNotFoundError):
            await loader.load(Path("/nonexistent/file.pdf"))


class TestPDFLoaderOCR:
    """Tests for OCR fallback in PDFLoader."""

    def test_page_needs_ocr_sufficient_text(self):
        """Pages with enough extractable text should NOT trigger OCR."""
        loader = PDFLoader(ocr_enabled=True)
        mock_page = MagicMock()
        mock_page.get_images.return_value = [("img1",)]

        assert loader._page_needs_ocr(mock_page, "A" * 100) is False

    def test_page_needs_ocr_little_text_with_images(self):
        """Pages with little text AND images should trigger OCR."""
        loader = PDFLoader(ocr_enabled=True)
        mock_page = MagicMock()
        mock_page.get_images.return_value = [("img1",), ("img2",)]

        assert loader._page_needs_ocr(mock_page, "short") is True

    def test_page_needs_ocr_blank_page_no_images(self):
        """Truly blank pages should NOT trigger OCR."""
        loader = PDFLoader(ocr_enabled=True)
        mock_page = MagicMock()
        mock_page.get_images.return_value = []

        assert loader._page_needs_ocr(mock_page, "") is False

    @pytest.mark.asyncio
    async def test_ocr_fallback_extracts_scanned_text(self, tmp_path):
        """When regular extraction fails, OCR should extract text from scanned pages."""
        loader = PDFLoader(ocr_enabled=True)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 1

        mock_page = MagicMock()
        mock_page.get_text.return_value = ""
        mock_page.get_images.return_value = [("img1",)]

        mock_textpage = MagicMock()
        mock_page.get_textpage_ocr.return_value = mock_textpage

        def get_text_side_effect(*args, **kwargs):
            if "textpage" in kwargs:
                return "Fireball deals 8d6 fire damage"
            return ""
        mock_page.get_text.side_effect = get_text_side_effect

        mock_pdf.__getitem__.return_value = mock_page

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf), \
             patch.object(loader, "_check_ocr_available", return_value=True):
            test_file = tmp_path / "scanned.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            assert "Fireball deals 8d6 fire damage" in doc.text
            assert doc.pages[0]["extraction_method"] == "ocr"

    @pytest.mark.asyncio
    async def test_ocr_disabled_skips_ocr(self, tmp_path):
        """When ocr_enabled=False, never attempt OCR."""
        loader = PDFLoader(ocr_enabled=False)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 2

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Normal text page"

        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = ""
        mock_page2.get_images.return_value = [("img1",)]

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf):
            test_file = tmp_path / "test.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            assert len(doc.pages) == 1
            assert doc.pages[0]["text"] == "Normal text page"
            mock_page2.get_textpage_ocr.assert_not_called()

    @pytest.mark.asyncio
    async def test_ocr_unavailable_degrades_gracefully(self, tmp_path, caplog):
        """When Tesseract isn't installed, skip OCR with a warning."""
        import logging
        caplog.set_level(logging.WARNING)

        loader = PDFLoader(ocr_enabled=True)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 2

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Text page content"

        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = ""
        mock_page2.get_images.return_value = [("img1",)]

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf), \
             patch.object(loader, "_check_ocr_available", return_value=False):
            test_file = tmp_path / "test.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            assert len(doc.pages) == 1
            assert "Text page content" in doc.text

    @pytest.mark.asyncio
    async def test_ocr_failure_continues_processing(self, tmp_path, caplog):
        """If OCR fails on one page, other pages should still process."""
        import logging
        caplog.set_level(logging.WARNING)

        loader = PDFLoader(ocr_enabled=True)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 2

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = ""
        mock_page1.get_images.return_value = [("img1",)]
        mock_page1.get_textpage_ocr.side_effect = RuntimeError("OCR engine error")

        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 has real text content here"

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf), \
             patch.object(loader, "_check_ocr_available", return_value=True):
            test_file = tmp_path / "test.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            assert len(doc.pages) == 1
            assert "Page 2 has real text content" in doc.text
            assert "OCR failed" in caplog.text

    @pytest.mark.asyncio
    async def test_extraction_method_tracked_in_pages(self, tmp_path):
        """Each page should record whether it was extracted via text or OCR."""
        loader = PDFLoader(ocr_enabled=True)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 2

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Normal text content that is long enough to pass the threshold easily"

        mock_page2 = MagicMock()
        mock_page2.get_images.return_value = [("img1",)]
        mock_textpage = MagicMock()
        mock_page2.get_textpage_ocr.return_value = mock_textpage

        def page2_get_text(*args, **kwargs):
            if "textpage" in kwargs:
                return "OCR extracted text from scanned page"
            return ""
        mock_page2.get_text.side_effect = page2_get_text

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf), \
             patch.object(loader, "_check_ocr_available", return_value=True):
            test_file = tmp_path / "mixed.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            assert len(doc.pages) == 2
            assert doc.pages[0]["extraction_method"] == "text"
            assert doc.pages[1]["extraction_method"] == "ocr"


# ---------------------------------------------------------------------------
# ExtractionMode + column-aware PDFLoader
# ---------------------------------------------------------------------------

def _make_two_column_page(left_blocks: list[tuple], right_blocks: list[tuple]) -> MagicMock:
    """Build a mock fitz.Page that simulates a two-column PDF page."""
    all_blocks = []
    for i, (x0, y0, x1, y1, text) in enumerate(left_blocks):
        all_blocks.append((x0, y0, x1, y1, text, i, 0))
    for i, (x0, y0, x1, y1, text) in enumerate(right_blocks, start=len(left_blocks)):
        all_blocks.append((x0, y0, x1, y1, text, i, 0))

    page = MagicMock()
    page.rect.width = 612.0
    page.get_text.side_effect = lambda mode=None, **kw: (
        all_blocks if mode == "blocks" else
        "\n".join(b[4] for b in all_blocks)
    )
    page.get_images.return_value = []
    return page


class TestPDFLoaderColumnAware:
    """Tests for ExtractionMode and column-aware text extraction."""

    def test_extraction_mode_enum_exists(self):
        """ExtractionMode enum should expose DEFAULT and COLUMN_AWARE values."""
        assert ExtractionMode.DEFAULT.value == "default"
        assert ExtractionMode.COLUMN_AWARE.value == "column_aware"

    def test_pdf_loader_default_mode(self):
        """PDFLoader should default to ExtractionMode.DEFAULT."""
        loader = PDFLoader()
        assert loader._extraction_mode == ExtractionMode.DEFAULT

    def test_pdf_loader_accepts_column_aware_mode(self):
        """PDFLoader should accept extraction_mode=ExtractionMode.COLUMN_AWARE."""
        loader = PDFLoader(extraction_mode=ExtractionMode.COLUMN_AWARE)
        assert loader._extraction_mode == ExtractionMode.COLUMN_AWARE

    @pytest.mark.asyncio
    async def test_column_aware_reads_left_column_before_right(self, tmp_path):
        """Column-aware mode should produce left-column text before right-column text."""
        left_blocks = [
            (0, 100, 300, 120, "Left row 1"),
            (0, 200, 300, 220, "Left row 2"),
        ]
        right_blocks = [
            (310, 100, 600, 120, "Right row 1"),
            (310, 200, 600, 220, "Right row 2"),
        ]

        loader = PDFLoader(extraction_mode=ExtractionMode.COLUMN_AWARE, ocr_enabled=False)
        mock_page = _make_two_column_page(left_blocks, right_blocks)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 1
        mock_pdf.__getitem__.return_value = mock_page

        test_file = tmp_path / "two_col.pdf"
        test_file.write_text("dummy")

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf):
            doc = await loader.load(test_file)

        text = doc.text
        assert text.index("Left row 1") < text.index("Right row 1")
        assert text.index("Left row 2") < text.index("Right row 1")

    @pytest.mark.asyncio
    async def test_default_mode_is_unchanged(self, tmp_path):
        """Default mode should call page.get_text() with no arguments."""
        loader = PDFLoader(extraction_mode=ExtractionMode.DEFAULT, ocr_enabled=False)

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Normal text content for the page"
        mock_page.get_images.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 1
        mock_pdf.__getitem__.return_value = mock_page

        test_file = tmp_path / "single.pdf"
        test_file.write_text("dummy")

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf):
            doc = await loader.load(test_file)

        mock_page.get_text.assert_called_once_with()
        assert "Normal text content" in doc.text

    @pytest.mark.asyncio
    async def test_column_aware_calls_get_text_blocks(self, tmp_path):
        """Column-aware mode should call page.get_text('blocks')."""
        left_blocks = [(0, 100, 300, 120, "Left content")]
        right_blocks = [(310, 100, 600, 120, "Right content")]

        loader = PDFLoader(extraction_mode=ExtractionMode.COLUMN_AWARE, ocr_enabled=False)
        mock_page = _make_two_column_page(left_blocks, right_blocks)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 1
        mock_pdf.__getitem__.return_value = mock_page

        test_file = tmp_path / "col.pdf"
        test_file.write_text("dummy")

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf):
            await loader.load(test_file)

        mock_page.get_text.assert_called_with("blocks")

    @pytest.mark.asyncio
    async def test_column_aware_filters_non_text_blocks(self, tmp_path):
        """Column-aware mode should skip image blocks (block_type != 0)."""
        all_blocks_raw = [
            (0, 50, 300, 70, "Text block left", 0, 0),
            (0, 100, 300, 120, "[image]", 1, 1),
            (310, 50, 600, 70, "Text block right", 2, 0),
        ]

        page = MagicMock()
        page.rect.width = 612.0
        page.get_text.side_effect = lambda mode=None, **kw: (
            all_blocks_raw if mode == "blocks" else "fallback"
        )
        page.get_images.return_value = []

        loader = PDFLoader(extraction_mode=ExtractionMode.COLUMN_AWARE, ocr_enabled=False)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 1
        mock_pdf.__getitem__.return_value = page

        test_file = tmp_path / "mixed_blocks.pdf"
        test_file.write_text("dummy")

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf):
            doc = await loader.load(test_file)

        assert "[image]" not in doc.text
        assert "Text block left" in doc.text
        assert "Text block right" in doc.text

    @pytest.mark.asyncio
    async def test_column_aware_left_sorted_by_y(self, tmp_path):
        """Left-column blocks should be sorted by Y coordinate (top to bottom)."""
        left_blocks = [
            (0, 300, 300, 320, "Left bottom"),
            (0, 100, 300, 120, "Left top"),
        ]
        right_blocks = [
            (310, 500, 600, 520, "Right content"),
        ]

        loader = PDFLoader(extraction_mode=ExtractionMode.COLUMN_AWARE, ocr_enabled=False)
        mock_page = _make_two_column_page(left_blocks, right_blocks)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 1
        mock_pdf.__getitem__.return_value = mock_page

        test_file = tmp_path / "sorted.pdf"
        test_file.write_text("dummy")

        with patch("dragonwizard.rag.sources.pdf.loader.fitz.open", return_value=mock_pdf):
            doc = await loader.load(test_file)

        assert doc.text.index("Left top") < doc.text.index("Left bottom")

    def test_extraction_mode_importable_from_sources_pdf_package(self):
        """ExtractionMode should be importable from dragonwizard.rag.sources.pdf."""
        from dragonwizard.rag.sources.pdf import ExtractionMode as EM
        assert EM.DEFAULT.value == "default"
        assert EM.COLUMN_AWARE.value == "column_aware"

    def test_extraction_mode_importable_from_rag_package(self):
        """ExtractionMode should be re-exported from dragonwizard.rag."""
        from dragonwizard.rag import ExtractionMode as EM
        assert EM.COLUMN_AWARE.value == "column_aware"
