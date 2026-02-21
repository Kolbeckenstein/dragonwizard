"""
Unit tests for document loaders.

These tests use mocked file I/O to avoid creating actual files,
making them fast and suitable for CI/CD pipelines.

The tests document expected behavior:
- File format detection
- Text extraction
- Metadata generation
- Error handling
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

from dragonwizard.rag.loaders import TextLoader, MarkdownLoader, PDFLoader
from dragonwizard.rag.loaders.pdf_loader import PDFLoadError


class TestTextLoader:
    """Test TextLoader functionality."""

    def test_supports_txt_format(self):
        """Should support .txt files."""
        loader = TextLoader()

        assert loader.supports_format(Path("document.txt")) is True
        assert loader.supports_format(Path("document.TXT")) is True
        assert loader.supports_format(Path("/path/to/file.txt")) is True

    def test_does_not_support_other_formats(self):
        """Should not support non-txt files."""
        loader = TextLoader()

        assert loader.supports_format(Path("document.pdf")) is False
        assert loader.supports_format(Path("document.md")) is False
        assert loader.supports_format(Path("document.docx")) is False

    @pytest.mark.asyncio
    async def test_load_text_file_successfully(self, tmp_path):
        """Should load text file and extract metadata."""
        loader = TextLoader()

        # Create test file
        test_file = tmp_path / "rules.txt"
        test_content = "Fireball deals 8d6 fire damage."
        test_file.write_text(test_content)

        # Load document
        doc = await loader.load(test_file)

        assert doc.text == test_content
        assert doc.metadata.source_file == str(test_file)
        assert doc.metadata.source_type == "txt"
        assert doc.metadata.title == "rules"
        assert doc.metadata.author is None
        assert doc.metadata.page_count is None
        assert doc.pages is None

    @pytest.mark.asyncio
    async def test_load_file_not_found_raises(self):
        """Should raise FileNotFoundError for missing files."""
        loader = TextLoader()

        with pytest.raises(FileNotFoundError):
            await loader.load(Path("/nonexistent/file.txt"))

    @pytest.mark.asyncio
    async def test_load_directory_raises(self, tmp_path):
        """Should raise ValueError when path is a directory."""
        loader = TextLoader()

        with pytest.raises(ValueError, match="not a file"):
            await loader.load(tmp_path)

    @pytest.mark.asyncio
    async def test_load_empty_file_raises(self, tmp_path):
        """Should raise ValueError for empty files."""
        loader = TextLoader()

        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        with pytest.raises(ValueError, match="empty"):
            await loader.load(test_file)

    @pytest.mark.asyncio
    async def test_load_whitespace_only_file_raises(self, tmp_path):
        """Should raise ValueError for whitespace-only files."""
        loader = TextLoader()

        test_file = tmp_path / "whitespace.txt"
        test_file.write_text("   \n\n  \t  ")

        with pytest.raises(ValueError, match="empty"):
            await loader.load(test_file)

    @pytest.mark.asyncio
    async def test_load_non_utf8_file_raises(self, tmp_path):
        """Should raise UnicodeDecodeError for non-UTF-8 files."""
        loader = TextLoader()

        # Create file with invalid UTF-8
        test_file = tmp_path / "invalid.txt"
        test_file.write_bytes(b'\xff\xfe\xfd')  # Invalid UTF-8 sequence

        with pytest.raises(UnicodeDecodeError):
            await loader.load(test_file)


class TestMarkdownLoader:
    """Test MarkdownLoader functionality."""

    def test_supports_md_formats(self):
        """Should support .md and .markdown files."""
        loader = MarkdownLoader()

        assert loader.supports_format(Path("document.md")) is True
        assert loader.supports_format(Path("document.MD")) is True
        assert loader.supports_format(Path("document.markdown")) is True
        assert loader.supports_format(Path("document.MARKDOWN")) is True

    def test_does_not_support_other_formats(self):
        """Should not support non-markdown files."""
        loader = MarkdownLoader()

        assert loader.supports_format(Path("document.txt")) is False
        assert loader.supports_format(Path("document.pdf")) is False

    @pytest.mark.asyncio
    async def test_load_markdown_file_successfully(self, tmp_path):
        """Should load markdown file and extract metadata."""
        loader = MarkdownLoader()

        test_file = tmp_path / "guide.md"
        test_content = "# D&D Guide\n\nSome content here."
        test_file.write_text(test_content)

        doc = await loader.load(test_file)

        assert doc.text == test_content
        assert doc.metadata.source_file == str(test_file)
        assert doc.metadata.source_type == "md"
        assert doc.metadata.title == "D&D Guide"  # Extracted from # heading
        assert doc.pages is None

    @pytest.mark.asyncio
    async def test_extract_title_from_first_heading(self, tmp_path):
        """Should extract title from first # heading."""
        loader = MarkdownLoader()

        test_file = tmp_path / "guide.md"
        test_content = """Some preamble text.

# Main Title

## Subsection

More content here."""
        test_file.write_text(test_content)

        doc = await loader.load(test_file)

        assert doc.metadata.title == "Main Title"

    @pytest.mark.asyncio
    async def test_fallback_to_filename_if_no_heading(self, tmp_path):
        """Should use filename as title if no # heading found."""
        loader = MarkdownLoader()

        test_file = tmp_path / "my_guide.md"
        test_content = "## Subsection\n\nNo level 1 heading."
        test_file.write_text(test_content)

        doc = await loader.load(test_file)

        assert doc.metadata.title == "my_guide"

    @pytest.mark.asyncio
    async def test_load_file_not_found_raises(self):
        """Should raise FileNotFoundError for missing files."""
        loader = MarkdownLoader()

        with pytest.raises(FileNotFoundError):
            await loader.load(Path("/nonexistent/file.md"))

    @pytest.mark.asyncio
    async def test_load_empty_markdown_raises(self, tmp_path):
        """Should raise ValueError for empty markdown files."""
        loader = MarkdownLoader()

        test_file = tmp_path / "empty.md"
        test_file.write_text("")

        with pytest.raises(ValueError, match="empty"):
            await loader.load(test_file)


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

        # Mock PyMuPDF
        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 2  # 2 pages

        # Mock pages
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 content: Fireball"

        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 content: Magic Missile"

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]

        with patch("dragonwizard.rag.loaders.pdf_loader.fitz.open", return_value=mock_pdf):
            test_file = tmp_path / "srd.pdf"
            test_file.write_text("dummy")  # Create file for exists() check

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

        with patch("dragonwizard.rag.loaders.pdf_loader.fitz.open", return_value=mock_pdf):
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

        # Page 1: has text
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 text"

        # Page 2: empty
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "   \n  "

        # Page 3: has text
        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = "Page 3 text"

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2, mock_page3]

        with patch("dragonwizard.rag.loaders.pdf_loader.fitz.open", return_value=mock_pdf):
            test_file = tmp_path / "test.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            # Should only have pages 1 and 3
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

        # Page 1: success
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 text"

        # Page 2: fails
        mock_page2 = MagicMock()
        mock_page2.get_text.side_effect = Exception("Corrupted page")

        # Page 3: success
        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = "Page 3 text"

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2, mock_page3]

        with patch("dragonwizard.rag.loaders.pdf_loader.fitz.open", return_value=mock_pdf):
            test_file = tmp_path / "partial.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            # Should have pages 1 and 3
            assert len(doc.pages) == 2
            assert "Failed to extract text from page 2" in caplog.text

    @pytest.mark.asyncio
    async def test_load_pdf_with_no_extractable_text_raises(self, tmp_path):
        """Should raise ValueError if PDF has no extractable text."""
        loader = PDFLoader()

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 2

        # All pages empty
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""
        mock_pdf.__getitem__.return_value = mock_page

        with patch("dragonwizard.rag.loaders.pdf_loader.fitz.open", return_value=mock_pdf):
            test_file = tmp_path / "empty.pdf"
            test_file.write_text("dummy")

            with pytest.raises(ValueError, match="no extractable text"):
                await loader.load(test_file)

    @pytest.mark.asyncio
    async def test_load_corrupted_pdf_raises(self, tmp_path):
        """Should raise PDFLoadError for corrupted PDFs."""
        loader = PDFLoader()

        with patch("dragonwizard.rag.loaders.pdf_loader.fitz.open") as mock_open:
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
    """
    Tests for OCR fallback in PDFLoader.

    The OCR feature auto-detects scanned/image pages and uses Tesseract
    (via PyMuPDF's built-in integration) as a fallback when regular text
    extraction returns insufficient text. All tests mock the OCR calls —
    no Tesseract installation required.

    Design rationale tests:
    - OCR detection uses a text-length heuristic (< 50 chars) plus
      image-presence check. This avoids wasting OCR on blank pages.
    - OCR is opt-in via settings but defaults to True. When Tesseract
      isn't installed, the loader gracefully degrades rather than crashing.
    - Tool failures on individual pages don't stop the entire document.
    """

    def test_page_needs_ocr_sufficient_text(self):
        """Pages with enough extractable text should NOT trigger OCR.

        The 50-char threshold catches truly scanned pages while allowing
        pages with minimal annotations (chapter titles, etc.) through.
        """
        loader = PDFLoader(ocr_enabled=True)
        mock_page = MagicMock()
        mock_page.get_images.return_value = [("img1",)]

        # 100+ chars of text — no OCR needed even though images are present
        assert loader._page_needs_ocr(mock_page, "A" * 100) is False

    def test_page_needs_ocr_little_text_with_images(self):
        """Pages with little text BUT containing images should trigger OCR.

        This is the core scanned-page detection: the page has image content
        (the scan) but almost no extractable text.
        """
        loader = PDFLoader(ocr_enabled=True)
        mock_page = MagicMock()
        mock_page.get_images.return_value = [("img1",), ("img2",)]

        assert loader._page_needs_ocr(mock_page, "short") is True

    def test_page_needs_ocr_blank_page_no_images(self):
        """Truly blank pages (no text AND no images) should NOT trigger OCR.

        There's nothing to OCR on a blank page — don't waste cycles on it.
        """
        loader = PDFLoader(ocr_enabled=True)
        mock_page = MagicMock()
        mock_page.get_images.return_value = []

        assert loader._page_needs_ocr(mock_page, "") is False

    @pytest.mark.asyncio
    async def test_ocr_fallback_extracts_scanned_text(self, tmp_path):
        """When regular extraction fails, OCR should extract text from scanned pages.

        This is the happy path: page has images but no text → OCR produces text.
        """
        loader = PDFLoader(ocr_enabled=True)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 1

        # Page with no regular text but has images (scanned)
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""
        mock_page.get_images.return_value = [("img1",)]

        # OCR returns extracted text
        mock_textpage = MagicMock()
        mock_page.get_textpage_ocr.return_value = mock_textpage
        # When get_text is called with textpage kwarg, return OCR text
        def get_text_side_effect(*args, **kwargs):
            if "textpage" in kwargs:
                return "Fireball deals 8d6 fire damage"
            return ""
        mock_page.get_text.side_effect = get_text_side_effect

        mock_pdf.__getitem__.return_value = mock_page

        with patch("dragonwizard.rag.loaders.pdf_loader.fitz.open", return_value=mock_pdf), \
             patch.object(loader, "_check_ocr_available", return_value=True):
            test_file = tmp_path / "scanned.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            assert "Fireball deals 8d6 fire damage" in doc.text
            assert doc.pages[0]["extraction_method"] == "ocr"

    @pytest.mark.asyncio
    async def test_ocr_disabled_skips_ocr(self, tmp_path):
        """When ocr_enabled=False, never attempt OCR even on scanned pages.

        Users may want to disable OCR for faster processing or in environments
        where Tesseract isn't available.
        """
        loader = PDFLoader(ocr_enabled=False)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 2

        # Page 1: has text
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Normal text page"

        # Page 2: scanned (no text, has images) — but OCR is disabled
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = ""
        mock_page2.get_images.return_value = [("img1",)]

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]

        with patch("dragonwizard.rag.loaders.pdf_loader.fitz.open", return_value=mock_pdf):
            test_file = tmp_path / "test.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            # Should only have the text page; scanned page skipped
            assert len(doc.pages) == 1
            assert doc.pages[0]["text"] == "Normal text page"
            # OCR method should never have been called
            mock_page2.get_textpage_ocr.assert_not_called()

    @pytest.mark.asyncio
    async def test_ocr_unavailable_degrades_gracefully(self, tmp_path, caplog):
        """When Tesseract isn't installed, skip OCR with a warning — don't crash.

        Graceful degradation: the pipeline still processes text-based pages.
        This is important for CI environments or minimal Docker images.
        """
        import logging
        caplog.set_level(logging.WARNING)

        loader = PDFLoader(ocr_enabled=True)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 2

        # Page 1: text
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Text page content"

        # Page 2: scanned
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = ""
        mock_page2.get_images.return_value = [("img1",)]

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]

        with patch("dragonwizard.rag.loaders.pdf_loader.fitz.open", return_value=mock_pdf), \
             patch.object(loader, "_check_ocr_available", return_value=False):
            test_file = tmp_path / "test.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            # Text page should still load fine
            assert len(doc.pages) == 1
            assert "Text page content" in doc.text

    @pytest.mark.asyncio
    async def test_ocr_failure_continues_processing(self, tmp_path, caplog):
        """If OCR fails on one page, other pages should still process.

        Same resilience pattern as the existing page extraction:
        individual page failures don't crash the entire document.
        """
        import logging
        caplog.set_level(logging.WARNING)

        loader = PDFLoader(ocr_enabled=True)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 2

        # Page 1: scanned, OCR fails
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = ""
        mock_page1.get_images.return_value = [("img1",)]
        mock_page1.get_textpage_ocr.side_effect = RuntimeError("OCR engine error")

        # Page 2: normal text
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 has real text content here"

        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]

        with patch("dragonwizard.rag.loaders.pdf_loader.fitz.open", return_value=mock_pdf), \
             patch.object(loader, "_check_ocr_available", return_value=True):
            test_file = tmp_path / "test.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            # Page 2 should still load despite page 1 OCR failure
            assert len(doc.pages) == 1
            assert "Page 2 has real text content" in doc.text
            assert "OCR failed" in caplog.text

    @pytest.mark.asyncio
    async def test_extraction_method_tracked_in_pages(self, tmp_path):
        """Each page should record whether it was extracted via text or OCR.

        This metadata is useful for debugging ingestion quality —
        you can see which pages needed OCR and inspect their output.
        """
        loader = PDFLoader(ocr_enabled=True)

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.__len__.return_value = 2

        # Page 1: regular text extraction
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Normal text content that is long enough to pass the threshold easily"

        # Page 2: OCR extraction
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

        with patch("dragonwizard.rag.loaders.pdf_loader.fitz.open", return_value=mock_pdf), \
             patch.object(loader, "_check_ocr_available", return_value=True):
            test_file = tmp_path / "mixed.pdf"
            test_file.write_text("dummy")

            doc = await loader.load(test_file)

            assert len(doc.pages) == 2
            assert doc.pages[0]["extraction_method"] == "text"
            assert doc.pages[1]["extraction_method"] == "ocr"
