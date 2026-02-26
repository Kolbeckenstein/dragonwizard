"""Unit tests for MarkdownLoader."""

from pathlib import Path

import pytest

from dragonwizard.rag.sources.markdown.loader import MarkdownLoader


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
        assert doc.metadata.title == "D&D Guide"
        assert doc.pages is None

    @pytest.mark.asyncio
    async def test_extract_title_from_first_heading(self, tmp_path):
        """Should extract title from first # heading."""
        loader = MarkdownLoader()

        test_file = tmp_path / "guide.md"
        test_content = "Some preamble text.\n\n# Main Title\n\n## Subsection\n\nMore content here."
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
