"""Unit tests for TextLoader."""

from pathlib import Path

import pytest

from dragonwizard.rag.sources.text.loader import TextLoader


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

        test_file = tmp_path / "rules.txt"
        test_content = "Fireball deals 8d6 fire damage."
        test_file.write_text(test_content)

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

        test_file = tmp_path / "invalid.txt"
        test_file.write_bytes(b'\xff\xfe\xfd')

        with pytest.raises(UnicodeDecodeError):
            await loader.load(test_file)
