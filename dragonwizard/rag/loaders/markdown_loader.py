"""
Markdown document loader.

Loads .md files and extracts heading structure.
"""

import logging
import re
from pathlib import Path

import aiofiles

from dragonwizard.config.logging import get_logger
from dragonwizard.rag.base import Document, DocumentLoader, DocumentMetadata

logger = get_logger(__name__)


class MarkdownLoader(DocumentLoader):
    """
    Loader for Markdown files (.md).

    Reads Markdown files and preserves formatting while extracting
    heading structure for better chunking context.

    Example:
        >>> loader = MarkdownLoader()
        >>> doc = await loader.load(Path("guide.md"))
        >>> print(doc.metadata.title)
        "Guide"  # From filename or # First Heading
    """

    def supports_format(self, file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        return file_path.suffix.lower() in [".md", ".markdown"]

    async def load(self, file_path: Path) -> Document:
        """
        Load a Markdown file.

        Args:
            file_path: Path to the .md file

        Returns:
            Document object with extracted text and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
            ValueError: If file is empty
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        logger.info(f"Loading markdown file: {file_path}")

        try:
            # Read file with UTF-8 encoding
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                text = await f.read()

        except UnicodeDecodeError:
            # Let encoding errors bubble up naturally - they're specific and helpful
            logger.error(f"Failed to decode markdown file (not UTF-8): {file_path}")
            raise

        except Exception as e:
            # Wrap truly generic I/O errors
            logger.error(f"Failed to load markdown file: {e}")
            raise IOError(f"Could not load markdown file '{file_path}': {e}") from e

        # Validate content (after try/except so ValueError isn't caught)
        if not text.strip():
            raise ValueError(f"Markdown file is empty: {file_path}")

        # Extract title from first # heading or use filename
        title = self._extract_title(text, file_path)

        # Extract metadata
        metadata = DocumentMetadata(
            source_file=str(file_path),
            source_type="md",
            title=title,
            author=None,
            page_count=None
        )

        logger.info(
            f"Loaded markdown file: {file_path.name} "
            f"(title: {title}, {len(text)} characters)"
        )

        return Document(
            text=text,
            metadata=metadata,
            pages=None
        )

    def _extract_title(self, text: str, file_path: Path) -> str:
        """
        Extract title from markdown content or filename.

        Looks for the first # heading. If not found, uses filename.

        Args:
            text: Markdown content
            file_path: Path to the markdown file

        Returns:
            Extracted or inferred title
        """
        # Look for first # heading (level 1)
        match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Fall back to filename without extension
        return file_path.stem
