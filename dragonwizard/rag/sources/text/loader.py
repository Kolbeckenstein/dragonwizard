"""
Plain text document loader.

Loads .txt files with UTF-8 encoding.
"""

import logging
from pathlib import Path

import aiofiles

from dragonwizard.config.logging import get_logger
from dragonwizard.rag.base import Document, DocumentLoader, DocumentMetadata

logger = get_logger(__name__)


class TextLoader(DocumentLoader):
    """
    Loader for plain text files (.txt).

    Reads UTF-8 encoded text files and extracts basic metadata.

    Example:
        >>> loader = TextLoader()
        >>> doc = await loader.load(Path("rules.txt"))
        >>> print(doc.metadata.title)
        "rules"
    """

    def supports_format(self, file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        return file_path.suffix.lower() == ".txt"

    async def load(self, file_path: Path) -> Document:
        """
        Load a plain text file.

        Args:
            file_path: Path to the .txt file

        Returns:
            Document object with extracted text and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
            ValueError: If file is empty
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        logger.info(f"Loading text file: {file_path}")

        try:
            # Read file with UTF-8 encoding
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                text = await f.read()

        except UnicodeDecodeError:
            # Let encoding errors bubble up naturally - they're specific and helpful
            logger.error(f"Failed to decode text file (not UTF-8): {file_path}")
            raise

        except Exception as e:
            # Wrap truly generic I/O errors
            logger.error(f"Failed to load text file: {e}")
            raise IOError(f"Could not load text file '{file_path}': {e}") from e

        # Validate content (after try/except so ValueError isn't caught)
        if not text.strip():
            raise ValueError(f"Text file is empty: {file_path}")

        # Extract metadata
        metadata = DocumentMetadata(
            source_file=str(file_path),
            source_type="txt",
            title=file_path.stem,  # Filename without extension
            author=None,
            page_count=None
        )

        logger.info(
            f"Loaded text file: {file_path.name} "
            f"({len(text)} characters)"
        )

        return Document(
            text=text,
            metadata=metadata,
            pages=None
        )
