"""
Text chunking strategies for the RAG system.

This module provides intelligent text chunking that:
- Splits text at sentence boundaries (preserves semantic units)
- Respects token limits (predictable LLM context usage)
- Creates overlapping chunks (preserves context across boundaries)
- Generates metadata for each chunk

Example:
    >>> chunker = SentenceAwareChunker(
    ...     target_tokens=512,
    ...     overlap_tokens=50,
    ...     encoding_name="cl100k_base"
    ... )
    >>> chunks = chunker.chunk_text(
    ...     text="Fireball is a 3rd-level spell. It deals 8d6 fire damage...",
    ...     metadata={"source_file": "srd.pdf", "title": "D&D 5e SRD"}
    ... )
    >>> for chunk in chunks:
    ...     print(f"Chunk {chunk.metadata.chunk_index}: {chunk.metadata.token_count} tokens")
"""

import logging
import uuid
from typing import Any

import tiktoken
from nltk.tokenize import sent_tokenize

from dragonwizard.config.logging import get_logger
from dragonwizard.rag.base import Chunk, ChunkMetadata

logger = get_logger(__name__)


class SentenceAwareChunker:
    """
    Chunks text at sentence boundaries with token-based limits.

    This chunker:
    1. Splits text into sentences using NLTK's Punkt tokenizer
    2. Groups sentences into chunks up to target_tokens
    3. Creates overlap by including last N tokens from previous chunk
    4. Generates ChunkMetadata for each chunk

    Sentence-aware chunking prevents mid-sentence cuts that would:
    - Break semantic meaning
    - Confuse the embedding model
    - Produce poor search results

    Attributes:
        target_tokens: Target size for each chunk (e.g., 512)
        overlap_tokens: Number of tokens to overlap between chunks (e.g., 50)
        encoding_name: Tiktoken encoding name (default: "cl100k_base" for GPT-4)
        _tokenizer: Tiktoken encoding instance
        _sentence_tokenizer: NLTK sentence tokenizer

    Example:
        >>> chunker = SentenceAwareChunker(target_tokens=512, overlap_tokens=50)
        >>> chunks = chunker.chunk_text(
        ...     text=long_document,
        ...     metadata={"source_file": "srd.pdf", "title": "D&D 5e SRD"}
        ... )
        >>> print(f"Created {len(chunks)} chunks")
    """

    def __init__(
        self,
        target_tokens: int = 512,
        overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the chunker.

        Args:
            target_tokens: Target chunk size in tokens (default: 512)
            overlap_tokens: Number of tokens to overlap (default: 50)
            encoding_name: Tiktoken encoding name (default: "cl100k_base")

        Note:
            The encoding "cl100k_base" is used by GPT-4 and GPT-3.5-turbo.
            It provides accurate token counts for modern LLMs.
        """
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding_name = encoding_name

        # Initialize tiktoken encoder
        try:
            self._tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.error(f"Failed to load tiktoken encoding '{encoding_name}': {e}")
            raise RuntimeError(f"Could not load tokenizer: {e}") from e

        # NLTK's sent_tokenize will handle loading the appropriate tokenizer
        # We just verify it's available by trying to use it
        try:
            # Test that NLTK punkt tokenizer is available
            sent_tokenize("Test sentence.")
        except LookupError:
            logger.error(
                "NLTK 'punkt_tab' tokenizer not found. "
                "Run: python -c \"import nltk; nltk.download('punkt_tab')\""
            )
            raise RuntimeError(
                "NLTK 'punkt_tab' tokenizer not installed. "
                "Install with: python -c \"import nltk; nltk.download('punkt_tab')\""
            )

        logger.debug(
            f"Initialized SentenceAwareChunker "
            f"(target_tokens={target_tokens}, overlap_tokens={overlap_tokens})"
        )

    def chunk_text(
        self,
        text: str,
        document_id: str,
        metadata: dict[str, Any]
    ) -> list[Chunk]:
        """
        Split text into chunks at sentence boundaries.

        Args:
            text: Text to chunk
            document_id: UUID of the parent document
            metadata: Base metadata to include in all chunks (source_file, title, etc.)

        Returns:
            List of Chunk objects with text, metadata, and no embeddings yet

        Raises:
            ValueError: If text is empty

        Example:
            >>> chunks = chunker.chunk_text(
            ...     text="Fireball is a spell. It deals damage. ...",
            ...     document_id="doc-123",
            ...     metadata={
            ...         "source_file": "srd.pdf",
            ...         "source_type": "pdf",
            ...         "title": "D&D 5e SRD",
            ...         "page_number": 241
            ...     }
            ... )
            >>> print(f"Created {len(chunks)} chunks")
            >>> print(f"First chunk: {chunks[0].metadata.token_count} tokens")
        """
        if not text or not text.strip():
            raise ValueError("Cannot chunk empty text")

        logger.debug(f"Chunking text ({len(text)} characters)")

        # Split into sentences
        sentences = self._split_into_sentences(text)
        logger.debug(f"Split into {len(sentences)} sentences")

        # Group sentences into chunks
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0
        overlap_sentences = []  # Sentences to carry over for overlap

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # Check if adding this sentence would exceed target
            if current_chunk_tokens + sentence_tokens > self.target_tokens and current_chunk_sentences:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(chunk_text)

                # Prepare overlap for next chunk
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences,
                    self.overlap_tokens
                )

                # Start new chunk with overlap
                current_chunk_sentences = overlap_sentences + [sentence]
                current_chunk_tokens = self._count_tokens(" ".join(current_chunk_sentences))
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens

        # Add final chunk if any sentences remain
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(chunk_text)

        logger.debug(f"Created {len(chunks)} chunks")

        # Convert to Chunk objects with metadata
        return self._create_chunk_objects(
            chunks=chunks,
            document_id=document_id,
            base_metadata=metadata
        )

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using NLTK Punkt tokenizer.

        Args:
            text: Text to split

        Returns:
            List of sentences (strings)

        Note:
            NLTK Punkt handles abbreviations (Dr., Mr., etc.) and
            edge cases better than naive splitting on periods.
        """
        # Use NLTK's sentence tokenizer
        sentences = sent_tokenize(text.strip())

        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self._tokenizer.encode(text))

    def _get_overlap_sentences(
        self,
        sentences: list[str],
        target_overlap_tokens: int
    ) -> list[str]:
        """
        Get sentences from end of list that total approximately target_overlap_tokens.

        Args:
            sentences: List of sentences
            target_overlap_tokens: Target number of tokens for overlap

        Returns:
            List of sentences from the end that fit within target_overlap_tokens

        Example:
            >>> sentences = ["Hello.", "World.", "This is a test."]
            >>> overlap = chunker._get_overlap_sentences(sentences, 5)
            >>> # Returns last sentence(s) totaling ~5 tokens
        """
        if not sentences:
            return []

        overlap_sentences = []
        overlap_tokens = 0

        # Work backwards from end of list
        for sentence in reversed(sentences):
            sentence_tokens = self._count_tokens(sentence)

            # Stop if adding this sentence would exceed target
            if overlap_tokens + sentence_tokens > target_overlap_tokens and overlap_sentences:
                break

            overlap_sentences.insert(0, sentence)  # Prepend to maintain order
            overlap_tokens += sentence_tokens

        return overlap_sentences

    def _create_chunk_objects(
        self,
        chunks: list[str],
        document_id: str,
        base_metadata: dict[str, Any]
    ) -> list[Chunk]:
        """
        Convert chunk texts to Chunk objects with metadata.

        Args:
            chunks: List of chunk text strings
            document_id: Parent document UUID
            base_metadata: Base metadata to include in all chunks

        Returns:
            List of Chunk objects with ChunkMetadata

        Note:
            Embeddings are not generated here - that happens in the pipeline.
        """
        chunk_objects = []

        for i, chunk_text in enumerate(chunks):
            # Generate unique chunk ID
            chunk_id = str(uuid.uuid4())

            # Count tokens for this chunk
            token_count = self._count_tokens(chunk_text)

            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                source_file=base_metadata.get("source_file", "unknown"),
                source_type=base_metadata.get("source_type", "unknown"),
                title=base_metadata.get("title", "Untitled"),
                chunk_index=i,
                total_chunks=len(chunks),
                token_count=token_count,
                page_number=base_metadata.get("page_number"),
                section=base_metadata.get("section"),
                char_start=base_metadata.get("char_start"),
                char_end=base_metadata.get("char_end")
            )

            # Create chunk (no embedding yet)
            chunk = Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
                embedding=None
            )

            chunk_objects.append(chunk)

        return chunk_objects
