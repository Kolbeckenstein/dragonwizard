"""
Unit tests for SentenceAwareChunker.

These tests document expected chunking behavior:
- Sentence boundary detection
- Token counting and limits
- Overlap between chunks
- Metadata generation
"""

from unittest.mock import MagicMock, patch

import pytest

from dragonwizard.rag.chunking import SentenceAwareChunker


class TestChunkerInitialization:
    """Test chunker initialization."""

    def test_initialize_with_defaults(self):
        """Should initialize with default parameters."""
        chunker = SentenceAwareChunker()

        assert chunker.target_tokens == 512
        assert chunker.overlap_tokens == 50
        assert chunker.encoding_name == "cl100k_base"
        assert chunker._tokenizer is not None

    def test_initialize_with_custom_params(self):
        """Should initialize with custom parameters."""
        chunker = SentenceAwareChunker(
            target_tokens=256,
            overlap_tokens=25,
            encoding_name="cl100k_base"
        )

        assert chunker.target_tokens == 256
        assert chunker.overlap_tokens == 25

    def test_initialize_raises_if_nltk_not_available(self):
        """Should raise RuntimeError if NLTK punkt tokenizer not installed."""
        with patch("dragonwizard.rag.chunking.sent_tokenize") as mock_sent_tokenize:
            mock_sent_tokenize.side_effect = LookupError("punkt_tab not found")

            with pytest.raises(RuntimeError, match="punkt_tab.*not installed"):
                SentenceAwareChunker()


class TestChunkText:
    """Test text chunking functionality."""

    def test_chunk_single_sentence(self):
        """Should create one chunk for short text."""
        chunker = SentenceAwareChunker(target_tokens=100, overlap_tokens=10)

        text = "Fireball deals 8d6 fire damage."
        metadata = {
            "source_file": "test.txt",
            "source_type": "txt",
            "title": "Test"
        }

        chunks = chunker.chunk_text(
            text=text,
            document_id="doc-123",
            metadata=metadata
        )

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].metadata.document_id == "doc-123"
        assert chunks[0].metadata.source_file == "test.txt"

    def test_chunk_multiple_sentences(self):
        """Should split long text into multiple chunks."""
        chunker = SentenceAwareChunker(target_tokens=20, overlap_tokens=5)

        # Text with multiple sentences that will force multiple chunks
        text = (
            "Fireball is a 3rd-level evocation spell. "
            "It deals 8d6 fire damage in a 20-foot radius. "
            "The fire spreads around corners. "
            "It ignites flammable objects in the area."
        )

        metadata = {
            "source_file": "test.txt",
            "source_type": "txt",
            "title": "Test"
        }

        chunks = chunker.chunk_text(
            text=text,
            document_id="doc-123",
            metadata=metadata
        )

        # Should create multiple chunks due to small target_tokens
        assert len(chunks) > 1

        # Check metadata consistency
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.total_chunks == len(chunks)
            assert chunk.metadata.document_id == "doc-123"
            assert chunk.metadata.source_file == "test.txt"

    def test_chunk_respects_token_limit(self):
        """Should not exceed target_tokens by too much."""
        chunker = SentenceAwareChunker(target_tokens=50, overlap_tokens=10)

        text = (
            "This is sentence one with some words. "
            "This is sentence two with more words. "
            "This is sentence three with even more words to test chunking. "
            "This is sentence four continuing the test."
        )

        metadata = {
            "source_file": "test.txt",
            "source_type": "txt",
            "title": "Test"
        }

        chunks = chunker.chunk_text(
            text=text,
            document_id="doc-123",
            metadata=metadata
        )

        # No chunk should exceed target by more than one sentence worth
        for chunk in chunks:
            # Allow some tolerance (up to 2x target) since we don't split mid-sentence
            assert chunk.metadata.token_count <= chunker.target_tokens * 2

    def test_chunk_empty_text_raises(self):
        """Should raise ValueError for empty text."""
        chunker = SentenceAwareChunker()

        with pytest.raises(ValueError, match="Cannot chunk empty text"):
            chunker.chunk_text(
                text="",
                document_id="doc-123",
                metadata={}
            )

        with pytest.raises(ValueError, match="Cannot chunk empty text"):
            chunker.chunk_text(
                text="   \n\n  ",
                document_id="doc-123",
                metadata={}
            )

    def test_chunk_includes_metadata_fields(self):
        """Should include all metadata fields in chunks."""
        chunker = SentenceAwareChunker(target_tokens=100)

        text = "Fireball deals damage."
        metadata = {
            "source_file": "srd.pdf",
            "source_type": "pdf",
            "title": "D&D 5e SRD",
            "page_number": 241,
            "section": "Spells"
        }

        chunks = chunker.chunk_text(
            text=text,
            document_id="doc-456",
            metadata=metadata
        )

        chunk = chunks[0]
        assert chunk.metadata.source_file == "srd.pdf"
        assert chunk.metadata.source_type == "pdf"
        assert chunk.metadata.title == "D&D 5e SRD"
        assert chunk.metadata.page_number == 241
        assert chunk.metadata.section == "Spells"

    def test_chunk_generates_unique_ids(self):
        """Should generate unique chunk IDs."""
        chunker = SentenceAwareChunker(target_tokens=20)

        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        metadata = {
            "source_file": "test.txt",
            "source_type": "txt",
            "title": "Test"
        }

        chunks = chunker.chunk_text(
            text=text,
            document_id="doc-789",
            metadata=metadata
        )

        # Collect all chunk IDs
        chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]

        # All should be unique
        assert len(chunk_ids) == len(set(chunk_ids))

        # All should be valid UUIDs (basic check: length and format)
        for chunk_id in chunk_ids:
            assert len(chunk_id) == 36  # UUID format: 8-4-4-4-12
            assert chunk_id.count('-') == 4

    def test_chunk_sets_no_embedding_initially(self):
        """Should not set embeddings (those are added by pipeline)."""
        chunker = SentenceAwareChunker()

        text = "Fireball deals damage."
        chunks = chunker.chunk_text(
            text=text,
            document_id="doc-123",
            metadata={"source_file": "test.txt", "source_type": "txt", "title": "Test"}
        )

        for chunk in chunks:
            assert chunk.embedding is None


class TestSentenceSplitting:
    """Test sentence boundary detection."""

    def test_split_simple_sentences(self):
        """Should split on period boundaries."""
        chunker = SentenceAwareChunker()

        text = "Sentence one. Sentence two. Sentence three."
        sentences = chunker._split_into_sentences(text)

        assert len(sentences) == 3
        assert "Sentence one." in sentences[0]
        assert "Sentence two." in sentences[1]
        assert "Sentence three." in sentences[2]

    def test_split_handles_abbreviations(self):
        """Should not split on abbreviations (Dr., Mr., etc.)."""
        chunker = SentenceAwareChunker()

        text = "Dr. Smith cast the spell. It dealt damage."
        sentences = chunker._split_into_sentences(text)

        # Should be 2 sentences, not 3 (Dr. shouldn't cause split)
        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0]

    def test_split_filters_empty_sentences(self):
        """Should filter out empty sentences."""
        chunker = SentenceAwareChunker()

        text = "Sentence one.   \n\n   Sentence two."
        sentences = chunker._split_into_sentences(text)

        assert len(sentences) == 2
        # All sentences should have content
        assert all(s.strip() for s in sentences)


class TestTokenCounting:
    """Test token counting functionality."""

    def test_count_tokens_simple_text(self):
        """Should count tokens correctly."""
        chunker = SentenceAwareChunker()

        text = "Hello world"
        count = chunker._count_tokens(text)

        # "Hello world" is 2 tokens in cl100k_base
        assert count == 2

    def test_count_tokens_empty_string(self):
        """Should return 0 for empty string, 1 for whitespace."""
        chunker = SentenceAwareChunker()

        assert chunker._count_tokens("") == 0
        # Whitespace counts as 1 token in tiktoken
        assert chunker._count_tokens("   ") == 1


class TestOverlapGeneration:
    """Test chunk overlap functionality."""

    def test_get_overlap_sentences_basic(self):
        """Should return last sentences up to target tokens."""
        chunker = SentenceAwareChunker()

        sentences = ["Short.", "Also short.", "This is a longer sentence."]
        overlap = chunker._get_overlap_sentences(sentences, target_overlap_tokens=10)

        # Should get last sentence(s) totaling ~10 tokens
        assert len(overlap) >= 1
        assert "sentence" in " ".join(overlap).lower()

    def test_get_overlap_sentences_empty_list(self):
        """Should return empty list for empty input."""
        chunker = SentenceAwareChunker()

        overlap = chunker._get_overlap_sentences([], target_overlap_tokens=10)
        assert overlap == []

    def test_get_overlap_sentences_respects_limit(self):
        """Should not exceed target overlap tokens significantly."""
        chunker = SentenceAwareChunker()

        sentences = [
            "First.",
            "Second.",
            "Third sentence here.",
            "Fourth sentence here too."
        ]

        overlap = chunker._get_overlap_sentences(sentences, target_overlap_tokens=5)

        # Count tokens in overlap
        overlap_text = " ".join(overlap)
        overlap_tokens = chunker._count_tokens(overlap_text)

        # Should be close to target (within reason due to whole sentence constraint)
        # Allow up to 3x target since we can't split sentences
        assert overlap_tokens <= 15  # 5 * 3


class TestChunkMetadataGeneration:
    """Test chunk metadata generation."""

    def test_creates_chunk_objects_with_metadata(self):
        """Should create Chunk objects with ChunkMetadata."""
        chunker = SentenceAwareChunker()

        chunks = ["Chunk one text.", "Chunk two text."]
        document_id = "doc-abc"
        base_metadata = {
            "source_file": "test.txt",
            "source_type": "txt",
            "title": "Test Document"
        }

        chunk_objects = chunker._create_chunk_objects(
            chunks=chunks,
            document_id=document_id,
            base_metadata=base_metadata
        )

        assert len(chunk_objects) == 2

        # Check first chunk
        chunk1 = chunk_objects[0]
        assert chunk1.text == "Chunk one text."
        assert chunk1.metadata.document_id == "doc-abc"
        assert chunk1.metadata.chunk_index == 0
        assert chunk1.metadata.total_chunks == 2
        assert chunk1.metadata.token_count > 0
        assert chunk1.embedding is None

        # Check second chunk
        chunk2 = chunk_objects[1]
        assert chunk2.text == "Chunk two text."
        assert chunk2.metadata.chunk_index == 1
        assert chunk2.metadata.total_chunks == 2

    def test_metadata_includes_optional_fields(self):
        """Should include optional metadata fields if provided."""
        chunker = SentenceAwareChunker()

        chunks = ["Chunk text."]
        base_metadata = {
            "source_file": "srd.pdf",
            "source_type": "pdf",
            "title": "SRD",
            "page_number": 42,
            "section": "Spells",
            "char_start": 0,
            "char_end": 100
        }

        chunk_objects = chunker._create_chunk_objects(
            chunks=chunks,
            document_id="doc-123",
            base_metadata=base_metadata
        )

        metadata = chunk_objects[0].metadata
        assert metadata.page_number == 42
        assert metadata.section == "Spells"
        assert metadata.char_start == 0
        assert metadata.char_end == 100
