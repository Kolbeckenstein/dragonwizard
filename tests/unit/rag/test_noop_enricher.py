"""
Unit tests for NoOpEnricher.

NoOpEnricher is the identity enricher — it returns chunks unchanged.
It exists so the pipeline has a concrete default and as a baseline
for comparison runs (--enricher none).
"""

import pytest

from dragonwizard.rag.base import Chunk, ChunkMetadata, Document, DocumentMetadata, NoOpEnricher


def _make_chunk(text: str, chunk_index: int = 0) -> Chunk:
    """Build a minimal Chunk for testing."""
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            chunk_id=f"chunk-{chunk_index}",
            document_id="doc-001",
            source_file="test.txt",
            source_type="txt",
            title="Test",
            chunk_index=chunk_index,
            total_chunks=3,
            token_count=10,
        ),
        embedding=None,
    )


def _make_document(text: str = "Test document text.") -> Document:
    """Build a minimal Document for testing."""
    return Document(
        text=text,
        metadata=DocumentMetadata(
            source_file="test.txt",
            source_type="txt",
            title="Test",
        ),
    )


class TestNoOpEnricher:
    """NoOpEnricher should pass chunks through without any modification."""

    @pytest.mark.asyncio
    async def test_returns_same_chunks(self):
        """Should return the exact same list of chunks."""
        enricher = NoOpEnricher()
        chunks = [_make_chunk("Chunk one."), _make_chunk("Chunk two.", 1)]
        document = _make_document()

        result = await enricher.enrich(chunks, document)

        assert result is chunks

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty(self):
        """Should return an empty list when given empty input."""
        enricher = NoOpEnricher()
        result = await enricher.enrich([], _make_document())
        assert result == []

    @pytest.mark.asyncio
    async def test_chunk_text_unchanged(self):
        """Chunk text should be identical after enrichment."""
        enricher = NoOpEnricher()
        original_texts = ["Alpha sentence.", "Beta sentence.", "Gamma sentence."]
        chunks = [_make_chunk(t, i) for i, t in enumerate(original_texts)]
        document = _make_document()

        result = await enricher.enrich(chunks, document)

        assert [c.text for c in result] == original_texts

    @pytest.mark.asyncio
    async def test_is_subclass_of_chunk_enricher(self):
        """NoOpEnricher should implement the ChunkEnricher ABC."""
        from dragonwizard.rag.base import ChunkEnricher
        assert isinstance(NoOpEnricher(), ChunkEnricher)

    @pytest.mark.asyncio
    async def test_importable_from_rag_package(self):
        """NoOpEnricher should be importable from dragonwizard.rag."""
        from dragonwizard.rag import NoOpEnricher as NE  # noqa: F401
        assert NE is NoOpEnricher
