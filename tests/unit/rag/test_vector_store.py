"""
Unit tests for ChromaVectorStore.

These tests use mocks to avoid creating actual ChromaDB files,
making them fast and suitable for CI/CD pipelines.

The tests document expected behavior:
- Store lifecycle (initialize, add, search, shutdown)
- Error handling (uninitialized store, invalid inputs)
- Search behavior (similarity scoring, filtering)
- Collection management (stats, deletion)
- Context manager support
"""

from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path

import pytest

from dragonwizard.rag.vector_store import ChromaVectorStore
from dragonwizard.rag.base import ChunkMetadata, SearchResult


class TestVectorStoreInitialization:
    """Test vector store initialization and lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_creates_client_and_collection(self, tmp_path):
        """Should create ChromaDB client and collection on initialize."""
        store = ChromaVectorStore(
            persist_directory=tmp_path / "db",
            collection_name="test_collection"
        )

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection

            await store.initialize()

            # Verify client was created with correct path
            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args.kwargs
            assert call_kwargs["path"] == str(tmp_path / "db")

            # Verify collection was created
            mock_client.get_or_create_collection.assert_called_once_with(
                name="test_collection",
                metadata={"hnsw:space": "cosine"}
            )

            assert store._initialized is True
            assert store._client is not None
            assert store._collection is not None

    @pytest.mark.asyncio
    async def test_initialize_creates_directory_if_missing(self, tmp_path):
        """Should create persist directory if it doesn't exist."""
        db_path = tmp_path / "nonexistent" / "db"
        assert not db_path.exists()

        store = ChromaVectorStore(persist_directory=db_path)

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()

        # Verify directory was created
        assert db_path.exists()
        assert db_path.is_dir()

    @pytest.mark.asyncio
    async def test_initialize_logs_success(self, tmp_path, caplog):
        """Should log initialization success."""
        import logging
        caplog.set_level(logging.INFO)

        store = ChromaVectorStore(
            persist_directory=tmp_path / "db",
            collection_name="test_collection"
        )

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()

        assert "Initializing ChromaDB" in caplog.text
        assert "test_collection" in caplog.text

    @pytest.mark.asyncio
    async def test_initialize_raises_on_failure(self, tmp_path):
        """Should raise RuntimeError if ChromaDB initialization fails."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client_class.side_effect = Exception("ChromaDB connection failed")

            with pytest.raises(RuntimeError, match="Could not initialize ChromaDB"):
                await store.initialize()

            assert store._initialized is False


class TestVectorStoreAdd:
    """Test adding documents to the vector store."""

    @pytest.mark.asyncio
    async def test_add_documents_successfully(self, tmp_path):
        """Should add documents with embeddings to ChromaDB."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()

            # Mock collection's add method
            store._collection.add = MagicMock()

            # Add documents
            ids = ["chunk-1", "chunk-2"]
            documents = ["Fireball deals damage", "Magic Missile never misses"]
            metadatas = [{"source": "srd.pdf"}, {"source": "phb.pdf"}]
            embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

            await store.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

            # Verify ChromaDB's add was called
            store._collection.add.assert_called_once_with(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

    @pytest.mark.asyncio
    async def test_add_without_initialize_raises(self, tmp_path):
        """Should raise RuntimeError if add() called before initialize()."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with pytest.raises(RuntimeError, match="not initialized"):
            await store.add(
                ids=["chunk-1"],
                documents=["text"],
                metadatas=[{}],
                embeddings=[[0.1]]
            )

    @pytest.mark.asyncio
    async def test_add_empty_lists_raises(self, tmp_path):
        """Should raise ValueError if given empty lists."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()

            with pytest.raises(ValueError, match="Cannot add empty lists"):
                await store.add(ids=[], documents=[], metadatas=[], embeddings=[])

    @pytest.mark.asyncio
    async def test_add_mismatched_lengths_raises(self, tmp_path):
        """Should raise ValueError if input lists have different lengths."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()

            with pytest.raises(ValueError, match="Input lists must have same length"):
                await store.add(
                    ids=["chunk-1", "chunk-2"],  # 2 items
                    documents=["text"],  # 1 item (mismatch!)
                    metadatas=[{}],
                    embeddings=[[0.1]]
                )

    @pytest.mark.asyncio
    async def test_add_handles_chromadb_failure(self, tmp_path):
        """Should raise RuntimeError if ChromaDB add fails."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()

            store._collection.add = MagicMock(side_effect=Exception("Duplicate ID"))

            with pytest.raises(RuntimeError, match="Failed to add documents"):
                await store.add(
                    ids=["chunk-1"],
                    documents=["text"],
                    metadatas=[{}],
                    embeddings=[[0.1]]
                )


class TestVectorStoreSearch:
    """Test semantic search functionality."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, tmp_path):
        """Should return SearchResult objects with correct data."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()

            # Mock ChromaDB query response
            store._collection.query = MagicMock(return_value={
                'ids': [['chunk-1', 'chunk-2']],
                'documents': [['Fireball deals fire damage', 'Magic Missile never misses']],
                'metadatas': [[
                    {
                        'document_id': 'doc-1',
                        'source_file': 'srd.pdf',
                        'source_type': 'pdf',
                        'title': 'D&D 5e SRD',
                        'chunk_index': 0,
                        'total_chunks': 10,
                        'token_count': 50,
                        'page_number': 241,
                        'ingestion_timestamp': '2026-01-09T00:00:00',
                        'pipeline_version': '0.1.0'
                    },
                    {
                        'document_id': 'doc-1',
                        'source_file': 'srd.pdf',
                        'source_type': 'pdf',
                        'title': 'D&D 5e SRD',
                        'chunk_index': 1,
                        'total_chunks': 10,
                        'token_count': 45,
                        'page_number': 257,
                        'ingestion_timestamp': '2026-01-09T00:00:00',
                        'pipeline_version': '0.1.0'
                    }
                ]],
                'distances': [[0.2, 0.5]]  # Cosine distances (lower = more similar)
            })

            # Execute search
            results = await store.search(
                query_embedding=[0.1, 0.2, 0.3],
                k=2
            )

            # Verify results
            assert len(results) == 2
            assert isinstance(results[0], SearchResult)

            # Check first result
            assert results[0].text == "Fireball deals fire damage"
            assert results[0].metadata.chunk_id == "chunk-1"
            assert results[0].metadata.page_number == 241
            assert 0.0 <= results[0].score <= 1.0
            assert results[0].citation is not None

            # Verify ChromaDB query was called correctly
            store._collection.query.assert_called_once()
            call_kwargs = store._collection.query.call_args.kwargs
            assert call_kwargs['query_embeddings'] == [[0.1, 0.2, 0.3]]
            assert call_kwargs['n_results'] == 2

    @pytest.mark.asyncio
    async def test_search_converts_distance_to_similarity(self, tmp_path):
        """Should convert ChromaDB cosine distance to similarity score."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()

            # Mock response with known distance
            store._collection.query = MagicMock(return_value={
                'ids': [['chunk-1']],
                'documents': [['text']],
                'metadatas': [[{
                    'document_id': 'doc-1',
                    'source_file': 'test.txt',
                    'source_type': 'txt',
                    'title': 'Test',
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'token_count': 10,
                    'ingestion_timestamp': '2026-01-09T00:00:00',
                    'pipeline_version': '0.1.0'
                }]],
                'distances': [[0.0]]  # Distance 0 = identical vectors
            })

            results = await store.search(query_embedding=[0.1], k=1)

            # Distance 0 should convert to similarity 1.0
            assert results[0].score == 1.0

    @pytest.mark.asyncio
    async def test_search_with_filters(self, tmp_path):
        """Should pass metadata filters to ChromaDB."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()

            store._collection.query = MagicMock(return_value={
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            })

            filters = {"source_type": "pdf"}
            await store.search(query_embedding=[0.1], k=5, filters=filters)

            # Verify filters were passed
            call_kwargs = store._collection.query.call_args.kwargs
            assert call_kwargs['where'] == filters

    @pytest.mark.asyncio
    async def test_search_without_initialize_raises(self, tmp_path):
        """Should raise RuntimeError if search() called before initialize()."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with pytest.raises(RuntimeError, match="not initialized"):
            await store.search(query_embedding=[0.1], k=5)

    @pytest.mark.asyncio
    async def test_search_with_invalid_k_raises(self, tmp_path):
        """Should raise ValueError if k <= 0."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()

            with pytest.raises(ValueError, match="k must be positive"):
                await store.search(query_embedding=[0.1], k=0)

            with pytest.raises(ValueError, match="k must be positive"):
                await store.search(query_embedding=[0.1], k=-5)

    @pytest.mark.asyncio
    async def test_search_handles_chromadb_failure(self, tmp_path):
        """Should raise RuntimeError if ChromaDB query fails."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()

            store._collection.query = MagicMock(side_effect=Exception("Index corrupted"))

            with pytest.raises(RuntimeError, match="Search failed"):
                await store.search(query_embedding=[0.1], k=5)


class TestVectorStoreCitations:
    """Test citation formatting."""

    @pytest.mark.asyncio
    async def test_format_citation_with_page_number(self, tmp_path):
        """Should format citation with page number for PDFs."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        metadata = ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_file="srd.pdf",
            source_type="pdf",
            title="D&D 5e SRD",
            chunk_index=5,
            total_chunks=20,
            token_count=100,
            page_number=241
        )

        citation = store._format_citation(metadata)

        assert "D&D 5e SRD" in citation
        assert "p.241" in citation
        assert "chunk 6/20" in citation  # chunk_index + 1

    @pytest.mark.asyncio
    async def test_format_citation_without_page_number(self, tmp_path):
        """Should format citation without page number for text files."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        metadata = ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_file="rules.txt",
            source_type="txt",
            title="House Rules",
            chunk_index=0,
            total_chunks=5,
            token_count=100,
            page_number=None  # No page number
        )

        citation = store._format_citation(metadata)

        assert "House Rules" in citation
        assert "p." not in citation  # No page number
        assert "chunk 1/5" in citation

    @pytest.mark.asyncio
    async def test_format_citation_with_section(self, tmp_path):
        """Should include section in citation if available."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        metadata = ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_file="readme.md",
            source_type="md",
            title="Campaign Guide",
            chunk_index=0,
            total_chunks=1,
            token_count=100,
            section="Character Creation"
        )

        citation = store._format_citation(metadata)

        assert "Campaign Guide" in citation
        assert "§Character Creation" in citation

    @pytest.mark.asyncio
    async def test_format_citation_includes_edition_when_set(self, tmp_path):
        """Edition tag should appear in citation so LLM and users know which ruleset.

        Format: "[Title (edition), p.N, chunk X/Y]"
        This helps the LLM avoid mixing 5e and 5.5e rules in a single answer,
        and lets users verify which edition a cited passage comes from.
        """
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        metadata = ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_file="data/raw/pdf/5e/phb.pdf",
            source_type="pdf",
            title="Player's Handbook",
            chunk_index=2,
            total_chunks=15,
            token_count=100,
            page_number=42,
            edition="5e",
        )

        citation = store._format_citation(metadata)

        assert "Player's Handbook (5e)" in citation
        assert "p.42" in citation

    @pytest.mark.asyncio
    async def test_format_citation_omits_edition_tag_when_none(self, tmp_path):
        """When edition is None, citation should not contain '(None)' or any edition tag.

        Documents without an edition (homebrew, SRD, forum content) should still
        produce clean citations — absence of edition is silently invisible.
        """
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        metadata = ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_file="homebrew.txt",
            source_type="txt",
            title="House Rules",
            chunk_index=0,
            total_chunks=3,
            token_count=80,
            edition=None,
        )

        citation = store._format_citation(metadata)

        assert "House Rules" in citation
        assert "(None)" not in citation
        assert "()" not in citation
        # Title should appear without any parenthetical suffix
        assert "House Rules (" not in citation


class TestVectorStoreManagement:
    """Test collection management operations."""

    @pytest.mark.asyncio
    async def test_get_stats_returns_document_count(self, tmp_path):
        """Should return statistics about the collection."""
        store = ChromaVectorStore(
            persist_directory=tmp_path / "db",
            collection_name="test_collection"
        )

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()

            # Mock collection count
            store._collection.count = MagicMock(return_value=42)

            stats = await store.get_stats()

            assert stats["document_count"] == 42
            assert stats["collection_name"] == "test_collection"

    @pytest.mark.asyncio
    async def test_get_stats_without_initialize_raises(self, tmp_path):
        """Should raise RuntimeError if get_stats() called before initialize()."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with pytest.raises(RuntimeError, match="not initialized"):
            await store.get_stats()

    @pytest.mark.asyncio
    async def test_delete_collection(self, tmp_path):
        """Should delete the entire collection."""
        store = ChromaVectorStore(
            persist_directory=tmp_path / "db",
            collection_name="test_collection"
        )

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            await store.initialize()
            await store.delete_collection()

            # Verify delete was called
            mock_client.delete_collection.assert_called_once_with(name="test_collection")
            # After deletion, the collection is recreated immediately
            # so that the store remains usable (e.g. for --clear-existing + ingest)
            assert store._collection is not None

    @pytest.mark.asyncio
    async def test_delete_collection_without_initialize_raises(self, tmp_path):
        """Should raise RuntimeError if delete_collection() called before initialize()."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with pytest.raises(RuntimeError, match="not initialized"):
            await store.delete_collection()


class TestVectorStoreShutdown:
    """Test store cleanup and resource management."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_references(self, tmp_path):
        """Should clear client and collection references."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()
            assert store._initialized is True
            assert store._client is not None
            assert store._collection is not None

            await store.shutdown()
            assert store._initialized is False
            assert store._client is None
            assert store._collection is None

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, tmp_path):
        """Should safely handle multiple shutdown() calls."""
        store = ChromaVectorStore(persist_directory=tmp_path / "db")

        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            await store.initialize()
            await store.shutdown()
            await store.shutdown()  # Should not raise

            assert store._initialized is False


class TestVectorStoreContextManager:
    """Test async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager_initializes_and_cleans_up(self, tmp_path):
        """Should initialize on entry and shutdown on exit."""
        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            async with ChromaVectorStore(persist_directory=tmp_path / "db") as store:
                # Inside context: store should be initialized
                assert store._initialized is True
                assert store._client is not None

            # Outside context: store should be shut down
            assert store._initialized is False
            assert store._client is None

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up_on_exception(self, tmp_path):
        """Should cleanup even if exception occurs inside context."""
        with patch("dragonwizard.rag.vector_store.chromadb.PersistentClient"):
            store = None
            try:
                async with ChromaVectorStore(persist_directory=tmp_path / "db") as s:
                    store = s
                    raise ValueError("Test exception")
            except ValueError:
                pass  # Expected

            # Store should still be shut down
            assert store._initialized is False
