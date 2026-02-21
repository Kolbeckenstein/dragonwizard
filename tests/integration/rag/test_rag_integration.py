"""
Integration tests for RAG system (Contract Definition).

These tests define the expected behavior of the RAG system.
They serve as executable specifications for:
- Document ingestion pipeline
- Vector storage and retrieval
- Query engine with semantic search

Following hybrid TDD: these contracts guide implementation, then we add
detailed unit tests to document edge cases.
"""

from pathlib import Path

import pytest

from dragonwizard.config.settings import RAGSettings
from dragonwizard.rag import RAGComponents


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_data_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent.parent.parent / "fixtures" / "sample_documents"


@pytest.fixture
def sample_text_file(test_data_dir):
    """Path to sample text file fixture."""
    return test_data_dir / "sample_rules.txt"


@pytest.fixture
def sample_markdown_file(test_data_dir):
    """Path to sample markdown file fixture."""
    return test_data_dir / "sample_readme.md"


@pytest.fixture
def rag_settings(tmp_path):
    """RAGSettings wired to isolated temp directories."""
    return RAGSettings(
        vector_db_path=str(tmp_path / "vector_db"),
        processed_data_path=str(tmp_path / "processed"),
    )


@pytest.fixture
def rag_factory(rag_settings):
    """RAGComponents factory built from test settings."""
    return RAGComponents(rag_settings)


# ---------------------------------------------------------------------------
# Contract Tests
# ---------------------------------------------------------------------------

# Contract Test 1: Full Pipeline from File to Vector Store
@pytest.mark.asyncio
async def test_ingest_text_file_end_to_end(sample_text_file, rag_factory):
    """
    CONTRACT: The system must be able to ingest a text file end-to-end.

    Given: A plain text file containing D&D rules
    When: We run the ingestion pipeline
    Then:
      - The file is loaded successfully
      - Text is chunked into smaller pieces
      - Embeddings are generated for each chunk
      - Chunks are stored in the vector database
      - We can verify the data was stored
    """
    async with rag_factory.create_embedding_model() as embedding_model, \
               rag_factory.create_vector_store() as vector_store:

        pipeline = rag_factory.create_pipeline(embedding_model, vector_store)

        # Execute: Ingest the sample file
        result = await pipeline.ingest_file(sample_text_file)

        # Verify: File was processed and chunks were created
        assert result > 0, "Should have created at least one chunk"

        # Verify: Chunks are actually in the vector store
        stats = await vector_store.get_stats()
        assert stats["document_count"] > 0, "Vector store should contain documents"
        assert stats["document_count"] == result, "Chunk count should match ingestion result"


# Contract Test 2: Semantic Search Returns Relevant Results
@pytest.mark.asyncio
async def test_search_returns_results_with_metadata(sample_text_file, rag_factory):
    """
    CONTRACT: The system must return semantically relevant results with metadata.

    Given: A vector database populated with D&D rules
    When: We search for relevant content
    Then:
      - Results are returned ranked by semantic similarity
      - Each result includes the source text
      - Each result includes metadata (filename, chunk index, etc.)
      - Results are formatted with citations
      - Score is present and normalised (0-1)
    """
    async with rag_factory.create_embedding_model() as embedding_model, \
               rag_factory.create_vector_store() as vector_store:

        pipeline = rag_factory.create_pipeline(embedding_model, vector_store)
        await pipeline.ingest_file(sample_text_file)

        engine = rag_factory.create_engine(embedding_model, vector_store)

        # Execute: Search for relevant content
        results = await engine.search("spell that deals fire damage", k=5)

        # Verify: Got results back
        assert len(results) > 0, "Should return at least one result"

        # Verify: Top result has metadata
        top_result = results[0]
        assert top_result.metadata is not None, "Result should have metadata"
        assert top_result.metadata.source_file is not None, "Should include source file"
        assert "sample_rules" in top_result.metadata.source_file, "Should reference correct source"

        # Verify: Results have citations
        assert top_result.citation is not None, "Result should have formatted citation"
        assert len(top_result.citation) > 0, "Citation should not be empty"

        # Verify: Score is present and reasonable
        assert top_result.score > 0, "Result should have a similarity score"
        assert top_result.score <= 1.0, "Similarity score should be normalised (0-1)"


# Contract Test 3: Vector Store Persists Data Across Restarts
@pytest.mark.asyncio
async def test_vector_store_persists_across_restarts(sample_text_file, rag_factory):
    """
    CONTRACT: Data must persist in the vector store across process restarts.

    Given: We've ingested documents into ChromaDB
    When: We shut down and restart the vector store
    Then:
      - Previously ingested documents are still retrievable
      - Search results are identical to before restart
      - No data is lost
    """
    # Phase 1: Ingest data with first instances
    async with rag_factory.create_embedding_model() as embedding_model1, \
               rag_factory.create_vector_store() as vector_store1:

        pipeline = rag_factory.create_pipeline(embedding_model1, vector_store1)
        chunk_count = await pipeline.ingest_file(sample_text_file)
        assert chunk_count > 0, "Should have ingested chunks"

        engine1 = rag_factory.create_engine(embedding_model1, vector_store1)
        results_before = await engine1.search("magic", k=3)
        assert len(results_before) > 0, "Should find results before restart"

    # Phase 2: Create NEW instances pointing to same database
    async with rag_factory.create_embedding_model() as embedding_model2, \
               rag_factory.create_vector_store() as vector_store2:

        # Verify: Data persisted
        stats = await vector_store2.get_stats()
        assert stats["document_count"] == chunk_count, "Should have same chunk count after restart"

        # Verify: Search still works with persisted data
        engine2 = rag_factory.create_engine(embedding_model2, vector_store2)
        results_after = await engine2.search("magic", k=3)
        assert len(results_after) == len(results_before), "Should return same number of results"

        # Verify: Results are semantically similar (same top result text)
        assert results_after[0].text == results_before[0].text, "Top result should be identical"


# Contract Test 4: Multiple File Types Can Be Ingested
@pytest.mark.asyncio
async def test_ingest_multiple_file_types(
    sample_text_file, sample_markdown_file, rag_factory
):
    """
    CONTRACT: The system must handle multiple document formats.

    Given: Text files (.txt) and Markdown files (.md)
    When: We ingest both types
    Then:
      - Both are processed successfully
      - We can search across both document types
      - Metadata correctly identifies source type
    """
    async with rag_factory.create_embedding_model() as embedding_model, \
               rag_factory.create_vector_store() as vector_store:

        pipeline = rag_factory.create_pipeline(embedding_model, vector_store)

        # Ingest both files
        chunks_txt = await pipeline.ingest_file(sample_text_file)
        chunks_md = await pipeline.ingest_file(sample_markdown_file)

        assert chunks_txt > 0, "Should have processed text file"
        assert chunks_md > 0, "Should have processed markdown file"

        # Verify: Can search across both
        engine = rag_factory.create_engine(embedding_model, vector_store)

        # Search for content from text file
        results_spell = await engine.search("evocation spell fire", k=5)
        assert len(results_spell) > 0, "Should find spell content"
        assert any(
            "sample_rules.txt" in r.metadata.source_file for r in results_spell
        ), "Should find results from text file"

        # Search for content from markdown file
        results_race = await engine.search("dwarf constitution darkvision", k=5)
        assert len(results_race) > 0, "Should find race content"
        assert any(
            "sample_readme.md" in r.metadata.source_file for r in results_race
        ), "Should find results from markdown file"
