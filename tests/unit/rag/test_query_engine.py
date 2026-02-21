"""
Unit tests for RAGEngine query engine.

Tests are organized by behavioral spec:
1. Initialization
2. Query preprocessing
3. Basic search
4. Score thresholding
5. Metadata filtering
6. Context formatting
7. Error handling

All tests use mocked EmbeddingModel and ChromaVectorStore to isolate
the RAGEngine's own logic (preprocessing, filtering, formatting).
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from dragonwizard.rag.base import ChunkMetadata, SearchResult
from dragonwizard.rag.engine import RAGEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_search_result(
    text: str = "Fireball deals 8d6 fire damage.",
    score: float = 0.85,
    source_file: str = "spells.md",
    source_type: str = "md",
    title: str = "Spells",
    chunk_index: int = 0,
    total_chunks: int = 5,
) -> SearchResult:
    """Helper to build a SearchResult with sensible defaults."""
    return SearchResult(
        text=text,
        score=score,
        metadata=ChunkMetadata(
            chunk_id=f"chunk-{chunk_index}",
            document_id="doc-001",
            source_file=source_file,
            source_type=source_type,
            title=title,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            token_count=30,
        ),
        citation=f"[{title}, {source_file}]",
    )


@pytest.fixture
def mock_embedding_model():
    """Mock EmbeddingModel that returns a deterministic 384-dim vector."""
    model = AsyncMock()
    # embed() returns a numpy array of shape (n_texts, 384)
    model.embed = AsyncMock(
        side_effect=lambda texts: np.random.default_rng(42).random((len(texts), 384))
    )
    return model


@pytest.fixture
def mock_vector_store():
    """Mock ChromaVectorStore whose search() returns controlled results."""
    store = AsyncMock()
    store.search = AsyncMock(return_value=[])
    return store


@pytest.fixture
def engine(mock_embedding_model, mock_vector_store):
    """RAGEngine wired up with mocked dependencies."""
    return RAGEngine(
        embedding_model=mock_embedding_model,
        vector_store=mock_vector_store,
    )


# ===========================================================================
# 1. Initialization
# ===========================================================================


class TestEngineInitialization:
    """RAGEngine should accept dependencies and configuration."""

    def test_initialize_with_defaults(self, mock_embedding_model, mock_vector_store):
        """Should initialize with default k and no score threshold."""
        engine = RAGEngine(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
        )
        assert engine.default_k == 5
        assert engine.score_threshold is None

    def test_initialize_with_custom_k(self, mock_embedding_model, mock_vector_store):
        """Should accept a custom default_k."""
        engine = RAGEngine(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
            default_k=10,
        )
        assert engine.default_k == 10

    def test_initialize_with_score_threshold(
        self, mock_embedding_model, mock_vector_store
    ):
        """Should accept an instance-level score threshold."""
        engine = RAGEngine(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
            score_threshold=0.5,
        )
        assert engine.score_threshold == 0.5


# ===========================================================================
# 2. Query preprocessing
# ===========================================================================


class TestQueryPreprocessing:
    """RAGEngine should normalise queries before embedding."""

    def test_strips_leading_and_trailing_whitespace(self, engine):
        """Should strip whitespace from both ends."""
        assert engine._preprocess_query("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self, engine):
        """Should collapse runs of whitespace into a single space."""
        assert engine._preprocess_query("how  does   fireball  work") == (
            "how does fireball work"
        )

    def test_empty_string_raises_value_error(self, engine):
        """Should reject an empty query."""
        with pytest.raises(ValueError, match="[Qq]uery.*empty"):
            engine._preprocess_query("")

    def test_whitespace_only_raises_value_error(self, engine):
        """Should reject a whitespace-only query (the bug you caught!)."""
        with pytest.raises(ValueError, match="[Qq]uery.*empty"):
            engine._preprocess_query("     ")

    def test_tabs_and_newlines_normalised(self, engine):
        """Should treat all whitespace characters uniformly."""
        assert engine._preprocess_query("fire\t\nball") == "fire ball"


# ===========================================================================
# 3. Basic search
# ===========================================================================


class TestBasicSearch:
    """RAGEngine.search() should embed the query and return results."""

    @pytest.mark.asyncio
    async def test_returns_search_results(self, engine, mock_vector_store):
        """Should return a list of SearchResult objects."""
        mock_vector_store.search.return_value = [
            _make_search_result(score=0.9),
            _make_search_result(score=0.7, chunk_index=1),
        ]
        results = await engine.search("fire damage spell")
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_embeds_the_query(self, engine, mock_embedding_model):
        """Should call embed() with the preprocessed query string."""
        await engine.search("fire damage spell")
        mock_embedding_model.embed.assert_awaited_once()
        call_args = mock_embedding_model.embed.call_args[0][0]
        assert call_args == ["fire damage spell"]

    @pytest.mark.asyncio
    async def test_passes_embedding_to_vector_store(
        self, engine, mock_embedding_model, mock_vector_store
    ):
        """Should forward the query embedding to vector_store.search()."""
        await engine.search("fireball")
        mock_vector_store.search.assert_awaited_once()
        # The embedding argument should be a list of floats
        call_kwargs = mock_vector_store.search.call_args
        query_embedding = call_kwargs.kwargs.get(
            "query_embedding", call_kwargs.args[0] if call_kwargs.args else None
        )
        assert query_embedding is not None
        assert len(query_embedding) == 384

    @pytest.mark.asyncio
    async def test_uses_default_k(self, engine, mock_vector_store):
        """Should use default_k when k is not specified."""
        await engine.search("fireball")
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["k"] == 5

    @pytest.mark.asyncio
    async def test_overrides_k_per_query(self, engine, mock_vector_store):
        """Should allow overriding k for a single query."""
        await engine.search("fireball", k=10)
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["k"] == 10

    @pytest.mark.asyncio
    async def test_results_ordered_by_score_descending(
        self, engine, mock_vector_store
    ):
        """Should return results highest-score-first."""
        mock_vector_store.search.return_value = [
            _make_search_result(score=0.9, chunk_index=0),
            _make_search_result(score=0.5, chunk_index=1),
            _make_search_result(score=0.7, chunk_index=2),
        ]
        results = await engine.search("fireball")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_preprocesses_query_before_embedding(
        self, engine, mock_embedding_model
    ):
        """Should normalise the query (collapse spaces) before embedding."""
        await engine.search("  fire   damage  ")
        call_args = mock_embedding_model.embed.call_args[0][0]
        assert call_args == ["fire damage"]


# ===========================================================================
# 4. Score thresholding
# ===========================================================================


class TestScoreThresholding:
    """RAGEngine should optionally filter out low-scoring results."""

    @pytest.mark.asyncio
    async def test_filters_below_instance_threshold(
        self, mock_embedding_model, mock_vector_store
    ):
        """Should exclude results below the instance-level threshold."""
        engine = RAGEngine(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
            score_threshold=0.6,
        )
        mock_vector_store.search.return_value = [
            _make_search_result(score=0.8, chunk_index=0),
            _make_search_result(score=0.3, chunk_index=1),  # Below threshold
            _make_search_result(score=0.7, chunk_index=2),
        ]
        results = await engine.search("fireball")
        assert len(results) == 2
        assert all(r.score >= 0.6 for r in results)

    @pytest.mark.asyncio
    async def test_per_query_threshold_overrides_instance(
        self, mock_embedding_model, mock_vector_store
    ):
        """Should use per-query threshold when provided."""
        engine = RAGEngine(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
            score_threshold=0.3,  # Instance default: lenient
        )
        mock_vector_store.search.return_value = [
            _make_search_result(score=0.8, chunk_index=0),
            _make_search_result(score=0.6, chunk_index=1),
            _make_search_result(score=0.4, chunk_index=2),
        ]
        results = await engine.search("fireball", score_threshold=0.7)
        assert len(results) == 1
        assert results[0].score >= 0.7

    @pytest.mark.asyncio
    async def test_no_threshold_returns_all(self, engine, mock_vector_store):
        """Should return all results when threshold is None (default)."""
        mock_vector_store.search.return_value = [
            _make_search_result(score=0.1, chunk_index=0),
            _make_search_result(score=0.01, chunk_index=1),
        ]
        results = await engine.search("fireball")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_below_threshold(
        self, mock_embedding_model, mock_vector_store
    ):
        """Should return empty list if every result is below threshold."""
        engine = RAGEngine(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
            score_threshold=0.9,
        )
        mock_vector_store.search.return_value = [
            _make_search_result(score=0.5, chunk_index=0),
            _make_search_result(score=0.6, chunk_index=1),
        ]
        results = await engine.search("fireball")
        assert results == []


# ===========================================================================
# 5. Metadata filtering
# ===========================================================================


class TestMetadataFiltering:
    """RAGEngine should pass metadata filters to the vector store."""

    @pytest.mark.asyncio
    async def test_passes_filters_to_vector_store(self, engine, mock_vector_store):
        """Should forward filters dict to vector_store.search()."""
        filters = {"source_type": "pdf"}
        await engine.search("fireball", filters=filters)
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["filters"] == filters

    @pytest.mark.asyncio
    async def test_no_filters_by_default(self, engine, mock_vector_store):
        """Should pass None filters when none are specified."""
        await engine.search("fireball")
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["filters"] is None


# ===========================================================================
# 6. Context formatting
# ===========================================================================


class TestContextFormatting:
    """RAGEngine.format_context() should produce LLM-ready strings."""

    def test_formats_numbered_results(self, engine):
        """Should produce numbered entries with text and citation."""
        results = [
            _make_search_result(
                text="Fireball deals 8d6 fire damage.",
                score=0.9,
                title="Spells",
                source_file="spells.md",
            ),
            _make_search_result(
                text="Fire damage ignites flammable objects.",
                score=0.7,
                title="Combat Rules",
                source_file="combat.md",
                chunk_index=1,
            ),
        ]
        formatted = engine.format_context(results)

        # Should contain the text of both results
        assert "Fireball deals 8d6 fire damage." in formatted
        assert "Fire damage ignites flammable objects." in formatted
        # Should contain numbering
        assert "[1]" in formatted
        assert "[2]" in formatted
        # Should contain score information
        assert "0.9" in formatted or "0.90" in formatted

    def test_empty_results_returns_no_results_message(self, engine):
        """Should return a meaningful message when there are no results."""
        formatted = engine.format_context([])
        assert "no relevant" in formatted.lower() or "no results" in formatted.lower()

    def test_format_includes_source_info(self, engine):
        """Should include source file and title for citations."""
        results = [
            _make_search_result(
                title="Player's Handbook",
                source_file="phb_spells.md",
                score=0.85,
            ),
        ]
        formatted = engine.format_context(results)
        assert "Player's Handbook" in formatted or "phb_spells.md" in formatted

    def test_format_respects_max_tokens(self, engine):
        """Should truncate output when max_tokens is specified."""
        results = [
            _make_search_result(text="A" * 500, score=0.9, chunk_index=i)
            for i in range(10)
        ]
        full = engine.format_context(results)
        truncated = engine.format_context(results, max_tokens=50)
        # Truncated should be shorter than full
        assert len(truncated) < len(full)


# ===========================================================================
# 7. Error handling
# ===========================================================================


class TestErrorHandling:
    """RAGEngine should handle and propagate errors correctly."""

    @pytest.mark.asyncio
    async def test_empty_query_raises_value_error(self, engine):
        """Should raise ValueError for empty queries."""
        with pytest.raises(ValueError, match="[Qq]uery.*empty"):
            await engine.search("")

    @pytest.mark.asyncio
    async def test_whitespace_query_raises_value_error(self, engine):
        """Should raise ValueError for whitespace-only queries."""
        with pytest.raises(ValueError, match="[Qq]uery.*empty"):
            await engine.search("    ")

    @pytest.mark.asyncio
    async def test_embedding_failure_raises_runtime_error(
        self, engine, mock_embedding_model
    ):
        """Should wrap embedding failures in RuntimeError."""
        mock_embedding_model.embed.side_effect = RuntimeError("Model crashed")
        with pytest.raises(RuntimeError, match="[Ee]mbed"):
            await engine.search("fireball")

    @pytest.mark.asyncio
    async def test_vector_store_failure_raises_runtime_error(
        self, engine, mock_vector_store
    ):
        """Should wrap vector store failures in RuntimeError."""
        mock_vector_store.search.side_effect = RuntimeError("ChromaDB down")
        with pytest.raises(RuntimeError, match="[Ss]earch"):
            await engine.search("fireball")
