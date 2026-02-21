"""
RAG Query Engine.

High-level interface for querying the RAG system. Orchestrates:
1. Query preprocessing (whitespace normalisation)
2. Query embedding (via EmbeddingModel)
3. Vector similarity search (via ChromaVectorStore)
4. Score filtering
5. Result formatting for LLM consumption
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dragonwizard.rag.base import SearchResult
from dragonwizard.rag.embeddings import EmbeddingModel
from dragonwizard.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Query engine for retrieval-augmented generation.

    Wraps an EmbeddingModel and ChromaVectorStore to provide a simple
    search interface: pass a natural-language query, get ranked results.

    Example:
        >>> engine = RAGEngine(embedding_model=model, vector_store=store)
        >>> results = await engine.search("how does fireball work?")
        >>> context = engine.format_context(results)
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore,
        default_k: int = 5,
        score_threshold: float | None = None,
    ):
        """
        Initialize the RAG query engine.

        Args:
            embedding_model: Initialized embedding model (same one used for ingestion)
            vector_store: Initialized vector store with ingested documents
            default_k: Default number of results to return (can be overridden per-query)
            score_threshold: Minimum similarity score (0.0-1.0). Results below this
                           are filtered out. None means no filtering.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.default_k = default_k
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Search for relevant document chunks.

        Args:
            query: Natural language query string
            k: Number of results (default: self.default_k)
            filters: Metadata filters passed to the vector store
            score_threshold: Override instance threshold for this query.
                           Pass explicitly to override; None falls back to
                           the instance-level self.score_threshold.

        Returns:
            List of SearchResult objects, ordered by descending score

        Raises:
            ValueError: If query is empty or whitespace-only
            RuntimeError: If embedding or vector search fails
        """
        # 1. Preprocess
        processed = self._preprocess_query(query)

        # 2. Embed
        try:
            embeddings: np.ndarray = await self.embedding_model.embed([processed])
            query_embedding = embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise RuntimeError(f"Embedding failed for query: {e}") from e

        # 3. Vector search
        effective_k = k if k is not None else self.default_k
        try:
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                k=effective_k,
                filters=filters,
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RuntimeError(f"Search failed: {e}") from e

        # 4. Score filtering
        effective_threshold = (
            score_threshold if score_threshold is not None else self.score_threshold
        )
        if effective_threshold is not None:
            results = [r for r in results if r.score >= effective_threshold]

        # 5. Ensure descending score order
        results.sort(key=lambda r: r.score, reverse=True)

        logger.debug(
            f"Query '{processed}' returned {len(results)} results "
            f"(k={effective_k}, threshold={effective_threshold})"
        )
        return results

    def format_context(
        self,
        results: list[SearchResult],
        max_tokens: int | None = None,
    ) -> str:
        """
        Format search results as context for LLM prompts.

        Produces a numbered list of results with text and citations,
        suitable for injecting into an LLM system/user prompt.

        Args:
            results: Search results to format
            max_tokens: Approximate token limit. When set, results are
                       included until the budget is exhausted.

        Returns:
            Formatted context string

        Example output::

            [1] Fireball deals 8d6 fire damage in a 20-foot radius...
                Source: Spells, spells.md (score: 0.87)

            [2] Fire damage ignites flammable objects...
                Source: Combat Rules, combat.md (score: 0.72)
        """
        if not results:
            return "No relevant results found."

        sections: list[str] = []
        token_estimate = 0

        for i, result in enumerate(results, start=1):
            section = (
                f"[{i}] {result.text}\n"
                f"    Source: {result.metadata.title}, "
                f"{result.metadata.source_file} "
                f"(score: {result.score:.2f})"
            )

            # Rough token estimate: ~4 chars per token
            section_tokens = len(section) // 4
            if max_tokens is not None and token_estimate + section_tokens > max_tokens:
                break

            sections.append(section)
            token_estimate += section_tokens

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query string before embedding.

        Normalises whitespace and rejects empty queries.

        Args:
            query: Raw user query

        Returns:
            Cleaned query string

        Raises:
            ValueError: If query is empty after normalisation
        """
        processed = " ".join(query.split())
        if not processed:
            raise ValueError("Query cannot be empty")
        return processed
