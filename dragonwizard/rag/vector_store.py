"""
Vector store implementation using ChromaDB.

This module provides a wrapper around ChromaDB for storing and retrieving
text embeddings with semantic search capabilities.

ChromaDB stores:
- Document text chunks
- Vector embeddings (384-dim for all-MiniLM-L6-v2)
- Metadata (source file, page numbers, timestamps, etc.)

Example:
    >>> async with ChromaVectorStore(persist_directory="./data/vector_db") as store:
    ...     # Add documents with embeddings
    ...     await store.add(
    ...         ids=["chunk-1", "chunk-2"],
    ...         documents=["Fireball deals 8d6 fire damage", "Magic Missile never misses"],
    ...         metadatas=[{"source": "srd.pdf"}, {"source": "srd.pdf"}],
    ...         embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]]
    ...     )
    ...
    ...     # Search for similar documents
    ...     results = await store.search(
    ...         query_embedding=[0.15, 0.25, ...],
    ...         k=5
    ...     )
"""

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from dragonwizard.config.logging import get_logger
from dragonwizard.rag.base import SearchResult, ChunkMetadata

logger = get_logger(__name__)


class ChromaVectorStore:
    """
    Wrapper for ChromaDB vector database.

    This class handles:
    - Persistent storage of embeddings and metadata
    - Semantic search with cosine similarity
    - Collection management (create, delete, stats)
    - Citation formatting for LLM prompts

    ChromaDB automatically handles:
    - Vector indexing (HNSW algorithm)
    - Similarity search optimization
    - Disk persistence

    Attributes:
        persist_directory: Path to ChromaDB storage directory
        collection_name: Name of the ChromaDB collection
        _client: ChromaDB client instance
        _collection: Active ChromaDB collection
        _initialized: Whether the store has been initialized

    Example:
        >>> store = ChromaVectorStore(
        ...     persist_directory=Path("./data/vector_db"),
        ...     collection_name="dnd_content"
        ... )
        >>> async with store:
        ...     await store.add(ids=[...], documents=[...], metadatas=[...], embeddings=[...])
        ...     results = await store.search(query_embedding=[...], k=5)
    """

    def __init__(
        self,
        persist_directory: Path | str,
        collection_name: str = "documents"
    ):
        """
        Initialize vector store (doesn't create connection yet).

        Args:
            persist_directory: Path to ChromaDB storage directory
            collection_name: Name of the collection (default: "documents")

        Note:
            The actual ChromaDB connection happens in initialize() to support
            async context manager pattern.
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize ChromaDB client and collection.

        Creates the persist directory if it doesn't exist.
        Creates or loads the specified collection.

        Raises:
            RuntimeError: If ChromaDB initialization fails
        """
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")

        try:
            # Create persist directory if needed
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client with persistence
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,  # Disable telemetry
                    allow_reset=True  # Allow collection deletion for testing
                )
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

            self._initialized = True
            logger.info(
                f"ChromaDB initialized successfully "
                f"(collection: {self.collection_name}, path: {self.persist_directory})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"Could not initialize ChromaDB: {e}") from e

    async def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]]
    ) -> None:
        """
        Add documents with embeddings to the vector store.

        All lists must have the same length and correspond element-wise.

        Args:
            ids: Unique identifiers for each chunk (e.g., UUID strings)
            documents: Text content of each chunk
            metadatas: Metadata dictionaries for each chunk
            embeddings: Vector embeddings for each chunk (384-dim for all-MiniLM-L6-v2)

        Raises:
            RuntimeError: If store not initialized
            ValueError: If input lists have mismatched lengths

        Example:
            >>> await store.add(
            ...     ids=["chunk-1", "chunk-2"],
            ...     documents=["Fireball deals 8d6 fire damage", "Magic Missile never misses"],
            ...     metadatas=[
            ...         {"source_file": "srd.pdf", "page_number": 241},
            ...         {"source_file": "srd.pdf", "page_number": 257}
            ...     ],
            ...     embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]]
            ... )
        """
        if not self._initialized or self._collection is None:
            raise RuntimeError(
                "Vector store not initialized. "
                "Use 'async with ChromaVectorStore(...) as store:' or call await store.initialize()"
            )

        # Validate input lengths
        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError(
                f"Input lists must have same length: "
                f"ids={len(ids)}, documents={len(documents)}, "
                f"metadatas={len(metadatas)}, embeddings={len(embeddings)}"
            )

        if not ids:
            raise ValueError("Cannot add empty lists to vector store")

        logger.debug(f"Adding {len(ids)} documents to collection '{self.collection_name}'")

        try:
            # ChromaDB's add() method takes lists
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

            logger.debug(f"Successfully added {len(ids)} documents")

        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise RuntimeError(f"Failed to add documents: {e}") from e

    async def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """
        Search for semantically similar documents.

        Uses cosine similarity to find the k most similar chunks.
        Returns results sorted by similarity (highest first).

        Args:
            query_embedding: Query vector (384-dim for all-MiniLM-L6-v2)
            k: Number of results to return (default: 5)
            filters: Optional metadata filters (e.g., {"source_type": "pdf"})

        Returns:
            List of SearchResult objects with text, score, metadata, and citation

        Raises:
            RuntimeError: If store not initialized
            ValueError: If k <= 0

        Example:
            >>> results = await store.search(
            ...     query_embedding=[0.1, 0.2, ...],
            ...     k=5,
            ...     filters={"source_type": "pdf"}
            ... )
            >>> for result in results:
            ...     print(f"{result.score:.3f}: {result.text[:50]}...")
            ...     print(f"Citation: {result.citation}")
        """
        if not self._initialized or self._collection is None:
            raise RuntimeError(
                "Vector store not initialized. "
                "Use 'async with ChromaVectorStore(...) as store:' or call await store.initialize()"
            )

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        logger.debug(f"Searching for {k} similar documents (filters: {filters})")

        try:
            # ChromaDB's query() method
            results = self._collection.query(
                query_embeddings=[query_embedding],  # Must be list of embeddings
                n_results=k,
                where=filters  # Optional metadata filtering
            )

            # Convert ChromaDB format to SearchResult objects
            search_results = []

            # ChromaDB returns nested lists: results['ids'][0] = list of ids
            ids = results['ids'][0] if results['ids'] else []
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []

            for i in range(len(ids)):
                # Convert distance to similarity score (0-1 range)
                # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
                # Convert to similarity: 1 - (distance / 2)
                similarity = 1.0 - (distances[i] / 2.0)
                similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

                # Reconstruct ChunkMetadata from dict
                metadata_dict = metadatas[i]
                metadata_dict['chunk_id'] = ids[i]  # Add chunk_id back
                chunk_metadata = ChunkMetadata(**metadata_dict)

                # Format citation
                citation = self._format_citation(chunk_metadata)

                search_results.append(SearchResult(
                    text=documents[i],
                    score=similarity,
                    metadata=chunk_metadata,
                    citation=citation
                ))

            logger.debug(f"Found {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed: {e}") from e

    def _format_citation(self, metadata: ChunkMetadata) -> str:
        """
        Format a citation string from chunk metadata.

        Creates human-readable citations for LLM prompts and UI display.

        Args:
            metadata: Chunk metadata

        Returns:
            Formatted citation string

        Example:
            >>> citation = store._format_citation(metadata)
            >>> print(citation)
            "[D&D 5e SRD, p.241, chunk 5/20]"
        """
        parts = []

        # Title with optional edition tag, e.g. "Player's Handbook (5e)"
        title_part = (
            f"{metadata.title} ({metadata.edition})"
            if metadata.edition
            else metadata.title
        )
        parts.append(title_part)

        # Page number (PDFs only)
        if metadata.page_number:
            parts.append(f"p.{metadata.page_number}")

        # Section (if available)
        if metadata.section:
            parts.append(f"§{metadata.section}")

        # Chunk index (optional detail)
        parts.append(f"chunk {metadata.chunk_index + 1}/{metadata.total_chunks}")

        return f"[{', '.join(parts)}]"

    async def delete_collection(self) -> None:
        """
        Delete the entire collection and all its data.

        WARNING: This is irreversible and will delete all documents in the collection.
        Use with caution.

        Raises:
            RuntimeError: If store not initialized
        """
        if not self._initialized or self._client is None:
            raise RuntimeError(
                "Vector store not initialized. "
                "Use 'async with ChromaVectorStore(...) as store:' or call await store.initialize()"
            )

        logger.warning(f"Deleting collection '{self.collection_name}'")

        try:
            self._client.delete_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted successfully")

            # Recreate the collection immediately to maintain initialized state
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Recreated collection '{self.collection_name}'")

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise RuntimeError(f"Failed to delete collection: {e}") from e

    async def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with:
            - document_count: Number of documents in the collection
            - collection_name: Name of the collection

        Raises:
            RuntimeError: If store not initialized

        Example:
            >>> stats = await store.get_stats()
            >>> print(f"Collection has {stats['document_count']} documents")
        """
        if not self._initialized or self._collection is None:
            raise RuntimeError(
                "Vector store not initialized. "
                "Use 'async with ChromaVectorStore(...) as store:' or call await store.initialize()"
            )

        try:
            count = self._collection.count()
            return {
                "document_count": count,
                "collection_name": self.collection_name
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise RuntimeError(f"Failed to get stats: {e}") from e

    async def shutdown(self) -> None:
        """
        Clean up resources and close ChromaDB connection.

        ChromaDB automatically persists data, so no explicit save is needed.
        """
        if self._client is not None:
            logger.debug("Shutting down ChromaDB")
            # ChromaDB doesn't have explicit close, but clear references
            self._collection = None
            self._client = None

        self._initialized = False
        logger.debug("ChromaDB shutdown complete")

    # Context manager support for async with

    async def __aenter__(self):
        """Enter async context manager (initializes store)."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager (cleans up store)."""
        await self.shutdown()
        return False  # Don't suppress exceptions
