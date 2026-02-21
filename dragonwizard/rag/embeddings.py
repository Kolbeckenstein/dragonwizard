"""
Embedding model wrapper using Sentence Transformers.

This module provides a wrapper around the Sentence Transformers library
for generating text embeddings locally (no API keys required).

Embeddings are vector representations of text that capture semantic meaning.
Texts with similar meanings produce similar vectors, enabling semantic search.

Example:
    >>> async with EmbeddingModel("all-MiniLM-L6-v2", device="cpu") as model:
    ...     embeddings = await model.embed(["Hello world", "Greetings"])
    ...     print(embeddings.shape)  # (2, 384) - 2 texts, 384 dimensions
"""

import logging
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from dragonwizard.config.logging import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Wrapper for Sentence Transformers embedding model.

    This class handles:
    - Loading pre-trained embedding models
    - Generating embeddings for text (with batching)
    - Device management (CPU vs CUDA)
    - Async context manager for lifecycle

    The model is loaded lazily on initialize() and cleaned up on shutdown().

    Attributes:
        model_name: Name of the Sentence Transformers model
        device: Device to run on ('cpu' or 'cuda')
        batch_size: Number of texts to process per batch
        _model: Loaded SentenceTransformer instance (None until initialized)
        _initialized: Whether the model has been loaded

    Example:
        >>> model = EmbeddingModel(
        ...     model_name="sentence-transformers/all-MiniLM-L6-v2",
        ...     device="cpu",
        ...     batch_size=32
        ... )
        >>> async with model:
        ...     embeddings = await model.embed(["Hello", "World"])
    """

    def __init__(
        self,
        model_name: str,
        device: Literal["cpu", "cuda"] = "cpu",
        batch_size: int = 32
    ):
        """
        Initialize embedding model (doesn't load weights yet).

        Args:
            model_name: Sentence Transformers model identifier
                Examples: "all-MiniLM-L6-v2", "all-mpnet-base-v2"
            device: Device for inference ('cpu' or 'cuda')
            batch_size: Number of texts to process in parallel

        Note:
            The actual model loading happens in initialize() to support
            async context manager pattern.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Load the embedding model into memory.

        This downloads the model weights if not cached locally.
        First-time downloads can be ~90MB for all-MiniLM-L6-v2.

        Raises:
            RuntimeError: If model loading fails
        """
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")

        try:
            # Load model (downloads if not cached)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._initialized = True

            # Log model info
            embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Embedding model loaded successfully "
                f"(dimension: {embedding_dim}, device: {self.device})"
            )

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not load embedding model '{self.model_name}': {e}") from e

    async def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Texts are processed in batches for efficiency. The embeddings are
        L2-normalized, making them suitable for cosine similarity search.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of shape (len(texts), embedding_dim)
            where embedding_dim is typically 384 for all-MiniLM-L6-v2

        Raises:
            RuntimeError: If model not initialized
            ValueError: If texts list is empty

        Example:
            >>> embeddings = await model.embed([
            ...     "Fireball deals 8d6 fire damage",
            ...     "Magic Missile never misses"
            ... ])
            >>> embeddings.shape
            (2, 384)
            >>> # Embeddings are normalized
            >>> np.linalg.norm(embeddings[0])
            1.0
        """
        if not self._initialized or self._model is None:
            raise RuntimeError(
                "Embedding model not initialized. "
                "Use 'async with EmbeddingModel(...) as model:' or call await model.initialize()"
            )

        if not texts:
            raise ValueError("Cannot embed empty list of texts")

        logger.debug(f"Generating embeddings for {len(texts)} texts (batch_size={self.batch_size})")

        try:
            # Generate embeddings with batching and normalization
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,  # Disable for production
                normalize_embeddings=True,  # L2 normalization for cosine similarity
                convert_to_numpy=True
            )

            logger.debug(f"Generated embeddings with shape {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    async def shutdown(self) -> None:
        """
        Clean up model resources.

        Releases GPU memory (if using CUDA) and marks model as uninitialized.
        """
        if self._model is not None:
            logger.debug("Shutting down embedding model")
            # SentenceTransformer doesn't have explicit cleanup,
            # but we can help garbage collection by removing references
            del self._model
            self._model = None

        self._initialized = False
        logger.debug("Embedding model shutdown complete")

    # Context manager support for async with

    async def __aenter__(self):
        """Enter async context manager (loads model)."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager (cleans up model)."""
        await self.shutdown()
        return False  # Don't suppress exceptions
