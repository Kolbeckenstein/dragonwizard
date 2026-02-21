"""
Unit tests for EmbeddingModel.

These tests use mocks to avoid loading the actual 90MB Sentence Transformers model,
making them fast and suitable for CI/CD pipelines.

The tests document expected behavior:
- Model lifecycle (initialize, embed, shutdown)
- Error handling (uninitialized model, empty inputs)
- Batch processing behavior
- Vector normalization
- Context manager support
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from dragonwizard.rag.embeddings import EmbeddingModel


class TestEmbeddingModelInitialization:
    """Test model initialization and lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_loads_model(self):
        """Should load SentenceTransformer model on initialize."""
        model = EmbeddingModel(
            model_name="test-model",
            device="cpu",
            batch_size=32
        )

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_st.return_value.get_sentence_embedding_dimension.return_value = 384

            await model.initialize()

            # Verify model was loaded
            mock_st.assert_called_once_with("test-model", device="cpu")
            assert model._initialized is True
            assert model._model is not None

    @pytest.mark.asyncio
    async def test_initialize_logs_model_info(self, caplog):
        """Should log model information on successful initialization."""
        import logging
        caplog.set_level(logging.INFO)  # Ensure we capture INFO level logs

        model = EmbeddingModel(
            model_name="test-model",
            device="cpu",
            batch_size=32
        )

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_st.return_value.get_sentence_embedding_dimension.return_value = 384

            await model.initialize()

            # Check logs
            assert "Loading embedding model: test-model" in caplog.text
            assert "dimension: 384" in caplog.text

    @pytest.mark.asyncio
    async def test_initialize_raises_on_failure(self):
        """Should raise RuntimeError if model loading fails."""
        model = EmbeddingModel(model_name="invalid-model", device="cpu")

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_st.side_effect = Exception("Model not found")

            with pytest.raises(RuntimeError, match="Could not load embedding model"):
                await model.initialize()

            assert model._initialized is False


class TestEmbeddingGeneration:
    """Test embedding generation."""

    @pytest.mark.asyncio
    async def test_embed_returns_correct_shape(self):
        """Should return numpy array with shape (num_texts, embedding_dim)."""
        model = EmbeddingModel(model_name="test-model", device="cpu", batch_size=2)

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            # Mock model to return 384-dim embeddings
            mock_instance = mock_st.return_value
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.return_value = np.random.rand(3, 384)

            await model.initialize()
            embeddings = await model.embed(["text1", "text2", "text3"])

            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (3, 384)

    @pytest.mark.asyncio
    async def test_embed_normalizes_vectors(self):
        """Should return L2-normalized vectors (norm = 1.0)."""
        model = EmbeddingModel(model_name="test-model", device="cpu")

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            # Create mock normalized embeddings
            embeddings = np.random.rand(2, 384)
            # Normalize them (as SentenceTransformer would)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            mock_instance = mock_st.return_value
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.return_value = embeddings

            await model.initialize()
            result = await model.embed(["text1", "text2"])

            # Verify normalization was requested
            mock_instance.encode.assert_called_once()
            call_kwargs = mock_instance.encode.call_args.kwargs
            assert call_kwargs["normalize_embeddings"] is True

            # Verify result is normalized
            norms = np.linalg.norm(result, axis=1)
            np.testing.assert_array_almost_equal(norms, [1.0, 1.0], decimal=5)

    @pytest.mark.asyncio
    async def test_embed_respects_batch_size(self):
        """Should pass batch_size to SentenceTransformer.encode()."""
        model = EmbeddingModel(model_name="test-model", device="cpu", batch_size=16)

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_instance = mock_st.return_value
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.return_value = np.random.rand(5, 384)

            await model.initialize()
            await model.embed(["text1", "text2", "text3", "text4", "text5"])

            # Verify batch_size was used
            call_kwargs = mock_instance.encode.call_args.kwargs
            assert call_kwargs["batch_size"] == 16

    @pytest.mark.asyncio
    async def test_embed_disables_progress_bar(self):
        """Should disable progress bar for production use."""
        model = EmbeddingModel(model_name="test-model", device="cpu")

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_instance = mock_st.return_value
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.return_value = np.random.rand(1, 384)

            await model.initialize()
            await model.embed(["text"])

            # Verify progress bar is disabled
            call_kwargs = mock_instance.encode.call_args.kwargs
            assert call_kwargs["show_progress_bar"] is False

    @pytest.mark.asyncio
    async def test_embed_without_initialize_raises(self):
        """Should raise RuntimeError if embed() called before initialize()."""
        model = EmbeddingModel(model_name="test-model", device="cpu")

        with pytest.raises(RuntimeError, match="not initialized"):
            await model.embed(["text"])

    @pytest.mark.asyncio
    async def test_embed_empty_list_raises(self):
        """Should raise ValueError if given empty text list."""
        model = EmbeddingModel(model_name="test-model", device="cpu")

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_st.return_value.get_sentence_embedding_dimension.return_value = 384

            await model.initialize()

            with pytest.raises(ValueError, match="Cannot embed empty list"):
                await model.embed([])

    @pytest.mark.asyncio
    async def test_embed_handles_encoding_failure(self):
        """Should raise RuntimeError if encoding fails."""
        model = EmbeddingModel(model_name="test-model", device="cpu")

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_instance = mock_st.return_value
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.side_effect = Exception("CUDA out of memory")

            await model.initialize()

            with pytest.raises(RuntimeError, match="Embedding generation failed"):
                await model.embed(["text"])


class TestEmbeddingModelShutdown:
    """Test model cleanup and resource management."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_model(self):
        """Should clear model reference and mark as uninitialized."""
        model = EmbeddingModel(model_name="test-model", device="cpu")

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_st.return_value.get_sentence_embedding_dimension.return_value = 384

            await model.initialize()
            assert model._initialized is True
            assert model._model is not None

            await model.shutdown()
            assert model._initialized is False
            assert model._model is None

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self):
        """Should safely handle multiple shutdown() calls."""
        model = EmbeddingModel(model_name="test-model", device="cpu")

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_st.return_value.get_sentence_embedding_dimension.return_value = 384

            await model.initialize()
            await model.shutdown()
            await model.shutdown()  # Should not raise

            assert model._initialized is False


class TestEmbeddingModelContextManager:
    """Test async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager_initializes_and_cleans_up(self):
        """Should initialize on entry and shutdown on exit."""
        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_instance = mock_st.return_value
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.return_value = np.random.rand(1, 384)

            async with EmbeddingModel("test-model", device="cpu") as model:
                # Inside context: model should be initialized
                assert model._initialized is True

                # Should be able to generate embeddings
                embeddings = await model.embed(["test"])
                assert embeddings.shape == (1, 384)

            # Outside context: model should be shut down
            assert model._initialized is False
            assert model._model is None

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up_on_exception(self):
        """Should cleanup even if exception occurs inside context."""
        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_instance = mock_st.return_value
            mock_instance.get_sentence_embedding_dimension.return_value = 384

            model = None
            try:
                async with EmbeddingModel("test-model", device="cpu") as m:
                    model = m
                    raise ValueError("Test exception")
            except ValueError:
                pass  # Expected

            # Model should still be shut down
            assert model._initialized is False


class TestEmbeddingModelDeviceSelection:
    """Test CPU vs CUDA device selection."""

    @pytest.mark.asyncio
    async def test_cpu_device_selection(self):
        """Should initialize model on CPU when specified."""
        model = EmbeddingModel(model_name="test-model", device="cpu")

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_st.return_value.get_sentence_embedding_dimension.return_value = 384

            await model.initialize()

            # Verify CPU was specified
            mock_st.assert_called_once_with("test-model", device="cpu")

    @pytest.mark.asyncio
    async def test_cuda_device_selection(self):
        """Should initialize model on CUDA when specified."""
        model = EmbeddingModel(model_name="test-model", device="cuda")

        with patch("dragonwizard.rag.embeddings.SentenceTransformer") as mock_st:
            mock_st.return_value.get_sentence_embedding_dimension.return_value = 384

            await model.initialize()

            # Verify CUDA was specified
            mock_st.assert_called_once_with("test-model", device="cuda")
