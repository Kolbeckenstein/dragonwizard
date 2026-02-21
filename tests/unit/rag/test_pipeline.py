"""
Unit tests for IngestionPipeline.

These tests encode the behavioral specification for the pipeline:
- Single file ingestion with document ID handling
- Directory ingestion with error resilience
- Duplicate detection and skipping
- Progress callback support
- Proper integration with components

Tests use mocks for embedding model and vector store to isolate pipeline logic.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import json

import pytest
import numpy as np

from dragonwizard.config.settings import RAGSettings
from dragonwizard.rag.pipeline import IngestionPipeline


class TestPipelineInitialization:
    """Test pipeline initialization."""

    def test_initialize_with_components(self):
        """Should initialize with provided components."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        vector_store = MagicMock()

        pipeline = IngestionPipeline(
            settings=settings,
            embedding_model=embedding_model,
            vector_store=vector_store
        )

        assert pipeline.settings == settings
        assert pipeline.embedding_model == embedding_model
        assert pipeline.vector_store == vector_store
        assert pipeline._chunker is not None
        assert len(pipeline._loaders) == 4  # .txt, .md, .markdown, .pdf


class TestSingleFileIngestion:
    """Test single file ingestion behavior."""

    @pytest.mark.asyncio
    async def test_ingest_file_generates_document_id_by_default(self, tmp_path):
        """Should generate UUID for document_id if not provided."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(1, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test sentence.")

        count = await pipeline.ingest_file(test_file)

        assert count > 0
        # Verify document_id was used (we can't assert the exact UUID, but we can check it was passed)
        vector_store.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_file_uses_custom_document_id(self, tmp_path):
        """Should use provided document_id instead of generating one."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(1, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test sentence.")

        custom_id = "custom-doc-123"
        count = await pipeline.ingest_file(test_file, document_id=custom_id)

        assert count > 0
        # Verify the custom ID was used in metadata
        call_args = vector_store.add.call_args
        metadatas = call_args.kwargs['metadatas']
        assert all(meta['document_id'] == custom_id for meta in metadatas)

    @pytest.mark.asyncio
    async def test_ingest_file_returns_chunk_count(self, tmp_path):
        """Should return the number of chunks created."""
        settings = RAGSettings(chunk_size=20)  # Small chunks to force multiple
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(3, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Sentence one. Sentence two. Sentence three.")

        count = await pipeline.ingest_file(test_file)

        # Should have multiple chunks due to small chunk_size
        assert count >= 1
        # Verify count matches what was inserted
        call_args = vector_store.add.call_args
        assert len(call_args.kwargs['ids']) == count

    @pytest.mark.asyncio
    async def test_ingest_file_calls_progress_callback(self, tmp_path):
        """Should call progress callback if provided."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(1, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content.")

        progress_callback = MagicMock()
        await pipeline.ingest_file(test_file, progress_callback=progress_callback)

        # Should have been called at least once (e.g., for completion)
        assert progress_callback.call_count > 0


class TestSingleFileErrorHandling:
    """Test error handling for single file ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_file_raises_for_missing_file(self):
        """Should raise FileNotFoundError for missing files."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        vector_store = MagicMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        with pytest.raises(FileNotFoundError):
            await pipeline.ingest_file(Path("/nonexistent/file.txt"))

    @pytest.mark.asyncio
    async def test_ingest_file_raises_for_directory(self, tmp_path):
        """Should raise ValueError when path is a directory."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        vector_store = MagicMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        with pytest.raises(ValueError, match="not a file"):
            await pipeline.ingest_file(tmp_path)

    @pytest.mark.asyncio
    async def test_ingest_file_raises_for_unsupported_format(self, tmp_path):
        """Should raise ValueError for unsupported file formats."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        vector_store = MagicMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        test_file = tmp_path / "document.docx"
        test_file.write_text("content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            await pipeline.ingest_file(test_file)


class TestDirectoryIngestion:
    """Test directory ingestion behavior."""

    @pytest.mark.asyncio
    async def test_ingest_directory_processes_all_files(self, tmp_path):
        """Should process all supported files in directory."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(1, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        # Create test files
        (tmp_path / "file1.txt").write_text("Content one.")
        (tmp_path / "file2.md").write_text("# Content two")
        (tmp_path / "ignored.docx").write_text("Should be ignored")

        results = await pipeline.ingest_directory(tmp_path)

        # Should process 2 files (.txt and .md, but not .docx)
        assert len(results) == 2
        assert any("file1.txt" in path for path in results.keys())
        assert any("file2.md" in path for path in results.keys())
        assert all(count > 0 for count in results.values())

    @pytest.mark.asyncio
    async def test_ingest_directory_recursive(self, tmp_path):
        """Should recursively process subdirectories when recursive=True."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(1, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        # Create nested structure
        (tmp_path / "file1.txt").write_text("Root file.")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("Nested file.")

        results = await pipeline.ingest_directory(tmp_path, recursive=True)

        # Should process both files
        assert len(results) == 2
        assert any("file1.txt" in path for path in results.keys())
        assert any("file2.txt" in path for path in results.keys())

    @pytest.mark.asyncio
    async def test_ingest_directory_non_recursive(self, tmp_path):
        """Should not process subdirectories when recursive=False."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(1, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        # Create nested structure
        (tmp_path / "file1.txt").write_text("Root file.")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("Nested file.")

        results = await pipeline.ingest_directory(tmp_path, recursive=False)

        # Should only process root file
        assert len(results) == 1
        assert any("file1.txt" in path for path in results.keys())

    @pytest.mark.asyncio
    async def test_ingest_directory_continues_on_file_error(self, tmp_path, caplog):
        """Should continue processing remaining files if one fails."""
        import logging
        caplog.set_level(logging.ERROR)

        settings = RAGSettings()
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(1, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        # Create files: one good, one that will fail
        (tmp_path / "good.txt").write_text("Good content.")
        (tmp_path / "bad.txt").write_text("")  # Empty file will raise ValueError

        results = await pipeline.ingest_directory(tmp_path)

        # Should process the good file
        assert len(results) == 1
        assert any("good.txt" in path for path in results.keys())
        # Should log the error
        assert "Failed to ingest" in caplog.text

    @pytest.mark.asyncio
    async def test_ingest_directory_calls_progress_callback(self, tmp_path):
        """Should call progress callback during directory ingestion."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(1, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        (tmp_path / "file1.txt").write_text("Content one.")
        (tmp_path / "file2.txt").write_text("Content two.")

        progress_callback = MagicMock()
        await pipeline.ingest_directory(tmp_path, progress_callback=progress_callback)

        # Should be called multiple times (per file or per operation)
        assert progress_callback.call_count > 0

    @pytest.mark.asyncio
    async def test_ingest_directory_returns_empty_dict_for_empty_dir(self, tmp_path):
        """Should return empty dict for directory with no supported files."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        vector_store = MagicMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        results = await pipeline.ingest_directory(tmp_path)

        assert results == {}


class TestDirectoryErrorHandling:
    """Test error handling for directory ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_directory_raises_for_missing_directory(self):
        """Should raise FileNotFoundError for missing directories."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        vector_store = MagicMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        with pytest.raises(FileNotFoundError):
            await pipeline.ingest_directory(Path("/nonexistent/dir/"))

    @pytest.mark.asyncio
    async def test_ingest_directory_raises_for_file_path(self, tmp_path):
        """Should raise ValueError when path is a file, not directory."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        vector_store = MagicMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        with pytest.raises(ValueError, match="not a directory"):
            await pipeline.ingest_directory(test_file)


class TestDuplicateDetection:
    """Test duplicate detection and skipping."""

    @pytest.mark.asyncio
    async def test_skip_already_processed_file(self, tmp_path):
        """Should skip files that have already been processed."""
        settings = RAGSettings(processed_data_path=str(tmp_path / "processed"))
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(1, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content.")

        # First ingestion
        count1 = await pipeline.ingest_file(test_file)
        assert count1 > 0

        # Second ingestion (should skip)
        count2 = await pipeline.ingest_file(test_file)
        assert count2 == 0  # Skipped

    @pytest.mark.asyncio
    async def test_reprocess_file_with_force_flag(self, tmp_path):
        """Should re-process files when force=True."""
        settings = RAGSettings(processed_data_path=str(tmp_path / "processed"))
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(1, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content.")

        # First ingestion
        count1 = await pipeline.ingest_file(test_file)
        assert count1 > 0

        # Second ingestion with force=True
        count2 = await pipeline.ingest_file(test_file, force=True)
        assert count2 > 0  # Re-processed

    @pytest.mark.asyncio
    async def test_compute_file_hash_correctly(self, tmp_path):
        """Should compute SHA-256 hash of file content."""
        settings = RAGSettings(processed_data_path=str(tmp_path / "processed"))
        embedding_model = MagicMock()
        vector_store = MagicMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content for hashing.")

        file_hash = pipeline._compute_file_hash(test_file)

        # Should be a valid SHA-256 hash (64 hex characters)
        assert len(file_hash) == 64
        assert all(c in "0123456789abcdef" for c in file_hash)

    @pytest.mark.asyncio
    async def test_persist_metadata_to_json(self, tmp_path):
        """Should persist processing metadata to metadata.json."""
        processed_dir = tmp_path / "processed"
        settings = RAGSettings(processed_data_path=str(processed_dir))
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(1, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content.")

        await pipeline.ingest_file(test_file)

        # Verify metadata.json was created
        metadata_file = processed_dir / "metadata.json"
        assert metadata_file.exists()

        # Verify it contains the file hash
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert str(test_file) in metadata
        assert "hash" in metadata[str(test_file)]
        assert "timestamp" in metadata[str(test_file)]


class TestComponentIntegration:
    """Test integration with embedding model and vector store."""

    @pytest.mark.asyncio
    async def test_uses_settings_for_chunking(self, tmp_path):
        """Should use settings.chunk_size and settings.chunk_overlap."""
        settings = RAGSettings(chunk_size=50, chunk_overlap=10)
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(2, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        # Verify chunker was initialized with correct settings
        assert pipeline._chunker.target_tokens == 50
        assert pipeline._chunker.overlap_tokens == 10

    @pytest.mark.asyncio
    async def test_passes_metadata_through_pipeline(self, tmp_path):
        """Should preserve metadata (source_file, title) through pipeline."""
        settings = RAGSettings()
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(1, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()

        pipeline = IngestionPipeline(settings, embedding_model, vector_store)

        test_file = tmp_path / "test_doc.txt"
        test_file.write_text("Content.")

        await pipeline.ingest_file(test_file)

        # Verify metadata was passed to vector store
        call_args = vector_store.add.call_args
        metadatas = call_args.kwargs['metadatas']

        assert all(str(test_file) in meta['source_file'] for meta in metadatas)
        assert all(meta['source_type'] == 'txt' for meta in metadatas)
        assert all(meta['title'] == 'test_doc' for meta in metadatas)
