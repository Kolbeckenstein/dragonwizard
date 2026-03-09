"""
Unit tests for IngestionPipeline.ingest_url() and _ingest_document().

These tests cover the new web-ingestion path added alongside the existing
file-based ingest_file() / ingest_directory() methods.

_ingest_document() is a private helper extracted from ingest_file() to avoid
duplicating the chunk→enrich→embed→store logic.

ingest_url() uses _ingest_document() + WebPageLoader + URL-based deduplication.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import json

import numpy as np
import pytest

from dragonwizard.config.settings import RAGSettings
from dragonwizard.rag.base import Document, DocumentMetadata
from dragonwizard.rag.pipeline import IngestionPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_URL = "https://dnd5e.wikidot.com/spell:fireball"
SAMPLE_HTML = """
<html><body>
  <div id="page-content">
    <h1 id="page-title">Fireball</h1>
    <p>3rd-level evocation. A bright streak flashes from your pointing finger
    to a point you choose within range, and then blossoms with a low roar into
    an explosion of flame. Each creature in a 20-foot-radius sphere centered on
    that point must make a Dexterity saving throw. A target takes 8d6 fire damage
    on a failed save, or half as much damage on a successful one.</p>
  </div>
</body></html>
"""


def _make_pipeline(tmp_path):
    settings = RAGSettings(
        vector_db_path=str(tmp_path / "chroma"),
        processed_data_path=str(tmp_path / "processed"),
    )
    embedding_model = MagicMock()
    embedding_model.embed = AsyncMock(return_value=np.random.rand(3, 384))
    vector_store = MagicMock()
    vector_store.add = AsyncMock()
    return IngestionPipeline(settings, embedding_model, vector_store)


def _make_session(html: str = SAMPLE_HTML, status: int = 200):
    response = AsyncMock()
    response.text = AsyncMock(return_value=html)
    if status >= 400:
        from aiohttp import ClientResponseError
        response.raise_for_status = MagicMock(
            side_effect=ClientResponseError(request_info=MagicMock(), history=())
        )
    else:
        response.raise_for_status = MagicMock()

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(return_value=ctx)
    return session


# ---------------------------------------------------------------------------
# _ingest_document tests
# ---------------------------------------------------------------------------

class TestIngestDocument:
    """Tests for the private _ingest_document() helper."""

    @pytest.mark.asyncio
    async def test_returns_chunk_count(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        # Ensure embed returns enough rows for the chunks that will be created
        pipeline.embedding_model.embed = AsyncMock(return_value=np.random.rand(5, 384))
        doc = Document(
            text="Fireball is a 3rd-level evocation spell. " * 10,
            metadata=DocumentMetadata(
                source_file=SAMPLE_URL,
                source_type="web",
                title="Fireball",
            ),
        )
        count = await pipeline._ingest_document(doc, "test-doc-id")
        assert count > 0

    @pytest.mark.asyncio
    async def test_calls_vector_store_add(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        doc = Document(
            text="Some D&D rules content.",
            metadata=DocumentMetadata(
                source_file=SAMPLE_URL,
                source_type="web",
                title="Test Page",
            ),
        )
        await pipeline._ingest_document(doc, "doc-123")
        pipeline.vector_store.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_applies_enrichers(self, tmp_path):
        enricher = MagicMock()
        enricher.enrich = AsyncMock(side_effect=lambda chunks, doc: chunks)
        settings = RAGSettings(
            vector_db_path=str(tmp_path / "chroma"),
            processed_data_path=str(tmp_path / "processed"),
        )
        embedding_model = MagicMock()
        embedding_model.embed = AsyncMock(return_value=np.random.rand(5, 384))
        vector_store = MagicMock()
        vector_store.add = AsyncMock()
        pipeline = IngestionPipeline(settings, embedding_model, vector_store, enrichers=[enricher])

        doc = Document(
            text="Content for enricher test. " * 5,
            metadata=DocumentMetadata(source_file=SAMPLE_URL, source_type="web", title="Test"),
        )
        await pipeline._ingest_document(doc, "doc-id")
        enricher.enrich.assert_called_once()


# ---------------------------------------------------------------------------
# ingest_url tests
# ---------------------------------------------------------------------------

class TestIngestUrl:
    @pytest.mark.asyncio
    async def test_ingest_url_returns_chunk_count(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline.embedding_model.embed = AsyncMock(return_value=np.random.rand(5, 384))
        session = _make_session()
        count = await pipeline.ingest_url(SAMPLE_URL, session)
        assert count > 0

    @pytest.mark.asyncio
    async def test_ingest_url_records_url_in_metadata(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline.embedding_model.embed = AsyncMock(return_value=np.random.rand(5, 384))
        session = _make_session()
        await pipeline.ingest_url(SAMPLE_URL, session)

        # metadata.json should contain a url: key
        metadata_file = tmp_path / "processed" / "metadata.json"
        assert metadata_file.exists()
        with open(metadata_file) as f:
            metadata = json.load(f)
        assert f"url:{SAMPLE_URL}" in metadata

    @pytest.mark.asyncio
    async def test_ingest_url_skips_already_processed(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline.embedding_model.embed = AsyncMock(return_value=np.random.rand(5, 384))
        session = _make_session()

        # First ingest
        count1 = await pipeline.ingest_url(SAMPLE_URL, session)
        assert count1 > 0

        # Second ingest — should skip
        count2 = await pipeline.ingest_url(SAMPLE_URL, session)
        assert count2 == 0

    @pytest.mark.asyncio
    async def test_force_flag_re_ingests_url(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline.embedding_model.embed = AsyncMock(return_value=np.random.rand(5, 384))
        session = _make_session()

        await pipeline.ingest_url(SAMPLE_URL, session)
        count = await pipeline.ingest_url(SAMPLE_URL, session, force=True)
        assert count > 0

    @pytest.mark.asyncio
    async def test_ingest_url_sets_edition(self, tmp_path):
        """The edition parameter should be stored in chunk metadata."""
        pipeline = _make_pipeline(tmp_path)
        pipeline.embedding_model.embed = AsyncMock(return_value=np.random.rand(5, 384))
        session = _make_session()
        await pipeline.ingest_url(SAMPLE_URL, session, edition="5e")

        # Verify add() was called with metadata containing edition="5e"
        add_call = pipeline.vector_store.add.call_args
        metadatas = add_call[1]["metadatas"]
        assert all(m.get("edition") == "5e" for m in metadatas)

    @pytest.mark.asyncio
    async def test_ingest_url_source_file_is_url(self, tmp_path):
        """source_file in stored chunks should be the URL, not a path."""
        pipeline = _make_pipeline(tmp_path)
        pipeline.embedding_model.embed = AsyncMock(return_value=np.random.rand(5, 384))
        session = _make_session()
        await pipeline.ingest_url(SAMPLE_URL, session)

        add_call = pipeline.vector_store.add.call_args
        metadatas = add_call[1]["metadatas"]
        assert all(m.get("source_file") == SAMPLE_URL for m in metadatas)

    @pytest.mark.asyncio
    async def test_ingest_url_source_type_is_web(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline.embedding_model.embed = AsyncMock(return_value=np.random.rand(5, 384))
        session = _make_session()
        await pipeline.ingest_url(SAMPLE_URL, session)

        add_call = pipeline.vector_store.add.call_args
        metadatas = add_call[1]["metadatas"]
        assert all(m.get("source_type") == "web" for m in metadatas)
