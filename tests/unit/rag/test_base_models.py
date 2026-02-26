"""
Unit tests for RAG base data models (DocumentMetadata, ChunkMetadata).

These tests focus on model field declarations, default values, and
the ChromaDB serialisation logic — behaviour that sits below the pipeline
and chunker but is relied on by everything above them.

Edition metadata tests are the primary focus here: verifying that edition
is a proper first-class field (not just an 'extra' field via ConfigDict)
and that it serialises / omits correctly in the ChromaDB metadata dict.
"""

import pytest

from dragonwizard.rag.base import ChunkMetadata, DocumentMetadata


def _minimal_chunk_metadata(**overrides) -> ChunkMetadata:
    """Helper to build a valid ChunkMetadata with sensible defaults."""
    defaults = dict(
        chunk_id="chunk-abc",
        document_id="doc-xyz",
        source_file="phb.pdf",
        source_type="pdf",
        title="Player's Handbook",
        chunk_index=0,
        total_chunks=10,
        token_count=50,
    )
    defaults.update(overrides)
    return ChunkMetadata(**defaults)


class TestDocumentMetadataEdition:
    """DocumentMetadata edition field declaration and defaults."""

    def test_edition_field_accepts_known_value(self):
        """Should accept a known edition string like '5e'."""
        meta = DocumentMetadata(
            source_file="phb.pdf",
            source_type="pdf",
            title="Player's Handbook",
            edition="5e",
        )
        assert meta.edition == "5e"

    def test_edition_field_accepts_5_5e(self):
        """Should accept '5.5e' for 2024 rules."""
        meta = DocumentMetadata(
            source_file="phb2024.pdf",
            source_type="pdf",
            title="Player's Handbook (2024)",
            edition="5.5e",
        )
        assert meta.edition == "5.5e"

    def test_edition_defaults_to_none(self):
        """Should default to None when not provided, covering non-edition-organised files."""
        meta = DocumentMetadata(
            source_file="homebrew.txt",
            source_type="txt",
            title="Homebrew Rules",
        )
        assert meta.edition is None


class TestChunkMetadataEdition:
    """ChunkMetadata edition field and ChromaDB serialisation.

    ChromaDB requires metadata values to be non-None (None crashes on insertion).
    The to_chromadb_dict() method filters None values, so edition=None should
    be absent from the serialised dict entirely.
    """

    def test_chunk_metadata_edition_included_in_chromadb_dict_when_set(self):
        """Edition value should appear in to_chromadb_dict() when set.

        This is what gets stored in ChromaDB, enabling filters={"edition": "5e"}.
        """
        meta = _minimal_chunk_metadata(edition="5e")
        chroma_dict = meta.to_chromadb_dict()

        assert "edition" in chroma_dict
        assert chroma_dict["edition"] == "5e"

    def test_chunk_metadata_edition_excluded_from_chromadb_dict_when_none(self):
        """Edition should be absent from the ChromaDB dict when None.

        ChromaDB does not support None values in metadata. The existing
        to_chromadb_dict() None-filtering covers this — we just verify
        edition participates in that filtering correctly.
        """
        meta = _minimal_chunk_metadata(edition=None)
        chroma_dict = meta.to_chromadb_dict()

        assert "edition" not in chroma_dict

    def test_chunk_metadata_edition_5_5e_in_chromadb_dict(self):
        """5.5e edition string should round-trip through ChromaDB serialisation."""
        meta = _minimal_chunk_metadata(edition="5.5e")
        chroma_dict = meta.to_chromadb_dict()

        assert chroma_dict["edition"] == "5.5e"

    def test_chunk_metadata_edition_defaults_to_none(self):
        """Edition should default to None (and be absent from chromadb dict)."""
        meta = _minimal_chunk_metadata()  # no edition kwarg
        assert meta.edition is None
        assert "edition" not in meta.to_chromadb_dict()
