"""
Base classes and data structures for the RAG system.

This module defines the core data structures used throughout the RAG pipeline:
- DocumentMetadata: Metadata for loaded documents
- ChunkMetadata: Metadata for text chunks
- Document: Represents a loaded source document
- Chunk: Represents a piece of a document with embeddings
- SearchResult: Represents a retrieved chunk with similarity score
- ChunkEnricher: Abstract base class for chunk post-processors
- NoOpEnricher: Identity enricher (null-object implementation of ChunkEnricher)
- DocumentLoader: Abstract base class for loading different file formats
"""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """
    Metadata for a loaded document.

    This is extracted by DocumentLoaders when reading files.
    """

    source_file: str = Field(description="Path to the original source file")
    source_type: str = Field(description="File type: pdf, txt, md")
    title: str = Field(description="Document title (from metadata or filename)")
    author: str | None = Field(None, description="Document author (if available)")
    page_count: int | None = Field(None, ge=1, description="Number of pages (PDFs only)")
    edition: str | None = Field(
        None,
        description='D&D edition, e.g. "5e" (2014 rules) or "5.5e" (2024 rules). '
                    'Inferred from source directory at ingestion time.',
    )

    model_config = ConfigDict(extra="allow")  # Allow extra fields for future extensibility


class ChunkMetadata(BaseModel):
    """
    Metadata for a text chunk.

    This is generated during the chunking process and stored in the vector database.
    ChromaDB will receive this as a plain dict (via .model_dump()).

    Note: chunk_id is stored separately in ChromaDB's 'ids' parameter,
    not in the metadata dict.
    """

    chunk_id: str = Field(description="Unique chunk identifier (UUID)")
    document_id: str = Field(description="Parent document UUID")
    source_file: str = Field(description="Path to original source file")
    source_type: str = Field(description="File type: pdf, txt, md")
    title: str = Field(description="Document title")
    chunk_index: int = Field(ge=0, description="Index of this chunk within the document")
    total_chunks: int = Field(gt=0, description="Total number of chunks in the document")
    token_count: int = Field(gt=0, description="Number of tokens in this chunk")

    # Optional fields (not present for all chunks)
    page_number: int | None = Field(None, ge=1, description="PDF page number (if applicable)")
    section: str | None = Field(None, description="Section heading (if detected)")
    char_start: int | None = Field(None, ge=0, description="Character offset in source document")
    char_end: int | None = Field(None, gt=0, description="Character end offset in source document")
    edition: str | None = Field(
        None,
        description='D&D edition tag, e.g. "5e" or "5.5e". Stored in ChromaDB and used for '
                    'metadata-level filtering: engine.search(filters={"edition": "5e"}). '
                    'None for documents not organised into an edition directory.',
    )

    # Tracking fields
    ingestion_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this chunk was processed"
    )
    pipeline_version: str = Field(default="0.1.0", description="Version of ingestion pipeline")

    model_config = ConfigDict(extra="allow")  # Allow extra fields for future metadata

    def to_chromadb_dict(self) -> dict[str, Any]:
        """
        Convert to ChromaDB metadata format.

        ChromaDB stores chunk_id separately in the 'ids' parameter,
        so we exclude it from the metadata dict. Also filters out None
        values as ChromaDB doesn't support them.

        Returns:
            Dictionary suitable for ChromaDB's metadatas parameter
        """
        metadata = self.model_dump(mode="json", exclude={"chunk_id"})
        # Filter out None values - ChromaDB doesn't support them
        return {k: v for k, v in metadata.items() if v is not None}


class Document(BaseModel):
    """
    Represents a loaded document before chunking.

    This is the output of DocumentLoaders (PDF, text, markdown loaders).
    It contains the full text plus metadata about the source.

    Example:
        >>> doc = Document(
        ...     text="Fireball is a 3rd-level evocation spell...",
        ...     metadata=DocumentMetadata(
        ...         source_file="srd.pdf",
        ...         source_type="pdf",
        ...         title="D&D 5e SRD",
        ...         page_count=241
        ...     )
        ... )
    """

    text: str = Field(min_length=1, description="Full extracted text content")
    metadata: DocumentMetadata = Field(description="Document metadata")
    pages: list[dict[str, Any]] | None = Field(
        None,
        description="Page-level data for PDFs: [{'page_num': 1, 'text': '...'}, ...]"
    )


class Chunk(BaseModel):
    """
    Represents a chunk of text with optional embedding.

    Chunks are created by the chunking strategy (sentence-aware, token-limited).
    Each chunk becomes a separate entry in the vector database.

    Example:
        >>> chunk = Chunk(
        ...     text="Fireball deals 8d6 fire damage...",
        ...     metadata=ChunkMetadata(
        ...         chunk_id="abc-123",
        ...         document_id="doc-456",
        ...         source_file="srd.pdf",
        ...         source_type="pdf",
        ...         title="D&D 5e SRD",
        ...         chunk_index=0,
        ...         total_chunks=10,
        ...         token_count=45,
        ...         page_number=241
        ...     ),
        ...     embedding=[0.1, 0.2, 0.3, ...]  # 384-dim vector
        ... )
    """

    text: str = Field(min_length=1, description="Chunk text content")
    metadata: ChunkMetadata = Field(description="Chunk metadata")
    embedding: list[float] | None = Field(None, description="Vector embedding (if generated)")

    def to_chromadb_format(self) -> tuple[str, str, dict[str, Any], list[float] | None]:
        """
        Convert chunk to ChromaDB format.

        ChromaDB's add() method expects separate lists for:
        - ids: List of unique identifiers
        - documents: List of text content
        - metadatas: List of metadata dictionaries
        - embeddings: List of embedding vectors

        Returns:
            Tuple of (id, document, metadata_dict, embedding) that can be
            unpacked into lists for batch insertion into ChromaDB.

        Example:
            >>> chunks = [chunk1, chunk2, chunk3]
            >>> ids, docs, metas, embeds = [], [], [], []
            >>> for chunk in chunks:
            ...     id, doc, meta, embed = chunk.to_chromadb_format()
            ...     ids.append(id)
            ...     docs.append(doc)
            ...     metas.append(meta)
            ...     embeds.append(embed)
            >>> collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
        """
        return (
            self.metadata.chunk_id,
            self.text,
            self.metadata.to_chromadb_dict(),
            self.embedding
        )


class SearchResult(BaseModel):
    """
    Represents a search result from the RAG engine.

    This is what the RAG engine returns when you query for information.
    It contains the relevant text chunk plus metadata for citations.

    Example:
        >>> result = SearchResult(
        ...     text="Fireball deals 8d6 fire damage...",
        ...     score=0.92,  # High similarity
        ...     metadata=ChunkMetadata(...),
        ...     citation="[D&D 5e SRD, p.241]"
        ... )
    """

    text: str = Field(description="Chunk text content")
    score: float = Field(ge=0.0, le=1.0, description="Similarity score (0.0 to 1.0)")
    metadata: ChunkMetadata = Field(description="Chunk metadata")
    citation: str = Field(description="Formatted citation string for LLM prompts")


class ChunkEnricher(ABC):
    """
    Abstract base class for chunk enrichers.

    Chunk enrichers post-process chunks after the chunking step and before
    embedding. Each enricher receives the full list of chunks plus the source
    document, and returns a (possibly modified) list of chunks.

    Enrichers must NOT mutate input chunks — use chunk.model_copy(update={...})
    to produce new Chunk instances (Pydantic v2 pattern).

    Async is required because some enrichers (e.g. LLMHeadingEnricher) call
    an LLM API inside enrich().

    Example:
        >>> class NoOpEnricher(ChunkEnricher):
        ...     async def enrich(self, chunks, document):
        ...         return chunks  # identity

    Intended enrichers:
        - NoOpEnricher: identity, for baseline comparisons
        - StatisticalHeadingEnricher: font-size heuristic heading injection
        - LLMHeadingEnricher: statistical + LLM confirmation for ambiguous cases
        - WeightedHeadingEnricher: like statistical, but adds confidence score
    """

    @abstractmethod
    async def enrich(self, chunks: list["Chunk"], document: "Document") -> list["Chunk"]:
        """
        Enrich a list of chunks using information from the source document.

        Args:
            chunks: List of Chunk objects produced by the chunker
            document: The source Document (text + metadata)

        Returns:
            A new list of Chunk objects (do NOT mutate the input list or chunks)
        """
        pass


class NoOpEnricher(ChunkEnricher):
    """
    Identity enricher that returns chunks unchanged.

    Used as:
    - Default enricher (no annotation overhead)
    - Baseline for comparison runs (--enricher none)

    Example:
        >>> enricher = NoOpEnricher()
        >>> result = await enricher.enrich(chunks, document)
        >>> result is chunks
        True
    """

    async def enrich(self, chunks: list["Chunk"], document: "Document") -> list["Chunk"]:
        """Return the chunk list unchanged."""
        return chunks


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.

    Document loaders are responsible for:
    1. Reading files from disk (PDF, text, markdown, etc.)
    2. Extracting text content
    3. Extracting metadata (title, author, page numbers, etc.)
    4. Returning a Document object

    Each file format has its own loader implementation:
    - PDFLoader: Uses PyMuPDF to extract text and page numbers
    - TextLoader: Reads UTF-8 plain text files
    - MarkdownLoader: Parses markdown and extracts headings

    The IngestionPipeline selects the appropriate loader based on file extension.

    Example:
        >>> loader = PDFLoader()
        >>> if loader.supports_format(Path("rules.pdf")):
        ...     doc = await loader.load(Path("rules.pdf"))
        ...     print(doc.metadata.title)
    """

    @abstractmethod
    async def load(self, file_path: Path) -> Document:
        """
        Load a document from a file path.

        Args:
            file_path: Path to the document file

        Returns:
            Document object with extracted text and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid or unsupported
            IOError: If file cannot be read
        """
        pass

    @abstractmethod
    def supports_format(self, file_path: Path) -> bool:
        """
        Check if this loader supports the given file format.

        Args:
            file_path: Path to check

        Returns:
            True if this loader can handle the file, False otherwise

        Example:
            >>> loader = PDFLoader()
            >>> loader.supports_format(Path("rules.pdf"))
            True
            >>> loader.supports_format(Path("rules.txt"))
            False
        """
        pass
