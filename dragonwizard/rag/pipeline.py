"""
Document ingestion pipeline for the RAG system.

This module orchestrates the end-to-end process of ingesting documents:
1. Load documents (PDF, text, markdown)
2. Chunk text into smaller pieces
3. Generate embeddings for chunks
4. Store chunks in vector database
5. Track processed files to avoid duplicates

Example:
    >>> async with EmbeddingModel(...) as embedding_model, \\
    ...           ChromaVectorStore(...) as vector_store:
    ...     pipeline = IngestionPipeline(
    ...         settings=settings,
    ...         embedding_model=embedding_model,
    ...         vector_store=vector_store
    ...     )
    ...     count = await pipeline.ingest_file(Path("srd.pdf"))
    ...     print(f"Ingested {count} chunks")
"""

import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

from dragonwizard.config.logging import get_logger
from dragonwizard.config.settings import RAGSettings
from dragonwizard.rag.base import Chunk, ChunkEnricher, Document
from dragonwizard.rag.chunking import SentenceAwareChunker
from dragonwizard.rag.embeddings import EmbeddingModel
from dragonwizard.rag.sources.markdown.loader import MarkdownLoader
from dragonwizard.rag.sources.pdf.loader import ExtractionMode, PDFLoader
from dragonwizard.rag.sources.text.loader import TextLoader
from dragonwizard.rag.vector_store import ChromaVectorStore

logger = get_logger(__name__)

# Parent directory names that map to known D&D editions.
# Files in other directories (homebrew, supplements, web-scraped content)
# receive edition=None and are not filtered by edition-scoped queries.
_KNOWN_EDITIONS: frozenset[str] = frozenset({"5e", "5.5e"})


class IngestionPipeline:
    """
    Orchestrates document ingestion from file to vector store.

    The pipeline:
    1. Checks for duplicate files (unless force=True)
    2. Loads document using appropriate loader
    3. Chunks text into manageable pieces
    4. Generates embeddings for each chunk
    5. Inserts chunks into vector store
    6. Records file metadata to prevent re-processing

    Attributes:
        settings: RAG configuration settings
        embedding_model: Model for generating embeddings
        vector_store: Vector database for storing chunks
        _chunker: Text chunking strategy
        _loaders: Registered document loaders by file extension

    Example:
        >>> settings = RAGSettings()
        >>> async with EmbeddingModel(...) as em, ChromaVectorStore(...) as vs:
        ...     pipeline = IngestionPipeline(settings, em, vs)
        ...     await pipeline.ingest_directory(Path("data/raw/srd/"))
    """

    def __init__(
        self,
        settings: RAGSettings,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore,
        enrichers: list[ChunkEnricher] | None = None,
        extraction_mode: ExtractionMode = ExtractionMode.DEFAULT,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            settings: RAG configuration settings
            embedding_model: Initialized embedding model
            vector_store: Initialized vector store
            enrichers: Optional list of ChunkEnrichers applied after chunking
            extraction_mode: How to read text from PDF pages (default/column_aware)
        """
        self.settings = settings
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self._enrichers = enrichers or []

        # Initialize chunker with settings
        self._chunker = SentenceAwareChunker(
            target_tokens=settings.chunk_size,
            overlap_tokens=settings.chunk_overlap,
            encoding_name="cl100k_base"
        )

        # Register document loaders (PDFLoader respects extraction_mode)
        self._loaders = {
            ".txt": TextLoader(),
            ".md": MarkdownLoader(),
            ".markdown": MarkdownLoader(),
            ".pdf": PDFLoader(ocr_enabled=settings.ocr_enabled, extraction_mode=extraction_mode)
        }

        # Ensure processed data directory exists
        self._processed_dir = Path(settings.processed_data_path)
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self._processed_dir / "metadata.json"

        logger.debug("Initialized IngestionPipeline")

    async def ingest_file(
        self,
        file_path: Path,
        document_id: Optional[str] = None,
        force: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> int:
        """
        Ingest a single document file.

        Args:
            file_path: Path to the document file
            document_id: Optional custom document ID (default: generate UUID)
            force: If True, re-process even if already ingested
            progress_callback: Optional callback for progress updates

        Returns:
            Number of chunks created and inserted (0 if skipped)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is not a file or format unsupported
            RuntimeError: If ingestion fails

        Example:
            >>> count = await pipeline.ingest_file(Path("srd.pdf"))
            >>> print(f"Created {count} chunks")
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Check for duplicates (unless force=True)
        if not force and self._is_already_processed(file_path):
            logger.info(f"Skipping already-processed file: {file_path.name}")
            if progress_callback:
                progress_callback(f"Skipped (already processed): {file_path.name}")
            return 0

        logger.info(f"Starting ingestion of: {file_path}")
        if progress_callback:
            progress_callback(f"Loading: {file_path.name}")

        try:
            # Step 1: Load document
            document = await self._load_document(file_path)
            # Annotate with edition after loading; loaders don't know about
            # directory layout, so this is the pipeline's responsibility.
            document.metadata.edition = self._infer_edition(file_path)
            logger.info(
                f"Loaded document: {document.metadata.title} "
                f"({len(document.text)} characters, edition={document.metadata.edition})"
            )
            if progress_callback:
                progress_callback(f"Loaded: {document.metadata.title}")

            # Step 2: Chunk document
            doc_id = document_id or str(uuid.uuid4())
            chunks = self._chunk_document(document, doc_id)
            logger.info(f"Created {len(chunks)} chunks")
            if progress_callback:
                progress_callback(f"Chunked: {len(chunks)} pieces")

            # Step 2b: Apply enrichers (heading injection, etc.)
            for enricher in self._enrichers:
                chunks = await enricher.enrich(chunks, document)
            if self._enrichers:
                logger.info(f"Enriched to {len(chunks)} chunks")

            # Step 3: Generate embeddings in batches
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self.embedding_model.embed(chunk_texts)
            logger.info(f"Generated embeddings ({embeddings.shape})")
            if progress_callback:
                progress_callback(f"Embedded: {len(chunks)} chunks")

            # Attach embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()

            # Step 4: Insert into vector store
            await self._insert_chunks(chunks)
            logger.info(f"Inserted {len(chunks)} chunks into vector store")
            if progress_callback:
                progress_callback(f"Stored: {len(chunks)} chunks")

            # Step 5: Record metadata
            self._record_processed_file(file_path)

            logger.info(
                f"Successfully ingested: {file_path.name} "
                f"({len(chunks)} chunks)"
            )
            if progress_callback:
                progress_callback(f"Completed: {file_path.name}")

            return len(chunks)

        except (FileNotFoundError, ValueError):
            # Let validation and file errors bubble up naturally
            raise

        except Exception as e:
            # Wrap truly unexpected errors
            logger.error(f"Failed to ingest file {file_path}: {e}")
            raise RuntimeError(f"Ingestion failed for '{file_path}': {e}") from e

    async def ingest_directory(
        self,
        directory_path: Path,
        recursive: bool = False,
        force: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> dict[str, int]:
        """
        Ingest all supported documents in a directory.

        Args:
            directory_path: Path to directory
            recursive: If True, process subdirectories recursively
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping file paths to chunk counts

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If path is not a directory

        Example:
            >>> results = await pipeline.ingest_directory(
            ...     Path("data/raw/srd/"),
            ...     recursive=True
            ... )
            >>> print(f"Ingested {len(results)} files")
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        logger.info(
            f"Starting directory ingestion: {directory_path} "
            f"(recursive={recursive})"
        )
        if progress_callback:
            progress_callback(f"Scanning: {directory_path}")

        # Discover supported files
        files = self._discover_files(directory_path, recursive)
        logger.info(f"Found {len(files)} supported files")
        if progress_callback:
            progress_callback(f"Found {len(files)} files")

        if not files:
            logger.warning(f"No supported files found in: {directory_path}")
            if progress_callback:
                progress_callback("No supported files found")
            return {}

        # Ingest each file
        results = {}
        for i, file_path in enumerate(files, 1):
            try:
                if progress_callback:
                    progress_callback(f"Processing {i}/{len(files)}: {file_path.name}")

                chunk_count = await self.ingest_file(
                    file_path,
                    force=force,
                    progress_callback=progress_callback
                )
                results[str(file_path)] = chunk_count

            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                # Continue with remaining files
                if progress_callback:
                    progress_callback(f"Failed: {file_path.name}")

        logger.info(
            f"Directory ingestion complete: {len(results)}/{len(files)} files succeeded"
        )
        if progress_callback:
            progress_callback(f"Complete: {len(results)}/{len(files)} files")

        return results

    def _discover_files(
        self,
        directory_path: Path,
        recursive: bool
    ) -> list[Path]:
        """
        Discover all supported files in a directory.

        Args:
            directory_path: Directory to search
            recursive: Whether to search subdirectories

        Returns:
            List of file paths with supported extensions
        """
        files = []

        if recursive:
            # Recursive glob
            for ext in self._loaders.keys():
                files.extend(directory_path.rglob(f"*{ext}"))
        else:
            # Non-recursive glob
            for ext in self._loaders.keys():
                files.extend(directory_path.glob(f"*{ext}"))

        return sorted(files)

    async def _load_document(self, file_path: Path) -> Document:
        """
        Load a document using the appropriate loader.

        Args:
            file_path: Path to document file

        Returns:
            Loaded Document object

        Raises:
            ValueError: If file format is not supported
        """
        # Select loader based on file extension
        ext = file_path.suffix.lower()
        loader = self._loaders.get(ext)

        if loader is None:
            supported = ", ".join(self._loaders.keys())
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {supported}"
            )

        # Load document
        return await loader.load(file_path)

    def _chunk_document(
        self,
        document: Document,
        document_id: str
    ) -> list[Chunk]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: Loaded document
            document_id: UUID for the document

        Returns:
            List of Chunk objects
        """
        # Prepare base metadata for chunks
        base_metadata = {
            "source_file": document.metadata.source_file,
            "source_type": document.metadata.source_type,
            "title": document.metadata.title,
            "edition": document.metadata.edition,
            "page_number": None  # Will be overridden for PDFs
        }

        # Chunk the full document text
        chunks = self._chunker.chunk_text(
            text=document.text,
            document_id=document_id,
            metadata=base_metadata
        )

        return chunks

    async def _insert_chunks(self, chunks: list[Chunk]) -> None:
        """
        Insert chunks into the vector store.

        Args:
            chunks: List of Chunk objects with embeddings

        Raises:
            RuntimeError: If insertion fails
        """
        # Convert chunks to ChromaDB format
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            chunk_id, document_text, metadata_dict, embedding = chunk.to_chromadb_format()
            ids.append(chunk_id)
            documents.append(document_text)
            metadatas.append(metadata_dict)
            embeddings.append(embedding)

        # Bulk insert
        await self.vector_store.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def _infer_edition(self, file_path: Path) -> str | None:
        """
        Infer D&D edition from the file's parent directory name.

        Relies on the established directory convention:
            data/raw/pdf/5e/   → "5e"   (2014 rules)
            data/raw/pdf/5.5e/ → "5.5e" (2024 rules)

        Returns None for paths that don't match a known edition directory,
        leaving those documents unfiltered by edition-scoped queries
        (e.g. homebrew content, web-scraped SRD, forum Q&A).
        """
        parent = file_path.parent.name
        return parent if parent in _KNOWN_EDITIONS else None

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA-256 hash of file content.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal hash string (64 characters)
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def _is_already_processed(self, file_path: Path) -> bool:
        """
        Check if file has already been processed.

        Args:
            file_path: Path to file

        Returns:
            True if file was already processed, False otherwise
        """
        if not self._metadata_file.exists():
            return False

        try:
            with open(self._metadata_file) as f:
                metadata = json.load(f)

            file_key = str(file_path)
            if file_key not in metadata:
                return False

            # Check if hash matches (file unchanged)
            current_hash = self._compute_file_hash(file_path)
            stored_hash = metadata[file_key].get("hash")

            return current_hash == stored_hash

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error reading metadata file: {e}")
            return False

    def _record_processed_file(self, file_path: Path) -> None:
        """
        Record file as processed in metadata.json.

        Args:
            file_path: Path to processed file
        """
        # Load existing metadata
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                metadata = {}
        else:
            metadata = {}

        # Add/update file entry
        file_key = str(file_path)
        metadata[file_key] = {
            "hash": self._compute_file_hash(file_path),
            "timestamp": datetime.now(UTC).isoformat(),
            "pipeline_version": "0.1.0"
        }

        # Save metadata
        with open(self._metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Recorded processed file: {file_path.name}")
