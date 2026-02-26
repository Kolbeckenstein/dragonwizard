"""
RAG component factory.

Centralises the construction of RAG components from settings,
eliminating duplicated wiring across CLI commands, tests, and
future entry points (Discord bot, etc.).
"""

from __future__ import annotations

from dragonwizard.config.settings import RAGSettings
from dragonwizard.rag.base import ChunkEnricher
from dragonwizard.rag.embeddings import EmbeddingModel
from dragonwizard.rag.engine import RAGEngine
from dragonwizard.rag.sources.pdf.loader import ExtractionMode
from dragonwizard.rag.pipeline import IngestionPipeline
from dragonwizard.rag.vector_store import ChromaVectorStore


class RAGComponents:
    """
    Factory for building RAG components from settings.

    Eliminates the repeated pattern of manually extracting config
    fields and passing them to constructors.

    Example::

        factory = RAGComponents(settings.rag)
        async with factory.create_embedding_model() as model, \\
                   factory.create_vector_store() as store:
            engine = factory.create_engine(model, store)
            results = await engine.search("fireball")
    """

    def __init__(self, settings: RAGSettings):
        self.settings = settings

    def create_embedding_model(self) -> EmbeddingModel:
        """Create an EmbeddingModel from settings."""
        return EmbeddingModel(
            model_name=self.settings.embedding_model,
            device=self.settings.embedding_device,
            batch_size=self.settings.embedding_batch_size,
        )

    def create_vector_store(self) -> ChromaVectorStore:
        """Create a ChromaVectorStore from settings."""
        return ChromaVectorStore(
            persist_directory=self.settings.vector_db_path,
            collection_name=self.settings.collection_name,
        )

    def create_pipeline(
        self,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore,
        enrichers: list[ChunkEnricher] | None = None,
        extraction_mode: ExtractionMode = ExtractionMode.DEFAULT,
    ) -> IngestionPipeline:
        """Create an IngestionPipeline from settings + initialized dependencies."""
        return IngestionPipeline(
            settings=self.settings,
            embedding_model=embedding_model,
            vector_store=vector_store,
            enrichers=enrichers,
            extraction_mode=extraction_mode,
        )

    def create_engine(
        self,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore,
    ) -> RAGEngine:
        """Create a RAGEngine from settings + initialized dependencies."""
        return RAGEngine(
            embedding_model=embedding_model,
            vector_store=vector_store,
            default_k=self.settings.default_k,
            score_threshold=self.settings.score_threshold,
        )
