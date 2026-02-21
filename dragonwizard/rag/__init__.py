"""
RAG (Retrieval-Augmented Generation) Engine Layer.

Provides document processing, vector store management, semantic search,
and context retrieval for D&D 5e rules lookups.
"""

# Public API exports
from dragonwizard.rag.base import Chunk, Document, SearchResult
from dragonwizard.rag.components import RAGComponents
from dragonwizard.rag.embeddings import EmbeddingModel
from dragonwizard.rag.engine import RAGEngine
from dragonwizard.rag.pipeline import IngestionPipeline
from dragonwizard.rag.vector_store import ChromaVectorStore

__all__ = [
    "Chunk",
    "Document",
    "SearchResult",
    "RAGComponents",
    "EmbeddingModel",
    "RAGEngine",
    "IngestionPipeline",
    "ChromaVectorStore",
]
