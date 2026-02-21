"""
Document loaders for the RAG system.

This package provides loaders for different file formats:
- TextLoader: Plain text files (.txt)
- MarkdownLoader: Markdown files (.md)
- PDFLoader: PDF files (.pdf)

Each loader implements the DocumentLoader interface and returns Document objects.
"""

from dragonwizard.rag.loaders.text_loader import TextLoader
from dragonwizard.rag.loaders.markdown_loader import MarkdownLoader
from dragonwizard.rag.loaders.pdf_loader import PDFLoader

__all__ = ["TextLoader", "MarkdownLoader", "PDFLoader"]
