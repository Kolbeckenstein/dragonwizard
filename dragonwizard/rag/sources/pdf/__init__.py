"""
PDF source: loader and enrichers.

All components here require PyMuPDF (fitz) and are specific to PDF files.
"""

from dragonwizard.rag.sources.pdf.loader import ExtractionMode, PDFLoadError, PDFLoader
from dragonwizard.rag.sources.pdf.llm_confirmed_enricher import LLMHeadingEnricher
from dragonwizard.rag.sources.pdf.statistical_enricher import StatisticalHeadingEnricher
from dragonwizard.rag.sources.pdf.weighted_enricher import WeightedHeadingEnricher

__all__ = [
    "ExtractionMode",
    "LLMHeadingEnricher",
    "PDFLoadError",
    "PDFLoader",
    "StatisticalHeadingEnricher",
    "WeightedHeadingEnricher",
]
