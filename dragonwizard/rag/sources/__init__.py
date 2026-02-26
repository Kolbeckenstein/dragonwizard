"""
Source-specific document loaders and enrichers.

Each subdirectory owns one data-source type — its loader(s) and any
enrichers that depend on that source's specific format or metadata.

    sources/pdf/      PDF loader (PyMuPDF) + font-heuristic enrichers
    sources/text/     Plain-text loader
    sources/markdown/ Markdown loader

Source-agnostic enrichers (e.g. NoOpEnricher) live in dragonwizard.rag.enrichers.
"""
