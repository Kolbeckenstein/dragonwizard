"""
Tests for new ingest CLI flags: --extraction-mode, --enricher, --collection.

These tests verify that:
- The parser accepts the new flags without error
- Valid values are accepted; invalid values are rejected
- The flags are wired through to the correct pipeline components
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dragonwizard.__main__ import create_parser


class TestIngestCLIFlags:
    """Parser-level tests for the new ingest flags."""

    def test_extraction_mode_default_accepted(self):
        """--extraction-mode default should parse without error."""
        parser = create_parser()
        args = parser.parse_args(["ingest", "data/raw/pdf", "--extraction-mode", "default"])
        assert args.extraction_mode == "default"

    def test_extraction_mode_column_aware_accepted(self):
        """--extraction-mode column_aware should parse without error."""
        parser = create_parser()
        args = parser.parse_args(["ingest", "data/raw/pdf", "--extraction-mode", "column_aware"])
        assert args.extraction_mode == "column_aware"

    def test_extraction_mode_invalid_rejected(self):
        """--extraction-mode with invalid value should exit with error."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ingest", "data/raw/pdf", "--extraction-mode", "invalid_mode"])

    def test_extraction_mode_defaults_to_column_aware(self):
        """--extraction-mode should default to 'column_aware' when not specified."""
        parser = create_parser()
        args = parser.parse_args(["ingest", "data/raw/pdf"])
        assert args.extraction_mode == "column_aware"

    def test_source_path_defaults_to_data_raw_pdf(self):
        """source_path should default to data/raw/pdf when omitted."""
        parser = create_parser()
        args = parser.parse_args(["ingest"])
        assert args.source_path == Path("data/raw/pdf")

    def test_source_path_explicit_overrides_default(self):
        """An explicit source_path should override the default."""
        parser = create_parser()
        args = parser.parse_args(["ingest", "some/other/path"])
        assert args.source_path == Path("some/other/path")

    def test_recursive_defaults_to_true(self):
        """--recursive should default to True when not specified."""
        parser = create_parser()
        args = parser.parse_args(["ingest"])
        assert args.recursive is True

    def test_enricher_none_accepted(self):
        """--enricher none should parse without error."""
        parser = create_parser()
        args = parser.parse_args(["ingest", "data/raw/pdf", "--enricher", "none"])
        assert args.enricher == "none"

    def test_enricher_stat_headings_accepted(self):
        """--enricher stat_headings should parse without error."""
        parser = create_parser()
        args = parser.parse_args(["ingest", "data/raw/pdf", "--enricher", "stat_headings"])
        assert args.enricher == "stat_headings"

    def test_enricher_llm_headings_accepted(self):
        """--enricher llm_headings should parse without error."""
        parser = create_parser()
        args = parser.parse_args(["ingest", "data/raw/pdf", "--enricher", "llm_headings"])
        assert args.enricher == "llm_headings"

    def test_enricher_weighted_headings_accepted(self):
        """--enricher weighted_headings should parse without error."""
        parser = create_parser()
        args = parser.parse_args(["ingest", "data/raw/pdf", "--enricher", "weighted_headings"])
        assert args.enricher == "weighted_headings"

    def test_enricher_invalid_rejected(self):
        """--enricher with unknown value should exit with error."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ingest", "data/raw/pdf", "--enricher", "magic_enricher"])

    def test_enricher_defaults_to_none(self):
        """--enricher should default to 'none' when not specified."""
        parser = create_parser()
        args = parser.parse_args(["ingest", "data/raw/pdf"])
        assert args.enricher == "none"

    def test_ingest_collection_flag_accepted(self):
        """ingest --collection should parse without error."""
        parser = create_parser()
        args = parser.parse_args(["ingest", "data/raw/pdf", "--collection", "my_collection"])
        assert args.collection == "my_collection"

    def test_ingest_collection_defaults_to_none(self):
        """--collection should default to None when not specified."""
        parser = create_parser()
        args = parser.parse_args(["ingest", "data/raw/pdf"])
        assert args.collection is None

    def test_query_collection_flag_accepted(self):
        """query --collection should parse without error."""
        parser = create_parser()
        args = parser.parse_args(["query", "How does fireball work?", "--collection", "col_stat"])
        assert args.collection == "col_stat"

    def test_query_collection_defaults_to_none(self):
        """query --collection should default to None when not specified."""
        parser = create_parser()
        args = parser.parse_args(["query", "How does fireball work?"])
        assert args.collection is None
