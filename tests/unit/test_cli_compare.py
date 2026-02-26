"""
Tests for the 'compare' CLI command.

The compare command:
  dragonwizard compare QUESTION --collections A,B,C [--k N]

It embeds the query once, searches each named collection, and prints
an ASCII side-by-side comparison table.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dragonwizard.__main__ import create_parser


class TestCompareCommandArgs:
    """Parser-level tests for the compare subcommand."""

    def test_compare_subcommand_exists(self):
        """compare should be a valid subcommand."""
        parser = create_parser()
        args = parser.parse_args(["compare", "Do orcs have darkvision?", "--collections", "A,B"])
        assert args.command == "compare"

    def test_compare_question_positional(self):
        """compare requires a positional QUESTION argument."""
        parser = create_parser()
        args = parser.parse_args(["compare", "Do orcs have darkvision?", "--collections", "A,B"])
        assert args.question == "Do orcs have darkvision?"

    def test_compare_collections_required(self):
        """compare requires --collections."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["compare", "Some question?"])

    def test_compare_collections_parsed_as_string(self):
        """--collections should be a comma-separated string."""
        parser = create_parser()
        args = parser.parse_args(["compare", "Q?", "--collections", "baseline,col_stat,col_llm"])
        assert args.collections == "baseline,col_stat,col_llm"

    def test_compare_k_defaults_to_3(self):
        """--k should default to 3 when not specified."""
        parser = create_parser()
        args = parser.parse_args(["compare", "Q?", "--collections", "A,B"])
        assert args.k == 3

    def test_compare_k_custom(self):
        """--k should accept custom integer values."""
        parser = create_parser()
        args = parser.parse_args(["compare", "Q?", "--collections", "A,B", "--k", "5"])
        assert args.k == 5

    def test_compare_edition_flag_accepted(self):
        """compare should accept --edition flag for filtering."""
        parser = create_parser()
        args = parser.parse_args(["compare", "Q?", "--collections", "A,B", "--edition", "5e"])
        assert args.edition == "5e"

    def test_compare_edition_defaults_to_none(self):
        """compare --edition should default to None."""
        parser = create_parser()
        args = parser.parse_args(["compare", "Q?", "--collections", "A,B"])
        assert args.edition is None


class TestCompareCommandExecution:
    """Integration tests for the compare command execution."""

    @pytest.mark.asyncio
    async def test_compare_queries_each_collection(self, tmp_path):
        """compare should search each named collection once."""
        from dragonwizard.__main__ import cmd_compare
        from dragonwizard.config.settings import Settings, load_settings

        settings = MagicMock()
        settings.rag.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        settings.rag.embedding_device = "cpu"
        settings.rag.embedding_batch_size = 32
        settings.rag.vector_db_path = str(tmp_path / "db")
        settings.rag.collection_name = "default"
        settings.rag.score_threshold = 0.0
        settings.rag.default_k = 5

        args = MagicMock()
        args.question = "Do orcs have darkvision?"
        args.collections = "baseline,col_stat"
        args.k = 3
        args.edition = None

        # Mock the embedding model and per-collection search
        mock_embedding_model = MagicMock()
        mock_embedding_model.__aenter__ = AsyncMock(return_value=mock_embedding_model)
        mock_embedding_model.__aexit__ = AsyncMock(return_value=False)

        mock_result = MagicMock()
        mock_result.text = "Orcs have darkvision of 60 feet."
        mock_result.score = 0.85
        mock_result.citation = "Monster Manual p.245"

        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(return_value=[mock_result])
        mock_engine.format_context = MagicMock(return_value="context")

        search_calls = []

        async def capturing_search(query, k, filters=None):
            search_calls.append(query)
            return [mock_result]

        mock_engine.search = capturing_search

        mock_vector_store = MagicMock()
        mock_vector_store.__aenter__ = AsyncMock(return_value=mock_vector_store)
        mock_vector_store.__aexit__ = AsyncMock(return_value=False)

        with patch("dragonwizard.__main__.RAGComponents") as MockRAG:
            factory = MagicMock()
            factory.create_embedding_model.return_value = mock_embedding_model
            factory.create_vector_store.return_value = mock_vector_store
            factory.create_engine.return_value = mock_engine
            MockRAG.return_value = factory

            exit_code = await cmd_compare(args, settings)

        # Should have created a vector store per collection (2 collections)
        assert factory.create_vector_store.call_count == 2
        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_compare_prints_ascii_table(self, tmp_path, capsys):
        """compare should print collection names and results."""
        from dragonwizard.__main__ import cmd_compare

        settings = MagicMock()
        settings.rag.vector_db_path = str(tmp_path / "db")
        settings.rag.collection_name = "default"
        settings.rag.score_threshold = 0.0

        args = MagicMock()
        args.question = "Darkvision?"
        args.collections = "baseline,col_stat"
        args.k = 2
        args.edition = None

        mock_embedding_model = MagicMock()
        mock_embedding_model.__aenter__ = AsyncMock(return_value=mock_embedding_model)
        mock_embedding_model.__aexit__ = AsyncMock(return_value=False)

        mock_result = MagicMock()
        mock_result.text = "Orc darkvision text."
        mock_result.score = 0.90
        mock_result.citation = "MM p.1"

        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(return_value=[mock_result])

        mock_vector_store = MagicMock()
        mock_vector_store.__aenter__ = AsyncMock(return_value=mock_vector_store)
        mock_vector_store.__aexit__ = AsyncMock(return_value=False)

        with patch("dragonwizard.__main__.RAGComponents") as MockRAG:
            factory = MagicMock()
            factory.create_embedding_model.return_value = mock_embedding_model
            factory.create_vector_store.return_value = mock_vector_store
            factory.create_engine.return_value = mock_engine
            MockRAG.return_value = factory

            await cmd_compare(args, settings)

        captured = capsys.readouterr()
        output = captured.out

        # Should mention both collection names
        assert "baseline" in output
        assert "col_stat" in output
        # Should show the question
        assert "Darkvision?" in output
