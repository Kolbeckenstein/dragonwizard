"""
DragonWizard CLI entry point.

Provides command-line interface for running the bot and utility commands.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from dragonwizard import __version__
from dragonwizard.config.logging import get_logger, setup_logging
from dragonwizard.config.settings import Settings, load_settings
from dragonwizard.rag.components import RAGComponents


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog="dragonwizard",
        description="LLM-powered Discord bot for D&D 5th edition rules assistance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"DragonWizard {__version__}",
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: .env in current directory)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Override logging level from config",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command (default - will start Discord bot in future)
    run_parser = subparsers.add_parser(
        "run",
        help="Run the Discord bot (not yet implemented)",
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Show current configuration",
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Ask a D&D rules question (RAG search + optional LLM response)",
    )
    query_parser.add_argument(
        "question",
        help='Question to ask, e.g. "How does advantage work?"',
    )
    query_parser.add_argument(
        "--edition",
        choices=["5e", "5.5e"],
        default=None,
        help="Filter results to a specific D&D edition (default: both editions)",
    )
    query_parser.add_argument(
        "--rag-only",
        action="store_true",
        help="Show retrieved chunks only — skip LLM, no API key needed",
    )
    query_parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: RAG__DEFAULT_K from config)",
    )
    query_parser.add_argument(
        "--collection",
        default=None,
        help="Query a specific named ChromaDB collection (default: settings collection)",
    )

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents into the RAG vector store",
    )
    ingest_parser.add_argument(
        "source_path",
        nargs="?",
        type=Path,
        default=Path("data/raw/pdf"),
        help="Path to file or directory to ingest (default: data/raw/pdf)",
    )
    ingest_parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Process directories recursively (default: True)",
    )
    ingest_parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear vector store before ingestion",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process already ingested documents",
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override embedding batch size from config",
    )
    ingest_parser.add_argument(
        "--extraction-mode",
        choices=["default", "column_aware"],
        default="column_aware",
        help="PDF text extraction: 'default' or 'column_aware' (left-column-first). "
             "Default: column_aware",
    )
    ingest_parser.add_argument(
        "--enricher",
        choices=["none", "stat_headings", "llm_headings", "weighted_headings"],
        default="none",
        help="Chunk enricher: 'none', 'stat_headings', 'llm_headings', "
             "or 'weighted_headings'. Default: none",
    )
    ingest_parser.add_argument(
        "--collection",
        default=None,
        help="Override the target ChromaDB collection name "
             "(default: settings collection).",
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare retrieval quality across multiple named collections",
    )
    compare_parser.add_argument(
        "question",
        help="Question to search for across all collections",
    )
    compare_parser.add_argument(
        "--collections",
        required=True,
        help="Comma-separated collection names, e.g. baseline,col_stat,col_llm",
    )
    compare_parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Chunks to retrieve per collection (default: 3)",
    )
    compare_parser.add_argument(
        "--edition",
        choices=["5e", "5.5e"],
        default=None,
        help="Filter results to a specific D&D edition (default: no filter)",
    )

    return parser


def _build_enricher(name: str, settings):
    """
    Map CLI enricher name to a ChunkEnricher instance.

    Args:
        name: One of 'none', 'stat_headings', 'llm_headings', 'weighted_headings'
        settings: Full Settings object (for LLM config when needed)

    Returns:
        ChunkEnricher instance, or None for 'none'
    """
    from dragonwizard.rag.sources.pdf import (
        StatisticalHeadingEnricher,
        LLMHeadingEnricher,
        WeightedHeadingEnricher,
    )

    if name == "none":
        return None
    elif name == "stat_headings":
        return StatisticalHeadingEnricher()
    elif name == "llm_headings":
        return LLMHeadingEnricher(llm_settings=settings.llm)
    elif name == "weighted_headings":
        return WeightedHeadingEnricher()
    else:
        raise ValueError(f"Unknown enricher: {name!r}")


def cmd_config(settings) -> int:
    """Show current configuration."""
    logger = get_logger(__name__)

    logger.info("Current Configuration:")
    logger.info("\n=== DragonWizard Configuration ===\n")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Log Level: {settings.log_level}")
    logger.info(f"Log File: {settings.log_file or 'None (console only)'}")
    logger.info(f"\nBot Name: {settings.bot.name}")
    logger.info(f"Command Prefix: {settings.bot.command_prefix}")
    logger.info(f"\nLLM Model: {settings.llm.model}")
    logger.info(f"LLM API Key: {'Set' if settings.llm.api_key else 'Not set'}")
    logger.info(f"\nRAG Vector DB: {settings.rag.vector_db}")
    logger.info(f"RAG Chunk Size: {settings.rag.chunk_size}")
    logger.info(f"RAG Default K: {settings.rag.default_k}")
    logger.info(f"RAG Embedding Model: {settings.rag.embedding_model}")
    logger.info(f"RAG Embedding Device: {settings.rag.embedding_device}")
    logger.info(f"RAG Embedding Batch Size: {settings.rag.embedding_batch_size}")
    logger.info(f"\nFeature Flags:")
    logger.info(f"  Character Sheets: {settings.features.character_sheets}")
    logger.info(f"  Campaign Context: {settings.features.campaign_context}")
    logger.info(f"  Session Memory: {settings.features.session_memory}")
    logger.info(f"\nTool Dice Server: {settings.tools.dice_server}")

    return 0


def cmd_run(settings) -> int:
    """Start the Discord bot."""
    logger = get_logger(__name__)

    if not settings.bot.token:
        logger.error(
            "Discord bot token not set. Add BOT__TOKEN=<your-token> to your .env file."
        )
        return 1

    if not settings.llm.api_key:
        logger.warning(
            "LLM API key not set (LLM__API_KEY). "
            "The bot will start but /ask will fail until this is configured."
        )

    from dragonwizard.bot import DragonWizardBot

    bot = DragonWizardBot(settings)
    logger.info(f"Starting {settings.bot.name}...")
    # log_handler=None: disable discord.py's default logging setup and use ours
    bot.run(settings.bot.token, log_handler=None)
    return 0


async def cmd_ingest(args, settings: Settings) -> int:
    """
    Ingest documents into the RAG vector store.

    Args:
        args: Parsed command-line arguments
        settings: Application settings

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = get_logger(__name__)

    # Validate source path
    source_path: Path = args.source_path
    if not source_path.exists():
        logger.error(f"Source path does not exist: {source_path}")
        return 1

    # Override batch size if specified
    if args.batch_size is not None:
        settings.rag.embedding_batch_size = args.batch_size
        logger.info(f"Overriding embedding batch size: {args.batch_size}")

    try:
        from dragonwizard.rag.sources.pdf.loader import ExtractionMode

        # Resolve enricher and extraction mode from CLI args
        enricher = _build_enricher(args.enricher, settings)
        enrichers = [enricher] if enricher is not None else []
        extraction_mode = ExtractionMode(args.extraction_mode)

        # Override collection name if --collection specified
        rag_settings = settings.rag
        if args.collection:
            rag_settings = rag_settings.model_copy(update={"collection_name": args.collection})

        # Initialize components via factory
        logger.info("Initializing RAG components...")
        logger.info(f"  Extraction mode: {args.extraction_mode}")
        logger.info(f"  Enricher: {args.enricher}")
        if args.collection:
            logger.info(f"  Collection: {args.collection}")
        factory = RAGComponents(rag_settings)

        async with factory.create_embedding_model() as embedding_model, \
                   factory.create_vector_store() as vector_store:

            # Clear existing data if requested
            if args.clear_existing:
                logger.warning("Clearing existing vector store...")
                await vector_store.delete_collection()
                logger.info("Vector store cleared")

            pipeline = factory.create_pipeline(
                embedding_model,
                vector_store,
                enrichers=enrichers,
                extraction_mode=extraction_mode,
            )

            # Progress callback for user feedback
            def progress_callback(message: str):
                logger.info(f"  {message}")

            # Start timing
            start_time = time.time()

            # Ingest file or directory
            if source_path.is_file():
                logger.info(f"Ingesting file: {source_path}")
                chunk_count = await pipeline.ingest_file(
                    source_path,
                    force=args.force,
                    progress_callback=progress_callback
                )

                elapsed = time.time() - start_time
                logger.info("\n=== Ingestion Complete ===")
                logger.info(f"File: {source_path.name}")
                logger.info(f"Chunks created: {chunk_count}")
                logger.info(f"Time elapsed: {elapsed:.2f}s")

            elif source_path.is_dir():
                logger.info(
                    f"Ingesting directory: {source_path} "
                    f"(recursive={args.recursive})"
                )
                results = await pipeline.ingest_directory(
                    source_path,
                    recursive=args.recursive,
                    force=args.force,
                    progress_callback=progress_callback
                )

                elapsed = time.time() - start_time
                total_chunks = sum(results.values())

                logger.info("\n=== Ingestion Complete ===")
                logger.info(f"Directory: {source_path}")
                logger.info(f"Files processed: {len(results)}")
                logger.info(f"Total chunks created: {total_chunks}")
                logger.info(f"Time elapsed: {elapsed:.2f}s")

                # Show per-file breakdown
                if results:
                    logger.info("\nPer-file results:")
                    for file_path, count in results.items():
                        logger.info(f"  {Path(file_path).name}: {count} chunks")
            else:
                logger.error(f"Source path is neither a file nor directory: {source_path}")
                return 1

            # Show vector store statistics
            stats = await vector_store.get_stats()
            logger.info("\nVector store statistics:")
            logger.info(f"  Total documents: {stats['document_count']}")
            logger.info(f"  Collection name: {stats['collection_name']}")

            return 0

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return 1


async def cmd_query(args, settings: Settings) -> int:
    """
    Query the RAG store with a natural-language question.

    Flow:
      1. Embed the query with the local Sentence Transformers model
      2. Search ChromaDB for the top-k most similar chunks
         (optionally filtered by --edition)
      3a. With --rag-only: print the chunks with citations and scores
      3b. Without --rag-only: feed context into LLMOrchestrator and
          print the LLM's answer with source citations

    The --rag-only mode is useful for validating ingestion quality
    before spending API tokens — you can see exactly which chunks the
    retrieval layer is finding for a given question.
    """
    logger = get_logger(__name__)

    question: str = args.question
    edition_filter = {"edition": args.edition} if args.edition else None
    k = args.k or settings.rag.default_k

    try:
        rag_settings = settings.rag
        if args.collection:
            rag_settings = rag_settings.model_copy(update={"collection_name": args.collection})
        factory = RAGComponents(rag_settings)

        async with factory.create_embedding_model() as embedding_model, \
                   factory.create_vector_store() as vector_store:

            engine = factory.create_engine(embedding_model, vector_store)

            logger.info(f"Searching for top {k} chunks" +
                        (f" (edition: {args.edition})" if args.edition else "") + "...")

            results = await engine.search(
                query=question,
                k=k,
                filters=edition_filter,
            )

            if not results:
                print("No relevant results found in the vector store.")
                print("Tip: Run 'dragonwizard ingest data/raw/pdf --recursive' first.")
                return 0

            if args.rag_only:
                edition_note = f"  edition filter: {args.edition}" if args.edition else ""
                print("\n=== RAG Search Results ===")
                print(f"Q: {question}")
                if edition_note:
                    print(edition_note)
                print(f"Retrieved: {len(results)} chunks\n")

                for i, result in enumerate(results, start=1):
                    preview = result.text[:300].replace("\n", " ").strip()
                    if len(result.text) > 300:
                        preview += "..."
                    print(f"[{i}] score={result.score:.3f}  {result.citation}")
                    print(f"    {preview}\n")

                return 0

            # Full LLM response path
            from pathlib import Path as _Path
            from dragonwizard.llm import LLMOrchestrator, LLMError

            system_template_path = _Path(__file__).parent / "prompts" / "system.txt"
            if not system_template_path.exists():
                logger.error(f"System prompt template not found: {system_template_path}")
                return 1
            system_template = system_template_path.read_text()

            context = engine.format_context(results)
            orchestrator = LLMOrchestrator(
                settings=settings.llm,
                system_template=system_template,
                # No tool adapter for CLI — dice rolling not wired here yet
            )

            logger.info(f"Sending to {settings.llm.model}...")

            try:
                response = await orchestrator.generate_response(
                    query=question,
                    context=context,
                )
            except LLMError as e:
                print(f"\nLLM error: {e}", file=sys.stderr)
                print("Tip: Set LLM__API_KEY in your .env file.", file=sys.stderr)
                return 1

            edition_note = f" [{args.edition}]" if args.edition else ""
            print(f"\n=== DragonWizard{edition_note} ===")
            print(f"Q: {question}\n")
            print(response.text)

            print(f"\n--- Sources ({len(results)} chunks) ---")
            for i, result in enumerate(results, start=1):
                print(f"  [{i}] {result.citation}  (score: {result.score:.3f})")

            if response.tool_calls:
                print("\n--- Tool Calls ---")
                for tc in response.tool_calls:
                    print(f"  {tc.name} → {tc.result}")

            print(f"\nTokens: {response.usage.total_tokens} "
                  f"(prompt {response.usage.prompt_tokens} "
                  f"+ completion {response.usage.completion_tokens})")

            return 0

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return 1


async def cmd_compare(args, settings) -> int:
    """
    Compare retrieval quality across multiple named ChromaDB collections.

    Embeds the query once, then searches each named collection and prints
    an ASCII side-by-side comparison table.

    Args:
        args: Parsed arguments (question, collections, k, edition)
        settings: Application settings

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = get_logger(__name__)

    question: str = args.question
    collection_names = [c.strip() for c in args.collections.split(",") if c.strip()]
    k: int = args.k
    edition_filter = {"edition": args.edition} if args.edition else None

    if not collection_names:
        logger.error("--collections must contain at least one collection name")
        return 1

    logger.info(f"Comparing {len(collection_names)} collections for: {question!r}")

    try:
        # Build a base factory (for the embedding model)
        base_rag_settings = settings.rag
        factory = RAGComponents(base_rag_settings)

        collection_results: dict[str, list] = {}

        async with factory.create_embedding_model() as embedding_model:
            for coll_name in collection_names:
                try:
                    # Each collection needs its own vector store
                    coll_settings = base_rag_settings.model_copy(
                        update={"collection_name": coll_name}
                    )
                    coll_factory = RAGComponents(coll_settings)

                    async with coll_factory.create_vector_store() as vector_store:
                        engine = coll_factory.create_engine(embedding_model, vector_store)
                        results = await engine.search(question, k=k, filters=edition_filter)
                        collection_results[coll_name] = results

                except Exception as e:
                    logger.warning(f"Failed to search collection {coll_name!r}: {e}")
                    collection_results[coll_name] = []

        # Print ASCII comparison table
        col_width = 60
        separator = "-" * (col_width * len(collection_names) + len(collection_names) + 1)

        print(f"\n=== Compare: {question!r} ===\n")
        print(f"Collections: {', '.join(collection_names)}  |  k={k}")
        print(separator)

        # Header row
        header = " | ".join(name.ljust(col_width) for name in collection_names)
        print(f"| {header} |")
        print(separator)

        # Result rows (up to k rows)
        for rank in range(k):
            row_parts = []
            for name in collection_names:
                results = collection_results.get(name, [])
                if rank < len(results):
                    r = results[rank]
                    preview = r.text[:col_width - 20].replace("\n", " ").strip()
                    cell = f"[{rank + 1}] {r.score:.2f}  {preview}"
                else:
                    cell = "(no result)"
                row_parts.append(cell.ljust(col_width))
            print(f"| {' | '.join(row_parts)} |")

        print(separator)
        return 0

    except Exception as e:
        logger.error(f"Compare failed: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Load settings
    try:
        settings = load_settings(env_file=args.env_file)
    except Exception as e:
        print(f"Error loading settings: {e}", file=sys.stderr)
        return 1

    # Override log level if specified
    if args.log_level:
        settings.log_level = args.log_level

    # Setup logging
    setup_logging(settings)
    logger = get_logger(__name__)

    # Execute command
    if args.command == "config":
        return cmd_config(settings)
    elif args.command == "run":
        return cmd_run(settings)
    elif args.command == "query":
        return asyncio.run(cmd_query(args, settings))
    elif args.command == "ingest":
        return asyncio.run(cmd_ingest(args, settings))
    elif args.command == "compare":
        return asyncio.run(cmd_compare(args, settings))
    else:
        # Default: show help
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
