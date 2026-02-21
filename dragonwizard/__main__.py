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

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents into the RAG vector store",
    )
    ingest_parser.add_argument(
        "source_path",
        type=Path,
        help="Path to file or directory to ingest",
    )
    ingest_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process directories recursively",
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

    return parser


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
    logger.info(f"\nLLM Provider: {settings.llm.provider}")
    logger.info(f"LLM Model: {settings.llm.model}")
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
    """Run the Discord bot (placeholder)."""
    logger = get_logger(__name__)
    logger.info("Starting DragonWizard bot...")
    logger.warning("Discord bot functionality not yet implemented (Phase 5)")
    logger.info("See implementation.md for development roadmap")
    return 1


async def cmd_ingest(args, settings: Settings) -> int:
    """
    Ingest documents into the RAG vector store.

    Args:
        args: Parsed command-line arguments
        settings: Application settings

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from dragonwizard.rag.components import RAGComponents

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
        # Initialize components via factory
        logger.info("Initializing RAG components...")
        factory = RAGComponents(settings.rag)

        async with factory.create_embedding_model() as embedding_model, \
                   factory.create_vector_store() as vector_store:

            # Clear existing data if requested
            if args.clear_existing:
                logger.warning("Clearing existing vector store...")
                await vector_store.delete_collection()
                logger.info("Vector store cleared")

            pipeline = factory.create_pipeline(embedding_model, vector_store)

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
    elif args.command == "ingest":
        return asyncio.run(cmd_ingest(args, settings))
    else:
        # Default: show help
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
