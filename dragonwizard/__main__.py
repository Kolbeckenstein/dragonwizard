"""
DragonWizard CLI entry point.

Provides command-line interface for running the bot and utility commands.
"""

import argparse
import sys
from pathlib import Path

from dragonwizard import __version__
from dragonwizard.config.logging import get_logger, setup_logging
from dragonwizard.config.settings import load_settings


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
    logger.info(f"RAG Retrieval K: {settings.rag.retrieval_k}")
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
    else:
        # Default: show help
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
