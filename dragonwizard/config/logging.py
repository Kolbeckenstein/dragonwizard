"""
Logging configuration and setup.

Provides structured logging with console and file output support.
"""

import logging
import sys
from pathlib import Path
from typing import Any

from dragonwizard.config.settings import Settings


class ColoredFormatter(logging.Formatter):
    """Formatter that adds color to console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color."""
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


def setup_logging(settings: Settings) -> None:
    """
    Configure logging based on settings.

    Args:
        settings: Application settings containing log configuration
    """
    # Create root logger
    root_logger = logging.getLogger("dragonwizard")
    root_logger.setLevel(getattr(logging, settings.log_level))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level))
    console_formatter = ColoredFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if configured
    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, settings.log_level))
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Don't propagate to root logger
    root_logger.propagate = False

    # Log initial setup
    root_logger.info(f"Logging initialized - Level: {settings.log_level}")
    if settings.log_file:
        root_logger.info(f"Logging to file: {settings.log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"dragonwizard.{name}")
