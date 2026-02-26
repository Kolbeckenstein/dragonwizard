"""
Application settings and configuration management.

Uses Pydantic Settings for validation and environment variable support.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    """Discord bot configuration."""

    name: str = Field(default="DragonWizard", description="Bot display name")
    command_prefix: str = Field(default="!", description="Command prefix for bot commands")
    token: str = Field(default="", description="Discord bot token")
    allowed_channel_ids: list[int] = Field(
        default_factory=list,
        description="If non-empty, bot only responds in these channel IDs. "
                    "Set via BOT__ALLOWED_CHANNEL_IDS='[123456,789012]'",
    )
    dev_guild_id: int | None = Field(
        default=None,
        description="If set, syncs slash commands to this guild instantly (dev mode). "
                    "If None, syncs globally (up to 1 hour propagation).",
    )


class LLMSettings(BaseSettings):
    """LLM API configuration."""

    model: str = Field(
        default="anthropic/claude-3-5-sonnet-20241022",
        description="LiteLLM model string, e.g. 'anthropic/claude-3-5-sonnet-20241022', "
                    "'openai/gpt-4o', 'ollama/llama3'. The provider prefix tells LiteLLM "
                    "which API to route the request to.",
    )
    max_tokens: int = Field(default=1024, description="Maximum tokens in response")
    temperature: float = Field(default=0.3, description="Sampling temperature")
    api_key: str = Field(default="", description="API key for the model's provider")

    model_config = SettingsConfigDict(env_prefix="LLM_")


class RAGSettings(BaseSettings):
    """RAG engine configuration."""

    vector_db: Literal["chromadb"] = Field(
        default="chromadb", description="Vector database to use"
    )
    collection_name: str = Field(
        default="dragonwizard", description="ChromaDB collection name"
    )
    chunk_size: int = Field(default=512, description="Document chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    default_k: int = Field(default=5, description="Number of chunks to retrieve")
    score_threshold: float | None = Field(
        default=None,
        description="Minimum similarity score (0.0-1.0). None means no filtering.",
    )

    # Data paths
    vector_db_path: str = Field(
        default="data/vector_db",
        description="Path to vector database storage"
    )
    processed_data_path: str = Field(
        default="data/processed",
        description="Path to store processing metadata"
    )

    # Local embeddings configuration (Sentence Transformers)
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence Transformers model name (local, no API key needed)"
    )
    embedding_device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Device for embedding generation (cpu or cuda)"
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )

    # OCR configuration
    ocr_enabled: bool = Field(
        default=True,
        description="Enable OCR fallback for scanned PDF pages (requires tesseract-ocr)"
    )

    model_config = SettingsConfigDict(env_prefix="RAG_")


class FeatureFlags(BaseSettings):
    """Feature toggles for optional functionality."""

    character_sheets: bool = Field(
        default=False, description="Enable character sheet features"
    )
    campaign_context: bool = Field(
        default=False, description="Enable campaign-specific context"
    )
    session_memory: bool = Field(
        default=False, description="Enable conversation session memory"
    )

    model_config = SettingsConfigDict(env_prefix="FEATURE_")


class ToolSettings(BaseSettings):
    """External tool configuration."""

    dice_server: str = Field(
        default="mcp://dice-roller", description="MCP dice server URI"
    )
    dice_server_path: str | None = Field(
        default=None,
        description="Path to MCP dice server index.js. "
                    "If set, enables LLM tool use (dice rolling mid-answer) and /roll command.",
    )

    model_config = SettingsConfigDict(env_prefix="TOOL_")


class Settings(BaseSettings):
    """Main application settings."""

    # Environment
    environment: Literal["development", "production"] = Field(
        default="development", description="Deployment environment"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_file: Path | None = Field(default=None, description="Log file path")

    # Sub-configurations
    bot: BotSettings = Field(default_factory=BotSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    tools: ToolSettings = Field(default_factory=ToolSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env (for future expansion)
    )


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_settings(env_file: str | Path | None = None) -> Settings:
    """
    Load settings from file and environment.

    Args:
        env_file: Path to .env file (optional)

    Returns:
        Loaded settings instance
    """
    global _settings
    if env_file:
        _settings = Settings(_env_file=env_file)
    else:
        _settings = Settings()
    return _settings
