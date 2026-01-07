# DragonWizard

DragonWizard is an LLM-powered Discord bot that provides accurate D&D 5th Edition rules answers using RAG (Retrieval-Augmented Generation) with local embeddings and external tool integrations.

## Features

- **Accurate Rules Lookup**: Uses RAG to retrieve relevant D&D 5e SRD content
- **LLM-Powered Answers**: Leverages Claude/GPT for natural, context-aware responses
- **Local Embeddings**: Uses Sentence Transformers (no API key needed, runs on CPU)
- **Tool Integration**: Can roll dice via MCP servers and demonstrate mechanics
- **Extensible Architecture**: Designed for future character sheets and campaign context
- **Privacy-First**: All embeddings generated locally, no data sent to third parties

## Project Status

🚧 **Currently in Phase 1.1 - Project Setup** ✅

See [implementation.md](implementation.md) for the full development roadmap.

## System Requirements

- **Python**: 3.13+
- **RAM**: 4GB minimum (for local embedding model)
- **CPU**: 4+ cores recommended
- **Disk**: ~500MB (models + data)
- **OS**: Linux, macOS, or Windows (WSL2)

The all-MiniLM-L6-v2 embedding model runs efficiently on CPU - no GPU required!

## Quick Start

### Prerequisites

**Option 1: Dev Container (Recommended)**
- [Docker](https://www.docker.com/products/docker-desktop)
- [VS Code](https://code.visualstudio.com/) with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

**Option 2: Local Installation**
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Node.js 20+ and npm (for MCP dice server)

### Installation

**Option 1: Dev Container (Easiest)**

1. Install Docker and VS Code with Dev Containers extension
2. Clone and open the repository:
```bash
git clone <repository-url>
cd dragonwizard
code .
```
3. When prompted, click "Reopen in Container" (or press F1 → "Dev Containers: Reopen in Container")
4. Wait for automatic setup (~5 minutes first time)
5. Done! All dependencies are installed and configured.

See [.devcontainer/README.md](.devcontainer/README.md) for details.

**Option 2: Local Installation**

1. Clone the repository:
```bash
git clone <repository-url>
cd dragonwizard
```

2. Run complete setup:
```bash
make setup
```

Or install manually:
```bash
# Install Python dependencies
uv sync

# Build MCP dice server
make install-dice-server

# Set up environment
cp .env.example .env
# Edit .env and add your API keys
```

3. Verify installation:
```bash
uv run dragonwizard --version
```

### Configuration

The bot is configured via environment variables. Copy `.env.example` to `.env` and configure:

**Required API Keys:**
- `LLM__API_KEY`: Anthropic Claude API key ([Get one here](https://console.anthropic.com/))
- `DISCORD_TOKEN`: Discord bot token ([Create bot here](https://discord.com/developers/applications))

**No Embeddings API Key Needed!** We use local Sentence Transformers (all-MiniLM-L6-v2) which:
- Runs on your CPU (no GPU required)
- Downloads automatically on first use (~90MB)
- Generates 384-dimensional embeddings
- Zero cost, fully private

**Configuration Options:**
- `LLM__PROVIDER`: "anthropic" or "openai"
- `LLM__MODEL`: Model to use (default: claude-3-5-sonnet-20241022)
- `RAG__CHUNK_SIZE`: Document chunk size (default: 512 tokens)
- `RAG__EMBEDDING_MODEL`: Sentence Transformers model (default: sentence-transformers/all-MiniLM-L6-v2)
- `RAG__EMBEDDING_DEVICE`: "cpu" or "cuda" (default: cpu)
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)

View current configuration:
```bash
uv run dragonwizard config
```

## Usage

### CLI Commands

```bash
# Show version
uv run dragonwizard --version

# Show current configuration
uv run dragonwizard config

# Run the Discord bot (not yet implemented)
uv run dragonwizard run
```

### Development

Run tests:
```bash
uv run pytest
```

Run tests with coverage:
```bash
uv run pytest --cov=dragonwizard
```

## Project Structure

```
dragonwizard/
├── dragonwizard/          # Main package
│   ├── bot/              # Discord bot layer
│   ├── rag/              # RAG engine (retrieval & vector store)
│   ├── llm/              # LLM orchestration & tool calling
│   ├── tools/            # External tool integrations (MCP, etc.)
│   ├── config/           # Configuration & settings management
│   └── __main__.py       # CLI entry point
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── data/                # Data storage
│   ├── raw/            # Source documents (SRD, etc.)
│   └── processed/      # Processed chunks & embeddings
├── pyproject.toml       # Project metadata & dependencies
├── .env.example         # Example environment configuration
└── implementation.md    # Development roadmap
```

## Architecture

DragonWizard follows a layered architecture:

1. **Discord Bot Layer**: Message handling, command parsing, response formatting
2. **Query Processing Layer**: Intent classification, query enrichment
3. **RAG Engine Layer**: Document retrieval, vector search, semantic matching
4. **LLM Orchestration Layer**: Prompt construction, tool calling, response generation
5. **Tool Integration Layer**: External tools (dice rolling, spell lookup, etc.)

**Key Technology Decisions:**
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2) - local, free, private
- **Vector DB**: ChromaDB - simple, local, SQLite-backed
- **LLM**: Anthropic Claude - strong reasoning, long context, tool use
- **Bot Framework**: discord.py - mature, async Python library

See [architecture.md](architecture.md) for detailed architecture documentation.

## Development Roadmap

See [implementation.md](implementation.md) for the complete implementation plan.

**Current Phase: 1.1 - Project Setup** ✅
- [x] Initialize Python project with uv
- [x] Create project structure
- [x] Set up configuration management
- [x] Set up logging framework
- [x] Create CLI entry point
- [x] Updated docs for local embeddings

**Next Phase: 1.2 - MCP Dice Server Integration**
- [ ] Install MCP Python SDK
- [ ] Create tool adapter pattern
- [ ] Implement dice rolling CLI

**Future Phases:**
- Phase 2: LLM API Integration & Tool Use
- Phase 3: RAG Engine (Document Processing, Vector DB, Embeddings)
- Phase 4: Query Processing Layer
- Phase 5: Discord Bot Integration (MVP Complete!)
- Phase 6: Enhancement & Polish
- Phase 7: Extensibility (Character Sheets, Campaign Context)

## Why Local Embeddings?

We chose Sentence Transformers over cloud APIs for several reasons:

| Factor | Sentence Transformers (Local) | OpenAI API |
|--------|------------------------------|-----------|
| **Cost** | Free | ~$0.0001 per 1K tokens |
| **Privacy** | All data stays local | Sent to OpenAI |
| **Setup** | ~90MB model download | API key only |
| **Speed** | Depends on CPU (fast enough) | Consistent, fast |
| **Requirements** | 2-4GB RAM | Internet connection |
| **Offline** | Yes | No |

For a D&D rules bot with ~1,000-2,000 document chunks, local embeddings are:
- ✅ Fast enough (queries take milliseconds)
- ✅ Zero ongoing cost
- ✅ Fully private
- ✅ Work offline
- ✅ Simple to set up

## Contributing

This is currently a personal project. Contributions guidelines will be added in the future.

## License

TBD

## Resources

- [D&D 5e SRD](https://dnd.wizards.com/resources/systems-reference-document)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [discord.py Documentation](https://discordpy.readthedocs.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
