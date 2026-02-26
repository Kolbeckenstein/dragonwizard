# DragonWizard

A Discord bot that answers D&D 5th Edition rules questions using retrieval-augmented generation (RAG). Ask it anything from spell mechanics to grappling rules — it searches your ingested rulebooks and generates a cited answer via an LLM. It can also roll dice mid-answer.

Note - This is a vibe-coded project created for hobby and personal research reasons.

## Features

- **`/ask`** — slash command for rules questions; responds with cited sources
- **`@DragonWizard <question>`** — mention the bot anywhere to ask a question
- **`/roll`** — roll dice using standard notation (`2d6+3`, `4d6kh3`, `d20`)
- **LLM tool use** — the bot can roll dice automatically while composing an answer
- **Channel restriction** — optionally limit responses to specific channels
- **Multiple collections** — ingest with different strategies and compare quality


## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Node.js (for the MCP dice server — optional)
- tesseract-ocr (optional, for scanned PDF pages)
- An Anthropic, OpenAI, or compatible LLM API key
- A Discord bot token


## Setup

### 1. Clone and install

```bash
git clone <repo-url> dragonwizard
cd dragonwizard
git submodule update --init    # pulls the dice-rolling-mcp submodule
uv sync                        # installs Python dependencies
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set BOT__TOKEN and LLM__API_KEY
```

See [.env.example](.env.example) for all available settings with inline documentation.

### 3. Ingest documents

Place your PDF rulebooks in `data/raw/pdf/` then run:

```bash
uv run dragonwizard ingest
```

This uses column-aware PDF extraction by default (better for multi-column rulebook layouts).

Options:

```bash
# Ingest a specific directory or file
uv run dragonwizard ingest path/to/pdfs/

# Add statistical heading detection (injects "[Section: ...]" prefixes into chunks)
uv run dragonwizard ingest --enricher stat_headings

# Ingest into a named collection (for A/B comparison)
uv run dragonwizard ingest --enricher stat_headings --collection col_stat

# Force re-process already-ingested files
uv run dragonwizard ingest --force
```

### 4. Discord developer portal setup

Do this once before running the bot for the first time.

#### 4a. Create the application and get a token

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications) and click **New Application**
2. Give it a name (e.g. DragonWizard), click **Create**
3. In the left sidebar, click **Bot**
4. Click **Reset Token** → copy the token → set it as `BOT__TOKEN` in your `.env`

#### 4b. Enable the Message Content privileged intent

Still on the **Bot** page, scroll to **Privileged Gateway Intents** and toggle on:

- **Message Content Intent** ← required for `@DragonWizard <question>` mentions

Click **Save Changes**.

> Without this, the bot starts but cannot read message text, so @mention questions won't work.

#### 4c. Generate an invite URL and add the bot to your server

1. In the left sidebar, go to **OAuth2 → URL Generator**
2. Under **Scopes**, check **both**:
   - `bot`
   - `applications.commands` ← required for `/ask` and `/roll` slash commands
3. Under **Bot Permissions**, check:
   - `Send Messages`
   - `Read Message History`
   - `Use Slash Commands`
4. Copy the generated URL at the bottom and open it in your browser to invite the bot to your server

> If you skip `applications.commands` the bot will start but slash commands will fail with a 403 error. You can re-run the invite URL at any time to update scopes without removing the bot.

#### 4d. (Optional) Set your dev guild ID for instant slash command sync

By default, slash commands are synced globally, which can take up to 1 hour to appear. During development, add your server's ID to `.env` for instant sync:

```ini
BOT__DEV_GUILD_ID=your-server-id-here
```

To find your server ID: in Discord, enable Developer Mode (Settings → Advanced → Developer Mode), then right-click your server icon → **Copy Server ID**.

### 5. Run the bot

```bash
uv run dragonwizard run
```

On startup the bot loads the RAG engine, syncs slash commands, and logs in. You should see:

```
INFO - RAG engine ready
INFO - LLM orchestrator ready (model: anthropic/claude-sonnet-4-6)
INFO - Cogs loaded
INFO - Slash commands synced to dev guild ... (instant)
INFO - Logged in as DragonWizard#1234 (id: ...)
```


## CLI Reference

```
dragonwizard [--env-file PATH] [--log-level LEVEL] <command>

Commands:
  run      Start the Discord bot
  ingest   Ingest documents into the RAG vector store
  query    Ask a question from the CLI (no Discord needed)
  compare  Compare retrieval quality across multiple collections
  config   Show current configuration
```

### `ingest`

```
dragonwizard ingest [SOURCE_PATH] [OPTIONS]

  SOURCE_PATH               Directory or file to ingest (default: data/raw/pdf)
  --extraction-mode         default | column_aware  (default: column_aware)
  --enricher                none | stat_headings | llm_headings | weighted_headings
  --collection NAME         Target ChromaDB collection (default: settings value)
  --force                   Re-process already-ingested files
  --clear-existing          Wipe collection before ingesting
  --batch-size N            Override embedding batch size
```

### `query`

```
dragonwizard query QUESTION [OPTIONS]

  --rag-only                Show retrieved chunks only, skip LLM
  --k N                     Number of chunks to retrieve
  --edition 5e|5.5e         Filter by edition
  --collection NAME         Query a specific collection
```

### `compare`

Compare retrieval quality across multiple ingestion strategies:

```bash
uv run dragonwizard compare "Do orcs have darkvision?" \
    --collections baseline,col_stat,col_llm --k 3
```


## Ingestion Strategies

DragonWizard supports two independent quality dimensions you can combine:

| Flag | Effect |
|---|---|
| `--extraction-mode default` | Standard PDF text extraction |
| `--extraction-mode column_aware` | Left-column-first ordering for multi-column layouts |
| `--enricher none` | No annotation |
| `--enricher stat_headings` | Detects headings via font size; prepends `[Section: ...]` to chunks |
| `--enricher weighted_headings` | Like `stat_headings` but includes a confidence score |
| `--enricher llm_headings` | Uses an LLM to confirm ambiguous heading candidates |

Recommended starting point for D&D rulebooks:

```bash
uv run dragonwizard ingest --enricher stat_headings
```


## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=dragonwizard --cov-report=term-missing

# Run a specific test file
uv run pytest tests/unit/bot/ -v

# Query without the bot (useful for iterating on RAG quality)
uv run dragonwizard query "How does grappling work?" --rag-only
```

### Project Structure

```
dragonwizard/
├── bot/
│   ├── client.py          # DragonWizardBot (discord.py Bot subclass)
│   └── cogs/
│       ├── rules.py       # /ask command + @mention listener
│       └── dice.py        # /roll command
├── config/
│   ├── settings.py        # Pydantic Settings (all config)
│   └── logging.py         # Logging setup
├── llm/
│   ├── orchestrator.py    # LLM tool-use loop (LiteLLM)
│   └── models.py          # LLMResponse, ToolCall, etc.
├── rag/
│   ├── pipeline.py        # Ingestion pipeline
│   ├── engine.py          # Search + context formatting
│   ├── chunking.py        # Sentence-aware chunker
│   ├── embeddings.py      # Sentence Transformers wrapper
│   ├── vector_store.py    # ChromaDB wrapper
│   └── sources/
│       ├── pdf/           # PDF loader + heading enrichers
│       ├── text/          # Plain text loader
│       └── markdown/      # Markdown loader
├── tools/
│   ├── base.py            # ToolAdapter ABC
│   └── dice_roller.py     # MCP dice server client
└── prompts/
    └── system.txt         # LLM system prompt template

data/
├── raw/pdf/               # Put your PDF rulebooks here
└── vector_db/             # ChromaDB storage (auto-created)

external/
└── dice-rolling-mcp/      # MCP dice server (git submodule)
```

### Environment Variables

All settings use double-underscore nesting: `SECTION__FIELD`.

| Variable | Default | Description |
|---|---|---|
| `BOT__TOKEN` | _(required)_ | Discord bot token |
| `BOT__DEV_GUILD_ID` | _(unset)_ | Guild ID for instant slash command sync |
| `BOT__ALLOWED_CHANNEL_IDS` | `[]` | Restrict responses to these channel IDs |
| `LLM__API_KEY` | _(required)_ | API key for LLM provider |
| `LLM__MODEL` | `anthropic/claude-sonnet-4-6` | LiteLLM model string |
| `LLM__MAX_TOKENS` | `1024` | Max tokens in LLM response |
| `LLM__TEMPERATURE` | `0.3` | LLM sampling temperature |
| `RAG__COLLECTION_NAME` | `dragonwizard` | ChromaDB collection |
| `RAG__DEFAULT_K` | `5` | Chunks retrieved per query |
| `RAG__EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `RAG__EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `RAG__OCR_ENABLED` | `true` | OCR fallback for scanned pages |
| `TOOL_DICE_SERVER_PATH` | _(unset)_ | Path to `dice-rolling-mcp/dist/index.js` |

See [.env.example](.env.example) for the full list with descriptions.
