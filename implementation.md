# DragonWizard — Implementation Status

## What's Built

All core layers are complete. The system is end-to-end functional.

| Component | Status | Notes |
|---|---|---|
| Project setup & config | ✅ Done | Pydantic Settings, uv, .env support |
| MCP dice server integration | ✅ Done | `DiceRollerTool` via stdio MCP client |
| LLM orchestration | ✅ Done | LiteLLM, tool-use loop, graceful error handling |
| RAG ingestion pipeline | ✅ Done | PDF/text/markdown loaders, sentence chunker |
| PDF heading enrichers | ✅ Done | Statistical, LLM-confirmed, weighted variants |
| Column-aware PDF extraction | ✅ Done | Left-column-first for multi-column layouts |
| ChromaDB vector store | ✅ Done | Local persistence, metadata filtering |
| Local embeddings | ✅ Done | Sentence Transformers (all-MiniLM-L6-v2, CPU) |
| CLI: `ingest` | ✅ Done | `--extraction-mode`, `--enricher`, `--collection` |
| CLI: `query` | ✅ Done | `--rag-only`, `--k`, `--edition`, `--collection` |
| CLI: `compare` | ✅ Done | ASCII table across multiple collections |
| Discord bot | ✅ Done | `/ask`, `/roll`, @mention, channel restriction |


## Architecture Overview

```
Discord user
    │
    ▼
DragonWizardBot (discord.py)
    ├── RulesCog  →  /ask  +  @mention
    │       │
    │       ├── RAGEngine.search()          (ChromaDB + SentenceTransformers)
    │       └── LLMOrchestrator.generate()  (LiteLLM → Claude/GPT/Ollama)
    │                   │
    │                   └── DiceRollerTool  (MCP stdio ← tool use loop)
    │
    └── DiceCog   →  /roll
            │
            └── DiceRollerTool  (MCP stdio)
```


## Key Design Decisions

**Local embeddings** — Sentence Transformers runs fully locally (~90 MB model, CPU).
No API key required for embeddings; only the LLM call costs money.

**LiteLLM** — Provider-agnostic LLM calls. Swap models by changing `LLM__MODEL`:
```
anthropic/claude-sonnet-4-6     ← default
openai/gpt-4o
ollama/llama3                   ← local, no API key
```

**Column-aware PDF extraction** — D&D rulebooks use two-column layouts. The default
extraction interleaves left/right column text, producing incoherent chunks. The
`column_aware` mode splits blocks by horizontal midpoint and reads left column first.

**Composable heading enrichers** — Enrichers are `ChunkEnricher` plugins applied
after chunking. They inject `[Section: ...]` prefixes to improve retrieval for
entity-specific queries (e.g. "Do orcs have darkvision?"). The enricher pipeline
uses six built-in filters to reject OCR artifacts, trailing punctuation, etc.

**AsyncExitStack for bot lifecycle** — The RAG embedding model, vector store, and
dice tool are all long-lived async context managers. The bot holds them open for
its entire lifetime via `contextlib.AsyncExitStack`, cleaning up on shutdown.


## File Layout

```
dragonwizard/
├── __main__.py              CLI entry point (run/ingest/query/compare/config)
├── bot/
│   ├── client.py            DragonWizardBot
│   └── cogs/
│       ├── rules.py         RulesCog (/ask + @mention)
│       └── dice.py          DiceCog (/roll)
├── config/
│   ├── settings.py          All settings (Pydantic Settings)
│   └── logging.py           Logging configuration
├── llm/
│   ├── orchestrator.py      LLM + tool-use loop
│   └── models.py            LLMResponse, ToolCall, LLMError
├── rag/
│   ├── base.py              ABCs: DocumentLoader, ChunkEnricher
│   ├── pipeline.py          IngestionPipeline
│   ├── engine.py            RAGEngine (search + format_context)
│   ├── chunking.py          SentenceAwareChunker
│   ├── embeddings.py        SentenceTransformerEmbedding
│   ├── vector_store.py      ChromaVectorStore
│   ├── components.py        RAGComponents factory
│   └── sources/
│       ├── pdf/
│       │   ├── loader.py              PDFLoader + ExtractionMode enum
│       │   ├── statistical_enricher.py  StatisticalHeadingEnricher
│       │   ├── llm_confirmed_enricher.py LLMHeadingEnricher
│       │   └── weighted_enricher.py   WeightedHeadingEnricher
│       ├── text/loader.py             TextLoader
│       └── markdown/loader.py         MarkdownLoader
├── tools/
│   ├── base.py              ToolAdapter ABC
│   └── dice_roller.py       DiceRollerTool (MCP stdio client)
└── prompts/
    └── system.txt           LLM system prompt template
```


## Future Work

See [architecture.md](architecture.md) for the full enhancement backlog. Priority items:

- **Conversation memory** — multi-turn context within a Discord thread
- **5e SRD web scraper** — ingest the SRD directly without PDF extraction noise
- **D&D Beyond forum loader** — paired Q&A embeddings for expert rule interpretations
- **Cross-encoder reranking** — better result ordering for ambiguous queries
- **Hybrid search** — combine semantic + BM25 for exact spell/rule name lookups
