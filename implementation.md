# DragonWizard Implementation Order

## Overview

This document outlines the recommended implementation order for DragonWizard, prioritizing independently runnable, testable, and usable behavior at each step. Each phase delivers working functionality that can be tested in isolation before moving to the next layer.

## Implementation Philosophy

- **Bottom-up with testable milestones**: Build foundational layers that can be tested independently
- **Vertical slices**: Each phase should deliver end-to-end functionality, even if limited
- **Fail fast**: Test integrations early to catch API, dependency, or architecture issues
- **Incremental value**: Each step produces something runnable and demonstrable

---

## Phase 1: Foundation & External Dependencies

### 1.1 Project Setup & Configuration
**Rationale**: Establish development environment and dependency management first

**Tasks**:
- Initialize Python project with `uv` or Poetry
- Create project structure (directories for bot, rag, llm, tools, config)
- Set up configuration management (TOML/YAML config file)
- Create `.env.example` for API keys and secrets
- Set up logging framework
- Create basic `README.md` with setup instructions

**Testable Outcome**:
- `python -m dragonwizard --version` runs successfully
- Config loads from file
- Logging outputs to console and file

**Why First**: Prevents dependency hell later; establishes patterns for the entire project

---

### 1.2 MCP Dice Server Integration
**Rationale**: Test external tool integration early with simplest use case

**Tasks**:
- Install MCP Python SDK
- Create `ToolAdapter` base class
- Implement `DiceRollerTool` adapter for MCP dice server
- Write unit tests for dice rolling (mock MCP responses)
- Write integration tests (requires running MCP dice server)
- Create CLI tool for manual dice rolling: `python -m dragonwizard.tools.dice roll "2d20kh1+5"`

**Testable Outcome**:
```bash
$ python -m dragonwizard.tools.dice roll "2d20kh1+5"
Rolling: 2d20kh1+5
Results: [18, 7] -> 18 + 5 = 23
```

**Why Second**:
- Validates MCP integration works before building around it
- Simple, deterministic testing
- No LLM costs to debug
- Establishes tool adapter pattern

---

## Phase 2: LLM Orchestration (Without RAG)

### 2.1 Basic LLM API Integration
**Rationale**: Test LLM API connectivity and response handling independently

**Tasks**:
- Install Anthropic Python SDK
- Create `LLMClient` class with basic query/response
- Implement API key management from environment
- Add error handling (rate limits, timeouts, API errors)
- Create simple CLI: `python -m dragonwizard.llm query "What is a saving throw?"`
- Write tests with mocked API responses

**Testable Outcome**:
```bash
$ export ANTHROPIC_API_KEY=sk-...
$ python -m dragonwizard.llm query "What is a saving throw?"
A saving throw is a d20 roll used to resist spells, traps, and other effects...
```

**Why Now**:
- Validates API keys and connectivity
- Tests token budget handling
- No complex RAG or Discord dependencies yet

---

### 2.2 LLM Tool Use (Dice Rolling)
**Rationale**: Integrate LLM with tool calling before adding RAG complexity

**Tasks**:
- Define tool schema for `roll_dice` in Claude format
- Implement tool call handling in `LLMClient`
- Connect LLM tool calls to `DiceRollerTool` adapter
- Add examples to prompt for when to use dice rolling
- Create CLI that demonstrates tool use: `python -m dragonwizard.llm query "Roll 2d20 with advantage"`

**Testable Outcome**:
```bash
$ python -m dragonwizard.llm query "Show me an example of rolling with advantage"
When you have advantage, you roll 2d20 and take the higher result.
Let me demonstrate: [calls tool]
Rolling 2d20kh1: [15, 8] -> 15
In this example, you'd use 15 as your result.
```

**Why Now**:
- Tests complete LLM orchestration loop
- Validates tool integration works with LLM
- Still no RAG or Discord complexity

---

## Phase 3: RAG Engine (Standalone)

### 3.1 Document Processing & Chunking
**Rationale**: Build document pipeline independently of vector DB

**Tasks**:
- Download D&D 5e SRD content (markdown or text format)
- Create `DocumentProcessor` class
- Implement chunking strategy (512 tokens, 50 token overlap)
- Add metadata extraction (source, page, category)
- Create CLI: `python -m dragonwizard.rag process-docs ./data/srd/`
- Write tests for chunking logic

**Testable Outcome**:
```bash
$ python -m dragonwizard.rag process-docs ./data/srd/
Processing: combat.md
  Created 45 chunks
Processing: spells.md
  Created 203 chunks
Total: 1,247 chunks
Output: ./data/processed/chunks.json
```

**Why Now**:
- No API costs during development
- Can inspect chunk quality manually
- Tests document processing logic independently

---

### 3.2 Vector Database & Embeddings
**Rationale**: Build indexing and retrieval independently

**Tasks**:
- Install ChromaDB and Sentence Transformers
- Download all-MiniLM-L6-v2 model (happens automatically on first use)
- Create `VectorStore` class
- Implement local embedding generation with Sentence Transformers
- Build indexing pipeline: chunks → embeddings → ChromaDB
- Implement retrieval: query → embedding → top-k chunks
- Create CLI: `python -m dragonwizard.rag index` and `python -m dragonwizard.rag search "advantage rules"`

**Testable Outcome**:
```bash
$ python -m dragonwizard.rag index
Loading embedding model: all-MiniLM-L6-v2...
Model loaded (90MB, 384 dimensions)
Indexing 1,247 chunks...
Generated embeddings: 1,247/1,247 (batch_size=32)
Indexed to ChromaDB: ✓
Time: ~30-60 seconds on CPU

$ python -m dragonwizard.rag search "how does advantage work"
Top 5 results:
1. [Score: 0.89] Advantage and Disadvantage (PHB p.173)
   "When you have advantage on a roll, roll two d20s..."
2. [Score: 0.82] Combat: Attack Rolls (PHB p.194)
   "If circumstances cause a roll to have both advantage..."
...
```

**Why Now**:
- Tests retrieval quality before LLM integration
- Can iterate on chunking strategy based on results
- Validates local embeddings work (no API dependency)
- Can manually assess relevance
- Model downloads and caches automatically

---

### 3.3 RAG + LLM Integration
**Rationale**: Combine retrieval with LLM for complete QA pipeline

**Tasks**:
- Create `RAGEngine` class that orchestrates retrieval + LLM
- Implement prompt construction with retrieved context
- Add citation extraction and formatting
- Create CLI: `python -m dragonwizard ask "How does advantage work?"`
- Test response quality and citation accuracy

**Testable Outcome**:
```bash
$ python -m dragonwizard ask "How does advantage work?"
**Advantage** means you roll two d20s and use the higher result.

When you have advantage on a roll, roll 2d20 and use the higher number.
For example, if you roll 18 and 7, you use 18.

This typically occurs when circumstances give you an edge, such as
attacking a prone enemy or having the Help action used on you.

📖 Source: Player's Handbook, p. 173
```

**Why Now**:
- Complete core functionality testable without Discord
- Can iterate on prompt engineering
- Validates entire RAG pipeline
- Establishes response quality baseline

---

## Phase 4: Query Processing Layer

### 4.1 Intent Classification & Query Enrichment
**Rationale**: Add intelligence layer before Discord to test independently

**Tasks**:
- Create `QueryProcessor` class
- Implement intent classification (rules lookup vs. general vs. combat scenario)
- Add query validation and sanitization
- Create query enrichment (future: context injection)
- Add CLI: `python -m dragonwizard process "explain sneak attack"`

**Testable Outcome**:
```bash
$ python -m dragonwizard process "explain sneak attack"
Intent: RULES_LOOKUP
Enriched Query: "What are the rules for sneak attack in D&D 5e?"
Classification confidence: 0.94
```

**Why Now**:
- Tests query processing logic without Discord noise
- Can tune classification before production use
- Establishes routing patterns for future features

---

## Phase 5: Discord Bot Layer

### 5.1 Basic Discord Bot (Echo/Ping)
**Rationale**: Test Discord integration with simplest possible bot

**Tasks**:
- Install discord.py
- Create `DiscordBot` class
- Implement bot authentication and connection
- Add basic command: `!ping` responds with "Pong!"
- Add message listener that echoes mentions
- Test in private Discord server

**Testable Outcome**:
```
User: !ping
Bot: Pong! 🏓

User: @DragonWizard hello
Bot: You said: hello
```

**Why Now**:
- Validates Discord API token and permissions
- Tests bot can connect and respond
- No complex logic to debug

---

### 5.2 Discord + RAG Integration
**Rationale**: Connect all pieces into working end-to-end system

**Tasks**:
- Integrate `RAGEngine` into Discord message handler
- Add command: `!ask <question>` or mention-based queries
- Implement response formatting for Discord (markdown, length limits)
- Add error handling and user-friendly error messages
- Handle rate limiting (Discord and LLM APIs)
- Add typing indicator while processing

**Testable Outcome**:
```
User: @DragonWizard How does advantage work?
Bot: [typing...]
Bot: **Advantage** means you roll two d20s and use the higher result.

When you have advantage on a roll, roll 2d20 and use the higher number...

📖 Source: Player's Handbook, p. 173
```

**Why Now**:
- First fully functional end-to-end system
- All components tested individually first
- Can deploy MVP and gather feedback

---

### 5.3 Discord + Tool Use Integration
**Rationale**: Add dice rolling to Discord responses

**Tasks**:
- Enable tool calling in Discord message flow
- Format tool results nicely for Discord
- Add examples to help LLM know when to roll dice
- Test with queries like "Show me an advantage roll"

**Testable Outcome**:
```
User: @DragonWizard show me an example advantage roll
Bot: When you have advantage, you roll 2d20 and take the higher result.
     Let me demonstrate:

     🎲 Rolling 2d20kh1: [15, 8] → 15

     In this example, you'd use 15 as your result.
```

**Why Complete MVP**:
- Full MVP functionality delivered
- All architecture layers implemented
- Ready for user testing and feedback

---

## Phase 6: Enhancement & Polish

### 6.1 Better Error Handling & Resilience
**Tasks**:
- Add retry logic for transient API failures
- Implement graceful degradation (RAG fails → use LLM only)
- Add user-friendly error messages
- Implement logging for debugging production issues
- Add health check endpoint/command

---

### 6.2 Conversation Memory (Session Context)
**Tasks**:
- Add in-memory session storage (user_id → conversation history)
- Implement context window management (sliding window)
- Add multi-turn clarification support
- Create session timeout and cleanup

---

### 6.3 Improved Citations & Formatting
**Tasks**:
- Add page numbers and section references to citations
- Implement multiple source aggregation
- Create Discord embeds for richer formatting
- Add confidence indicators for uncertain answers

---

### 6.4 Expanded Document Corpus
**Tasks**:
- Add more SRD content (spells, monsters, items)
- Include community FAQs and errata
- Create update pipeline for new content
- Version control for rule sources

---

## Phase 7: Future Extensibility

### 7.1 Context Provider System
**Tasks**:
- Create `ContextProvider` abstract base class
- Implement plugin registry
- Add configuration for enabling/disabling providers

---

### 7.2 Character Sheet Integration
**Tasks**:
- Design character sheet schema
- Implement storage (SQLite or JSON)
- Create CRUD commands (`!character create`, `!character show`)
- Add `CharacterContextProvider` to inject character data into queries

---

### 7.3 Campaign-Specific Rules
**Tasks**:
- Create homebrew rule storage
- Implement rule override system
- Add per-server configuration
- Create admin commands for managing rules

---

## Testing Strategy by Phase

| Phase | Unit Tests | Integration Tests | Manual Testing |
|-------|-----------|-------------------|----------------|
| 1.1 Setup | Config loading | - | CLI commands work |
| 1.2 MCP Dice | Mock MCP responses | Live MCP server | CLI dice rolling |
| 2.1 LLM Basic | Mock API responses | Live API calls | CLI queries |
| 2.2 LLM Tools | Mock tool calls | Live LLM + MCP | CLI with tools |
| 3.1 Docs | Chunking logic | - | Inspect chunk quality |
| 3.2 Vector DB | Embedding generation | ChromaDB operations | Search quality |
| 3.3 RAG + LLM | Prompt construction | Full RAG pipeline | Response quality |
| 4.1 Query Processing | Intent classification | - | CLI processing |
| 5.1 Discord Basic | Message parsing | Live Discord bot | Discord commands |
| 5.2 Discord + RAG | Response formatting | Full E2E | Discord Q&A |
| 5.3 Discord + Tools | Tool result formatting | Full E2E with tools | Discord dice rolls |

---

## Success Criteria by Phase

### Phase 1 Complete:
- ✅ Project runs with dependencies installed
- ✅ Can roll dice via CLI using MCP server

### Phase 2 Complete:
- ✅ Can query LLM via CLI
- ✅ LLM can call dice rolling tool

### Phase 3 Complete:
- ✅ Can search indexed documents
- ✅ Local embeddings working (no API needed)
- ✅ Can get RAG-enhanced answers via CLI
- ✅ Citations are accurate

### Phase 4 Complete:
- ✅ Query intent classification works
- ✅ Queries are sanitized and enriched

### Phase 5 Complete (MVP):
- ✅ Bot responds to Discord messages
- ✅ Bot answers D&D questions with RAG
- ✅ Bot can roll dice when appropriate
- ✅ Deployed and usable in Discord server

---

## Estimated Complexity by Phase

| Phase | Complexity | New Dependencies | External APIs |
|-------|-----------|------------------|---------------|
| 1.1 Setup | Low | Python 3.13+, uv | - |
| 1.2 MCP Dice | Low | MCP SDK | MCP dice server |
| 2.1 LLM Basic | Medium | Anthropic SDK | Claude API |
| 2.2 LLM Tools | Medium | - | Claude + MCP |
| 3.1 Docs | Medium | - | - |
| 3.2 Vector DB | High | ChromaDB, Sentence Transformers, PyTorch | None (local) |
| 3.3 RAG + LLM | Medium | - | Claude only |
| 4.1 Query Processing | Low | - | - |
| 5.1 Discord Basic | Medium | discord.py | Discord API |
| 5.2 Discord + RAG | Low | - | Claude + Discord |
| 5.3 Discord + Tools | Low | - | All above |

---

## Key Advantages of This Order

1. **Early Risk Reduction**: External dependencies (MCP, LLM API, Discord API) tested early
2. **Independent Testing**: Each layer has CLI interface for isolated testing
3. **Incremental Integration**: Combine components only after individual validation
4. **Fast Feedback**: Can test RAG quality before building Discord bot
5. **Flexible Development**: Can pause at any phase and have working components
6. **Cost Management**: Local embeddings = zero cost; only LLM API calls cost money
7. **Privacy**: All embeddings generated locally, no data sent to third parties
8. **Debugging Simplification**: Issues isolated to single layer, not tangled across system

---

## Alternative Approaches Considered

### ❌ Top-Down (Discord First)
**Rejected because**:
- Can't test LLM/RAG quality without Discord noise
- Harder to debug when all layers involved
- Discord rate limits slow development iteration

### ❌ Horizontal Layers (All infrastructure first)
**Rejected because**:
- No working features until very end
- Risk of over-engineering unused components
- Can't validate architecture decisions early

### ✅ Bottom-Up with Vertical Slices (Chosen)
**Advantages**:
- Working software at each milestone
- Early validation of risky integrations
- CLI tools useful for debugging and testing
- Can deploy partial system if needed

---

## Next Steps

To begin implementation:

1. Review this plan with stakeholders
2. Set up development environment (Phase 1.1)
3. Ensure system requirements:
   - Python 3.13+
   - 4GB+ RAM (for local embeddings)
   - 4+ CPU cores recommended
   - ~500MB disk space (models + data)
4. Obtain necessary API keys:
   - Anthropic Claude API key
   - Discord bot token
   - ~~OpenAI API key~~ (NOT needed - using local embeddings)
5. Start with Phase 1.1 project setup
6. Work through phases sequentially, testing thoroughly at each step

Each phase should be committed to version control separately, with working tests and documentation updated accordingly.
