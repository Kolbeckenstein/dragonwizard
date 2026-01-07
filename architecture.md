# DragonWizard Architecture

## Overview

DragonWizard is a Discord bot that provides accurate D&D 5th Edition rules answers to players and GMs. It leverages modern LLM technology with RAG (Retrieval-Augmented Generation) to deliver precise, context-aware responses and can invoke external tools like dice servers for practical examples.

## Design Principles

- **Extensible**: Built to support future features like character sheets and campaign context
- **Modular**: Clear separation between Discord interface, LLM logic, and external integrations
- **Simple First**: Start with core Q&A functionality, expand as needed
- **Cost-Effective**: API-based approach suitable for small-scale personal/group use

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Discord User                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Discord Bot Layer                         │
│  - Message handling & command parsing                        │
│  - User context extraction                                   │
│  - Response formatting                                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Query Processing Layer                     │
│  - Intent classification (rules vs. general)                 │
│  - Query enrichment & reformulation                          │
│  - Context management (future: session/character state)      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     RAG Engine Layer                         │
│  ┌─────────────────┐  ┌──────────────────┐                 │
│  │  Vector Store   │  │  Document Store  │                 │
│  │  (embeddings)   │  │  (source texts)  │                 │
│  └────────┬────────┘  └────────┬─────────┘                 │
│           │                     │                            │
│           └─────────┬───────────┘                            │
│                     ▼                                        │
│           ┌──────────────────┐                              │
│           │ Retrieval Engine │                              │
│           │  - Semantic search                              │
│           │  - Chunk ranking │                              │
│           └─────────┬────────┘                              │
└─────────────────────┼──────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Orchestration Layer                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              LLM API (Claude/GPT)                      │ │
│  │  - Receives query + retrieved context                  │ │
│  │  - Generates response with citations                   │ │
│  │  - Can trigger tool calls                              │ │
│  └────────────────────────┬───────────────────────────────┘ │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool Integration Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ MCP Dice     │  │  Future:     │  │  Future:     │     │
│  │ Server       │  │  Spell DB    │  │  Character   │     │
│  │              │  │              │  │  Sheet API   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Discord Bot Layer

**Technology**: discord.py or discord.js

**Responsibilities**:
- Listen for messages/commands in configured channels
- Parse user intent (mentions, slash commands, DMs)
- Extract context: user ID, channel ID, server ID
- Format and send responses with proper Discord markdown
- Handle rate limiting and error states

**Key Interfaces**:
```
on_message(message) -> process_query()
on_command(command, args) -> process_query()
send_response(channel, formatted_response)
```

### 2. Query Processing Layer

**Technology**: Python service module

**Responsibilities**:
- Classify query type (rules lookup, general D&D question, combat scenario)
- Enrich queries with context (future: current character, session state)
- Validate and sanitize input
- Route to appropriate handler

**Key Functions**:
```
classify_intent(query: str) -> QueryType
enrich_query(query: str, context: dict) -> EnrichedQuery
validate_query(query: str) -> bool
```

**Extensibility Points**:
- Context providers (future: CharacterContextProvider, CampaignContextProvider)
- Query enrichers (add homebrew rules, session memory)

### 3. RAG Engine Layer

**Technology**:
- Vector DB: ChromaDB (local), Pinecone, or Weaviate (if scaling)
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2) - local, no API required
- Document store: PostgreSQL or local file system

**Responsibilities**:
- Index D&D 5e SRD and supplemental materials
- Convert queries to embeddings
- Retrieve top-k relevant document chunks
- Rank and filter results by relevance

**Data Sources**:
- D&D 5e System Reference Document (SRD)
- Rules compendiums (legally obtained)
- Community-curated FAQs and errata
- Future: homebrew content, campaign-specific rules

**Key Operations**:
```
index_documents(documents: List[Document])
retrieve(query_embedding: Vector, k: int) -> List[Chunk]
rerank(chunks: List[Chunk], query: str) -> List[Chunk]
```

**Chunking Strategy**:
- Logical units: individual rules, spell descriptions, class features
- Overlap: 50-100 tokens between chunks to preserve context
- Metadata: source book, page number, rule category

### 4. LLM Orchestration Layer

**Technology**:
- Primary: Anthropic Claude API (Claude 3.5 Sonnet or Opus)
- Alternative: OpenAI GPT-4
- SDK: Anthropic Python SDK with tool use support

**Responsibilities**:
- Construct prompts with retrieved context
- Manage conversation history (if multi-turn)
- Handle tool calls (dice rolling, lookups)
- Extract and format citations
- Manage token budgets and context windows

**Prompt Structure**:
```
System: You are a D&D 5e expert assistant...
Context: [Retrieved rule chunks with source citations]
User Query: [Original question]
Instructions: Answer based on provided context, cite sources...
```

**Tool Definitions**:
- `roll_dice(expression: str)`: Call MCP dice server
- `lookup_spell(name: str)`: Retrieve spell details
- `calculate_modifier(score: int)`: Compute ability modifier

**Key Functions**:
```
generate_response(query: str, context: List[Chunk]) -> Response
handle_tool_call(tool_name: str, args: dict) -> ToolResult
format_with_citations(response: str, sources: List[Source]) -> str
```

### 5. Tool Integration Layer

**Technology**: MCP (Model Context Protocol) + custom adapters

**Current Tools**:
- **MCP Dice Server**: Roll dice expressions, return results
  - Input: `2d20kh1+5` (advantage roll with +5 modifier)
  - Output: `[15, 8] -> 15 + 5 = 20`

**Future Tools**:
- Spell database API
- Character sheet CRUD operations
- Initiative tracker
- Combat simulator

**Integration Pattern**:
```python
class ToolAdapter:
    async def call(self, tool_name: str, params: dict) -> dict:
        # Route to appropriate tool implementation
        # Handle errors and timeouts
        # Return structured result
```

---

## Data Flow

### Example: User asks "How does advantage work in D&D?"

1. **Discord Bot Layer**:
   - Receives message: "@DragonWizard How does advantage work?"
   - Extracts: `user_id`, `channel_id`, query text

2. **Query Processing Layer**:
   - Classifies as: `QueryType.RULES_LOOKUP`
   - Enriches: No additional context needed (future: check for house rules)

3. **RAG Engine Layer**:
   - Embeds query
   - Retrieves top 5 chunks:
     - "Advantage and Disadvantage" (PHB p.173)
     - "Rolling Dice" section
     - Related examples
   - Reranks by semantic relevance

4. **LLM Orchestration Layer**:
   - Constructs prompt with retrieved context
   - Calls Claude API
   - Receives response with explanation
   - LLM may trigger tool call: `roll_dice("2d20kh1")` for example

5. **Tool Integration Layer**:
   - Executes dice roll via MCP
   - Returns: `[18, 7] -> 18`

6. **LLM Orchestration Layer**:
   - Incorporates tool result into response
   - Formats with citations

7. **Discord Bot Layer**:
   - Formats response with Discord markdown
   - Sends to channel

**Final Response**:
```
**Advantage** means you roll two d20s and use the higher result.

When you have advantage on a roll, roll 2d20 and use the higher number.
For example, if you roll 18 and 7, you use 18.

This typically occurs when circumstances give you an edge, such as attacking
a prone enemy or having the Help action used on you.

📖 Source: Player's Handbook, p. 173
```

---

## Technology Stack

### Core Components

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Discord Bot | discord.py | Mature Python library, async support |
| Backend Language | Python 3.13+ | Rich ML/AI ecosystem, async capabilities |
| LLM API | Anthropic Claude | Strong reasoning, long context, tool use |
| Vector Database | ChromaDB | Simple local setup, SQLite-backed, sufficient for small scale |
| Document Store | JSON files / SQLite | Simple, version-controllable, adequate for SRD content |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) | Free, runs locally, no API key needed, 384-dim vectors, ~90MB model |
| MCP Integration | MCP Python SDK | Standard protocol for tool integration |

### Supporting Infrastructure

- **Environment Management**: uv (fast, modern Python package manager)
- **Configuration**: Pydantic Settings with TOML/YAML support
- **Logging**: Python logging + optional Discord webhook
- **Deployment**: Docker container, can run on small VPS or local machine (4GB+ RAM recommended)
- **Secrets**: Environment variables + `.env` file
- **Local ML**: PyTorch + Sentence Transformers for embeddings (CPU sufficient)

---

## Scalability Considerations

### Current Design (Small Scale)
- Single process handling all requests
- ChromaDB with local persistence
- Local Sentence Transformers embeddings (CPU-based, cached in memory)
- In-memory caching for frequently accessed rules
- Rate limiting per Discord's requirements
- Minimum hardware: 4GB RAM, 4-core CPU, ~500MB disk for models + data

### Future Scaling Options (If Needed)
- **Medium Scale** (10-50 servers):
  - Move to managed vector DB (Pinecone)
  - Add Redis for caching and session state
  - GPU acceleration for embeddings (optional)
  - Horizontal scaling with queue-based architecture

- **Large Scale** (100+ servers):
  - Microservices architecture
  - Dedicated RAG service with GPU
  - Consider cloud embeddings API (OpenAI/Cohere) for cost/performance trade-off
  - Load balancer for bot instances
  - Cloud hosting (AWS/GCP)

---

## Extensibility Architecture

### Plugin System for Future Features

**Context Providers** (future):
```python
class ContextProvider(ABC):
    @abstractmethod
    async def get_context(self, user_id: str, query: str) -> dict:
        pass

class CharacterContextProvider(ContextProvider):
    async def get_context(self, user_id: str, query: str) -> dict:
        # Fetch character sheet for user
        # Return relevant stats/features
        pass

class CampaignContextProvider(ContextProvider):
    async def get_context(self, user_id: str, query: str) -> dict:
        # Fetch campaign-specific rules
        # Return homebrew modifications
        pass
```

**Tool Registry**:
```python
class ToolRegistry:
    def register(self, tool: Tool):
        # Add tool to available set

    def get_tool(self, name: str) -> Tool:
        # Retrieve tool by name
```

### Configuration-Driven Behavior

```toml
[bot]
name = "DragonWizard"
command_prefix = "!"

[llm]
provider = "anthropic"
model = "claude-3-5-sonnet-20241022"
max_tokens = 1024
temperature = 0.3

[rag]
vector_db = "chromadb"
chunk_size = 512
chunk_overlap = 50
retrieval_k = 5

[features]
character_sheets = false  # Enable when ready
campaign_context = false  # Enable when ready
session_memory = false    # Enable when ready

[tools]
dice_server = "mcp://dice-roller"
# Future tools added here
```

---

## Security & Privacy

### Considerations

1. **User Data**: Currently only processing transient queries, no PII storage
2. **API Keys**: Stored in environment variables, never committed
3. **Rate Limiting**: Implement per-user rate limits to prevent abuse
4. **Content Filtering**: Validate queries to prevent prompt injection
5. **Future**: When storing character sheets, implement proper access controls

### Best Practices

- Use Discord's interaction tokens for authentication
- Sanitize all user input before processing
- Don't log sensitive data (API keys, user tokens)
- Implement timeouts for external API calls
- Monitor API usage and costs

---

## Development Phases

### Phase 1: MVP (Current Focus)
- ✅ Core bot framework
- ✅ Basic question answering with Claude API
- ✅ RAG pipeline with 5e SRD
- ✅ MCP dice server integration
- ✅ Discord response formatting

### Phase 2: Enhancement
- Improved citation formatting
- Conversation memory within session
- Multi-turn clarification
- Expanded document corpus
- Better error handling and fallbacks

### Phase 3: Extensibility
- Character sheet storage and lookup
- Campaign-specific rule overrides
- Session state management
- Additional tool integrations

### Phase 4: Polish
- Admin dashboard for configuration
- Usage analytics
- A/B testing for prompt improvements
- Community-contributed content integration

---

## Open Questions & Future Decisions

1. **Embeddings Strategy**: ✅ **DECIDED** - Using local Sentence Transformers
   - Benefits: Free, private, no API dependency, sufficient quality for D&D rules
   - Trade-off accepted: Slightly slower than cloud APIs, requires ~90MB disk + 2-4GB RAM

2. **Document Updating**: How to handle errata and new content?
   - Need versioning strategy for rule sources

3. **Multi-turn Conversations**: How much context to maintain?
   - Balance between coherence and cost (token usage)

4. **Homebrew Content**: Storage format and validation?
   - Need schema for user-contributed rules

5. **Tool Expansion**: What tools provide most value?
   - Prioritize based on user feedback

---

## Success Metrics

- **Accuracy**: % of responses that correctly cite rules
- **Latency**: Time from query to response (target: <5s for 90th percentile)
- **Cost**: API costs per 1000 queries
- **User Satisfaction**: Feedback from Discord users
- **Reliability**: Uptime and error rates

---

## References

- [D&D 5e SRD](https://dnd.wizards.com/resources/systems-reference-document)
- [Discord Bot Documentation](https://discord.com/developers/docs)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [RAG Best Practices](https://www.anthropic.com/index/claude-2-1-prompting)
