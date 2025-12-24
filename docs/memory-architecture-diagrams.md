# Memory System Architecture Diagrams

## 1. Repository Relationships

```
                                    ┌─────────────────────────┐
                                    │     USER / AGENT        │
                                    │   (Claude, GPT, etc.)   │
                                    └───────────┬─────────────┘
                                                │
                                    ┌───────────▼─────────────┐
                                    │    MCP Client           │
                                    │  (mcp-remote, IDE)      │
                                    └───────────┬─────────────┘
                                                │ MCP Protocol
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
        ┌───────────▼───────────┐   ┌───────────▼───────────┐   ┌───────────▼───────────┐
        │                       │   │                       │   │                       │
        │  af-component-fastmcp │   │     mcp-gateway       │   │  agentic-application  │
        │      -backend         │   │                       │   │      -template        │
        │                       │   │                       │   │                       │
        │  DataRobot Platform   │   │  Standalone Deploy    │   │  Recipe/Template      │
        │  Custom Model Deploy  │   │  DataRobot SDK        │   │  User Customization   │
        │                       │   │                       │   │                       │
        └───────────┬───────────┘   └───────────┬───────────┘   └───────────┬───────────┘
                    │                           │                           │
                    │     imports               │     imports               │     imports
                    │                           │                           │
                    └───────────────────────────┼───────────────────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │                       │
                                    │   datarobot-genai     │
                                    │                       │
                                    │   Core MCP Server     │
                                    │   Tools Library       │
                                    │   Resource Store      │
                                    │   Memory API Base     │
                                    │                       │
                                    └───────────────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │                       │
                                    │  recipe-fastmcp       │
                                    │     -template         │
                                    │                       │
                                    │   FastMCP Framework   │
                                    │   MCP Protocol        │
                                    │                       │
                                    └───────────────────────┘
```

## 2. Memory System Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              MCP SERVER                                              │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          MCP TOOLS LAYER (Jeremy)                               │ │
│  │                                                                                 │ │
│  │   ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐  │ │
│  │   │memory_write │  │memory_read  │  │memory_search │  │memory_summarize     │  │ │
│  │   │             │  │             │  │              │  │                     │  │ │
│  │   │ @mcp_tool() │  │ @mcp_tool() │  │ @mcp_tool()  │  │ @mcp_tool()         │  │ │
│  │   └──────┬──────┘  └──────┬──────┘  └───────┬──────┘  └──────────┬──────────┘  │ │
│  │          │                │                 │                    │             │ │
│  └──────────┼────────────────┼─────────────────┼────────────────────┼─────────────┘ │
│             │                │                 │                    │               │
│             ▼                ▼                 ▼                    ▼               │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                         MEMORY API LAYER (Memory Team)                          │ │
│  │                                                                                 │ │
│  │   ┌─────────────────────────────────────────────────────────────────────────┐  │ │
│  │   │                      MemoryAPI (Extended)                                │  │ │
│  │   │                                                                         │  │ │
│  │   │  • add(messages, user_id, agent_id, metadata)                          │  │ │
│  │   │  • search(query, user_id, filters, top_k)                              │  │ │
│  │   │  • get_all(user_id, agent_id)                                          │  │ │
│  │   │  • update(memory_id, data)                                             │  │ │
│  │   │  • delete(memory_id)                                                   │  │ │
│  │   │  • summarize(user_id, strategy)                                        │  │ │
│  │   │  • history(memory_id)                                                  │  │ │
│  │   │                                                                         │  │ │
│  │   └─────────────────────────────────┬───────────────────────────────────────┘  │ │
│  │                                     │                                          │ │
│  └─────────────────────────────────────┼──────────────────────────────────────────┘ │
│                                        │                                            │
│                                        ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                    AGENTIC RESOURCE SYSTEM (ARS) - This PR                      │ │
│  │                                                                                 │ │
│  │   ┌──────────────────────┐    ┌──────────────────────────────────────────────┐ │ │
│  │   │   ResourceStore      │    │     ResourceStoreBackedResourceManager       │ │ │
│  │   │                      │    │                                              │ │ │
│  │   │  • put()             │◄───│  • Extends FastMCP ResourceManager           │ │ │
│  │   │  • get()             │    │  • Hooks into mcp.add_resource()             │ │ │
│  │   │  • query()           │    │  • Enables MCP resource discovery            │ │ │
│  │   │  • delete()          │    │                                              │ │ │
│  │   └──────────┬───────────┘    └──────────────────────────────────────────────┘ │ │
│  │              │                                                                  │ │
│  └──────────────┼──────────────────────────────────────────────────────────────────┘ │
│                 │                                                                    │
└─────────────────┼────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE BACKENDS (Memory Team)                             │
│                                                                                      │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐    │
│   │  Filesystem   │   │      S3       │   │   Postgres    │   │    Redis      │    │
│   │               │   │               │   │   + pgvector  │   │   (cache)     │    │
│   │  Local dev    │   │  Production   │   │  Vector search│   │  Hot data     │    │
│   │  Testing      │   │  Blob storage │   │  Structured   │   │  Sessions     │    │
│   └───────────────┘   └───────────────┘   └───────────────┘   └───────────────┘    │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

## 3. Memory Lifecycle & Jobs

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            MEMORY LIFECYCLE                                          │
│                                                                                      │
│                                                                                      │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐                 │
│    │  CREATE  │────▶│  ACTIVE  │────▶│  STALE   │────▶│ ARCHIVED │                 │
│    └──────────┘     └──────────┘     └──────────┘     └──────────┘                 │
│         │                │                │                │                        │
│         │                │                │                │                        │
│         ▼                ▼                ▼                ▼                        │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐                 │
│    │ Generate │     │  Access  │     │Summarize │     │  Delete  │                 │
│    │Embedding │     │  Update  │     │Consolidate│    │  Cleanup │                 │
│    └──────────┘     └──────────┘     └──────────┘     └──────────┘                 │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              JOB SYSTEM                                              │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          Job Scheduler                                       │   │
│   │                     (APScheduler / Celery)                                   │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                              │
│           ┌──────────────────────────┼──────────────────────────┐                   │
│           │                          │                          │                   │
│           ▼                          ▼                          ▼                   │
│   ┌───────────────┐          ┌───────────────┐          ┌───────────────┐          │
│   │ Summarization │          │ Consolidation │          │    Cleanup    │          │
│   │     Job       │          │     Job       │          │     Job       │          │
│   │               │          │               │          │               │          │
│   │ • Trigger:    │          │ • Trigger:    │          │ • Trigger:    │          │
│   │   - Time      │          │   - Threshold │          │   - TTL       │          │
│   │   - Count     │          │   - Similarity│          │   - Schedule  │          │
│   │   - Manual    │          │               │          │               │          │
│   │               │          │ • Action:     │          │ • Action:     │          │
│   │ • Action:     │          │   - Merge     │          │   - Delete    │          │
│   │   - LLM call  │          │   - Dedupe    │          │   - Archive   │          │
│   │   - Compress  │          │               │          │               │          │
│   └───────────────┘          └───────────────┘          └───────────────┘          │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 4. mem0 Compatibility Layer

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         AGENT FRAMEWORK INTEGRATION                                  │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                    Agent Frameworks                                          │   │
│   │                                                                              │   │
│   │  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────────┐  │   │
│   │  │ CrewAI  │  │LangGraph │  │LlamaIndex │  │   NAT    │  │  Custom/mem0  │  │   │
│   │  └────┬────┘  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └───────┬───────┘  │   │
│   │       │            │              │             │                │          │   │
│   └───────┼────────────┼──────────────┼─────────────┼────────────────┼──────────┘   │
│           │            │              │             │                │              │
│           ▼            ▼              ▼             ▼                ▼              │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                       Framework Adapters                                     │   │
│   │                                                                              │   │
│   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │   │
│   │  │ResourceStore│ │ResourceStore│ │ResourceStore│ │ResourceStore│            │   │
│   │  │   Storage   │ │ Checkpoint  │ │  ChatStore  │ │MemoryEditor │            │   │
│   │  │  (CrewAI)   │ │   Saver     │ │MemoryBlock  │ │   (NAT)     │            │   │
│   │  │             │ │ (LangGraph) │ │(LlamaIndex) │ │             │            │   │
│   │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘            │   │
│   │         │               │               │               │                    │   │
│   └─────────┼───────────────┼───────────────┼───────────────┼────────────────────┘   │
│             │               │               │               │                        │
│             └───────────────┴───────────────┴───────────────┘                        │
│                                     │                                                │
│                                     ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          MemoryAPI                                           │   │
│   │                    (mem0-compatible interface)                               │   │
│   │                                                                              │   │
│   │      add() | search() | get() | update() | delete() | history()             │   │
│   │                                                                              │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                                │
│                                     ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                       ResourceStore                                          │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

## 5. MCP Resource Tracking for Tracing

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        TRACING & LINEAGE FLOW                                        │
│                                                                                      │
│                                                                                      │
│    Agent Request                                                                     │
│         │                                                                            │
│         ▼                                                                            │
│   ┌───────────────┐                                                                  │
│   │  MCP Tool     │──────────────────────┐                                          │
│   │  Invocation   │                      │                                          │
│   └───────┬───────┘                      │                                          │
│           │                              │                                          │
│           │ OpenTelemetry Span           │                                          │
│           │ [tool.memory_write]          │                                          │
│           │                              │                                          │
│           ▼                              ▼                                          │
│   ┌───────────────┐              ┌───────────────┐                                  │
│   │  MemoryAPI    │──────────────│   Telemetry   │                                  │
│   │  Operation    │              │   Middleware  │                                  │
│   └───────┬───────┘              └───────────────┘                                  │
│           │                              │                                          │
│           │ Span [memory.write]          │                                          │
│           │                              │                                          │
│           ▼                              ▼                                          │
│   ┌───────────────┐              ┌───────────────┐                                  │
│   │ ResourceStore │              │  MCP Resource │                                  │
│   │    put()      │──────────────│  Registration │                                  │
│   └───────┬───────┘              └───────┬───────┘                                  │
│           │                              │                                          │
│           │ Span [store.put]             │ Resource ID linked                       │
│           │                              │ to span                                  │
│           ▼                              ▼                                          │
│   ┌───────────────┐              ┌───────────────┐                                  │
│   │   Backend     │              │  Discoverable │                                  │
│   │   Storage     │              │  via MCP      │                                  │
│   └───────────────┘              │ list_resources│                                  │
│                                  └───────────────┘                                  │
│                                                                                      │
│   Result:                                                                            │
│   • Full trace: Agent → Tool → Memory → Store → Backend                             │
│   • Resource discoverable via MCP protocol                                          │
│   • Lineage: Which tool created which memory                                        │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

## 6. Suggested Package Structure (mcp-gateway)

```
mcp-gateway/
├── pyproject.toml
├── README.md
│
└── src/
    └── mcp_gateway/
        ├── __init__.py
        ├── server.py                 # Extends DataRobotMCPServer
        │
        ├── memory/                   # ← MEMORY TEAM'S DOMAIN
        │   ├── __init__.py
        │   │
        │   ├── api.py               # Extended MemoryAPI
        │   │   │
        │   │   ├── class MemoryClient:
        │   │   │   ├── add()
        │   │   │   ├── search()
        │   │   │   ├── get_all()
        │   │   │   ├── get()
        │   │   │   ├── update()
        │   │   │   ├── delete()
        │   │   │   ├── delete_all()
        │   │   │   └── history()
        │   │   │
        │   │   └── class MemoryConfig:
        │   │       ├── backend_type
        │   │       ├── embedding_model
        │   │       └── job_settings
        │   │
        │   ├── backends/            # Storage implementations
        │   │   ├── __init__.py
        │   │   ├── base.py          # Extended backend interface
        │   │   ├── s3.py            # S3/MinIO
        │   │   ├── postgres.py      # PostgreSQL + pgvector
        │   │   ├── redis.py         # Redis cache layer
        │   │   └── composite.py     # Multi-tier (Redis → Postgres → S3)
        │   │
        │   ├── embeddings/          # Embedding generation
        │   │   ├── __init__.py
        │   │   ├── openai.py
        │   │   ├── sentence_transformers.py
        │   │   └── local.py
        │   │
        │   ├── jobs/                # Background jobs
        │   │   ├── __init__.py
        │   │   ├── scheduler.py     # Job scheduler (APScheduler)
        │   │   ├── summarization.py # Auto-summarize
        │   │   ├── consolidation.py # Merge similar
        │   │   ├── cleanup.py       # TTL cleanup
        │   │   └── embedding.py     # Background embedding
        │   │
        │   ├── search/              # Search implementations
        │   │   ├── __init__.py
        │   │   ├── semantic.py      # Vector similarity
        │   │   ├── keyword.py       # Full-text search
        │   │   └── hybrid.py        # Combined ranking
        │   │
        │   └── mem0_compat.py       # mem0 drop-in replacement
        │
        └── config.py                # Gateway configuration
```

## 7. Interface Contracts

```python
# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY API INTERFACE (mem0-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryClient(Protocol):
    """mem0-compatible memory interface."""
    
    async def add(
        self,
        messages: list[dict],          # [{"role": "user", "content": "..."}]
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict | None = None,
        filters: dict | None = None,
    ) -> dict:
        """Add memories from messages. Returns {"results": [...]}"""
        ...
    
    async def search(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """Search memories. Returns [{"id": ..., "memory": ..., "score": ...}]"""
        ...
    
    async def get_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get all memories for user/agent."""
        ...
    
    async def get(self, memory_id: str) -> dict | None:
        """Get specific memory by ID."""
        ...
    
    async def update(self, memory_id: str, data: str) -> dict:
        """Update memory content."""
        ...
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        ...
    
    async def delete_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Delete all memories. Returns count deleted."""
        ...
    
    async def history(self, memory_id: str) -> list[dict]:
        """Get memory version history."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND INTERFACE (extends ResourceBackend)
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryBackend(ResourceBackend, Protocol):
    """Extended backend interface for memory operations."""
    
    async def vector_search(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
        score_threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Semantic search. Returns [(resource_id, score), ...]"""
        ...
    
    async def get_by_scope(
        self,
        scope_id: str,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> list[Resource]:
        """Efficient scope-based queries."""
        ...
    
    async def bulk_put(
        self,
        items: list[tuple[Scope, str, bytes, dict]],
    ) -> list[str]:
        """Batch write. Returns list of resource IDs."""
        ...
    
    async def count(
        self,
        scope_id: str | None = None,
        kind: str | None = None,
    ) -> int:
        """Count resources matching criteria."""
        ...
```

---

These diagrams should help the memory team understand:
1. Where their code fits in the overall system
2. How data flows through the layers
3. What interfaces they need to implement
4. How to maintain separation for future extraction
