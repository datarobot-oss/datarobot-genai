# Memory Team Onboarding Guide

## Executive Summary

Welcome to the Memory API team! You'll be building the memory subsystem for DataRobot's MCP (Model Context Protocol) server. This document explains the overall architecture, how the repositories connect, and where your code should live.

**Your mission**: Build a production-ready Memory API that is:
- **mem0-compatible** for easy integration with agent frameworks
- **Pluggable** with multiple storage backends
- **Lifecycle-aware** with auto-summarization and memory management jobs
- **MCP-integrated** for tracing and lineage tracking
- **Separable** so it can be split into its own service later

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DEPLOYMENT LAYER                                        │
│                                                                                   │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐│
│  │ af-component-fastmcp │    │   mcp-gateway        │    │ agentic-application ││
│  │     -backend         │    │  (DataRobot SDK)     │    │    -template        ││
│  │                      │    │                      │    │                     ││
│  │ • Agentic Framework  │    │ • Standalone deploy  │    │ • Full app template ││
│  │ • Custom Models      │    │ • Extends datarobot  │    │ • Recipe-based      ││
│  │ • DR Platform        │    │   -genai             │    │                     ││
│  └──────────┬───────────┘    └──────────┬───────────┘    └──────────┬──────────┘│
│             │                           │                           │            │
│             └───────────────────────────┼───────────────────────────┘            │
│                                         ▼                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        datarobot-genai (THIS REPO)                               │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        DataRobotMCPServer                                    ││
│  │                                                                              ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────────────┐││
│  │  │   MCP Tools     │  │  MCP Resources  │  │      Memory API (NEW)        │││
│  │  │                 │  │                 │  │                              │││
│  │  │ • Predictive    │  │ • HttpResource  │  │  • MemoryAPI class           │││
│  │  │ • DataRobot SDK │  │ • Conversation  │  │  • Storage Backends          │││
│  │  │ • Memory Tools  │◄─┼─• Memory        │◄─┼──• Lifecycle Jobs            │││
│  │  │   (your tools)  │  │ • Artifacts     │  │  • mem0 compatibility        │││
│  │  └────────┬────────┘  └────────┬────────┘  └──────────────┬───────────────┘││
│  │           │                    │                          │                 ││
│  │           └────────────────────┼──────────────────────────┘                 ││
│  │                                ▼                                            ││
│  │  ┌─────────────────────────────────────────────────────────────────────────┐││
│  │  │             Agentic Resource System (ARS) - THIS PR                     │││
│  │  │                                                                         │││
│  │  │  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │││
│  │  │  │  ResourceStore   │  │ ResourceManager  │  │  Framework Adapters   │ │││
│  │  │  │                  │  │   (Extended)     │  │                       │ │││
│  │  │  │ • put/get/query  │  │ • MCP protocol   │  │ • CrewAI adapter      │ │││
│  │  │  │ • delete         │  │ • add_resource   │  │ • LangGraph adapter   │ │││
│  │  │  │ • scoped storage │  │ • list_resources │  │ • LlamaIndex adapter  │ │││
│  │  │  └────────┬─────────┘  └──────────────────┘  │ • NAT adapter         │ │││
│  │  │           │                                  └───────────────────────┘ │││
│  │  │           ▼                                                            │││
│  │  │  ┌─────────────────────────────────────────────────────────────────┐  │││
│  │  │  │               Pluggable Storage Backends                         │  │││
│  │  │  │                                                                  │  │││
│  │  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │  │││
│  │  │  │  │ Filesystem │  │    S3      │  │  Postgres  │  │   Redis    │ │  │││
│  │  │  │  │ (current)  │  │  (future)  │  │  (future)  │  │  (future)  │ │  │││
│  │  │  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │  │││
│  │  │  └─────────────────────────────────────────────────────────────────┘  │││
│  │  └─────────────────────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Map

### 1. `datarobot-genai` (This Repo) - Core Library
**Purpose**: The core MCP server and tools library

```
src/datarobot_genai/drmcp/
├── core/
│   ├── dr_mcp_server.py      # Main server class
│   ├── mcp_instance.py       # FastMCP instance & decorators
│   ├── config.py             # Configuration
│   ├── credentials.py        # Credentials management
│   │
│   └── resource_store/       # ← AGENTIC RESOURCE SYSTEM (ARS)
│       ├── store.py          # ResourceStore - unified storage
│       ├── backend.py        # Abstract backend interface
│       ├── models.py         # Scope, Resource, Lifetime models
│       ├── memory.py         # MemoryAPI - YOUR STARTING POINT
│       ├── resource_manager.py   # Extended FastMCP ResourceManager
│       ├── mcp_integration.py    # MCP protocol integration
│       │
│       ├── backends/         # Storage implementations
│       │   └── filesystem.py # Current filesystem backend
│       │
│       └── adapters/         # Framework integrations
│           ├── crewai_adapter.py
│           ├── langgraph_adapter.py
│           ├── llamaindex_adapter.py
│           └── nat_adapter.py
│
├── tools/                    # MCP Tools
│   ├── predictive/           # DataRobot ML tools
│   └── (memory_tools/)       # ← YOU WILL ADD memory tools here
│
└── test_utils/               # Testing utilities
```

### 2. `mcp-gateway` (Your Extension Point)
**Purpose**: Standalone deployment via DataRobot SDTK

```
mcp-gateway/
├── src/
│   └── mcp_gateway/
│       ├── __init__.py
│       ├── server.py         # Extends DataRobotMCPServer
│       │
│       └── memory/           # ← YOUR NEW CODE GOES HERE
│           ├── __init__.py
│           ├── api.py        # Extended MemoryAPI
│           ├── backends/     # Additional storage backends
│           │   ├── s3.py
│           │   ├── postgres.py
│           │   └── redis.py
│           ├── jobs/         # Background job system
│           │   ├── scheduler.py
│           │   ├── summarization.py
│           │   └── cleanup.py
│           └── tools/        # Memory-specific MCP tools
│               ├── write.py
│               ├── search.py
│               └── manage.py
│
├── pyproject.toml            # Depends on datarobot-genai
└── README.md
```

### 3. `af-component-fastmcp-backend`
**Purpose**: Agentic Framework deployment wrapper
- Packages the MCP server as a DataRobot Custom Model
- Used for platform deployments
- You likely won't modify this directly

### 4. `recipe-fastmcp-template` / `agentic-application-template`
**Purpose**: User-facing templates
- Show users how to build MCP servers
- Import from `datarobot-genai`
- Good reference for API design

---

## The Agentic Resource System (ARS) - What This PR Adds

### Core Concepts

```
┌─────────────────────────────────────────────────────────────────┐
│                         SCOPE                                    │
│  Organizes resources by context                                 │
│                                                                  │
│  Types:                                                         │
│  • "conversation" - Chat session resources                      │
│  • "memory"       - Persistent memory entries                   │
│  • "resource"     - Generic MCP resources                       │
│  • "custom"       - Framework-specific (e.g., "langgraph:xxx")  │
│                                                                  │
│  Example: Scope(type="memory", id="user_123")                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RESOURCE                                  │
│  The stored data unit                                           │
│                                                                  │
│  Fields:                                                        │
│  • id           - Unique identifier                             │
│  • scope        - Where it belongs                              │
│  • kind         - Type: "message", "note", "preference", etc.   │
│  • lifetime     - "ephemeral" or "persistent"                   │
│  • contentType  - MIME type                                     │
│  • metadata     - Tags, embeddings, custom fields               │
│  • contentRef   - Pointer to actual data                        │
│  • ttlSeconds   - Auto-cleanup time for ephemeral               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RESOURCE STORE                                 │
│  Unified storage API                                            │
│                                                                  │
│  Operations:                                                    │
│  • put(scope, kind, data, ...)  → Store data                    │
│  • get(resource_id)             → Retrieve data                 │
│  • query(scope, kind, metadata) → Search resources              │
│  • delete(resource_id)          → Remove resource               │
│                                                                  │
│  Pluggable Backend:                                             │
│  • FilesystemBackend (current)                                  │
│  • S3Backend, PostgresBackend, RedisBackend (you'll build)      │
└─────────────────────────────────────────────────────────────────┘
```

### Current MemoryAPI

We've laid the groundwork with a basic `MemoryAPI` class:

```python
# src/datarobot_genai/drmcp/core/resource_store/memory.py

class MemoryAPI:
    """Memory API for persistent storage using ResourceStore."""
    
    async def write(scope_id, kind, content, metadata) -> str:
        """Store a memory entry. Returns resource_id."""
    
    async def read(resource_id) -> dict:
        """Read a memory entry by ID."""
    
    async def search(scope_id, kind, metadata) -> list[dict]:
        """Search memory entries."""
    
    async def delete(resource_id) -> bool:
        """Delete a memory entry."""
```

**Your job**: Extend this into a production-ready Memory API.

---

## mem0 Compatibility

To integrate with agent frameworks, we may want to follow the [mem0](https://github.com/mem0ai/mem0) interface pattern:

```python
# mem0-compatible interface (what you should target)

class MemoryClient:
    def add(self, messages, user_id=None, agent_id=None, run_id=None, metadata=None):
        """Add memories from messages."""
    
    def search(self, query, user_id=None, agent_id=None, limit=10):
        """Search memories semantically."""
    
    def get_all(self, user_id=None, agent_id=None):
        """Get all memories for a user/agent."""
    
    def get(self, memory_id):
        """Get specific memory by ID."""
    
    def update(self, memory_id, data):
        """Update a memory."""
    
    def delete(self, memory_id):
        """Delete a memory."""
    
    def delete_all(self, user_id=None, agent_id=None):
        """Delete all memories for user/agent."""
    
    def history(self, memory_id):
        """Get memory history/versions."""
```

We've already built adapters for:
- **CrewAI** (`adapters/crewai_adapter.py`)
- **LangGraph** (`adapters/langgraph_adapter.py`)  
- **LlamaIndex** (`adapters/llamaindex_adapter.py`)
- **NVIDIA NAT** (`adapters/nat_adapter.py`)

---

## Work Division

### Team 1: Memory API Core (Jeremy - MCP Tools Owner)
I'll handle the MCP tools layer:
- `memory_write` tool
- `memory_read` tool
- `memory_search` tool
- `memory_delete` tool
- Tool registration and decorators

### Team 2: Memory API & Storage (Memory Team)
You'll handle:

#### Phase 1: Extended MemoryAPI (in `mcp-gateway`)
```
mcp-gateway/src/mcp_gateway/memory/
├── api.py                    # Extended MemoryAPI
│   ├── add()                 # mem0-compatible add
│   ├── search()              # Semantic search
│   ├── get_all()             # List all for user
│   ├── update()              # Update memory
│   ├── history()             # Version history
│   └── summarize()           # Trigger summarization
```

#### Phase 2: Storage Backends
```
mcp-gateway/src/mcp_gateway/memory/backends/
├── base.py                   # Abstract backend interface
├── s3.py                     # S3/MinIO storage
├── postgres.py               # PostgreSQL with pgvector
├── redis.py                  # Redis for caching layer
└── composite.py              # Multi-tier storage
```

**Backend Interface** (extend from `resource_store/backend.py`):
```python
class MemoryBackend(ResourceBackend):
    """Extended backend with memory-specific operations."""
    
    async def vector_search(self, embedding, top_k, filters) -> list:
        """Semantic similarity search."""
    
    async def get_by_user(self, user_id, limit, offset) -> list:
        """Efficient user-scoped queries."""
    
    async def bulk_write(self, items) -> list[str]:
        """Batch write optimization."""
```

#### Phase 3: Job System
```
mcp-gateway/src/mcp_gateway/memory/jobs/
├── scheduler.py              # Job scheduling (hook into the workflow api, or use covalent SDK directly)
├── summarization.py          # Auto-summarize old memories
├── consolidation.py          # Merge similar memories
├── cleanup.py                # TTL-based cleanup
└── embedding.py              # Background embedding generation
```

#### Phase 4: mem0 Adapter
```
mcp-gateway/src/mcp_gateway/memory/
└── mem0_adapter.py           # Drop-in mem0 replacement
```

---

## Integration Points with MCP

### Why Everything Goes Through MCP Resources

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRACING & LINEAGE                            │
│                                                                  │
│  Every memory operation creates an MCP Resource because:        │
│                                                                  │
│  1. TRACING: OpenTelemetry spans are attached to resources      │
│  2. LINEAGE: We can track: Tool → Memory → Resource → Backend   │
│  3. DISCOVERY: Clients can list_resources to see all memories   │
│  4. PROTOCOL: Standard MCP clients can read memory content      │
│                                                                  │
│  Flow:                                                          │
│  Agent calls memory_write tool                                  │
│       ↓                                                         │
│  Tool creates memory via MemoryAPI                              │
│       ↓                                                         │
│  MemoryAPI stores in ResourceStore                              │
│       ↓                                                         │
│  ResourceStore creates Resource + stores in backend             │
│       ↓                                                         │
│  Resource registered with MCP (discoverable via list_resources) │
│       ↓                                                         │
│  Telemetry span records the operation                           │
└─────────────────────────────────────────────────────────────────┘
```

### Resource Scoping for Memory

```python
# Memory uses scope.type = "memory"
# scope.id = user identifier

# Examples:
Scope(type="memory", id="user_123")           # User memories
Scope(type="memory", id="agent_456")          # Agent-specific
Scope(type="memory", id="session_789")        # Session-scoped
Scope(type="memory", id="global")             # Shared memories
```

---

## Getting Started Checklist

### Day 1: Environment Setup
- [ ] Clone `datarobot-genai` and `mcp-gateway`
- [ ] Run `task drmcp-unit` to verify tests pass
- [ ] Read `resource_store/INTEGRATION.md`
- [ ] Read `resource_store/memory.py` - understand current MemoryAPI

### Day 2: Architecture Deep Dive
- [ ] Trace a memory write: Tool → MemoryAPI → ResourceStore → Backend
- [ ] Understand the adapters in `resource_store/adapters/`
- [ ] Review how `ResourceStoreBackedResourceManager` extends FastMCP

### Day 3: First Code
- [ ] Create `mcp-gateway/src/mcp_gateway/memory/` directory structure
- [ ] Implement extended MemoryAPI skeleton
- [ ] Write first unit tests

### Week 1: Core Memory API
- [ ] Implement mem0-compatible interface
- [ ] Add user/agent scoping
- [ ] Implement search with filters

### Week 2: Storage Backends
- [ ] Design backend interface extension
- [ ] Implement PostgreSQL backend (recommended first)
- [ ] Add connection pooling and error handling

### Week 3: Job System
- [ ] Choose job framework (APScheduler for simple, Celery for distributed)
- [ ] Implement auto-summarization job
- [ ] Add cleanup job for ephemeral memories

### Week 4: Integration Testing
- [ ] End-to-end tests with real backends
- [ ] Performance benchmarks
- [ ] mem0 compatibility tests

---

## Code Placement Rules

| What | Where | Why |
|------|-------|-----|
| Core models (Scope, Resource) | `datarobot-genai/resource_store/models.py` | Shared foundation |
| Basic MemoryAPI | `datarobot-genai/resource_store/memory.py` | Core library |
| Framework adapters | `datarobot-genai/resource_store/adapters/` | Reusable |
| MCP memory tools | `datarobot-genai/tools/memory/` | Jeremy owns |
| Extended MemoryAPI | `mcp-gateway/memory/api.py` | Your domain |
| Storage backends | `mcp-gateway/memory/backends/` | Your domain |
| Job system | `mcp-gateway/memory/jobs/` | Your domain |
| mem0 adapter | `mcp-gateway/memory/mem0_adapter.py` | Your domain |

---

## Key Design Principles

### 1. Separation of Concerns
```
MCP Tools (interface) ←→ MemoryAPI (business logic) ←→ Backend (storage)
```

### 2. Backend Agnostic
```python
# Good: Backend is injected
memory_api = MemoryAPI(backend=PostgresBackend(connection_string))

# Bad: Hardcoded backend
memory_api = MemoryAPI()  # Uses filesystem internally
```

### 3. MCP Resource Tracking
```python
# Every memory operation should create a trackable resource
async def write(self, ...):
    resource = await self.store.put(...)
    # Resource is now discoverable via MCP list_resources
    return resource.id
```

### 4. mem0 Interface First
```python
# Design your API to match mem0, then adapt internally
class MemoryClient:
    def add(self, messages, user_id=None, ...):
        # Internally uses our MemoryAPI/ResourceStore
        pass
```

---

## Questions to Answer

As you design, consider:

1. **Scoping**: How do we handle multi-tenant memory? (user_id, org_id, project_id)
2. **Embedding**: Which embedding model? Where does it run? Async?
3. **Search**: Full-text vs semantic vs hybrid?
4. **Summarization**: LLM-based? Rule-based? When to trigger?
5. **Versioning**: Do we keep memory history? How long?
6. **Privacy**: How to handle PII in memories?
7. **Scale**: Expected memory volume per user? Query patterns?

---

## Contact Points

- **MCP Tools & Core**: Jeremy (keeps ownership of tool layer)
- **Memory API**: Your team
- **ARS Foundation**: This PR establishes the base

---

## Quick Reference

### Run Tests
```bash
task drmcp-unit                    # Unit tests
task drmcp-integration             # Integration tests
task fix-ruff                      # Lint & format
```

### Key Files to Read First
1. `resource_store/store.py` - ResourceStore class
2. `resource_store/memory.py` - Current MemoryAPI
3. `resource_store/backend.py` - Backend interface
4. `resource_store/adapters/crewai_adapter.py` - Example adapter
5. `core/dr_mcp_server.py` - How server initializes ResourceStore

### Import Pattern
```python
# In mcp-gateway, import from datarobot-genai
from datarobot_genai.drmcp.core.resource_store import (
    ResourceStore,
    MemoryAPI,
    Scope,
    Resource,
)
from datarobot_genai.drmcp.core.resource_store.backends.filesystem import (
    FilesystemBackend,
)
```

---

Welcome aboard! 🚀
