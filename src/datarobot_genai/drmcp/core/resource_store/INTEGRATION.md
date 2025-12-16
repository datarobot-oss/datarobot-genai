# ResourceStore Integration with FastMCP Resources

## Overview

The ResourceStore system provides unified storage for conversation state, memory, and MCP resources. This document explains how it integrates with FastMCP's Resource classes by **extending FastMCP's ResourceManager** rather than creating a separate integration layer.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastMCP MCP Server                        │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         MCP Protocol Handlers                        │  │
│  │  • list_resources()                                  │  │
│  │  • read_resource(uri)                                │  │
│  └──────────────────────────────────────────────────────┘  │
│                        ↕                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         FastMCP Resource Classes                    │  │
│  │  • HttpResource (uri, url, name, mime_type)         │  │
│  │  • mcp.add_resource(resource)                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                        ↕                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ResourceStoreBackedResourceManager                  │  │
│  │  (extends FastMCP's ResourceManager)                 │  │
│  │  • add_resource() - stores in ResourceStore          │  │
│  │  • get_resource_data() - reads from ResourceStore    │  │
│  │  • list_resources_for_scope() - queries ResourceStore│  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────────────────┐
│                    ResourceStore                            │
│  • Unified storage backend (FilesystemBackend, etc.)       │
│  • Stores: conversation state, memory, resources           │
│  • Provides: put(), get(), query(), delete()              │
└─────────────────────────────────────────────────────────────┘
```

## How They Work Together

### 1. FastMCP Resources (MCP Protocol Layer)

FastMCP provides:
- **`HttpResource`**: A class representing an MCP resource with:
  - `uri`: Unique identifier (e.g., `"mcp://resources/abc123"`)
  - `url`: Accessible URL (can be external or internal)
  - `name`: Human-readable name
  - `mime_type`: Content type

- **`mcp.add_resource(resource)`**: Registers a resource with the MCP server, making it discoverable via `list_resources`

- **Built-in handlers**: FastMCP automatically handles MCP protocol methods:
  - `list_resources`: Returns all registered resources
  - `read_resource(uri)`: Retrieves resource content by URI

### 2. ResourceStore (Storage Backend)

ResourceStore provides:
- **Unified storage**: Single backend for all resource types
- **Scoped storage**: Resources organized by scope (conversation, memory, resource)
- **Metadata**: Rich metadata support (tags, embeddings, etc.)
- **Lifetime management**: Ephemeral vs persistent resources with TTL

### 3. Extended ResourceManager (`ResourceStoreBackedResourceManager`)

We **extend FastMCP's ResourceManager** to use ResourceStore as the backend.

**Automatic Initialization:**

ResourceStore is automatically initialized during `DataRobotMCPServer.__init__`. The server:
1. Creates a `FilesystemBackend` with the configured storage path
2. Creates a `ResourceStore` with the backend
3. Creates a `ResourceStoreBackedResourceManager` with the store
4. Replaces FastMCP's `_resource_manager` with our ResourceStore-backed one

This means all calls to `mcp.add_resource()` automatically use the ResourceStore backend.

**Accessing the ResourceManager:**

Tools that need the ResourceManager can access it directly:

```python
from datarobot_genai.drmcp.core.mcp_instance import mcp

# Access the ResourceStore-backed ResourceManager
resource_manager = mcp._resource_manager

# Access the underlying ResourceStore
store = mcp._resource_manager.store
```

**When storing a resource:**
```python
from datarobot_genai.drmcp.core.mcp_instance import mcp
from fastmcp.resources import HttpResource
import uuid

# Tool creates large output
data = generate_large_csv()

# Create HttpResource
resource_id = str(uuid.uuid4())
resource = HttpResource(
    uri=f"mcp://resources/{resource_id}",
    url=f"mcp://resources/{resource_id}",
    name="Prediction Results",
    mime_type="text/csv",
)

# This stores in ResourceStore AND registers with FastMCP
mcp._resource_manager.add_resource(
    resource,
    data=data,
    scope_id="conversation_123",
    lifetime="ephemeral",
    ttl_seconds=86400,
)

# Now:
# 1. Data is stored in ResourceStore (backend)
# 2. HttpResource is registered with FastMCP (via parent class)
# 3. Clients can discover it via list_resources
```

**Retrieving resource data:**
```python
# Get the data back from ResourceStore
data, content_type = await mcp._resource_manager.get_resource_data("abc123")
```

## Usage Patterns

### Pattern 1: Tool Creates Large Output

```python
from datarobot_genai.drmcp.core.mcp_instance import mcp
from fastmcp.resources import HttpResource
import uuid

@dr_mcp_tool()
async def predict_large_dataset(deployment_id: str) -> dict:
    # Generate large results
    results = await make_predictions(deployment_id)
    csv_data = results.to_csv()
    
    # Create HttpResource
    resource_id = str(uuid.uuid4())
    resource = HttpResource(
        uri=f"mcp://resources/{resource_id}",
        url=f"mcp://resources/{resource_id}",
        name=f"Predictions for {deployment_id}",
        mime_type="text/csv",
    )
    
    # Store and register (ResourceStoreBackedResourceManager stores in ResourceStore)
    mcp._resource_manager.add_resource(
        resource,
        data=csv_data,
        scope_id=f"deployment_{deployment_id}",
        lifetime="ephemeral",
        ttl_seconds=3600,  # 1 hour
    )
    
    # Return resource reference
    return {
        "status": "completed",
        "resource_id": resource_id,
        "resource_uri": resource.uri,
        "message": "Results available as MCP resource"
    }
```

### Pattern 2: Replacing Current S3-Based Resources

Currently, tools do:
```python
# Current approach (utils.py)
resource = HttpResource(
    uri="predictions://" + uuid.uuid4().hex,
    url=s3_url,  # External S3 URL
    name=resource_name,
    mime_type="text/csv",
)
mcp.add_resource(resource)
```

With ResourceStore:
```python
# New approach - data stored in ResourceStore, not S3
from datarobot_genai.drmcp.core.mcp_instance import mcp

resource = HttpResource(
    uri=f"mcp://resources/{resource_id}",
    url=f"mcp://resources/{resource_id}",  # Points to ResourceStore, not S3
    name=resource_name,
    mime_type="text/csv",
)
mcp._resource_manager.add_resource(
    resource,
    data=csv_data,  # Store actual data in ResourceStore
    scope_id=conversation_id,
    lifetime="ephemeral",
    ttl_seconds=86400,
)
```

### Pattern 3: Dynamic Resource Listing

```python
from datarobot_genai.drmcp.core.mcp_instance import mcp

# List all resources for a conversation
resources = await mcp._resource_manager.list_resources_for_scope("conversation_123")

# Convert to MCP format
mcp_resources = [
    {
        "uri": r.uri,
        "name": r.name,
        "mimeType": r.mime_type,
    }
    for r in resources
]
```

## Benefits

1. **Unified Storage**: All resources (conversation state, memory, MCP resources) use the same backend
2. **Flexible Backends**: Can swap FilesystemBackend → S3Backend → PostgresBackend without changing tool code
3. **Rich Metadata**: Store embeddings, tags, and other metadata with resources
4. **Lifetime Management**: Automatic cleanup of ephemeral resources
5. **MCP Protocol Compliance**: Still works with FastMCP's resource system
6. **No Boilerplate**: ResourceStore is automatically initialized, no setup needed in tools

## Key Advantages of Extending ResourceManager

1. **Clean Architecture**: Extends FastMCP's ResourceManager rather than creating parallel system
2. **Backward Compatible**: Tools can still use ResourceManager interface
3. **Automatic Setup**: Initialized during server startup, not a "side thing"
4. **Type Safety**: Same interface as FastMCP's ResourceManager
5. **Simple Access**: Just use `mcp._resource_manager` directly

## FastMCP Resource Protocol

FastMCP handles MCP resources automatically:
- When you call `mcp.add_resource(resource)`, FastMCP registers it
- MCP clients can call `list_resources` to discover all registered resources
- MCP clients can call `read_resource(uri)` to retrieve resource content
- FastMCP routes these calls to the appropriate handlers

ResourceStore enhances this by:
- Providing persistent storage (not just in-memory)
- Adding metadata and querying capabilities
- Supporting scoped resources (conversation, memory, etc.)
- Managing resource lifetimes (ephemeral vs persistent)
