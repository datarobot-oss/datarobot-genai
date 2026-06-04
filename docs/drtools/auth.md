# Tool authentication (`drtools`)

`drtools` resolves secrets (DataRobot API tokens, third-party API keys, OAuth tokens) through a single
injection + resolution model:

1. **Runtime adapter** injects per-request data into `drtools.core.auth` (`set_request_headers`, `set_auth_context`).
2. **Tool code** calls resolvers (`resolve_datarobot_token`, `resolve_secret`, OAuth helpers).
3. **`auth_resolution_strategy`** (from `ToolsAuthCredentials`) chooses headers vs config.

Set the strategy via environment variable:

```bash
export AUTH_RESOLUTION_STRATEGY=http
```

| Strategy | Behavior |
|---|---|
| `http` (default) | Injected headers only |
| `config` | Config/env only (ignore headers) |

Config is defined in `datarobot_genai.drtools.core.credentials.ToolsAuthCredentials`.

## Headers tools expect

When `AUTH_RESOLUTION_STRATEGY=http` (the default), tools read:

| Secret | Header examples |
|---|---|
| DataRobot API token | `Authorization: Bearer <token>`, `x-datarobot-api-token` |
| Tavily | `x-tavily-api-key` |
| Perplexity | `x-perplexity-api-key` |
| OAuth provider fallback | `x-datarobot-<provider>-access-token` |
| Authorization context (OAuth OBO) | `x-datarobot-authorization-context` (JWT) |

Header names are matched case-insensitively.

---

## FastMCP runtime

FastMCP servers inject headers automatically via middleware. You do **not** call `set_request_headers`
manually inside tool handlers when middleware is registered.

### Template MCP server (`drmcp`)

The DataRobot MCP template registers OAuth + header injection at startup:

```python
from datarobot_genai.drmcp.core.middleware import initialize_oauth_middleware

# After creating the FastMCP instance:
initialize_oauth_middleware(mcp)
```

This wires `drmcpbase` middleware to `drtools.core.auth`:

- `read_http_headers()` → `set_request_headers()`
- JWT in `x-datarobot-authorization-context` → `set_auth_context()`
- Auth context also stored on FastMCP context state for OAuth OBO

Deploy with HTTP auth:

```bash
export AUTH_RESOLUTION_STRATEGY=http
```

Clients must send credentials on each MCP HTTP request (streamable-http / SSE transport).

### Custom FastMCP server (`drmcpbase` + `drtools`)

For a standalone FastMCP server built on `drmcpbase` e.g. Global MCP, wire the same hooks explicitly:

```python
from fastmcp import FastMCP

from datarobot_genai.drmcpbase.middleware import OAuthMiddleWare
from datarobot_genai.drmcpbase.middleware import register_oauth_middleware
from datarobot_genai.drtools.core.auth import extract_auth_context_from_headers
from datarobot_genai.drtools.core.auth import set_auth_context
from datarobot_genai.drtools.core.auth import set_request_headers

mcp = FastMCP("my-server")

register_oauth_middleware(
    mcp,
    OAuthMiddleWare(
        inject_headers=set_request_headers,
        extract_auth_context=extract_auth_context_from_headers,
        set_auth_context=set_auth_context,
    ),
)
```

For `drmcp`, the helper is equivalent:

```python
from datarobot_genai.drmcp.core.middleware import create_oauth_middleware, register_oauth_middleware

register_oauth_middleware(mcp, create_oauth_middleware())
```

### FastMCP client (calling your server)

Pass headers on the MCP HTTP connection so middleware can inject them:

```python
from langchain_mcp_adapters.sessions import StreamableHttpConnection

connection = StreamableHttpConnection(
    url="https://my-mcp.example.com/mcp",
    headers={
        "Authorization": "Bearer <datarobot-api-token>",
        "x-datarobot-authorization-context": "<jwt-auth-context>",
        "x-tavily-api-key": "<optional-third-party-key>",
    },
)
```

On the server, `AUTH_RESOLUTION_STRATEGY=http` ensures tools use these headers only.

---

## LangChain runtime

LangChain does not inject headers into `drtools` automatically. Choose one of the patterns below.

### Option A — Remote MCP tools (recommended)

Run tools behind a FastMCP server (above) and load them with `langchain-mcp-adapters`.
Headers travel with the MCP HTTP request; the FastMCP middleware injects them server-side.

**Low-level** (`langchain-mcp-adapters`):

```python
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools

connection = StreamableHttpConnection(
    transport="streamable_http",
    url="https://my-mcp.example.com/mcp",
    headers={
        "Authorization": "Bearer <datarobot-api-token>",
        "x-datarobot-authorization-context": "<jwt-auth-context>",
    },
)

async with create_session(connection=connection) as session:
    tools = await load_mcp_tools(session=session)
```

**In this repo** (LangGraph + DRAgent), incoming request headers are forwarded via
`MCPConfig` and `mcp_tools_context`:

```python
from datarobot_genai.core.mcp import MCPConfig
from datarobot_genai.langgraph.mcp import mcp_tools_context
from datarobot_genai.nat.helpers import extract_authorization_from_context
from datarobot_genai.nat.helpers import extract_datarobot_headers_from_context

forwarded_headers = extract_datarobot_headers_from_context()
authorization_context = extract_authorization_from_context()
mcp_config = MCPConfig(
    forwarded_headers=forwarded_headers,
    authorization_context=authorization_context,
)

async with mcp_tools_context(mcp_config) as tools:
    agent = MyAgent(llm=llm, forwarded_headers=forwarded_headers, tools=tools)
    ...
```

Set on the **MCP server**: `AUTH_RESOLUTION_STRATEGY=http`.

### Option B — In-process `drtools` tools

When LangChain calls `drtools` Python functions directly in the same process, inject headers
**before each tool invocation** (or once per incoming HTTP request if all tools share one request).

#### Wrap `drtools` functions as LangChain tools

`drtools` exports plain async functions you can pass to LangChain's `tool()` decorator and use with
`create_agent` or call directly:

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

from datarobot_genai.drtools.tavily.tools import tavily_search

search_tool = tool(tavily_search)
agent = create_agent(model, tools=[search_tool])

# Direct call (same auth rules as when the agent invokes the tool):
result = await search_tool.ainvoke({
    "query": "server setup",
    "max_results": 5,
})
```

The same pattern works for any `drtools` tool (for example
`search_datarobot_agentic_docs` from `datarobot_genai.drtools.dr_docs`, which needs no API key).

**Tools that require secrets** (Tavily, Perplexity, DataRobot API, OAuth providers) still go through
`drtools.core.auth` resolvers. LangChain does not inject headers for you—you must provide credentials
via config/env or request headers depending on `AUTH_RESOLUTION_STRATEGY`.

**Config/env** (local scripts, `AUTH_RESOLUTION_STRATEGY=config`):

```bash
export TAVILY_API_KEY=tvly-...
```

```python
search_tool = tool(tavily_search)
result = await search_tool.ainvoke({"query": "server setup", "max_results": 5})
```

**HTTP headers** (multi-tenant apps, `AUTH_RESOLUTION_STRATEGY=http`):

```python
from datarobot_genai.drtools.core.auth import set_request_headers

set_request_headers({"x-tavily-api-key": "<per-user-tavily-key>"})
try:
    result = await search_tool.ainvoke({"query": "server setup", "max_results": 5})
finally:
    set_request_headers({})
```

When the agent runs inside an HTTP handler, bind headers once for the whole request so every tool
call in that agent run sees the same credentials (see below).

#### Request context helpers

```python
import os

os.environ["AUTH_RESOLUTION_STRATEGY"] = "http"

from datarobot_genai.drtools.core.auth import set_auth_context
from datarobot_genai.drtools.core.auth import set_request_headers


def bind_drtools_request_context(
    headers: dict[str, str],
    auth_context=None,
) -> None:
    """Inject headers/auth context for the current async task or thread."""
    set_request_headers(headers)
    set_auth_context(auth_context)


def clear_drtools_request_context() -> None:
    set_request_headers({})
    set_auth_context(None)
```

**Per HTTP request** (e.g. FastAPI / Starlette handler wrapping a LangChain agent):

```python
from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/chat")
async def chat(request: Request, body: dict):
    headers = {k.lower(): v for k, v in request.headers.items()}
    bind_drtools_request_context(headers)
    try:
        return await agent.ainvoke(body["message"])
    finally:
        clear_drtools_request_context()
```

**Per tool call** (wrap individual tools that use `drtools`):

```python
from functools import wraps
from langchain.tools import tool


def with_drtools_headers(get_headers):
    """Decorator: inject headers from a callable before the tool runs."""

    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            bind_drtools_request_context(get_headers())
            try:
                return await fn(*args, **kwargs)
            finally:
                clear_drtools_request_context()

        return wrapper

    return decorator


@tool
@with_drtools_headers(lambda: _CURRENT_REQUEST_HEADERS.get())
async def my_datarobot_tool(query: str) -> str:
    ...
```

Store incoming request headers in a `contextvars.ContextVar` in your web layer and read them in
`get_headers()` so concurrent requests stay isolated.

### Option C — Tests and scripts

Tests and one-off scripts can inject headers directly (same API as production runtimes):

```python
from datarobot_genai.drtools.core.auth import set_request_headers

set_request_headers(
    {
        "authorization": "Bearer test-token",
        "x-tavily-api-key": "tvly-test",
    }
)
```

---

## OAuth (authorization context)

OAuth On-Behalf-Of tokens require an `AuthCtx` in addition to headers. FastMCP middleware sets this
via `set_auth_context()` when `x-datarobot-authorization-context` is present.

For in-process LangChain (option B), parse and inject explicitly:

```python
from datarobot_genai.drtools.core.auth import extract_auth_context_from_headers

auth_ctx = extract_auth_context_from_headers(headers)
bind_drtools_request_context(headers, auth_context=auth_ctx)
```

If OBO is unavailable, tools fall back to provider-specific
`x-datarobot-<provider>-access-token` headers when using the `http` strategy.

Google Drive and Microsoft Graph **require** `AUTH_RESOLUTION_STRATEGY=http` (OAuth OBO or header
tokens). They do not support `config`.
