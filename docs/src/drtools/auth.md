# Tool authentication (`drtools`)

`drtools` resolves secrets through a single model:

1. **Runtime adapter** injects per-request data (`set_request_headers`, `set_auth_context`).
2. **Tool code** calls resolvers (`resolve_datarobot_token`, `resolve_secret`, OAuth helpers).
3. **`auth_resolution_strategy`** chooses headers vs config/env.

```bash
export AUTH_RESOLUTION_STRATEGY=http   # default
# or
export AUTH_RESOLUTION_STRATEGY=config
```

| Strategy | Behavior |
|---|---|
| `http` (default) | Injected request headers only |
| `config` | Config/env only (ignore headers) |

> **Personal agents only:** Use `config` for single-user, personal setups (local scripts, your own dev agent). Do **not** use it for shared or multi-tenant deployments—every caller would use the same server-side tokens from env/config, so you end up sharing your API keys and OAuth tokens with everyone who can reach that agent.

Credentials are defined in `datarobot_genai.drtools.core.credentials.ToolsAuthCredentials`.

### Common headers (`http` strategy)

| Secret | Header examples |
|---|---|
| DataRobot API token | `x-datarobot-api-token: <token>`, `x-datarobot-authorization: Bearer <token>` |
| Tavily | `x-tavily-api-key` |
| Perplexity | `x-perplexity-api-key` |
| OAuth fallback | `x-datarobot-<provider>-access-token` |
| OAuth OBO context | `x-datarobot-authorization-context` (JWT) |

Header names are matched case-insensitively.

### Server app token fallback

When `AUTH_RESOLUTION_STRATEGY=http` and no request headers are present, callers can pass
`headers_auth_only=False` to `get_datarobot_access_token()` / `request_user_dr_client()` to use
the server's `DATAROBOT_API_TOKEN`. This is used for dynamic tool and prompt registration at MCP
startup.

---

## MCP server (`http`)

FastMCP middleware injects headers automatically. Register it at server startup:

```python
from datarobot_genai.drmcp.core.middleware import initialize_oauth_middleware

initialize_oauth_middleware(mcp)
```

Deploy with:

```bash
export AUTH_RESOLUTION_STRATEGY=http
```

Clients send credentials on each MCP HTTP request:

```python
from langchain_mcp_adapters.sessions import StreamableHttpConnection

connection = StreamableHttpConnection(
    url="https://my-mcp.example.com/mcp",
    headers={
        "x-datarobot-api-token": "<datarobot-api-token>",
        "x-datarobot-authorization-context": "<jwt-auth-context>",
    },
)
```

---

## LangChain in-process (`http`)

LangChain does not inject headers automatically. Bind them once per incoming request:

```python
import os

os.environ["AUTH_RESOLUTION_STRATEGY"] = "http"

from fastapi import FastAPI, Request
from langchain.agents import create_agent
from langchain_core.tools import tool

from datarobot_genai.drtools.core.auth import set_auth_context
from datarobot_genai.drtools.core.auth import set_request_headers
from datarobot_genai.drtools.tavily.tools import tavily_search

app = FastAPI()
search_tool = tool(tavily_search)
agent = create_agent(model, tools=[search_tool])


@app.post("/chat")
async def chat(request: Request, body: dict):
    headers = {k.lower(): v for k, v in request.headers.items()}
    set_request_headers(headers)
    set_auth_context(None)  # or parse JWT from x-datarobot-authorization-context
    try:
        return await agent.ainvoke(body["message"])
    finally:
        set_request_headers({})
        set_auth_context(None)
```

---

## LangChain in-process (`config`)

For local scripts and tests on a **personal agent** (not a shared deployment), set credentials via env and skip header injection:

```bash
export AUTH_RESOLUTION_STRATEGY=config
export DATAROBOT_API_TOKEN=your-token
export TAVILY_API_KEY=tvly-...
```

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

from datarobot_genai.drtools.tavily.tools import tavily_search

search_tool = tool(tavily_search)
agent = create_agent(model, tools=[search_tool])

result = await search_tool.ainvoke({"query": "server setup", "max_results": 5})
```

---

## OAuth

OAuth On-Behalf-Of requires an `AuthCtx` in addition to headers. FastMCP middleware sets this via
`set_auth_context()` when `x-datarobot-authorization-context` is present. If OBO is unavailable,
tools fall back to `x-datarobot-<provider>-access-token` headers under the `http` strategy.

Google Drive and Microsoft Graph require `AUTH_RESOLUTION_STRATEGY=http` (OAuth OBO or header
tokens). They do not support `config`.
