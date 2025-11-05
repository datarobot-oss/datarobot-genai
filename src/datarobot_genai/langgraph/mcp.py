import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langchain.tools import BaseTool
from langchain_mcp_adapters.sessions import SSEConnection
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools

from datarobot_genai.core.mcp.common import MCPConfig


@asynccontextmanager
async def mcp_tools_context(
    api_base: str | None = None, api_key: str | None = None
) -> AsyncGenerator[list[BaseTool], None]:
    """Yield a list of LangChain BaseTool instances loaded via MCP.

    If no configuration or loading fails, yields an empty list without raising.
    """
    mcp_config = MCPConfig(api_base=api_base, api_key=api_key)
    server_config = mcp_config.server_config

    if not server_config:
        print("No MCP server configured, using empty tools list", flush=True)
        yield []
        return

    url = server_config["url"]
    print(f"Connecting to MCP server: {url}", flush=True)
    if mcp_config.external_mcp_transport == "streamable-http":
        connection = StreamableHttpConnection(transport="streamable_http", **server_config)
    elif mcp_config.external_mcp_transport == "sse":
        connection = SSEConnection(transport="sse", **server_config)
    else:
        raise RuntimeError("Unsupported MCP transport specified.")

    try:
        async with create_session(connection=connection) as session:
            # Use the connection to load available MCP tools
            tools = await load_mcp_tools(session=session)
            print(f"Successfully loaded {len(tools)} MCP tools", flush=True)
            yield tools
    except Exception as e:
        print(
            f"Warning: Failed to load MCP tools from {url}: {e}\n{traceback.format_exc()}",
            flush=True,
        )
        yield []
