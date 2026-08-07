# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP integration tests for the shared ``GET /metadata`` route (global-mcp)."""

from typing import Any

from fastmcp import FastMCP
from starlette.testclient import TestClient

from datarobot_genai.drmcputils.routes.metadata import register_metadata_routes


def _make_server(gate: Any = None, config_provider: Any = None) -> FastMCP:
    mcp = FastMCP("metadata-test")

    @mcp.tool(tags={"beta", "alpha"})
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @mcp.tool(meta={"tool_category": "PROXIED_USER_MCP"})
    def proxied_tool() -> str:
        """Stand-in for a tool proxied from a user's MCP server."""
        return "proxied"

    @mcp.tool(meta={"tool_category": "USER_TOOL_DEPLOYMENT"})
    def deployment_tool() -> str:
        """Stand-in for a DataRobot deployment tool."""
        return "deployed"

    @mcp.prompt
    def greeting() -> str:
        """Say hello."""
        return "hello"

    register_metadata_routes(mcp, gate=gate, config_provider=config_provider)
    return mcp


class TestMetadataRoute:
    def test_catalog_items_counts_and_markers(self) -> None:
        mcp = _make_server()
        with TestClient(mcp.http_app()) as client:
            resp = client.get("/metadata")
        assert resp.status_code == 200
        body = resp.json()

        tools = {t["name"]: t for t in body["tools"]["items"]}
        assert body["tools"]["count"] == 3
        assert tools["add"]["tags"] == ["alpha", "beta"]  # sorted
        assert tools["add"]["toolCategory"] is None
        assert tools["proxied_tool"]["toolCategory"] == "PROXIED_USER_MCP"
        assert tools["deployment_tool"]["toolCategory"] == "USER_TOOL_DEPLOYMENT"

        assert body["prompts"]["count"] == 1
        assert body["prompts"]["items"][0]["name"] == "greeting"
        assert body["resources"] == {"items": [], "count": 0}

    def test_no_config_block_by_default(self) -> None:
        mcp = _make_server()
        with TestClient(mcp.http_app()) as client:
            body = client.get("/metadata").json()
        assert "config" not in body

    def test_config_provider_result_is_attached(self) -> None:
        async def config_provider(_request: Any) -> dict[str, Any]:
            return {"server": {"name": "global-mcp"}}

        mcp = _make_server(config_provider=config_provider)
        with TestClient(mcp.http_app()) as client:
            body = client.get("/metadata").json()
        assert body["config"] == {"server": {"name": "global-mcp"}}

    def test_custom_base_path_is_honored(self) -> None:
        prefixed = FastMCP("metadata-prefixed")
        register_metadata_routes(prefixed, base_path="/globalmcp/metadata")
        with TestClient(prefixed.http_app()) as client:
            assert client.get("/globalmcp/metadata").status_code == 200
            assert client.get("/metadata").status_code == 404


class TestMetadataGate:
    def test_gate_denies_returns_404(self) -> None:
        async def deny(_request: Any) -> bool:
            return False

        mcp = _make_server(gate=deny)
        with TestClient(mcp.http_app()) as client:
            assert client.get("/metadata").status_code == 404

    def test_gate_raising_fails_closed_to_404(self) -> None:
        async def boom(_request: Any) -> bool:
            raise RuntimeError("down")

        mcp = _make_server(gate=boom)
        with TestClient(mcp.http_app()) as client:
            assert client.get("/metadata").status_code == 404

    def test_gate_allows_serves_catalog(self) -> None:
        async def allow(_request: Any) -> bool:
            return True

        mcp = _make_server(gate=allow)
        with TestClient(mcp.http_app()) as client:
            assert client.get("/metadata").status_code == 200
