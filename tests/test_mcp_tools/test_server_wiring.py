"""Integration tests for mcp_tools registry wiring into the server.

Tests that _load_mcp_tools_registry() correctly registers tools from
the unified registry into the FastMCP instance, and that sandbox
configuration follows MCP_SERVER_MODE.
"""
from __future__ import annotations

import importlib
import os
import pytest
from unittest.mock import MagicMock, patch

from datarobot_genai.mcp_tools._registry import clear_registry, get_all_tools


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


def _make_minimal_server(mcp_server_mode: str = "template") -> "DataRobotMCPServer":  # type: ignore[name-defined]
    """Create a minimal DataRobotMCPServer with mocked infrastructure."""
    from fastmcp import FastMCP
    from datarobot_genai.drmcp.core.dr_mcp_server import DataRobotMCPServer

    mock_mcp = MagicMock(spec=FastMCP)
    mock_mcp.tool.return_value = lambda fn: fn  # tool() as decorator

    mock_config = MagicMock()
    mock_config.app_log_level = "INFO"
    mock_config.enable_predictive_tools = False
    mock_config.enable_memory_management = False
    mock_config.resource_store_storage_path = None
    mock_config.mcp_server_name = "test-mcp"
    mock_config.tool_registration_duplicate_behavior = "warn"
    mock_config.prompt_registration_duplicate_behavior = "warn"

    mock_creds = MagicMock()
    mock_creds.has_aws_credentials.return_value = False

    patch_targets = [
        patch("datarobot_genai.drmcp.core.dr_mcp_server.get_config", return_value=mock_config),
        patch("datarobot_genai.drmcp.core.dr_mcp_server.get_credentials", return_value=mock_creds),
        patch("datarobot_genai.drmcp.core.dr_mcp_server.MCPLogging"),
        patch("datarobot_genai.drmcp.core.dr_mcp_server.initialize_telemetry"),
        patch("datarobot_genai.drmcp.core.dr_mcp_server.initialize_oauth_middleware"),
        patch("datarobot_genai.drmcp.core.dr_mcp_server.register_routes"),
        patch.dict(os.environ, {"MCP_SERVER_MODE": mcp_server_mode}),
    ]
    # initialize_resource_store may not be in the module namespace in all versions
    import datarobot_genai.drmcp.core.dr_mcp_server as _srv_mod
    if hasattr(_srv_mod, "initialize_resource_store"):
        patch_targets.insert(-1, patch("datarobot_genai.drmcp.core.dr_mcp_server.initialize_resource_store"))

    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in patch_targets:
            stack.enter_context(p)
        server = DataRobotMCPServer(mcp=mock_mcp, transport="streamable-http")

    return server


class TestMCPServerMode:
    def test_template_mode_uses_inprocess_sandbox(self):
        from datarobot_genai.mcp_tools.wren_tools.code_execution import (
            InProcessSandbox,
            _sandbox,
        )

        # Reload to get fresh module state
        import datarobot_genai.mcp_tools.wren_tools.code_execution as ce_mod
        importlib.reload(ce_mod)

        with patch.dict(os.environ, {"MCP_SERVER_MODE": "template"}):
            _make_minimal_server("template")

        from datarobot_genai.mcp_tools.wren_tools.code_execution import _sandbox as s
        assert isinstance(s, ce_mod.InProcessSandbox)

    def test_global_mode_uses_noop_sandbox(self):
        import datarobot_genai.mcp_tools.wren_tools.code_execution as ce_mod
        importlib.reload(ce_mod)

        with patch.dict(os.environ, {"MCP_SERVER_MODE": "global"}):
            _make_minimal_server("global")

        from datarobot_genai.mcp_tools.wren_tools.code_execution import _sandbox as s
        assert isinstance(s, ce_mod.NoopSandbox)

    def test_is_multi_tenant_reads_env_var(self):
        server = _make_minimal_server("template")
        assert server._is_multi_tenant() is False

        with patch.dict(os.environ, {"MCP_SERVER_MODE": "global"}):
            assert server._is_multi_tenant() is True


class TestToolsRegisteredIntoFastMCP:
    def test_registry_tools_are_wired_to_mcp(self):
        """Pre-seed the registry, then verify server wires those tools into FastMCP."""
        from datarobot_genai.mcp_tools._registry import register_tool

        async def _fake_tool(x: str) -> str:
            return x

        register_tool("fake_tool_a", _fake_tool, "A fake tool", "test_category")
        register_tool("fake_tool_b", _fake_tool, "B fake tool", "test_category")

        server = _make_minimal_server()

        # The server should have called mcp.tool() for each registered tool
        call_names = [
            call.kwargs.get("name")
            for call in server._mcp.tool.call_args_list
        ]
        assert "fake_tool_a" in call_names
        assert "fake_tool_b" in call_names
