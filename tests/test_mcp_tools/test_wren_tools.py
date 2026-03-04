"""Tests for wren_tools registry integration and sandbox implementations."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from datarobot_genai.mcp_tools._registry import clear_registry, get_all_tools, get_tools_by_category
from datarobot_genai.mcp_tools.wren_tools.code_execution import (
    InProcessSandbox,
    NoopSandbox,
    set_sandbox_provider,
    execute_code,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


class TestSandboxProviders:
    @pytest.mark.asyncio
    async def test_noop_sandbox_returns_error(self):
        sandbox = NoopSandbox()
        result = await sandbox.execute("print('hello')", "test-session")
        assert result["error"] is not None
        assert "not available" in result["error"]
        assert result["result"] is None

    @pytest.mark.asyncio
    async def test_inprocess_sandbox_stdout(self):
        sandbox = InProcessSandbox()
        result = await sandbox.execute("print('hello world')", "test-session")
        assert result["error"] is None
        assert "hello world" in result["stdout"]

    @pytest.mark.asyncio
    async def test_inprocess_sandbox_result_variable(self):
        sandbox = InProcessSandbox()
        result = await sandbox.execute("result = 1 + 2", "test-session")
        assert result["error"] is None
        assert result["result"] == 3

    @pytest.mark.asyncio
    async def test_inprocess_sandbox_exception_captured(self):
        sandbox = InProcessSandbox()
        result = await sandbox.execute("raise ValueError('oops')", "test-session")
        assert result["error"] is not None
        assert "ValueError" in result["error"]

    @pytest.mark.asyncio
    async def test_inprocess_sandbox_timeout(self):
        sandbox = InProcessSandbox()
        result = await sandbox.execute(
            "import time; time.sleep(10)", "test-session", timeout_seconds=1
        )
        assert result["error"] is not None
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_set_sandbox_provider_switches_implementation(self):
        set_sandbox_provider(InProcessSandbox())
        result = await execute_code("result = 42")
        assert result["result"] == 42

        set_sandbox_provider(NoopSandbox())
        result = await execute_code("result = 42")
        assert result["error"] is not None

        # Reset to default
        set_sandbox_provider(NoopSandbox())


def _reload_wren_tools() -> None:
    """Force re-registration of all wren tools after registry has been cleared."""
    import importlib
    import datarobot_genai.mcp_tools.wren_tools.code_execution as m1
    import datarobot_genai.mcp_tools.wren_tools.data as m2
    import datarobot_genai.mcp_tools.wren_tools.deployment as m3
    import datarobot_genai.mcp_tools.wren_tools.model as m4
    import datarobot_genai.mcp_tools.wren_tools.optimization as m5
    import datarobot_genai.mcp_tools.wren_tools.search as m6
    import datarobot_genai.mcp_tools.wren_tools.use_case as m7
    import datarobot_genai.mcp_tools.wren_tools.vdb as m8

    for mod in (m1, m2, m3, m4, m5, m6, m7, m8):
        importlib.reload(mod)


class TestWrenToolsRegistration:
    def test_wren_tools_register_on_import(self):
        _reload_wren_tools()
        wren = get_tools_by_category("wren_tools")
        assert len(wren) > 0

    def test_expected_tools_are_registered(self):
        _reload_wren_tools()
        tools = get_all_tools()
        expected = [
            "execute_code",
            "list_datarobot_datasets",
            "get_datarobot_dataset",
            "upload_dataset_to_datarobot",
            "list_datastores",
            "browse_datastore",
            "query_datastore",
            "get_deployment_info",
            "predict_with_deployment",
            "deploy_model",
            "get_prediction_history",
            "list_models",
            "get_model_info",
            "run_autopilot",
            "is_eligible_for_timeseries_training",
            "list_use_cases",
            "list_use_case_assets",
            "list_vector_databases",
            "query_vector_database",
            "web_search",
            "cuopt_solve",
        ]
        for name in expected:
            assert name in tools, f"Expected tool '{name}' not registered"

    def test_no_duplicate_tool_names(self):
        _reload_wren_tools()
        _reload_wren_tools()  # double-reload should not duplicate (registry deduplicates)
        wren = get_tools_by_category("wren_tools")
        names = list(wren.keys())
        assert len(names) == len(set(names)), "Duplicate tool names detected"
