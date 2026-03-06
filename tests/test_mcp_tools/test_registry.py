"""Tests for the tool registry."""
import pytest
from datarobot_genai.mcp_tools._registry import (
    register_tool,
    get_all_tools,
    get_tools_by_category,
    clear_registry,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


async def _dummy_tool(x: str) -> str:
    return x


def test_register_and_retrieve():
    register_tool("my_tool", _dummy_tool, "A test tool", "test_category")
    tools = get_all_tools()
    assert "my_tool" in tools
    assert tools["my_tool"].category == "test_category"


def test_duplicate_keeps_first():
    register_tool("dup", _dummy_tool, "First", "cat_a")
    register_tool("dup", _dummy_tool, "Second", "cat_b")
    tools = get_all_tools()
    assert tools["dup"].description == "First"
    assert tools["dup"].category == "cat_a"


def test_get_by_category():
    register_tool("tool_a", _dummy_tool, "A", "alpha")
    register_tool("tool_b", _dummy_tool, "B", "beta")
    register_tool("tool_c", _dummy_tool, "C", "alpha")
    alpha_tools = get_tools_by_category("alpha")
    assert len(alpha_tools) == 2
    assert "tool_a" in alpha_tools
    assert "tool_c" in alpha_tools


def test_empty_registry():
    assert get_all_tools() == {}
