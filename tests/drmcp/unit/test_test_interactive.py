# Copyright 2025 DataRobot, Inc.
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

"""Unit tests for test_interactive.py module."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from mcp import ClientSession
from mcp.types import CallToolResult
from mcp.types import ListToolsResult
from mcp.types import TextContent
from mcp.types import Tool
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call_function import (
    ChatCompletionMessageToolCallFunction,
)

from datarobot_genai.drmcp.test_utils.test_interactive import LLMMCPClient
from datarobot_genai.drmcp.test_utils.test_interactive import LLMResponse
from datarobot_genai.drmcp.test_utils.test_interactive import ToolCall


# Helper functions for creating mocks
def create_mock_tool(
    name: str, description: str = "", input_schema: dict | None = None
) -> MagicMock:
    """Create a mock Tool object."""
    mock_tool = MagicMock(spec=Tool)
    mock_tool.name = name
    mock_tool.description = description
    mock_tool.inputSchema = input_schema or {"type": "object"}
    return mock_tool


def create_mock_text_content(text: str) -> MagicMock:
    """Create a mock TextContent object."""
    mock_content = MagicMock(spec=TextContent)
    mock_content.text = text
    return mock_content


def create_mock_tool_call(
    tool_name: str, arguments: str = "{}", call_id: str = "call_123"
) -> MagicMock:
    """Create a mock ChatCompletionMessageToolCall."""
    mock_function = MagicMock(spec=ChatCompletionMessageToolCallFunction)
    mock_function.name = tool_name
    mock_function.arguments = arguments

    mock_tool_call = MagicMock(spec=ChatCompletionMessageToolCall)
    mock_tool_call.id = call_id
    mock_tool_call.function = mock_function
    return mock_tool_call


def create_mock_chat_completion(
    content: str | None = None,
    tool_calls: list[ChatCompletionMessageToolCall] | None = None,
) -> MagicMock:
    """Create a mock ChatCompletion object."""
    mock_message = MagicMock(spec=ChatCompletionMessage)
    mock_message.content = content
    mock_message.tool_calls = tool_calls

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock(spec=ChatCompletion)
    mock_response.choices = [mock_choice]
    return mock_response


def create_mock_call_tool_result(content: list | None = None) -> MagicMock:
    """Create a mock CallToolResult."""
    mock_result = MagicMock(spec=CallToolResult)
    mock_result.content = content or []
    return mock_result


# Fixtures
@pytest.fixture
def mock_openai_patch():
    """Fixture to patch OpenAI and return the mock."""
    with patch("datarobot_genai.drmcp.test_utils.test_interactive.openai.OpenAI") as mock:
        yield mock


@pytest.fixture
def mock_azure_openai_patch():
    """Fixture to patch AzureOpenAI and return the mock."""
    with patch("datarobot_genai.drmcp.test_utils.test_interactive.openai.AzureOpenAI") as mock:
        yield mock


@pytest.fixture
def basic_client_config() -> dict:
    """Return basic client configuration."""
    return {"openai_api_key": "test-key"}


@pytest.fixture
def llm_client(mock_openai_patch, basic_client_config) -> LLMMCPClient:
    """Create a basic LLMMCPClient instance."""
    return LLMMCPClient(str(basic_client_config))


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock ClientSession."""
    return AsyncMock(spec=ClientSession)


@pytest.fixture
def mock_openai_client_instance(mock_openai_patch) -> MagicMock:
    """Create a mock OpenAI client instance."""
    mock_instance = MagicMock()
    mock_openai_patch.return_value = mock_instance
    return mock_instance


class TestToolCall:
    """Test cases for ToolCall class."""

    def test_tool_call_creation(self) -> None:
        """Test ToolCall creation."""
        tool_call = ToolCall("test_tool", {"param": "value"}, "test reasoning")

        assert tool_call.tool_name == "test_tool"
        assert tool_call.parameters == {"param": "value"}
        assert tool_call.reasoning == "test reasoning"


class TestLLMResponse:
    """Test cases for LLMResponse class."""

    def test_llm_response_creation(self) -> None:
        """Test LLMResponse creation."""
        tool_calls = [ToolCall("tool1", {}, "reasoning1")]
        tool_results = ["result1"]

        response = LLMResponse("content", tool_calls, tool_results)

        assert response.content == "content"
        assert response.tool_calls == tool_calls
        assert response.tool_results == tool_results


class TestLLMMCPClient:
    """Test cases for LLMMCPClient class."""

    def test_init_with_openai_config(self, mock_openai_patch) -> None:
        """Test LLMMCPClient initialization with OpenAI config."""
        config = {
            "openai_api_key": "test-key",
            "model": "gpt-4",
            "save_llm_responses": False,
        }

        client = LLMMCPClient(str(config))

        assert client.model == "gpt-4"
        assert client.save_llm_responses is False
        assert client.available_tools == []
        mock_openai_patch.assert_called_once_with(api_key="test-key")

    def test_init_with_azure_config(self, mock_azure_openai_patch) -> None:
        """Test LLMMCPClient initialization with Azure OpenAI config."""
        config = {
            "openai_api_key": "test-key",
            "openai_api_base": "https://test.openai.azure.com",
            "openai_api_deployment_id": "deployment-123",
            "openai_api_version": "2024-01-01",
        }

        client = LLMMCPClient(str(config))

        assert client.model == "deployment-123"
        mock_azure_openai_patch.assert_called_once_with(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-01-01",
        )

    def test_init_with_dict_config(self, mock_openai_patch) -> None:
        """Test LLMMCPClient initialization with dict config."""
        config = {"openai_api_key": "test-key", "model": "gpt-3.5-turbo"}

        client = LLMMCPClient(config)

        assert client.model == "gpt-3.5-turbo"
        mock_openai_patch.assert_called_once_with(api_key="test-key")

    def test_init_with_defaults(self, mock_openai_patch) -> None:
        """Test LLMMCPClient initialization with default values."""
        config = {"openai_api_key": "test-key"}

        client = LLMMCPClient(str(config))

        assert client.model == "gpt-3.5-turbo"
        assert client.save_llm_responses is True

    @pytest.mark.asyncio
    async def test_add_mcp_tool_to_available_tools(self, llm_client, mock_session) -> None:
        """Test adding MCP tools to available tools list."""
        mock_tool1 = create_mock_tool("tool1", "Description 1", {"type": "object"})
        mock_tool2 = create_mock_tool(
            "tool2", "Description 2", {"type": "object", "properties": {}}
        )

        mock_tools_result = MagicMock(spec=ListToolsResult)
        mock_tools_result.tools = [mock_tool1, mock_tool2]
        mock_session.list_tools.return_value = mock_tools_result

        await llm_client._add_mcp_tool_to_available_tools(mock_session)

        assert len(llm_client.available_tools) == 2
        assert llm_client.available_tools[0]["function"]["name"] == "tool1"
        assert llm_client.available_tools[1]["function"]["name"] == "tool2"

    @pytest.mark.asyncio
    async def test_call_mcp_tool_with_text_content(self, llm_client, mock_session) -> None:
        """Test calling MCP tool with TextContent result."""
        mock_text_content = create_mock_text_content("Tool result text")
        mock_result = create_mock_call_tool_result([mock_text_content])
        mock_session.call_tool.return_value = mock_result

        result = await llm_client._call_mcp_tool("test_tool", {"param": "value"}, mock_session)

        assert result == "Tool result text"
        mock_session.call_tool.assert_called_once_with("test_tool", {"param": "value"})

    @pytest.mark.asyncio
    async def test_call_mcp_tool_with_non_text_content(self, llm_client, mock_session) -> None:
        """Test calling MCP tool with non-TextContent result."""
        mock_result = create_mock_call_tool_result([{"type": "image", "data": "base64data"}])
        mock_session.call_tool.return_value = mock_result

        result = await llm_client._call_mcp_tool("test_tool", {}, mock_session)

        assert isinstance(result, str)
        assert "image" in result or "base64data" in result

    @pytest.mark.asyncio
    async def test_call_mcp_tool_with_empty_content(self, llm_client, mock_session) -> None:
        """Test calling MCP tool with empty content."""
        mock_result = create_mock_call_tool_result([])
        mock_session.call_tool.return_value = mock_result

        result = await llm_client._call_mcp_tool("test_tool", {}, mock_session)

        assert result == "[]"

    @pytest.mark.asyncio
    async def test_call_mcp_tool_with_none_content(self, llm_client, mock_session) -> None:
        """Test calling MCP tool with None content."""
        mock_result = create_mock_call_tool_result(None)
        mock_session.call_tool.return_value = mock_result

        result = await llm_client._call_mcp_tool("test_tool", {}, mock_session)

        assert "None" in result or result == "[]"

    @pytest.mark.asyncio
    async def test_process_tool_calls_with_tool_calls(self, llm_client, mock_session) -> None:
        """Test processing tool calls from LLM response."""
        mock_text_content = create_mock_text_content("Tool result")
        mock_tool_result = create_mock_call_tool_result([mock_text_content])
        mock_session.call_tool.return_value = mock_tool_result

        mock_tool_call = create_mock_tool_call("test_tool", '{"param": "value"}')
        mock_response = create_mock_chat_completion(tool_calls=[mock_tool_call])

        messages = []
        tool_calls, tool_results = await llm_client._process_tool_calls(
            mock_response, messages, mock_session
        )

        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "test_tool"
        assert tool_calls[0].parameters == {"param": "value"}
        assert len(tool_results) == 1
        assert tool_results[0] == "Tool result"
        assert len(messages) == 2  # Assistant message + tool result

    @pytest.mark.asyncio
    async def test_process_tool_calls_with_error(self, llm_client, mock_session) -> None:
        """Test processing tool calls when tool call raises exception."""
        mock_session.call_tool.side_effect = Exception("Tool error")

        mock_tool_call = create_mock_tool_call("test_tool")
        mock_response = create_mock_chat_completion(tool_calls=[mock_tool_call])

        messages = []
        tool_calls, tool_results = await llm_client._process_tool_calls(
            mock_response, messages, mock_session
        )

        assert len(tool_calls) == 1
        assert len(tool_results) == 1
        assert "Error calling test_tool" in tool_results[0]
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_get_llm_response_with_tools(
        self, llm_client, mock_openai_client_instance
    ) -> None:
        """Test getting LLM response with tools enabled."""
        llm_client.available_tools = [{"type": "function", "function": {"name": "test_tool"}}]

        mock_completion = MagicMock()
        mock_openai_client_instance.chat.completions.create.return_value = mock_completion

        messages = [{"role": "user", "content": "test"}]
        response = await llm_client._get_llm_response(messages, allow_tool_calls=True)

        assert response == mock_completion
        mock_openai_client_instance.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client_instance.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_get_llm_response_without_tools(
        self, llm_client, mock_openai_client_instance
    ) -> None:
        """Test getting LLM response without tools."""
        mock_completion = MagicMock()
        mock_openai_client_instance.chat.completions.create.return_value = mock_completion

        messages = [{"role": "user", "content": "test"}]
        response = await llm_client._get_llm_response(messages, allow_tool_calls=False)

        assert response == mock_completion
        call_kwargs = mock_openai_client_instance.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.test_utils.test_interactive.save_response_to_file")
    async def test_process_prompt_with_mcp_support_single_response(
        self, mock_save_file, mock_openai_patch, mock_session, mock_openai_client_instance
    ) -> None:
        """Test process_prompt_with_mcp_support with single LLM response (no tool calls)."""
        config = {"openai_api_key": "test-key", "save_llm_responses": True}
        client = LLMMCPClient(str(config))

        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        mock_response = create_mock_chat_completion(content="Final response")
        mock_openai_client_instance.chat.completions.create.return_value = mock_response

        result = await client.process_prompt_with_mcp_support("test prompt", mock_session)

        assert result.content == "final response"
        assert len(result.tool_calls) == 0
        mock_save_file.assert_called_once()

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.test_utils.test_interactive.save_response_to_file")
    async def test_process_prompt_with_mcp_support_with_tool_calls(
        self,
        mock_save_file,
        mock_openai_patch,
        mock_session,
        mock_openai_client_instance,
    ) -> None:
        """Test process_prompt_with_mcp_support with tool calls."""
        config = {"openai_api_key": "test-key", "save_llm_responses": True}
        client = LLMMCPClient(str(config))

        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

        mock_tool_call = create_mock_tool_call("test_tool")
        mock_response_with_tools = create_mock_chat_completion(tool_calls=[mock_tool_call])
        mock_response_final = create_mock_chat_completion(content="Final answer")

        mock_openai_client_instance.chat.completions.create.side_effect = [
            mock_response_with_tools,
            mock_response_final,
        ]

        mock_text_content = create_mock_text_content("Tool result")
        mock_tool_result = create_mock_call_tool_result([mock_text_content])
        mock_session.call_tool.return_value = mock_tool_result

        result = await client.process_prompt_with_mcp_support("test prompt", mock_session)

        assert result.content == "final answer"
        assert len(result.tool_calls) == 1
        mock_save_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_prompt_with_mcp_support_content_cleaning(
        self, mock_openai_patch, mock_session, mock_openai_client_instance
    ) -> None:
        """Test that process_prompt_with_mcp_support cleans content (removes * and lowercases)."""
        config = {"openai_api_key": "test-key", "save_llm_responses": False}
        client = LLMMCPClient(str(config))

        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        mock_response = create_mock_chat_completion(content="Final *Response* With CAPS")
        mock_openai_client_instance.chat.completions.create.return_value = mock_response

        result = await client.process_prompt_with_mcp_support("test", mock_session)

        # Should remove * and lowercase
        assert result.content == "final response with caps"
        assert "*" not in result.content

    @pytest.mark.asyncio
    async def test_process_prompt_with_mcp_support_no_save(
        self, mock_openai_patch, mock_session, mock_openai_client_instance
    ) -> None:
        """Test process_prompt_with_mcp_support without saving responses."""
        config = {"openai_api_key": "test-key", "save_llm_responses": False}
        client = LLMMCPClient(str(config))

        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        mock_response = create_mock_chat_completion(content="Response")
        mock_openai_client_instance.chat.completions.create.return_value = mock_response

        with patch(
            "datarobot_genai.drmcp.test_utils.test_interactive.save_response_to_file"
        ) as mock_save:
            result = await client.process_prompt_with_mcp_support("test", mock_session)

            assert result.content == "response"
            mock_save.assert_not_called()
