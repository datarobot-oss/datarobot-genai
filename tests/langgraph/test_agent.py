# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import cached_property
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from ag_ui.core import BaseEvent
from ag_ui.core import EventType
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.types import Command

from datarobot_genai.core.chat.responses import CustomModelChatResponse
from datarobot_genai.core.chat.responses import to_custom_model_chat_response
from datarobot_genai.langgraph.agent import LangGraphAgent


@pytest.fixture
def authorization_context() -> dict[str, Any]:
    return {"user": {"id": "123", "name": "bar"}}


class SimpleLangGraphAgent(LangGraphAgent):
    @cached_property
    def workflow(self) -> StateGraph[MessagesState]:
        async def mock_stream_generator():
            # stream the first agent
            # tool call and respose
            yield (
                "first_agent",
                "messages",
                (
                    AIMessageChunk(
                        content="",
                        id="000",
                        tool_call_chunks=[
                            {"name": "get_info_about_city", "id": "tool_call_111", "args": None}
                        ],
                    ),
                    {},
                ),
            )
            yield (
                "first_agent",
                "messages",
                (
                    AIMessageChunk(
                        content="",
                        id="000",
                        tool_call_chunks=[{"name": "", "id": "", "args": "{'name': 'Paris'}"}],
                    ),
                    {},
                ),
            )
            yield (
                "first_agent",
                "messages",
                (
                    ToolMessage(
                        tool_call_id="tool_call_111",
                        id="000",
                        content="Paris is the capital city of France.",
                    ),
                    {},
                ),
            )
            # tool call end
            yield (
                "first_agent",
                "messages",
                (AIMessageChunk(content="Here is the information", id="111"), {}),
            )
            yield (
                "first_agent",
                "messages",
                (AIMessageChunk(content=" you requested about Paris.....", id="111"), {}),
            )

            yield (
                "first_agent",
                "updates",
                {
                    "first_agent": {
                        "usage": {
                            "total_tokens": 100,
                            "prompt_tokens": 50,
                            "completion_tokens": 50,
                        },
                        "messages": [
                            HumanMessage(content="Hi, tell me about Paris."),
                            AIMessage(
                                content="Here is the information you requested about Paris.....",
                                id="111",
                            ),
                        ],
                    }
                },
            )
            yield (
                "final_agent",
                "messages",
                (AIMessageChunk(content="Paris is the capital", id="222"), {}),
            )
            yield (
                "final_agent",
                "messages",
                (AIMessageChunk(content=" city of France.", id="222"), {}),
            )
            yield (
                "final_agent",
                "updates",
                {
                    "final_agent": {
                        "usage": {
                            "total_tokens": 100,
                            "prompt_tokens": 50,
                            "completion_tokens": 50,
                        },
                        "messages": [
                            HumanMessage(content="Hi, tell me about Paris."),
                            AIMessage(content="Paris is the capital city of France.", id="222"),
                        ],
                    }
                },
            )

        mock_graph_stream = Mock(astream=Mock(return_value=mock_stream_generator()))

        mock_state_graph = Mock(compile=Mock(return_value=mock_graph_stream))
        return mock_state_graph

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Tell user about {topic}.",
                },
                {"role": "user", "content": "Hi, tell me about {topic}."},
            ]
        )

    @property
    def langgraph_config(self) -> dict[str, Any]:
        return {}


async def test_langgraph_non_streaming():
    # GIVEN a simple langgraph agent implementation
    agent = SimpleLangGraphAgent()

    # WHEN invoking the agent with a completion create params
    completion_create_params = {
        "model": "test-model",
        "messages": [{"role": "user", "content": '{"topic": "Artificial Intelligence"}'}],
        "environment_var": True,
    }
    response_text, pipeline_interactions, usage_metrics = await agent.invoke(
        completion_create_params
    )

    # THEN agent.workflow is called with expected arguments
    expected_command = Command(
        update={
            "messages": [
                SystemMessage(
                    content="You are a helpful assistant. Tell user about Artificial Intelligence."
                ),
                HumanMessage(content="Hi, tell me about Artificial Intelligence."),
            ]
        }
    )
    agent.workflow.compile().astream.assert_called_once_with(
        input=expected_command,
        config={},
        debug=True,
        stream_mode=["updates", "messages"],
        subgraphs=True,
    )

    # THEN the response is a custom model chat response
    response = to_custom_model_chat_response(
        response_text, pipeline_interactions, usage_metrics, model="test-model"
    )
    assert isinstance(response, CustomModelChatResponse)
    # THEN the last message is the final message
    assert response.choices[0].message.content == "Paris is the capital city of France."
    # THEN the pipeline interactions are not None
    assert response.pipeline_interactions is not None
    assert response.usage.completion_tokens == 100
    assert response.usage.prompt_tokens == 100
    assert response.usage.total_tokens == 200


async def test_langgraph_streaming():
    # GIVEN a simple langgraph agent implementation
    agent = SimpleLangGraphAgent()

    # WHEN invoking the agent with a completion create params
    completion_create_params = {
        "model": "test-model",
        "messages": [{"role": "user", "content": '{"topic": "AI"}'}],
        "environment_var": True,
        "stream": True,
    }
    streaming_response_iterator = await agent.invoke(completion_create_params)

    # THEN the streaming response iterator returns the expected responses
    # Iterate directly over the async generator to avoid event loop conflicts
    # Note: With the new async with implementation, _invoke is called when we start consuming
    first_item_consumed = False
    events = []
    async for (
        response_event,
        pipeline_interactions,
        usage_metrics,
    ) in streaming_response_iterator:
        # Check that agent.workflow is called with expected arguments after first consumption
        if not first_item_consumed:
            expected_command = Command(
                update={
                    "messages": [
                        SystemMessage(content="You are a helpful assistant. Tell user about AI."),
                        HumanMessage(content="Hi, tell me about AI."),
                    ]
                }
            )
            agent.workflow.compile().astream.assert_called_once_with(
                input=expected_command,
                config={},
                debug=True,
                stream_mode=["updates", "messages"],
                subgraphs=True,
            )
            first_item_consumed = True

        assert not isinstance(response_event, str) or pipeline_interactions is not None

        if isinstance(response_event, BaseEvent):
            events.append(response_event)

    assert len(events) == 12
    assert events[0].type == EventType.TOOL_CALL_START
    assert events[0].tool_call_id == "tool_call_111"
    assert events[0].tool_call_name == "get_info_about_city"
    assert events[0].parent_message_id == "000"
    assert events[1].type == EventType.TOOL_CALL_ARGS
    assert events[1].tool_call_id == "tool_call_111"
    assert events[1].delta == "{'name': 'Paris'}"
    assert events[2].type == EventType.TOOL_CALL_END
    assert events[2].tool_call_id == "tool_call_111"
    assert events[3].type == EventType.TOOL_CALL_RESULT
    assert events[3].tool_call_id == "tool_call_111"
    assert events[3].content == "Paris is the capital city of France."
    assert events[3].role == "tool"
    assert events[4].type == EventType.TEXT_MESSAGE_START
    assert events[4].message_id == "111"
    assert events[5].type == EventType.TEXT_MESSAGE_CONTENT
    assert events[5].delta == "Here is the information"
    assert events[5].message_id == "111"
    assert events[6].type == EventType.TEXT_MESSAGE_CONTENT
    assert events[6].delta == " you requested about Paris....."
    assert events[6].message_id == "111"
    assert events[7].type == EventType.TEXT_MESSAGE_END
    assert events[7].message_id == "111"
    assert events[8].type == EventType.TEXT_MESSAGE_START
    assert events[8].message_id == "222"
    assert events[9].type == EventType.TEXT_MESSAGE_CONTENT
    assert events[9].delta == "Paris is the capital"
    assert events[9].message_id == "222"
    assert events[10].type == EventType.TEXT_MESSAGE_CONTENT
    assert events[10].delta == " city of France."
    assert events[10].message_id == "222"
    assert events[11].type == EventType.TEXT_MESSAGE_END
    assert events[11].message_id == "222"

    assert pipeline_interactions is not None
    assert usage_metrics is not None
    assert usage_metrics["total_tokens"] == 200
    assert usage_metrics["prompt_tokens"] == 100
    assert usage_metrics["completion_tokens"] == 100


async def test_invoke_calls_mcp_tools_context_and_sets_tools(authorization_context):
    """Test that invoke method calls mcp_tools_context and sets tools correctly."""
    # GIVEN a simple langgraph agent implementation
    agent = SimpleLangGraphAgent(authorization_context=authorization_context)

    # Mock the mcp_tools_context to return mock tools
    mock_tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]

    with patch("datarobot_genai.langgraph.agent.mcp_tools_context") as mock_mcp_context:
        # Configure the mock context manager
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_tools
        mock_context_manager.__aexit__.return_value = None
        mock_mcp_context.return_value = mock_context_manager

        with patch.object(agent, "set_mcp_tools") as mock_set_mcp_tools:
            # WHEN invoking the agent
            completion_create_params = {
                "model": "test-model",
                "messages": [{"role": "user", "content": '{"topic": "AI"}'}],
                "environment_var": True,
            }

            await agent.invoke(completion_create_params)

            # THEN mcp_tools_context is called with correct parameters
            mock_mcp_context.assert_called_once_with(
                authorization_context=authorization_context,
                forwarded_headers={},
            )

            # THEN set_mcp_tools is called with the tools from context
            mock_set_mcp_tools.assert_called_once_with(mock_tools)


async def test_invoke_passes_forwarded_headers_to_mcp_context(authorization_context):
    """Test that invoke method passes forwarded headers to MCP context."""
    # GIVEN a simple langgraph agent with forwarded headers
    forwarded_headers = {
        "x-datarobot-api-key": "scoped-token-123",
    }
    agent = SimpleLangGraphAgent(
        authorization_context=authorization_context, forwarded_headers=forwarded_headers
    )

    # Mock the mcp_tools_context to return mock tools
    mock_tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]

    with patch("datarobot_genai.langgraph.agent.mcp_tools_context") as mock_mcp_context:
        # Configure the mock context manager
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_tools
        mock_context_manager.__aexit__.return_value = None
        mock_mcp_context.return_value = mock_context_manager

        # WHEN invoking the agent
        completion_create_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": '{"topic": "AI"}'}],
            "environment_var": True,
        }

        await agent.invoke(completion_create_params)

        # THEN mcp_tools_context is called with forwarded headers
        mock_mcp_context.assert_called_once_with(
            authorization_context=authorization_context,
            forwarded_headers=forwarded_headers,
        )


async def test_invoke_streaming_passes_forwarded_headers_to_mcp_context(authorization_context):
    """Test that streaming invoke method passes forwarded headers to MCP context."""
    # GIVEN a simple langgraph agent with forwarded headers
    forwarded_headers = {"x-datarobot-api-key": "scoped-token-456"}
    agent = SimpleLangGraphAgent(
        authorization_context=authorization_context, forwarded_headers=forwarded_headers
    )

    # Mock the mcp_tools_context to return mock tools
    mock_tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]

    with patch("datarobot_genai.langgraph.agent.mcp_tools_context") as mock_mcp_context:
        # Configure the mock context manager
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_tools
        mock_context_manager.__aexit__.return_value = None
        mock_mcp_context.return_value = mock_context_manager

        # WHEN invoking the agent with streaming
        completion_create_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": '{"topic": "AI"}'}],
            "environment_var": True,
            "stream": True,
        }

        streaming_response_iterator = await agent.invoke(completion_create_params)

        # WHEN consuming the streaming generator
        items_consumed = 0
        async for _ in streaming_response_iterator:
            items_consumed += 1
            # THEN mcp_tools_context is called with forwarded headers
            if items_consumed == 1:
                mock_mcp_context.assert_called_once_with(
                    authorization_context=authorization_context,
                    forwarded_headers=forwarded_headers,
                )
                break


async def test_invoke_streaming_calls_mcp_tools_context_and_cleans_up(authorization_context):
    """Test that streaming invoke method uses async with correctly and cleans up context."""
    # GIVEN a simple langgraph agent implementation
    agent = SimpleLangGraphAgent(authorization_context=authorization_context)

    # Mock the mcp_tools_context to return mock tools
    mock_tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]

    with patch("datarobot_genai.langgraph.agent.mcp_tools_context") as mock_mcp_context:
        # Configure the mock context manager
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_tools
        mock_context_manager.__aexit__.return_value = None
        mock_mcp_context.return_value = mock_context_manager

        with patch.object(agent, "set_mcp_tools") as mock_set_mcp_tools:
            # WHEN invoking the agent with streaming
            completion_create_params = {
                "model": "test-model",
                "messages": [{"role": "user", "content": '{"topic": "AI"}'}],
                "environment_var": True,
                "stream": True,
            }

            streaming_response_iterator = await agent.invoke(completion_create_params)

            # THEN context is not entered yet (it's entered inside the generator)
            mock_context_manager.__aenter__.assert_not_called()
            mock_mcp_context.assert_not_called()

            # WHEN consuming the streaming generator
            items_consumed = 0
            async for _ in streaming_response_iterator:
                items_consumed += 1
                # THEN mcp_tools_context is called and context is entered when generator starts
                if items_consumed == 1:
                    # Verify mcp_tools_context is called with correct parameters
                    mock_mcp_context.assert_called_once_with(
                        authorization_context=authorization_context,
                        forwarded_headers={},
                    )
                    # Verify context is entered and tools are set
                    mock_context_manager.__aenter__.assert_called_once()
                    mock_set_mcp_tools.assert_called_once_with(mock_tools)
                    # Context should still be open during streaming
                    mock_context_manager.__aexit__.assert_not_called()

            # THEN context is properly exited after generator is exhausted
            # Verify __aexit__ was called with None, None, None (no exception)
            mock_context_manager.__aexit__.assert_called_once_with(None, None, None)


def test_create_pipeline_interactions_from_events_filters_tool_messages() -> None:
    # None returns None
    assert LangGraphAgent.create_pipeline_interactions_from_events(None) is None

    # Prepare events structure expected by the function using real message classes
    t1 = ToolMessage(content="tool", tool_call_id="tc_1")
    human = HumanMessage(content="hi")
    ai = AIMessage(content="ok")
    events: list[dict[str, Any]] = [
        {
            "node1": {
                "messages": [t1, human],
            }
        },
        {"node2": {"messages": [ai]}},
    ]

    sample = LangGraphAgent.create_pipeline_interactions_from_events(events)
    assert sample is not None
    # ToolMessage filtered out; order preserved
    msgs = sample.user_input
    assert len(msgs) == 2
