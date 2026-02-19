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
from ag_ui.core import RunAgentInput
from ag_ui.core import AssistantMessage
from ag_ui.core import SystemMessage as AgSystemMessage
from ag_ui.core import UserMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import MessagesState
from langgraph.graph.state import Command
from langgraph.graph.state import StateGraph

from datarobot_genai.langgraph.agent import LangGraphAgent


@pytest.fixture
def authorization_context() -> dict[str, Any]:
    return {"user": {"id": "123", "name": "bar"}}


@pytest.fixture
def run_agent_input() -> RunAgentInput:
    return RunAgentInput(
        messages=[UserMessage(content='{"topic": "AI"}', id="message_id")],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )


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


class HistoryAwareLangGraphAgent(LangGraphAgent):
    """LangGraph agent whose prompt template exposes {chat_history}."""

    @cached_property
    def workflow(self) -> StateGraph[MessagesState]:
        # Reuse the same mock workflow behaviour as SimpleLangGraphAgent; tests
        # that rely on streaming/invoke semantics use the original class.
        return SimpleLangGraphAgent().workflow

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                {
                    "role": "system",
                    "content": "History transcript:\n{chat_history}",
                },
                {
                    "role": "user",
                    "content": "Latest request: {topic}",
                },
            ]
        )

    @property
    def langgraph_config(self) -> dict[str, Any]:
        return {}


def test_convert_input_message_includes_history() -> None:
    """convert_input_message should preserve prior turns as chat history."""
    agent = SimpleLangGraphAgent()
    run_agent_input = RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are a helper."),
            UserMessage(id="user_1", content='{"topic": "First question"}'),
            AssistantMessage(id="asst_1", content="First answer"),
            UserMessage(id="user_2", content='{"topic": "Follow-up"}'),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    command = agent.convert_input_message(run_agent_input)
    all_messages = command.update["messages"]

    # Expect 3 history messages (system, first user, assistant)
    # plus the templated final user messages (system + user).
    assert len(all_messages) == 5
    assert isinstance(all_messages[0], SystemMessage)
    assert isinstance(all_messages[1], HumanMessage)
    assert isinstance(all_messages[2], AIMessage)
    assert isinstance(all_messages[-2], SystemMessage)
    assert isinstance(all_messages[-1], HumanMessage)


def test_convert_input_message_truncates_history() -> None:
    """History is truncated to MAX_HISTORY_MESSAGES prior turns."""
    agent = SimpleLangGraphAgent()
    max_history = agent.MAX_HISTORY_MESSAGES

    # Create more messages than MAX_HISTORY_MESSAGES, all user role so that
    # everything before the final one is treated as history.
    messages = [
        UserMessage(id=f"user_{i}", content=f'{{"topic": "Topic {i}"}}')
        for i in range(max_history + 10)
    ]
    run_agent_input = RunAgentInput(
        messages=messages,
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    command = agent.convert_input_message(run_agent_input)
    all_messages = command.update["messages"]

    # We expect MAX_HISTORY_MESSAGES history messages plus the 2 templated
    # messages for the final user turn.
    assert len(all_messages) == max_history + 2

def test_convert_input_message_injects_chat_history_variable() -> None:
    """convert_input_message should provide {chat_history} to the prompt template."""
    agent = HistoryAwareLangGraphAgent()
    run_agent_input = RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are a helper."),
            UserMessage(id="user_1", content='{"topic": "First question"}'),
            AssistantMessage(id="asst_1", content="First answer"),
            UserMessage(id="user_2", content='{"topic": "Follow-up"}'),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    command = agent.convert_input_message(run_agent_input)
    all_messages = command.update["messages"]

    # Find the system message generated from the prompt template and assert that
    # it contains a rendered history transcript (prior turns only).
    history_msgs = [
        m
        for m in all_messages
        if isinstance(m, SystemMessage) and "History transcript:" in str(m.content)
    ]
    assert len(history_msgs) == 1
    history_text = str(history_msgs[0].content)

    assert "system: You are a helper." in history_text
    assert 'user: {"topic": "First question"}' in history_text
    assert "assistant: First answer" in history_text
    # The latest user turn should not appear inside the history transcript.
    assert "Follow-up" not in history_text


def test_convert_input_message_zero_history_disables_history() -> None:
    """When MAX_HISTORY_MESSAGES is 0, no prior turns are included."""
    agent = SimpleLangGraphAgent()
    agent.MAX_HISTORY_MESSAGES = 0

    run_agent_input = RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are a helper."),
            UserMessage(id="user_1", content='{"topic": "First question"}'),
            AssistantMessage(id="asst_1", content="First answer"),
            UserMessage(id="user_2", content='{"topic": "Follow-up"}'),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    command = agent.convert_input_message(run_agent_input)
    all_messages = command.update["messages"]

    # Only the templated messages for the final user turn should remain.
    assert len(all_messages) == 2
    assert isinstance(all_messages[0], SystemMessage)
    assert isinstance(all_messages[1], HumanMessage)


async def test_langgraph_non_streaming(run_agent_input):
    # GIVEN a simple langgraph agent implementation
    agent = SimpleLangGraphAgent()

    # WHEN invoking the agent
    streaming_response_iterator = agent.invoke(run_agent_input)

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


async def test_invoke_calls_mcp_tools_context_and_sets_tools_and_cleans_up(
    authorization_context, run_agent_input
):
    """Test that invoke method calls mcp_tools_context and sets tools correctly."""
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

        with patch.object(agent, "set_mcp_tools") as mock_set_mcp_tools:
            # WHEN invoking the agent
            streaming_response_iterator = agent.invoke(run_agent_input)

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
                        forwarded_headers=forwarded_headers,
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
