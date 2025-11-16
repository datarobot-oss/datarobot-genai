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
import time
import uuid
from functools import cached_property
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import MessagesState
from langgraph.graph.state import Command
from langgraph.graph.state import StateGraph
from openai.types import CompletionUsage
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from datarobot_genai.core.chat.responses import CustomModelChatResponse
from datarobot_genai.core.chat.responses import CustomModelStreamingResponse
from datarobot_genai.core.chat.responses import to_custom_model_chat_response
from datarobot_genai.langgraph.agent import LangGraphAgent


@pytest.fixture
def authorization_context() -> dict[str, Any]:
    return {"user": {"id": "123", "name": "bar"}}


class SimpleLangGraphAgent(LangGraphAgent):
    @cached_property
    def workflow(self) -> StateGraph[MessagesState]:
        def to_message_chunk(s):
            return AIMessageChunk(content=s)

        async def mock_stream_generator():
            # stream the first agent
            yield (
                "first_agent",
                "messages",
                (to_message_chunk("Here is the information"), {}),
            )
            yield (
                "first_agent",
                "messages",
                (to_message_chunk(" you requested about Paris....."), {}),
            )
            yield (
                "first_agent",
                "updates",
                {
                    "first_agent": {
                        "messages": [
                            HumanMessage(content="Hi, tell me about Paris."),
                            AIMessage(
                                content="Here is the information you requested about Paris....."
                            ),
                        ]
                    }
                },
            )
            yield (
                "final_agent",
                "messages",
                (to_message_chunk("Paris is the capital"), {}),
            )
            yield (
                "final_agent",
                "messages",
                (to_message_chunk(" city of France."), {}),
            )
            yield (
                "final_agent",
                "updates",
                {
                    "final_agent": {
                        "messages": [
                            HumanMessage(content="Hi, tell me about Paris."),
                            AIMessage(content="Paris is the capital city of France."),
                        ]
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
    assert response.usage.completion_tokens == 0
    assert response.usage.prompt_tokens == 0
    assert response.usage.total_tokens == 0


async def test_langgraph_streaming():
    # GIVEN a simple langgraph agent implementation
    agent = SimpleLangGraphAgent()

    # WHEN invoking the agent with a completion create params
    completion_create_params = {
        "model": "test-model",
        "messages": [{"role": "user", "content": '{"topic": "Artificial Intelligence"}'}],
        "environment_var": True,
        "stream": True,
    }
    streaming_response_iterator = await agent.invoke(completion_create_params)

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

    # THEN the streaming response iterator returns the expected responses
    # Iterate directly over the async generator to avoid event loop conflicts
    idx = 0
    async for (
        response_text,
        pipeline_interactions,
        usage_metrics,
    ) in streaming_response_iterator:
        # Create the streaming response manually for testing
        completion_id = str(uuid.uuid4())
        created = int(time.time())

        if response_text:
            choice = ChunkChoice(
                index=0,
                delta=ChoiceDelta(role="assistant", content=response_text),
                finish_reason=None,
            )
            response = CustomModelStreamingResponse(
                id=completion_id,
                object="chat.completion.chunk",
                created=created,
                model="test-model",
                choices=[choice],
                usage=CompletionUsage(**usage_metrics) if usage_metrics else None,
            )

            assert isinstance(response, CustomModelStreamingResponse)
            if idx == 0:
                assert response.choices[0].delta.content == "Here is the information"
                assert response.choices[0].finish_reason is None
                assert response.pipeline_interactions is None
            elif idx == 1:
                assert response.choices[0].delta.content == " you requested about Paris....."
                assert response.choices[0].finish_reason is None
                assert response.pipeline_interactions is None
            elif idx == 2:
                assert response.choices[0].delta.content == "Paris is the capital"
                assert response.choices[0].finish_reason is None
                assert response.pipeline_interactions is None
            elif idx == 3:
                assert response.choices[0].delta.content == " city of France."
                assert response.choices[0].finish_reason is None
                assert response.pipeline_interactions is None
        else:
            # Final chunk
            choice = ChunkChoice(
                index=0,
                delta=ChoiceDelta(role="assistant"),
                finish_reason="stop",
            )
            response = CustomModelStreamingResponse(
                id=completion_id,
                object="chat.completion.chunk",
                created=created,
                model="test-model",
                choices=[choice],
                usage=CompletionUsage(**usage_metrics) if usage_metrics else None,
                pipeline_interactions=pipeline_interactions.model_dump_json()
                if pipeline_interactions
                else None,
            )
            assert response.choices[0].delta.content is None
            assert response.choices[0].finish_reason == "stop"
            assert response.pipeline_interactions is not None
        idx += 1


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
                api_base=agent.api_base,
                api_key=agent.api_key,
                authorization_context=authorization_context,
            )

            # THEN set_mcp_tools is called with the tools from context
            mock_set_mcp_tools.assert_called_once_with(mock_tools)


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
