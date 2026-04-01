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

import os
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from ag_ui.core import AssistantMessage
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import SystemMessage as AgSystemMessage
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import UserMessage
from ag_ui.core.types import FunctionCall as AgFunctionCall
from ag_ui.core.types import ToolCall as AgToolCall
from ag_ui.core.types import ToolMessage as AgToolMessage
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.profiler.callbacks.token_usage_base_model import TokenUsageBaseModel
from ragas import MultiTurnSample
from ragas.messages import AIMessage
from ragas.messages import HumanMessage

from datarobot_genai.nat.agent import NatAgent


@pytest.fixture
def workflow_path():
    return Path("some_path") / "workflow.yaml"


@pytest.fixture
def agent(workflow_path):
    return NatAgent(
        workflow_path=workflow_path,
    )


@pytest.fixture
def agent_with_headers(workflow_path):
    return NatAgent(
        workflow_path=workflow_path,
        forwarded_headers={
            "h1": "v1",
            "Authorization": "Bearer test-api-key",
        },
    )


@pytest.fixture
def agent_steps():
    start_step = IntermediateStep(
        parent_id="some_parent_id",
        function_ancestry=InvocationNode(
            function_id="some_function_id", function_name="some_function"
        ),
        payload=IntermediateStepPayload(
            event_type=IntermediateStepType.LLM_START,
            data=StreamEventData(
                input=[
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "user prompt"},
                ]
            ),
        ),
    )
    new_token_step = IntermediateStep(
        parent_id="some_parent_id",
        function_ancestry=InvocationNode(
            function_id="some_function_id", function_name="some_function"
        ),
        payload=IntermediateStepPayload(
            event_type=IntermediateStepType.LLM_NEW_TOKEN,
            usage_info=UsageInfo(
                token_usage=TokenUsageBaseModel(
                    total_tokens=2, completion_tokens=1, prompt_tokens=1
                )
            ),
        ),
    )
    end_step = IntermediateStep(
        parent_id="some_parent_id",
        function_ancestry=InvocationNode(
            function_id="some_function_id", function_name="some_function"
        ),
        payload=IntermediateStepPayload(
            event_type=IntermediateStepType.LLM_END,
            data=StreamEventData(output="LLM response"),
            usage_info=UsageInfo(
                token_usage=TokenUsageBaseModel(
                    total_tokens=2, completion_tokens=1, prompt_tokens=1
                )
            ),
        ),
    )
    return [start_step, new_token_step, end_step, end_step]


@pytest.fixture
def mock_intermediate_structured(agent_steps):
    with patch(
        "datarobot_genai.nat.agent.pull_intermediate_structured", new_callable=AsyncMock
    ) as mock_intermediate_structured:
        mock_intermediate_structured.return_value = agent_steps
        yield mock_intermediate_structured


@pytest.fixture
def mock_load_workflow():
    async def mock_result_stream():
        yield "chunk1"
        yield "chunk2"

    mock_run = MagicMock()
    mock_run.return_value.__aenter__.return_value = AsyncMock(result_stream=mock_result_stream)
    mock_session = MagicMock()
    mock_session.return_value.__aenter__.return_value = AsyncMock(run=mock_run)
    with patch(
        "datarobot_genai.nat.agent.load_workflow", new_callable=MagicMock
    ) as mock_load_workflow:
        mock_load_workflow.return_value.__aenter__.return_value = AsyncMock(session=mock_session)
        yield mock_load_workflow


@pytest.fixture
def run_agent_input() -> RunAgentInput:
    return RunAgentInput(
        messages=[UserMessage(content="Artificial Intelligence", id="message_id")],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )


@pytest.fixture
def patch_environment_variables():
    deployment_id = "abc123def456789012345678"
    api_base = "https://app.datarobot.com/api/v2"
    api_key = "test-api-key"

    with patch.dict(
        os.environ,
        {
            "MCP_DEPLOYMENT_ID": deployment_id,
            "DATAROBOT_ENDPOINT": api_base,
            "DATAROBOT_API_TOKEN": api_key,
        },
        clear=True,
    ):
        yield


@pytest.mark.usefixtures("mock_intermediate_structured", "mock_load_workflow")
async def test_invoke_includes_chat_history(workflow_path, mock_load_workflow):
    """NatAgent.invoke() passes structured messages for multi-turn conversations."""
    agent = NatAgent(workflow_path=workflow_path)

    run_agent_input = RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are a helper."),
            UserMessage(id="user_1", content="First question"),
            AssistantMessage(id="asst_1", content="First answer"),
            UserMessage(id="user_2", content="Follow-up"),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    _ = [event async for event in agent.invoke(run_agent_input)]

    # Get the ChatRequest passed to session.run()
    mock_workflow = mock_load_workflow.return_value.__aenter__.return_value
    mock_session = mock_workflow.session.return_value.__aenter__.return_value
    chat_request = mock_session.run.call_args[0][0]

    # Multi-turn path: structured messages, not a single concatenated string
    assert len(chat_request.messages) == 4
    assert chat_request.messages[0].role.value == "system"
    assert chat_request.messages[0].content == "You are a helper."
    assert chat_request.messages[1].role.value == "user"
    assert chat_request.messages[1].content == "First question"
    assert chat_request.messages[2].role.value == "assistant"
    assert chat_request.messages[2].content == "First answer"
    assert chat_request.messages[3].role.value == "user"
    assert chat_request.messages[3].content == "Follow-up"


@pytest.mark.usefixtures("mock_intermediate_structured", "mock_load_workflow")
async def test_invoke_mcp_headers(agent, run_agent_input):
    # GIVEN: an agent with run agent input
    # WHEN: invoke the agent with a run agent input
    streaming_response_iterator = agent.invoke(run_agent_input)

    # THEN: the response is an async generator
    assert isinstance(streaming_response_iterator, AsyncGenerator)
    events = [event async for event in streaming_response_iterator]

    # THEN: events follow AG-UI lifecycle pattern
    # RunStarted, TextMessageStart, TextMessageContent(chunk1), TextMessageContent(chunk2),
    # TextMessageEnd, RunFinished
    ag_events, pipeline_interactions_list, usage_list = zip(*events)

    # THEN: first event is RunStartedEvent
    assert isinstance(ag_events[0], RunStartedEvent)

    # THEN: text message events contain the chunks
    assert isinstance(ag_events[1], TextMessageStartEvent)
    assert isinstance(ag_events[2], TextMessageContentEvent)
    assert ag_events[2].delta == "chunk1"
    assert isinstance(ag_events[3], TextMessageContentEvent)
    assert ag_events[3].delta == "chunk2"
    assert isinstance(ag_events[4], TextMessageEndEvent)

    # THEN: last event is RunFinishedEvent with pipeline interactions
    assert isinstance(ag_events[-1], RunFinishedEvent)
    assert isinstance(pipeline_interactions_list[-1], MultiTurnSample)
    assert pipeline_interactions_list[-1] == MultiTurnSample(
        user_input=[
            HumanMessage(content="user prompt"),
            AIMessage(content="LLM response"),
            AIMessage(content="LLM response"),
        ]
    )

    # THEN: final usage has accumulated token counts
    assert usage_list[-1] == {
        "completion_tokens": 2,
        "prompt_tokens": 2,
        "total_tokens": 4,
    }


@pytest.mark.usefixtures("mock_intermediate_structured")
async def test_invoke_uses_thread_id_as_session_user_id(agent, run_agent_input, mock_load_workflow):
    # GIVEN: an agent and run agent input with a known thread_id
    _ = [event async for event in agent.invoke(run_agent_input)]

    # THEN: workflow.session() is called with user_id equal to the thread_id
    mock_workflow = mock_load_workflow.return_value.__aenter__.return_value
    mock_workflow.session.assert_called_once_with(user_id=run_agent_input.thread_id)


@pytest.mark.usefixtures("mock_intermediate_structured", "patch_environment_variables")
async def test_streaming_mcp_headers(
    agent_with_headers, run_agent_input, mock_load_workflow, workflow_path
):
    # GIVEN: an agent with headers and a run agent input
    # WHEN: invoke the agent with a run agent input
    streaming_response_iterator = agent_with_headers.invoke(run_agent_input)

    # THEN: the response is an async generator
    assert isinstance(streaming_response_iterator, AsyncGenerator)
    events = [event async for event in streaming_response_iterator]

    # THEN: events follow AG-UI lifecycle with text content chunks
    ag_events, _, _ = zip(*events)
    content_events = [e for e in ag_events if isinstance(e, TextMessageContentEvent)]
    assert [e.delta for e in content_events] == ["chunk1", "chunk2"]

    # THEN: the mcp load workflow is called with the expected headers
    expected_headers = {
        "h1": "v1",
        "Authorization": "Bearer test-api-key",
    }
    mock_load_workflow.assert_called_once_with(
        workflow_path,
        headers=expected_headers,
    )


@pytest.mark.usefixtures("mock_intermediate_structured", "mock_load_workflow")
async def test_invoke_multi_turn_passes_structured_messages(workflow_path, mock_load_workflow):
    """Multi-turn conversations pass structured NAT messages, not a single string."""
    agent = NatAgent(workflow_path=workflow_path)

    run_agent_input = RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are helpful."),
            UserMessage(id="user_1", content="search cats"),
            AssistantMessage(
                id="asst_1",
                content=None,
                tool_calls=[
                    AgToolCall(
                        id="call_1",
                        function=AgFunctionCall(name="search", arguments='{"q": "cats"}'),
                    )
                ],
            ),
            AgToolMessage(id="tool_1", content="found cats", tool_call_id="call_1"),
            UserMessage(id="user_2", content="tell me more"),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    _ = [event async for event in agent.invoke(run_agent_input)]

    # Get the ChatRequest that was passed to session.run()
    mock_workflow = mock_load_workflow.return_value.__aenter__.return_value
    mock_session = mock_workflow.session.return_value.__aenter__.return_value
    mock_run = mock_session.run
    assert mock_run.called
    chat_request = mock_run.call_args[0][0]

    # Should have multiple messages, not a single user message
    assert len(chat_request.messages) > 1
    # First should be system
    assert chat_request.messages[0].role.value == "system"
