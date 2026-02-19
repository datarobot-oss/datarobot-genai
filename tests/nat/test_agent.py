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
from ag_ui.core import SystemMessage as AgSystemMessage
from ag_ui.core import UserMessage
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
        forwarded_headers={"h1": "v1"},
        authorization_context={"c1": "v2"},
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
    with patch(
        "datarobot_genai.nat.agent.load_workflow", new_callable=MagicMock
    ) as mock_load_workflow:
        mock_load_workflow.return_value.__aenter__.return_value = AsyncMock(run=mock_run)
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


@patch.dict(os.environ, {}, clear=True)
def test_init_with_additional_kwargs(workflow_path):
    """Test initialization with additional keyword arguments."""
    # Setup
    additional_kwargs = {"extra_param1": "value1", "extra_param2": 42}

    # Execute
    agent = NatAgent(workflow_path=workflow_path, **additional_kwargs)

    # Verify that the extra parameters don't create attributes
    with pytest.raises(AttributeError):
        _ = agent.extra_param1


def test_make_chat_request_includes_history(workflow_path):
    """NatAgent.make_chat_request should include history only via `{chat_history}`."""
    agent = NatAgent(workflow_path=workflow_path)

    run_agent_input = RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are a helper."),
            UserMessage(id="user_1", content="First question"),
            AssistantMessage(id="asst_1", content="First answer"),
            UserMessage(id="user_2", content="History:\n{chat_history}\n\nLatest: Follow-up"),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )

    chat_request = agent.make_chat_request(run_agent_input)
    text = chat_request.messages[0].content
    assert "system: You are a helper." in text
    assert "user: First question" in text
    assert "assistant: First answer" in text
    assert "{chat_history}" not in text
    assert "Latest: Follow-up" in text


@pytest.mark.usefixtures("mock_intermediate_structured", "mock_load_workflow")
async def test_invoke_mcp_headers(agent, run_agent_input):
    # GIVEN: an agent with run agent input
    # WHEN: invoke the agent with a run agent input
    streaming_response_iterator = agent.invoke(run_agent_input)

    # THEN: the response is an async generator
    assert isinstance(streaming_response_iterator, AsyncGenerator)
    events = [event async for event in streaming_response_iterator]

    # THEN: the events are tuples of (result, pipeline interactions, usage)
    deltas, pipeline_interactions_list, usage_list = zip(*events)

    # THEN: the deltas are the expected result
    assert deltas == ("chunk1", "chunk2", "")

    # THEN: the pipeline interactions are the expected pipeline interactions
    assert pipeline_interactions_list == (
        None,
        None,
        MultiTurnSample(
            user_input=[
                HumanMessage(content="user prompt"),
                AIMessage(content="LLM response"),
                AIMessage(content="LLM response"),
            ]
        ),
    )

    # THEN: the usage is the expected usage
    assert usage_list == (
        {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
        {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
        {
            "completion_tokens": 2,
            "prompt_tokens": 2,
            "total_tokens": 4,
        },
    )


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

    # THEN: the events are tuples of (result, pipeline interactions, usage)
    deltas, _, _ = zip(*events)

    # THEN: the deltas are the expected result
    assert deltas == ("chunk1", "chunk2", "")

    # THEN: the mcp load workflow is called with the expected headers
    expected_headers = {
        "h1": "v1",
        "Authorization": "Bearer test-api-key",
        "X-DataRobot-Authorization-Context": (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjMSI6InYyIn0"
            ".5Elh7RxbEZV1JdUZi9duxJwXUkFRdzKhtyXyfTIj4Ms"
        ),
    }
    mock_load_workflow.assert_called_once_with(
        workflow_path,
        headers=expected_headers,
    )
