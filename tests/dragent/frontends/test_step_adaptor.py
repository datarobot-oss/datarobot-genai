# Copyright 2026 DataRobot, Inc. and its affiliates.
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

import json
import uuid
from types import SimpleNamespace

import pytest
from ag_ui.core import CustomEvent
from ag_ui.core import ReasoningEndEvent
from ag_ui.core import ReasoningMessageContentEvent
from ag_ui.core import ReasoningMessageEndEvent
from ag_ui.core import ReasoningMessageStartEvent
from ag_ui.core import ReasoningStartEvent
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import StepFinishedEvent
from ag_ui.core import StepStartedEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallStartEvent
from ag_ui.core.events import ToolCallResultEvent
from nat.builder.context import IntermediateStep
from nat.builder.context import IntermediateStepPayload
from nat.builder.context import IntermediateStepType
from nat.builder.context import InvocationNode
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import TokenUsageBaseModel
from nat.data_models.intermediate_step import TraceMetadata
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.step_adaptor import StepAdaptorConfig
from nat.data_models.step_adaptor import StepAdaptorMode

from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.frontends.step_adaptor import DRAgentNestedReasoningStepAdaptor


@pytest.fixture
def step_adaptor():
    return DRAgentNestedReasoningStepAdaptor(StepAdaptorConfig())


def test_step_adaptor_processes_custom_event(step_adaptor):
    # GIVEN a custom event
    step = IntermediateStep(
        parent_id="root",
        function_ancestry=InvocationNode(
            function_id="root", function_name="root", parent_id=None, parent_name=None
        ),
        payload=IntermediateStepPayload(
            event_type=IntermediateStepType.CUSTOM_START,
            name="custom_event",
            UUID=str(uuid.uuid4()),
            data=StreamEventData(input=None, output=None),
        ),
    )
    # WHEN the step is processed
    response = step_adaptor.process(step)
    # THEN the response is a custom event
    expected_response = DRAgentEventResponse(
        events=[CustomEvent(name=IntermediateStepType.CUSTOM_START, value=step.payload)],
        usage_metrics={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )
    assert response == expected_response


# BELOW is a huge test which covers a nested workflow. It is based on logs I captured
# while implementing the step adaptor.
@pytest.fixture
def intermediate_steps_ids():
    return {
        "agent_message_id": str(uuid.uuid4()),
        "planner_message_id": str(uuid.uuid4()),
        "writer_message_id": str(uuid.uuid4()),
        "content_writer_tool_call_id": str(uuid.uuid4()),
        "planner_tool_call_id": str(uuid.uuid4()),
        "writer_tool_call_id": str(uuid.uuid4()),
    }


@pytest.fixture
def payloads():
    return {
        "agent_llm_text": (
            "I'll use the content writing tool to help me provide a comprehensive response."
        ),
        "planner_outline": (
            "Blog Outline: Understanding AI Assistants\n\nKey Points:\n• AI assistants are software"
            " programs..."
        ),
        "writer_content": ("# Understanding AI Assistants\n\n## Decoding the Digital Companion..."),
        "tool_args_content_writer": (
            '{"input_message": "Write a detailed description of an AI assistant."}'
        ),
        "tool_args_planner": '{"input": "Write a detailed description of an AI assistant."}',
    }


@pytest.fixture
def expected_responses(intermediate_steps_ids, payloads):
    return [
        DRAgentEventResponse(
            events=[
                RunStartedEvent(run_id="", thread_id=""),
                StepStartedEvent(step_name="tool_calling_agent"),
            ]
        ),
        # FUNCTION_START at root level (function_level 0->1): empty (no CustomEvent fallthrough)
        DRAgentEventResponse(events=[]),
        # Root-level LLM events suppressed (stream converter handles these)
        DRAgentEventResponse(events=[]),
        DRAgentEventResponse(events=[]),
        # Root-level TOOL_START suppressed (stream converter handles these)
        DRAgentEventResponse(events=[]),
        # FUNCTION_START at level 1->2: empty (no CustomEvent fallthrough)
        DRAgentEventResponse(events=[]),
        DRAgentEventResponse(
            events=[
                ToolCallStartEvent(
                    tool_call_name="planner",
                    tool_call_id=intermediate_steps_ids["planner_tool_call_id"],
                ),
                ToolCallArgsEvent(
                    tool_call_id=intermediate_steps_ids["planner_tool_call_id"],
                    delta=payloads["tool_args_planner"],
                ),
            ]
        ),
        # FUNCTION_START (planner, level 2->3): empty
        DRAgentEventResponse(events=[]),
        DRAgentEventResponse(
            events=[
                ReasoningStartEvent(message_id=intermediate_steps_ids["planner_message_id"]),
                ReasoningMessageStartEvent(
                    message_id=intermediate_steps_ids["planner_message_id"], role="reasoning"
                ),
            ],
            model="vertex_ai/claude-3-5-haiku@20241022",
            usage_metrics={
                "prompt_tokens": 200,
                "completion_tokens": 300,
                "total_tokens": 500,
            },
        ),
        DRAgentEventResponse(
            events=[
                ReasoningMessageContentEvent(
                    message_id=intermediate_steps_ids["planner_message_id"],
                    delta=payloads["planner_outline"],
                ),
                ReasoningMessageEndEvent(message_id=intermediate_steps_ids["planner_message_id"]),
                ReasoningEndEvent(message_id=intermediate_steps_ids["planner_message_id"]),
            ],
            model="claude-3-5-haiku@20241022",
        ),
        # FUNCTION_END (planner, level 3->2): empty
        DRAgentEventResponse(events=[]),
        DRAgentEventResponse(
            events=[
                ToolCallEndEvent(tool_call_id=intermediate_steps_ids["planner_tool_call_id"]),
                ToolCallResultEvent(
                    message_id=intermediate_steps_ids["planner_tool_call_id"],
                    tool_call_id=intermediate_steps_ids["planner_tool_call_id"],
                    content=payloads["planner_outline"],
                    role="tool",
                ),
            ]
        ),
        DRAgentEventResponse(
            events=[
                ToolCallStartEvent(
                    tool_call_name="writer",
                    tool_call_id=intermediate_steps_ids["writer_tool_call_id"],
                ),
                ToolCallArgsEvent(
                    tool_call_id=intermediate_steps_ids["writer_tool_call_id"],
                    delta=json.dumps({"outline": payloads["planner_outline"]}),
                ),
            ]
        ),
        # FUNCTION_START (writer, level 2->3): empty
        DRAgentEventResponse(events=[]),
        DRAgentEventResponse(
            events=[
                ReasoningStartEvent(message_id=intermediate_steps_ids["writer_message_id"]),
                ReasoningMessageStartEvent(
                    message_id=intermediate_steps_ids["writer_message_id"], role="reasoning"
                ),
            ],
            model="vertex_ai/claude-3-5-haiku@20241022",
        ),
        DRAgentEventResponse(
            events=[
                ReasoningMessageContentEvent(
                    message_id=intermediate_steps_ids["writer_message_id"],
                    delta=payloads["writer_content"],
                ),
                ReasoningMessageEndEvent(message_id=intermediate_steps_ids["writer_message_id"]),
                ReasoningEndEvent(message_id=intermediate_steps_ids["writer_message_id"]),
            ],
            model="claude-3-5-haiku@20241022",
            usage_metrics={
                "prompt_tokens": 300,
                "completion_tokens": 400,
                "total_tokens": 700,
            },
        ),
        # FUNCTION_END (writer, level 3->2): empty
        DRAgentEventResponse(events=[]),
        DRAgentEventResponse(
            events=[
                ToolCallEndEvent(tool_call_id=intermediate_steps_ids["writer_tool_call_id"]),
                ToolCallResultEvent(
                    message_id=intermediate_steps_ids["writer_tool_call_id"],
                    tool_call_id=intermediate_steps_ids["writer_tool_call_id"],
                    content=payloads["writer_content"],
                    role="tool",
                ),
            ]
        ),
        # FUNCTION_END (content_writer_pipeline, level 2->1): empty
        DRAgentEventResponse(events=[]),
        # Root-level TOOL_END (content_writer_pipeline) suppressed
        DRAgentEventResponse(events=[]),
        # FUNCTION_END (tool_calling_agent, level 1->0): empty
        DRAgentEventResponse(events=[]),
        DRAgentEventResponse(
            events=[
                StepFinishedEvent(step_name="tool_calling_agent"),
                RunFinishedEvent(run_id="", thread_id=""),
            ]
        ),
    ]


# Based on the intermediate_steps.txt file
@pytest.fixture
def intermediate_steps_for_nested_reasoning(intermediate_steps_ids, payloads):
    """Return steps from workflow log: root -> tool_calling_agent -> content_writer_pipeline."""
    root_ancestry = InvocationNode(
        function_id="root", function_name="root", parent_id=None, parent_name=None
    )
    agent_ancestry = InvocationNode(
        function_id="31d55980-fc9c-453d-b3b3-934963938bd9",
        function_name="tool_calling_agent",
        parent_id="root",
        parent_name="root",
    )
    pipeline_ancestry = InvocationNode(
        function_id="33f4f8be-ff88-41e1-9b0f-0ecc4b95fc70",
        function_name="content_writer_pipeline",
        parent_id="31d55980-fc9c-453d-b3b3-934963938bd9",
        parent_name="tool_calling_agent",
    )
    planner_ancestry = InvocationNode(
        function_id="7d21befe-ed77-4c98-a7a5-149fe51bdc0f",
        function_name="planner",
        parent_id="33f4f8be-ff88-41e1-9b0f-0ecc4b95fc70",
        parent_name="content_writer_pipeline",
    )
    writer_ancestry = InvocationNode(
        function_id="89873563-a998-4a6e-8970-4681d9fc82c0",
        function_name="writer",
        parent_id="33f4f8be-ff88-41e1-9b0f-0ecc4b95fc70",
        parent_name="content_writer_pipeline",
    )

    return [
        # WORKFLOW_START
        IntermediateStep(
            parent_id="root",
            function_ancestry=root_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.WORKFLOW_START,
                name="tool_calling_agent",
                UUID="f0af669d-919f-4eb2-854c-51e9078d8e2d",
                data=StreamEventData(input=None, output=None),
            ),
        ),
        # FUNCTION_START tool_calling_agent
        IntermediateStep(
            parent_id="f0af669d-919f-4eb2-854c-51e9078d8e2d",
            function_ancestry=agent_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.FUNCTION_START,
                name="tool_calling_agent",
                UUID="31d55980-fc9c-453d-b3b3-934963938bd9",
                data=StreamEventData(input=None, output=None),
            ),
        ),
        # LLM_START (agent)
        IntermediateStep(
            parent_id="31d55980-fc9c-453d-b3b3-934963938bd9",
            function_ancestry=agent_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                name="vertex_ai/claude-3-5-haiku@20241022",
                UUID=intermediate_steps_ids["agent_message_id"],
                data=StreamEventData(input=["system", "user message"], output=None),
            ),
        ),
        # LLM_END (agent) with tool_calls
        IntermediateStep(
            parent_id="31d55980-fc9c-453d-b3b3-934963938bd9",
            function_ancestry=agent_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                name="claude-3-5-haiku@20241022",
                UUID=intermediate_steps_ids["agent_message_id"],
                data=StreamEventData(
                    input="who are you?",
                    output=payloads["agent_llm_text"],
                    payload=SimpleNamespace(
                        text=payloads["agent_llm_text"],
                        message=SimpleNamespace(tool_calls=[]),
                    ),
                ),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(
                        prompt_tokens=100,
                        completion_tokens=200,
                        cached_tokens=0,
                        reasoning_tokens=0,
                        total_tokens=300,
                    ),
                    num_llm_calls=0,
                    seconds_between_calls=0,
                ),
            ),
        ),
        # TOOL_START content_writer_pipeline
        IntermediateStep(
            parent_id="31d55980-fc9c-453d-b3b3-934963938bd9",
            function_ancestry=agent_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_START,
                name="content_writer_pipeline",
                UUID=intermediate_steps_ids["content_writer_tool_call_id"],
                data=StreamEventData(
                    input=payloads["tool_args_content_writer"],
                    output=None,
                ),
                metadata=TraceMetadata(
                    tool_inputs=json.loads(payloads["tool_args_content_writer"]),
                    tool_info={
                        "name": "content_writer_pipeline",
                        "description": "A tool that plans and writes content.",
                    },
                ),
            ),
        ),
        # FUNCTION_START content_writer_pipeline
        IntermediateStep(
            parent_id=intermediate_steps_ids["content_writer_tool_call_id"],
            function_ancestry=pipeline_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.FUNCTION_START,
                name="content_writer_pipeline",
                UUID="33f4f8be-ff88-41e1-9b0f-0ecc4b95fc70",
                data=StreamEventData(
                    input=payloads["tool_args_content_writer"],
                    output=None,
                ),
            ),
        ),
        # TOOL_START planner
        IntermediateStep(
            parent_id="33f4f8be-ff88-41e1-9b0f-0ecc4b95fc70",
            function_ancestry=pipeline_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_START,
                name="planner",
                UUID=intermediate_steps_ids["planner_tool_call_id"],
                data=StreamEventData(
                    input="Write a detailed description of an AI assistant.",
                    output=None,
                ),
                metadata=TraceMetadata(
                    tool_inputs={"input": "Write a detailed description of an AI assistant."},
                    tool_info={"name": "planner", "description": "planner"},
                ),
            ),
        ),
        # FUNCTION_START planner
        IntermediateStep(
            parent_id=intermediate_steps_ids["planner_tool_call_id"],
            function_ancestry=planner_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.FUNCTION_START,
                name="planner",
                UUID="7d21befe-ed77-4c98-a7a5-149fe51bdc0f",
                data=StreamEventData(
                    input="Write a detailed description of an AI assistant.",
                    output=None,
                ),
            ),
        ),
        # LLM_START (planner - nested)
        IntermediateStep(
            parent_id="7d21befe-ed77-4c98-a7a5-149fe51bdc0f",
            function_ancestry=planner_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                name="vertex_ai/claude-3-5-haiku@20241022",
                UUID=intermediate_steps_ids["planner_message_id"],
                data=StreamEventData(input=["HumanMessage(...)"], output=None),
            ),
        ),
        # LLM_END (planner - nested, no tool_calls)
        IntermediateStep(
            parent_id="7d21befe-ed77-4c98-a7a5-149fe51bdc0f",
            function_ancestry=planner_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                name="claude-3-5-haiku@20241022",
                UUID=intermediate_steps_ids["planner_message_id"],
                data=StreamEventData(
                    input="You are a content planner...",
                    output=payloads["planner_outline"],
                    payload=SimpleNamespace(
                        text=payloads["planner_outline"],
                        message=SimpleNamespace(tool_calls=[]),
                    ),
                ),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(
                        prompt_tokens=200,
                        completion_tokens=300,
                        cached_tokens=0,
                        reasoning_tokens=0,
                        total_tokens=500,
                    ),
                    num_llm_calls=0,
                    seconds_between_calls=0,
                ),
            ),
        ),
        # FUNCTION_END planner
        IntermediateStep(
            parent_id=intermediate_steps_ids["planner_tool_call_id"],
            function_ancestry=planner_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.FUNCTION_END,
                name="planner",
                UUID="7d21befe-ed77-4c98-a7a5-149fe51bdc0f",
                data=StreamEventData(
                    input="Write a detailed description of an AI assistant.",
                    output=payloads["planner_outline"],
                ),
            ),
        ),
        # TOOL_END planner
        IntermediateStep(
            parent_id="33f4f8be-ff88-41e1-9b0f-0ecc4b95fc70",
            function_ancestry=pipeline_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_END,
                name="planner",
                UUID=intermediate_steps_ids["planner_tool_call_id"],
                data=StreamEventData(
                    input="Write a detailed description of an AI assistant.",
                    output=payloads["planner_outline"],
                ),
                metadata=TraceMetadata(tool_outputs=payloads["planner_outline"]),
            ),
        ),
        # TOOL_START writer
        IntermediateStep(
            parent_id="33f4f8be-ff88-41e1-9b0f-0ecc4b95fc70",
            function_ancestry=pipeline_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_START,
                name="writer",
                UUID=intermediate_steps_ids["writer_tool_call_id"],
                data=StreamEventData(input=payloads["planner_outline"], output=None),
                metadata=TraceMetadata(
                    tool_inputs={"outline": payloads["planner_outline"]},
                    tool_info={"name": "writer", "description": "writer"},
                ),
            ),
        ),
        # FUNCTION_START writer
        IntermediateStep(
            parent_id=intermediate_steps_ids["writer_tool_call_id"],
            function_ancestry=writer_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.FUNCTION_START,
                name="writer",
                UUID="89873563-a998-4a6e-8970-4681d9fc82c0",
                data=StreamEventData(input=payloads["planner_outline"], output=None),
            ),
        ),
        # LLM_START (writer - nested)
        IntermediateStep(
            parent_id="89873563-a998-4a6e-8970-4681d9fc82c0",
            function_ancestry=writer_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                name="vertex_ai/claude-3-5-haiku@20241022",
                UUID=intermediate_steps_ids["writer_message_id"],
                data=StreamEventData(input=["HumanMessage(...)"], output=None),
            ),
        ),
        # LLM_END (writer - nested)
        IntermediateStep(
            parent_id="89873563-a998-4a6e-8970-4681d9fc82c0",
            function_ancestry=writer_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                name="claude-3-5-haiku@20241022",
                UUID=intermediate_steps_ids["writer_message_id"],
                data=StreamEventData(
                    input="You are a content writer...",
                    output=payloads["writer_content"],
                    payload=SimpleNamespace(
                        text=payloads["writer_content"],
                        message=SimpleNamespace(tool_calls=[]),
                    ),
                ),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(
                        prompt_tokens=300,
                        completion_tokens=400,
                        cached_tokens=0,
                        reasoning_tokens=0,
                        total_tokens=700,
                    ),
                    num_llm_calls=0,
                    seconds_between_calls=0,
                ),
            ),
        ),
        # FUNCTION_END writer
        IntermediateStep(
            parent_id="019c7ba7-6881-7f31-ad5a-0b135aa558b7",
            function_ancestry=writer_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.FUNCTION_END,
                name="writer",
                UUID="89873563-a998-4a6e-8970-4681d9fc82c0",
                data=StreamEventData(
                    input=payloads["planner_outline"], output=payloads["writer_content"]
                ),
            ),
        ),
        # TOOL_END writer
        IntermediateStep(
            parent_id="33f4f8be-ff88-41e1-9b0f-0ecc4b95fc70",
            function_ancestry=pipeline_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_END,
                name="writer",
                UUID=intermediate_steps_ids["writer_tool_call_id"],
                data=StreamEventData(
                    input=payloads["planner_outline"], output=payloads["writer_content"]
                ),
                metadata=TraceMetadata(tool_outputs=payloads["writer_content"]),
            ),
        ),
        # FUNCTION_END content_writer_pipeline
        IntermediateStep(
            parent_id="019c7ba7-4f3c-7073-ae6d-fbb39f4abf73",
            function_ancestry=pipeline_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.FUNCTION_END,
                name="content_writer_pipeline",
                UUID="33f4f8be-ff88-41e1-9b0f-0ecc4b95fc70",
                data=StreamEventData(
                    input={"input_message": "Write a detailed description of an AI assistant."},
                    output=payloads["writer_content"],
                ),
            ),
        ),
        # TOOL_END content_writer_pipeline (back at agent)
        IntermediateStep(
            parent_id="31d55980-fc9c-453d-b3b3-934963938bd9",
            function_ancestry=agent_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_END,
                name="content_writer_pipeline",
                UUID=intermediate_steps_ids["content_writer_tool_call_id"],
                data=StreamEventData(
                    input="{'input_message': \"...\"}",
                    output=payloads["writer_content"],
                ),
                metadata=TraceMetadata(tool_outputs=payloads["writer_content"]),
            ),
        ),
        # FUNCTION_END tool_calling_agent
        IntermediateStep(
            parent_id="f0af669d-919f-4eb2-854c-51e9078d8e2d",
            function_ancestry=agent_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.FUNCTION_END,
                name="tool_calling_agent",
                UUID="31d55980-fc9c-453d-b3b3-934963938bd9",
                data=StreamEventData(input=None, output=None),
            ),
        ),
        # WORKFLOW_END
        IntermediateStep(
            parent_id="root",
            function_ancestry=root_ancestry,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.WORKFLOW_END,
                name="tool_calling_agent",
                UUID="f0af669d-919f-4eb2-854c-51e9078d8e2d",
                data=StreamEventData(input=None, output=None),
            ),
        ),
    ]


def test_step_adaptor_processes_nested_reasoning_steps(
    step_adaptor, intermediate_steps_for_nested_reasoning, expected_responses
):
    actual_responses = []
    for step in intermediate_steps_for_nested_reasoning:
        result = step_adaptor.process(step)
        assert result is not None
        assert isinstance(result, DRAgentEventResponse)
        actual_responses.append(result)

    assert len(actual_responses) == len(expected_responses), (
        f"Response count: actual {len(actual_responses)} != expected {len(expected_responses)}"
    )

    for i, (actual_resp, expected_resp) in enumerate(zip(actual_responses, expected_responses)):
        assert len(actual_resp.events) == len(expected_resp.events), (
            f"Response {i}: event count mismatch"
        )
        for j, (actual_ev, expected_ev) in enumerate(zip(actual_resp.events, expected_resp.events)):
            assert actual_ev == expected_ev, f"Response {i} event {j}: {actual_ev} != {expected_ev}"


def test_step_adaptor_sends_nothing_in_mode_off(intermediate_steps_for_nested_reasoning):
    step_adaptor = DRAgentNestedReasoningStepAdaptor(StepAdaptorConfig(mode=StepAdaptorMode.OFF))
    for step in intermediate_steps_for_nested_reasoning:
        result = step_adaptor.process(step)
        assert result is None


def _make_tool_start_step(
    tool_call_id: str, data_input, metadata: TraceMetadata
) -> IntermediateStep:
    return IntermediateStep(
        parent_id="root",
        function_ancestry=InvocationNode(
            function_id="root", function_name="root", parent_id=None, parent_name=None
        ),
        payload=IntermediateStepPayload(
            event_type=IntermediateStepType.TOOL_START,
            name="my_tool",
            UUID=tool_call_id,
            data=StreamEventData(input=data_input, output=None),
            metadata=metadata,
        ),
    )


class TestHandleToolArgsEncoding:
    """Tests for the tool_inputs / data.input fallback logic in _handle_tool."""

    @pytest.fixture(autouse=True)
    def _set_nested_level(self, step_adaptor):
        """Set function_level > 1 so root-level tool suppression is bypassed."""
        step_adaptor.function_level = 2

    def test_tool_inputs_dict_used_when_present(self, step_adaptor):
        """metadata.tool_inputs (dict) takes precedence; result must be valid JSON."""
        tool_call_id = str(uuid.uuid4())
        inputs = {"param": "value", "count": 3}
        step = _make_tool_start_step(
            tool_call_id,
            data_input="{'param': 'value', 'count': 3}",  # LangChain-style non-JSON string
            metadata=TraceMetadata(tool_inputs=inputs),
        )
        response = step_adaptor.process(step)
        args_event = next(e for e in response.events if isinstance(e, ToolCallArgsEvent))
        assert args_event.delta == json.dumps(inputs)

    def test_data_input_dict_serialized_to_json(self, step_adaptor):
        """When tool_inputs is absent, a dict data.input is JSON-serialized."""
        tool_call_id = str(uuid.uuid4())
        input_dict = {"key": "val"}
        step = _make_tool_start_step(
            tool_call_id,
            data_input=input_dict,
            metadata=TraceMetadata(),
        )
        response = step_adaptor.process(step)
        args_event = next(e for e in response.events if isinstance(e, ToolCallArgsEvent))
        assert args_event.delta == json.dumps(input_dict)

    def test_data_input_valid_json_string_passed_through(self, step_adaptor):
        """When tool_inputs is absent and data.input is a valid JSON string, it is used as-is."""
        tool_call_id = str(uuid.uuid4())
        valid_json = '{"key": "value"}'
        step = _make_tool_start_step(
            tool_call_id,
            data_input=valid_json,
            metadata=TraceMetadata(),
        )
        response = step_adaptor.process(step)
        args_event = next(e for e in response.events if isinstance(e, ToolCallArgsEvent))
        assert args_event.delta == valid_json

    def test_data_input_invalid_json_string_falls_back_to_empty_object(self, step_adaptor):
        """When tool_inputs is absent and data.input is a non-JSON string (e.g. Python repr),
        delta falls back to '{}' and a warning is logged.
        """
        tool_call_id = str(uuid.uuid4())
        repr_string = "{'key': 'value'}"  # single-quoted Python repr, not valid JSON
        step = _make_tool_start_step(
            tool_call_id,
            data_input=repr_string,
            metadata=TraceMetadata(),
        )
        response = step_adaptor.process(step)
        args_event = next(e for e in response.events if isinstance(e, ToolCallArgsEvent))
        assert args_event.delta == "{}"

    def test_data_input_none_produces_empty_json_object(self, step_adaptor):
        """When both tool_inputs and data.input are absent, delta defaults to '{}'."""
        tool_call_id = str(uuid.uuid4())
        step = _make_tool_start_step(
            tool_call_id,
            data_input=None,
            metadata=TraceMetadata(),
        )
        response = step_adaptor.process(step)
        args_event = next(e for e in response.events if isinstance(e, ToolCallArgsEvent))
        assert args_event.delta == "{}"
