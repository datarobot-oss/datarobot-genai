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
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
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

from datarobot_genai.dragent.response import DRAgentEventResponse
from datarobot_genai.dragent.step_adaptor import DRAgentNestedReasoningStepAdaptor


@pytest.mark.parametrize(
    "config",
    [
        StepAdaptorConfig(
            mode=StepAdaptorMode.CUSTOM, custom_event_types=[IntermediateStepType.CUSTOM_START]
        ),
        StepAdaptorConfig(mode=StepAdaptorMode.OFF),
    ],
)
def test_step_adaptor_init_fails_with_non_default_config(config):
    with pytest.raises(ValueError):
        DRAgentNestedReasoningStepAdaptor(config)


@pytest.mark.parametrize(
    "config",
    [
        StepAdaptorConfig(mode=StepAdaptorMode.DEFAULT),
        StepAdaptorConfig(),
    ],
)
def test_step_adaptor_init_succeeds_with_default_config(config):
    adaptor = DRAgentNestedReasoningStepAdaptor(config)
    assert adaptor.config == StepAdaptorConfig()


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
        "tool_args_planner": "Write a detailed description of an AI assistant.",
    }


@pytest.fixture
def expected_responses(intermediate_steps_ids, payloads):
    return [
        DRAgentEventResponse(events=[RunStartedEvent(run_id="", thread_id="")]),
        DRAgentEventResponse(events=[StepStartedEvent(step_name="tool_calling_agent")]),
        DRAgentEventResponse(
            events=[],
            model="vertex_ai/claude-3-5-haiku@20241022",
        ),
        DRAgentEventResponse(
            events=[
                TextMessageStartEvent(message_id=intermediate_steps_ids["agent_message_id"]),
                TextMessageContentEvent(
                    message_id=intermediate_steps_ids["agent_message_id"],
                    delta=payloads["agent_llm_text"],
                ),
                TextMessageEndEvent(message_id=intermediate_steps_ids["agent_message_id"]),
            ],
            model="claude-3-5-haiku@20241022",
            usage_metrics={
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
            },
        ),
        DRAgentEventResponse(
            events=[
                ToolCallStartEvent(
                    tool_call_name="content_writer_pipeline",
                    tool_call_id=intermediate_steps_ids["content_writer_tool_call_id"],
                ),
                ToolCallArgsEvent(
                    tool_call_id=intermediate_steps_ids["content_writer_tool_call_id"],
                    delta=payloads["tool_args_content_writer"],
                ),
            ]
        ),
        DRAgentEventResponse(events=[StepStartedEvent(step_name="content_writer_pipeline")]),
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
        DRAgentEventResponse(events=[StepStartedEvent(step_name="planner")]),
        DRAgentEventResponse(
            events=[
                ReasoningStartEvent(message_id=intermediate_steps_ids["planner_message_id"]),
                ReasoningMessageStartEvent(
                    message_id=intermediate_steps_ids["planner_message_id"], role="assistant"
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
        DRAgentEventResponse(events=[StepFinishedEvent(step_name="planner")]),
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
                    delta=payloads["planner_outline"],
                ),
            ]
        ),
        DRAgentEventResponse(events=[StepStartedEvent(step_name="writer")]),
        DRAgentEventResponse(
            events=[
                ReasoningStartEvent(message_id=intermediate_steps_ids["writer_message_id"]),
                ReasoningMessageStartEvent(
                    message_id=intermediate_steps_ids["writer_message_id"], role="assistant"
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
        DRAgentEventResponse(events=[StepFinishedEvent(step_name="writer")]),
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
        DRAgentEventResponse(events=[StepFinishedEvent(step_name="content_writer_pipeline")]),
        DRAgentEventResponse(
            events=[
                ToolCallEndEvent(
                    tool_call_id=intermediate_steps_ids["content_writer_tool_call_id"]
                ),
                ToolCallResultEvent(
                    message_id=intermediate_steps_ids["content_writer_tool_call_id"],
                    tool_call_id=intermediate_steps_ids["content_writer_tool_call_id"],
                    content=payloads["writer_content"],
                    role="tool",
                ),
            ]
        ),
        DRAgentEventResponse(events=[StepFinishedEvent(step_name="tool_calling_agent")]),
        DRAgentEventResponse(events=[RunFinishedEvent(run_id="", thread_id="")]),
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
                metadata=TraceMetadata(tool_info={"name": "planner", "description": "planner"}),
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
                metadata=TraceMetadata(tool_info={"name": "writer", "description": "writer"}),
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


def _make_llm_step(event_type, uuid, *, text=None, chunk=None, tool_calls=None):
    """Build an LLM-related IntermediateStep."""
    root_ancestry = InvocationNode(
        function_id="root", function_name="root", parent_id=None, parent_name=None
    )
    payload_ns = SimpleNamespace(
        text=text,
        message=SimpleNamespace(tool_calls=tool_calls or []),
    )
    return IntermediateStep(
        parent_id="root",
        function_ancestry=root_ancestry,
        payload=IntermediateStepPayload(
            event_type=event_type,
            name="test-model",
            UUID=uuid,
            data=StreamEventData(input=None, output=text, payload=payload_ns, chunk=chunk),
        ),
    )


def _enter_primary_function(step_adaptor, function_name="test_function"):
    """Enter primary function scope for LLM primary-path tests."""
    root_ancestry = InvocationNode(
        function_id="root", function_name="root", parent_id=None, parent_name=None
    )
    function_step = IntermediateStep(
        parent_id="root",
        function_ancestry=root_ancestry,
        payload=IntermediateStepPayload(
            event_type=IntermediateStepType.FUNCTION_START,
            name=function_name,
            UUID=str(uuid.uuid4()),
            data=StreamEventData(input=None, output=None),
        ),
    )
    step_adaptor.process(function_step)


def test_llm_end_with_empty_text_skips_content_event(step_adaptor):
    """LLM_END with empty text and no streamed tokens should emit no text events."""
    msg_id = str(uuid.uuid4())

    _enter_primary_function(step_adaptor)
    # Prime the adaptor with LLM_START so seen_llm_new_token is reset
    start_step = _make_llm_step(IntermediateStepType.LLM_START, msg_id)
    step_adaptor.process(start_step)

    # WHEN LLM_END arrives with empty text
    end_step = _make_llm_step(IntermediateStepType.LLM_END, msg_id, text="")
    response = step_adaptor.process(end_step)

    # THEN no text events are emitted
    assert response is not None
    assert (response.events or []) == []


def test_llm_new_token_with_empty_chunk_skips_content_event(step_adaptor):
    """LLM_NEW_TOKEN with an empty chunk should not emit text events."""
    msg_id = str(uuid.uuid4())

    _enter_primary_function(step_adaptor)
    # Prime with LLM_START
    start_step = _make_llm_step(IntermediateStepType.LLM_START, msg_id)
    step_adaptor.process(start_step)

    # WHEN LLM_NEW_TOKEN arrives with an empty chunk
    token_step = _make_llm_step(IntermediateStepType.LLM_NEW_TOKEN, msg_id, chunk="")
    response = step_adaptor.process(token_step)

    # THEN no text events are emitted
    assert response is not None
    assert (response.events or []) == []


def test_llm_end_after_only_empty_stream_chunks_emits_no_text_events(step_adaptor):
    """LLM_END after only empty streamed chunks should emit no text events."""
    msg_id = str(uuid.uuid4())

    _enter_primary_function(step_adaptor)
    step_adaptor.process(_make_llm_step(IntermediateStepType.LLM_START, msg_id))
    step_adaptor.process(_make_llm_step(IntermediateStepType.LLM_NEW_TOKEN, msg_id, chunk=""))
    response = step_adaptor.process(_make_llm_step(IntermediateStepType.LLM_END, msg_id, text=""))

    assert response is not None
    assert (response.events or []) == []


def test_llm_new_token_with_non_empty_chunk_emits_content_event(step_adaptor):
    """LLM_NEW_TOKEN with a non-empty chunk should emit a TextMessageContentEvent."""
    msg_id = str(uuid.uuid4())

    _enter_primary_function(step_adaptor)
    start_step = _make_llm_step(IntermediateStepType.LLM_START, msg_id)
    step_adaptor.process(start_step)

    token_step = _make_llm_step(IntermediateStepType.LLM_NEW_TOKEN, msg_id, chunk="hello")
    response = step_adaptor.process(token_step)

    assert response is not None
    content_events = [e for e in (response.events or []) if isinstance(e, TextMessageContentEvent)]
    assert len(content_events) == 1
    assert content_events[0].delta == "hello"
    assert content_events[0].message_id == msg_id


def test_llm_end_with_text_and_no_new_tokens_emits_content_event(step_adaptor):
    """LLM_END with non-empty text (and no prior LLM_NEW_TOKEN) should emit a content event."""
    msg_id = str(uuid.uuid4())

    _enter_primary_function(step_adaptor)
    start_step = _make_llm_step(IntermediateStepType.LLM_START, msg_id)
    step_adaptor.process(start_step)

    end_step = _make_llm_step(IntermediateStepType.LLM_END, msg_id, text="response text")
    response = step_adaptor.process(end_step)

    assert response is not None
    content_events = [e for e in (response.events or []) if isinstance(e, TextMessageContentEvent)]
    assert len(content_events) == 1
    assert content_events[0].delta == "response text"


def test_adaptor_processes_nested_reasoning_steps(
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
        assert len(actual_resp.events or []) == len(expected_resp.events or []), (
            f"Response {i}: event count mismatch"
        )
        for j, (actual_ev, expected_ev) in enumerate(
            zip(actual_resp.events or [], expected_resp.events or [])
        ):
            assert actual_ev == expected_ev, f"Response {i} event {j}: {actual_ev} != {expected_ev}"
