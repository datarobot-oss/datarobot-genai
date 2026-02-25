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

import logging

from ag_ui.core import CustomEvent
from ag_ui.core import Event
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
from nat.data_models.api_server import ResponseSerializable
from nat.data_models.intermediate_step import IntermediateStepCategory
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.step_adaptor import StepAdaptorConfig
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.retriever.models import GlobalTypeConverter

from datarobot_genai.dragent.response import DRAgentEventResponse

logger = logging.getLogger(__name__)


class DRAgentNestedReasoningStepAdaptor(StepAdaptor):
    """
    Convert native agent steps to DRAgent AG-UI events.

    This adaptor interprets LLM events from the root level function as TEXT, and
    downstream functions as REASONING.

    TODO: make it configurable to support different behavior
    """

    def __init__(self, config: StepAdaptorConfig) -> None:
        # Override the config to default
        default_config = StepAdaptorConfig()
        if default_config != config:
            raise ValueError(
                f"Step config {config} is not supported for nested reasoning processing. "
                f"Using default config {default_config}",
                UserWarning,
            )
        super().__init__(default_config)
        self.function_level = 0
        self.seen_llm_new_token = False

    def process(self, step: IntermediateStep) -> ResponseSerializable | None:
        result = super().process(step)

        try:
            payload = step.payload
            ancestry = step.function_ancestry

            if step.event_category == IntermediateStepCategory.WORKFLOW:
                result = self._handle_workflow(payload, ancestry)
            # If we have not handle it yet, handle it as custom
            if result is None:
                result = self._handle_custom(payload, ancestry)

        except Exception as e:
            logger.exception("Error processing intermediate step: %s", e)
            return None

        if result is not None:
            result.usage_metrics = self._get_usage_metrics(step.usage_info)
            return result

        return result

    def _get_usage_metrics(self, usage_info: UsageInfo | None) -> dict[str, int]:
        if usage_info is None:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        return usage_info.token_usage.model_dump()

    def _unknown_step_type(self, payload: IntermediateStepPayload) -> Exception:
        return ValueError(
            f"Unsupported intermediate step type: {payload.event_type}, payload: {{payload}}"
        )

    def _handle_llm(
        self, payload: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        # Find the start in the history with matching run_id

        if self.function_level == 1:
            events = self._handle_llm_primary_function(payload)
        else:
            events = self._handle_llm_nested_function(payload)

        response = DRAgentEventResponse(
            events=events,
            # Only for llm events we actually know the model name
            # And its is passed as name
            model=payload.name,
        )
        return response

    def _handle_llm_primary_function(self, payload: IntermediateStepPayload) -> list[Event]:
        events = []

        if payload.event_type == IntermediateStepType.LLM_START:
            events.append(TextMessageStartEvent(message_id=payload.UUID))
            self.seen_llm_new_token = False
        # Text might be sent in both LLM_END and LLM_NEW_TOKEN steps
        # we need to send content only once
        elif payload.event_type == IntermediateStepType.LLM_END:
            if not self.seen_llm_new_token:
                events.append(
                    TextMessageContentEvent(
                        message_id=payload.UUID, delta=payload.data.payload.text
                    )
                )

            events.append(TextMessageEndEvent(message_id=payload.UUID))
        elif payload.event_type == IntermediateStepType.LLM_NEW_TOKEN:
            self.seen_llm_new_token = True
            events.append(
                TextMessageContentEvent(message_id=payload.UUID, delta=payload.data.chunk)
            )
        else:
            raise self._unknown_step_type(payload)

        return events

    def _handle_llm_nested_function(self, payload: IntermediateStepPayload) -> list[Event]:
        events = []
        if payload.event_type == IntermediateStepType.LLM_START:
            events.append(ReasoningStartEvent(message_id=payload.UUID))
            events.append(ReasoningMessageStartEvent(message_id=payload.UUID, role="assistant"))
            self.seen_llm_new_token = False
        elif payload.event_type == IntermediateStepType.LLM_END:
            if not self.seen_llm_new_token:
                events.append(
                    ReasoningMessageContentEvent(
                        message_id=payload.UUID, delta=payload.data.payload.text
                    )
                )
            events.append(ReasoningMessageEndEvent(message_id=payload.UUID))
            events.append(ReasoningEndEvent(message_id=payload.UUID))
        elif payload.event_type == IntermediateStepType.LLM_NEW_TOKEN:
            self.seen_llm_new_token = True
            events.append(
                ReasoningMessageContentEvent(message_id=payload.UUID, delta=payload.data.chunk)
            )
        else:
            raise self._unknown_step_type(payload)

        return events

    def _handle_workflow(
        self, payload: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        event = None
        # run id and thread id are set on the API level, should not be set here
        if payload.event_type == IntermediateStepType.WORKFLOW_START:
            event = RunStartedEvent(run_id="", thread_id="")
        elif payload.event_type == IntermediateStepType.WORKFLOW_END:
            event = RunFinishedEvent(run_id="", thread_id="")
        else:
            raise self._unknown_step_type(payload)

        response = DRAgentEventResponse(events=[event])
        return response

    def _handle_tool(
        self, payload: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        events = []

        # run id and thread id are set on the API level, should not be set here
        if payload.event_type == IntermediateStepType.TOOL_START:
            events.append(
                ToolCallStartEvent(tool_call_name=payload.name, tool_call_id=payload.UUID)
            )
            events.append(ToolCallArgsEvent(tool_call_id=payload.UUID, delta=payload.data.input))
        elif payload.event_type == IntermediateStepType.TOOL_END:
            events.append(ToolCallEndEvent(tool_call_id=payload.UUID))
            tool_outputs = GlobalTypeConverter.get().convert(payload.metadata.tool_outputs, str)
            events.append(
                ToolCallResultEvent(
                    message_id=payload.UUID,
                    tool_call_id=payload.UUID,
                    content=tool_outputs,
                    role="tool",
                )
            )
        else:
            raise self._unknown_step_type(payload)

        response = DRAgentEventResponse(events=events)
        return response

    def _handle_function(
        self, payload: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        # NAT function -> AG-UI step
        event = None
        # run id and thread id are set on the API level, should not be set here
        if payload.event_type == IntermediateStepType.FUNCTION_START:
            self.function_level += 1
            event = StepStartedEvent(step_name=payload.name)
        elif payload.event_type == IntermediateStepType.FUNCTION_END:
            self.function_level -= 1
            event = StepFinishedEvent(step_name=payload.name)
        else:
            raise self._unknown_step_type(payload)

        response = DRAgentEventResponse(events=[event])
        return response

    def _handle_custom(
        self, payload: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        event = CustomEvent(name=payload.event_type, value=payload)
        response = DRAgentEventResponse(events=[event])
        return response


class DRAgentEmptyStepAdaptor(StepAdaptor):
    def process(self, step: IntermediateStep) -> ResponseSerializable | None:
        return None
