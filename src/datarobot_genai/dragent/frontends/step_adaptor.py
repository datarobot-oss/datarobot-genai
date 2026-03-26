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

import json
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
from nat.data_models.step_adaptor import StepAdaptorMode
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.retriever.models import GlobalTypeConverter

from .response import DRAgentEventResponse

logger = logging.getLogger(__name__)


class DRAgentNestedReasoningStepAdaptor(StepAdaptor):
    """
    Convert native agent steps to DRAgent AG-UI events.

    This adaptor interprets LLM events from the root level function as TEXT, and
    downstream functions as REASONING.
    """

    def __init__(self, config: StepAdaptorConfig) -> None:
        super().__init__(config)
        self.function_level = 0
        self.seen_llm_new_token = False

    def _step_matches_filter(self, step: IntermediateStep, config: StepAdaptorConfig) -> bool:  # noqa: PLR0911
        """Returns True if this intermediate step should be included (based on the config.mode)."""  # noqa: D401
        if config.mode == StepAdaptorMode.OFF:
            return False

        if config.mode == StepAdaptorMode.DEFAULT:
            # Process all steps
            return True

        if config.mode == StepAdaptorMode.CUSTOM:
            # pass only what the user explicitly listed
            return step.event_type in config.custom_event_types

        return False

    def process(self, step: IntermediateStep) -> ResponseSerializable | None:
        result = super().process(step)

        if not self._step_matches_filter(step, self.config):
            return result

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

    def _get_usage_metrics(self, usage_info: UsageInfo | None) -> dict[str, int]:
        if usage_info is None:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        return usage_info.token_usage.model_dump()

    @staticmethod
    def _extract_payload_text(payload: IntermediateStepPayload) -> str:
        """Extract text from an LLM_END payload across different frameworks.

        LangChain payloads expose ``.text`` (a ``TextAccessor``), while
        LlamaIndex ``ChatResponse`` objects use ``.message.content``
        Always returns a plain ``str``
        """
        data = payload.data.payload
        if hasattr(data, "text"):
            return str(data.text)
        if hasattr(data, "message"):
            return str(getattr(data.message, "content", data.message))
        return str(data)

    def _unknown_step_type(self, payload: IntermediateStepPayload) -> Exception:
        return ValueError(
            f"Unsupported intermediate step type: {payload.event_type}, payload: {payload}"
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
                text = self._extract_payload_text(payload)
                if text:
                    events.append(TextMessageContentEvent(message_id=payload.UUID, delta=text))

            events.append(TextMessageEndEvent(message_id=payload.UUID))
        elif payload.event_type == IntermediateStepType.LLM_NEW_TOKEN:
            self.seen_llm_new_token = True
            if payload.data.chunk != "":
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
            events.append(ReasoningMessageStartEvent(message_id=payload.UUID, role="reasoning"))
            self.seen_llm_new_token = False
        elif payload.event_type == IntermediateStepType.LLM_END:
            if not self.seen_llm_new_token:
                text = self._extract_payload_text(payload)
                if text:
                    events.append(ReasoningMessageContentEvent(message_id=payload.UUID, delta=text))
            events.append(ReasoningMessageEndEvent(message_id=payload.UUID))
            events.append(ReasoningEndEvent(message_id=payload.UUID))
        elif payload.event_type == IntermediateStepType.LLM_NEW_TOKEN:
            self.seen_llm_new_token = True
            if payload.data.chunk != "":
                events.append(
                    ReasoningMessageContentEvent(message_id=payload.UUID, delta=payload.data.chunk)
                )
        else:
            raise self._unknown_step_type(payload)

        return events

    def _handle_workflow(
        self, payload: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        events = []
        # run id and thread id are set on the API level, should not be set here
        if payload.event_type == IntermediateStepType.WORKFLOW_START:
            events.append(RunStartedEvent(run_id="", thread_id=""))
            events.append(StepStartedEvent(step_name=payload.name))
            # Open the TextMessage lifecycle for the raw string stream from NAT native.
            # The converter emits only TextMessageContentEvent (stateless); Start/End
            # are managed here so there is no cross-component hidden state.
            events.append(TextMessageStartEvent(message_id="default_nat_response"))
        elif payload.event_type == IntermediateStepType.WORKFLOW_END:
            events.append(TextMessageEndEvent(message_id="default_nat_response"))
            events.append(StepFinishedEvent(step_name=payload.name))
            events.append(RunFinishedEvent(run_id="", thread_id=""))
        else:
            raise self._unknown_step_type(payload)

        response = DRAgentEventResponse(events=events)
        return response

    @staticmethod
    def _serialize_tool_args(payload: IntermediateStepPayload) -> str:
        """Extract tool call arguments as a JSON string.

        Tries metadata.tool_inputs first, then data.input. Returns "{}"
        when no usable arguments are found. Non-serializable values
        (e.g. CrewStructuredTool leaked via nvidia-nat-crewai) are logged
        and skipped.
        """
        tool_inputs = getattr(payload.metadata, "tool_inputs", None)
        if isinstance(tool_inputs, dict):
            raw = tool_inputs
        else:
            raw = payload.data.input
        if raw is None:
            return "{}"

        if isinstance(raw, str):
            try:
                json.loads(raw)
                return raw
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "Tool args not valid JSON for %s, skipping",
                    payload.name,
                )
                return "{}"

        try:
            return json.dumps(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Tool args not serializable for %s, skipping",
                payload.name,
            )
            return "{}"

    def _handle_tool(
        self, payload: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        events = []

        # run id and thread id are set on the API level, should not be set here
        if payload.event_type == IntermediateStepType.TOOL_START:
            events.append(
                ToolCallStartEvent(tool_call_name=payload.name, tool_call_id=payload.UUID)
            )
            args_delta = self._serialize_tool_args(payload)
            events.append(ToolCallArgsEvent(tool_call_id=payload.UUID, delta=args_delta))
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
        response_events: list[Event] = []

        if payload.event_type == IntermediateStepType.FUNCTION_START:
            if payload.name != "<workflow>" and self.function_level > 0:
                response_events.extend(
                    [
                        ToolCallStartEvent(tool_call_name=payload.name, tool_call_id=payload.UUID),
                        ToolCallArgsEvent(
                            tool_call_id=payload.UUID,
                            delta=self._serialize_tool_args(payload),
                        ),
                    ]
                )
            self.function_level += 1
        elif payload.event_type == IntermediateStepType.FUNCTION_END:
            if payload.name != "<workflow>" and self.function_level > 1:
                tool_output = payload.data.output
                if tool_output is None:
                    tool_output = ""

                response_events.extend(
                    [
                        ToolCallEndEvent(tool_call_id=payload.UUID),
                        ToolCallResultEvent(
                            message_id=payload.UUID,
                            tool_call_id=payload.UUID,
                            content=GlobalTypeConverter.get().convert(tool_output, str),
                            role="tool",
                        ),
                    ]
                )
            self.function_level -= 1
        else:
            raise self._unknown_step_type(payload)

        if response_events:
            return DRAgentEventResponse(events=response_events)

        return None

    def _handle_custom(
        self, payload: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        event = CustomEvent(name=payload.event_type, value=payload)
        response = DRAgentEventResponse(events=[event])
        return response
