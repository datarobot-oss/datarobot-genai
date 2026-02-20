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

from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from nat.builder.context import IntermediateStep
from nat.builder.context import IntermediateStepPayload
from nat.builder.context import IntermediateStepType
from nat.builder.context import InvocationNode
from nat.data_models.api_server import ResponseSerializable
from nat.data_models.intermediate_step import IntermediateStepCategory
from nat.data_models.intermediate_step import UsageInfo
from nat.front_ends.fastapi.step_adaptor import StepAdaptor

from datarobot_genai.dragent.response import DRAgentEventResponse

logger = logging.getLogger(__name__)


class DRAgentStepAdaptor(StepAdaptor):
    """Convert native agent steps to DRAgent events."""

    def process(self, step: IntermediateStep) -> ResponseSerializable | None:
        # Do not process steps from native agents
        if isinstance(step, DRAgentEventResponse):
            return step
        if isinstance(step, str):
            return DRAgentEventResponse(delta=step)

        result = super().process(step)

        try:
            payload = step.payload
            ancestry = step.function_ancestry

            if step.event_category == IntermediateStepCategory.WORKFLOW:
                result = self._handle_workflow(payload, ancestry)

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

    def _handle_llm(
        self, payload: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        # Find the start in the history with matching run_id
        event = None
        if payload.event_type == IntermediateStepType.LLM_START:
            event = TextMessageStartEvent(message_id=payload.UUID)
        elif payload.event_type == IntermediateStepType.LLM_END:
            event = TextMessageEndEvent(message_id=payload.UUID)
        elif payload.event_type == IntermediateStepType.LLM_NEW_TOKEN:
            event = TextMessageContentEvent(message_id=payload.UUID, delta=payload.data.chunk)
        else:
            raise ValueError(
                f"Unsupported LLM event type: {payload.event_type}, payload: {payload}"
            )

        response = DRAgentEventResponse(event=event)
        return response

    def _handle_workflow(
        self, payload: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        event = None
        # TODO: need to set the thread id from the payload
        # TODO: handle parent run, input, output
        if payload.event_type == IntermediateStepType.WORKFLOW_START:
            event = RunStartedEvent(run_id=payload.UUID, thread_id="")
        elif payload.event_type == IntermediateStepType.WORKFLOW_END:
            event = RunFinishedEvent(run_id=payload.UUID, thread_id="")
        else:
            raise ValueError(
                f"Unsupported LLM event type: {payload.event_type}, payload: {payload}"
            )

        response = DRAgentEventResponse(event=event)
        return response

    def _handle_tool(
        self, step: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        # TODO: Implement tool handling
        return None

    def _handle_function(
        self, step: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        # TODO: Implement function handling
        return None

    def _handle_custom(
        self, step: IntermediateStepPayload, ancestry: InvocationNode
    ) -> ResponseSerializable | None:
        # TODO: Implement custom handling
        return None


class DRAgentEmptyStepAdaptor(StepAdaptor):
    def process(self, step: IntermediateStep) -> ResponseSerializable | None:
        return None
