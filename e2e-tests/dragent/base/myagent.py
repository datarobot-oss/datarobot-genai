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

import uuid

from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from datarobot_genai.core.agents import BaseAgent
from datarobot_genai.core.agents import InvokeReturn
from datarobot_genai.core.agents import UsageMetrics
from datarobot_genai.core.agents import default_usage_metrics


class MyAgent(BaseAgent[None]):
    """Base agent that returns a static success response for e2e testing."""

    async def invoke(self, run_agent_input: RunAgentInput) -> InvokeReturn:
        usage_metrics: UsageMetrics = default_usage_metrics()
        thread_id = run_agent_input.thread_id
        run_id = run_agent_input.run_id
        message_id = str(uuid.uuid4())

        yield (
            RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id),
            None,
            usage_metrics,
        )
        yield (
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=message_id,
                role="assistant",
            ),
            None,
            usage_metrics,
        )
        yield (
            TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=message_id,
                delta="Success",
            ),
            None,
            usage_metrics,
        )
        yield (
            TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=message_id,
            ),
            None,
            usage_metrics,
        )
        yield (
            RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id),
            None,
            usage_metrics,
        )
