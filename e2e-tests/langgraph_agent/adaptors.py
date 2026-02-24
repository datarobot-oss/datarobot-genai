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

from collections.abc import AsyncGenerator

from ag_ui.core import BaseEvent
from ag_ui.core import Event
from ag_ui.core import RunAgentInput
from datarobot_genai.core.agents.base import BaseAgent


class AGUIAdaptor:
    """Bridges a BaseAgent to AG-UI event streaming for NAT."""

    def __init__(self, agent: BaseAgent) -> None:
        self.agent = agent

    async def chat(self, chat_input: RunAgentInput) -> AsyncGenerator[Event, None]:
        # invoke() is an async generator (yields events), not a coroutine
        async for event, _, _ in self.agent.invoke(chat_input):
            if isinstance(event, BaseEvent):
                yield event
