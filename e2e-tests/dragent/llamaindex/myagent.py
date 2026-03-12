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

from typing import Any

from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.llama_index.agent import LlamaIndexAgent
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

from dragent.common import calculator as _calculator_fn

calculator = FunctionTool.from_defaults(fn=_calculator_fn, name="calculator")


class MyAgent(LlamaIndexAgent):
    """Planner -> Writer LlamaIndex agent with calculator tool for e2e testing."""

    def __init__(
        self,
        llm: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._llm = llm

    def build_workflow(self) -> AgentWorkflow:
        planner = FunctionAgent(
            name="planner",
            description="Creates short bullet-point outlines",
            system_prompt=make_system_prompt(
                "You are a content planner. Given a topic, produce a short bullet-point "
                "outline with 3-5 key points. No paragraphs, no explanations — just the list. "
                "When done, hand off to the writer agent."
            ),
            llm=self._llm,
            tools=self.mcp_tools,
            can_handoff_to=["writer"],
        )
        writer = FunctionAgent(
            name="writer",
            description="Writes concise responses based on outlines",
            system_prompt=make_system_prompt(
                "You are a concise writer. Using the planner's outline, write a short response "
                "in 2-3 sentences. Use the calculator tool when asked to compute math."
            ),
            llm=self._llm,
            tools=[calculator] + self.mcp_tools,
        )
        return AgentWorkflow(agents=[planner, writer], root_agent="planner")

    def extract_response_text(self, result_state: Any, events: list[Any]) -> str:
        for event in reversed(events):
            resp = getattr(event, "response", None)
            if resp is not None:
                content = getattr(resp, "content", None)
                if content:
                    return str(content)
        return ""
