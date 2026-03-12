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
    """Single LlamaIndex agent with calculator tool for e2e testing."""

    def __init__(
        self,
        llm: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._llm = llm

    def build_workflow(self) -> AgentWorkflow:
        assistant = FunctionAgent(
            name="assistant",
            description="Answers questions and computes math expressions",
            system_prompt=make_system_prompt(
                "You are a helpful assistant. Answer questions concisely. "
                "Use the calculator tool when asked to compute math expressions."
            ),
            llm=self._llm,
            tools=[calculator] + self.mcp_tools,
        )
        return AgentWorkflow(agents=[assistant], root_agent="assistant")

    def extract_response_text(self, result_state: Any, events: list[Any]) -> str:
        for event in reversed(events):
            resp = getattr(event, "response", None)
            if resp is not None:
                content = getattr(resp, "content", None)
                if content:
                    return str(content)
        return ""
