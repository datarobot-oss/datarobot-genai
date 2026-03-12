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

from crewai import Agent
from crewai import Crew
from crewai import Task
from crewai.tools import tool
from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.crewai.agent import CrewAIAgent

from dragent.common import calculator as _calculator_fn


@tool
def calculator(expression: str) -> str:
    """Calculate a math expression, e.g. '15 * 7'."""
    return _calculator_fn(expression)


class MyAgent(CrewAIAgent):
    """Single CrewAI agent with calculator tool for e2e testing."""

    def __init__(
        self,
        llm: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._llm = llm

    @property
    def agents(self) -> list[Any]:
        assistant = Agent(
            role="Assistant",
            goal="Answer questions concisely. Use the calculator tool for math: {topic}.",
            backstory=make_system_prompt(
                "You are a helpful assistant. Answer questions concisely. "
                "Use the calculator tool when asked to compute math expressions."
            ),
            llm=self._llm,
            tools=[calculator] + self.mcp_tools,
            verbose=self.verbose,
        )
        return [assistant]

    @property
    def tasks(self) -> list[Any]:
        return self._tasks_for(self.agents)

    def crew(self) -> Crew:
        agents = self.agents
        return Crew(agents=agents, tasks=self._tasks_for(agents), verbose=self.verbose)

    def _tasks_for(self, agents: list[Any]) -> list[Any]:
        (assistant,) = agents
        return [
            Task(
                description=(
                    "Answer the following: {topic}. "
                    "Prior conversation context (may be empty): {chat_history}"
                ),
                expected_output="A concise answer to the question.",
                agent=assistant,
            ),
        ]

    def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
        return {
            "topic": user_prompt_content,
            "chat_history": "",
        }
