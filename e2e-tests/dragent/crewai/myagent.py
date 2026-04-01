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
from crewai import Task
from crewai.tools import tool
from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.crewai.agent import CrewAIAgent
from datarobot_genai.drtools.calculator import calculator

from dragent.tool import generate_objectid

generate_objectid_tool = tool(generate_objectid)
calculator_tool = tool(calculator)


class MyAgent(CrewAIAgent):
    """Planner -> Writer CrewAI agent with calculator tool for e2e testing."""

    def __init__(
        self,
        llm: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._llm = llm

    @property
    def agents(self) -> list[Any]:
        planner = Agent(
            role="Content Planner",
            goal="Create a short bullet-point outline with 3-5 key points about: {topic}.",
            backstory=make_system_prompt(
                "You are a content planner. Given a topic, produce a short bullet-point "
                "outline with 3-5 key points. No paragraphs, no explanations — just the list. "
                "Use the generate_objectid tool when asked to generate an object ID for a "
                "deployment. Use the calculator tool when asked to compute a mathematical "
                "expression."
            ),
            llm=self._llm,
            function_calling_llm=self._llm,
            tools=[generate_objectid_tool, calculator_tool] + self.tools,
            verbose=self.verbose,
        )
        writer = Agent(
            role="Content Writer",
            goal="Write a 2-3 sentence response based on the planner's outline about: {topic}.",
            backstory=make_system_prompt(
                "You are a concise writer. Using the planner's outline, write a short response "
                "in 2-3 sentences. Use the generate_objectid tool when asked to "
                "generate an object ID for a deployment. Use the calculator tool when asked to "
                "compute a mathematical expression."
            ),
            llm=self._llm,
            function_calling_llm=self._llm,
            tools=[generate_objectid_tool, calculator_tool] + self.tools,
            verbose=self.verbose,
        )
        return [planner, writer]

    @property
    def tasks(self) -> list[Any]:
        return self._tasks_for(self.agents)

    def _tasks_for(self, agents: list[Any]) -> list[Any]:
        planner, writer = agents
        return [
            Task(
                description=(
                    "Create a short outline about: {topic}. "
                    "Execute tool calls if requested instead of this task."
                ),
                expected_output=(
                    "A bullet-point outline with 3-5 key points. Or the result of a tool call."
                ),
                agent=planner,
            ),
            Task(
                description=(
                    "Using the planner's outline, write a short response about: {topic}. "
                    "Execute tool calls if requested instead of this task."
                ),
                expected_output="A concise 2-3 sentence response. Or the result of a tool call.",
                agent=writer,
            ),
        ]

    def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
        return {
            "topic": user_prompt_content,
        }
