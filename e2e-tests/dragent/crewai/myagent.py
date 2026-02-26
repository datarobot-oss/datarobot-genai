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

from datetime import datetime
from functools import cached_property
from typing import Any

from crewai import Agent
from crewai import Task
from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.crewai.agent import CrewAIAgent


class MyAgent(CrewAIAgent):
    """CrewAI planner/writer agent with NAT."""

    def __init__(
        self,
        llm: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._llm = llm

    @cached_property
    def agents(self) -> list[Any]:
        """Planner and writer agents."""
        planner = Agent(
            role="Content Planner",
            goal=(
                "Create a brief, structured outline for a blog article on the topic: {topic}. "
                "Make sure to find relevant information given the current year is "
                f"{datetime.now().year}."
            ),
            backstory=make_system_prompt(
                "You are a content planner. You create brief, structured outlines for blog "
                "articles. You identify the most important points and cite relevant sources. "
                "Keep it simple and to the point - this is just an outline for the writer.\n\n"
                "Create a simple outline with:\n"
                "1. 10-15 key points or facts (bullet points only, no paragraphs)\n"
                "2. 2-3 relevant sources or references\n"
                "3. A brief suggested structure (intro, 2-3 sections, conclusion)\n\n"
                "Do NOT write paragraphs or detailed explanations. Just provide a focused list."
            ),
            llm=self._llm,
            tools=self.mcp_tools,
            verbose=self.verbose,
        )

        writer = Agent(
            role="Content Writer",
            goal="Write a compelling blog post based on the planner's outline about: {topic}.",
            backstory=make_system_prompt(
                "You are a content writer working with a planner colleague. "
                "You write opinion pieces based on the planner's outline and context. "
                "You provide objective and impartial insights backed by the planner's "
                "information. You acknowledge when your statements are opinions versus "
                "objective facts.\n\n"
                "1. Use the content plan to craft a compelling blog post.\n"
                "2. Structure with an engaging introduction, insightful body, and "
                "summarizing conclusion.\n"
                "3. Sections/Subtitles are properly named in an engaging manner.\n"
                "4. CRITICAL: Keep the total output under 500 words. Each section should "
                "have 1-2 brief paragraphs.\n\n"
                "Write in markdown format, ready for publication."
            ),
            llm=self._llm,
            tools=self.mcp_tools,
            verbose=self.verbose,
        )

        return [planner, writer]

    @property
    def tasks(self) -> list[Any]:
        """Plan then write tasks."""
        planner, writer = self.agents

        plan_task = Task(
            description=(
                "Research and create a structured outline for a blog article about: {topic}. "
                "Prior conversation context (may be empty): {chat_history}"
            ),
            expected_output=(
                "A structured outline with 10-15 key points, 2-3 sources, "
                "and a brief suggested structure."
            ),
            agent=planner,
        )

        write_task = Task(
            description=(
                "Using the planner's outline, write a compelling blog post about: {topic}. "
                "Prior conversation context (may be empty): {chat_history}"
            ),
            expected_output=(
                "A well-structured blog post in markdown format, under 500 words, "
                "with an engaging introduction, insightful body, and summarizing conclusion."
            ),
            agent=writer,
        )

        return [plan_task, write_task]

    def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
        """Build inputs for Crew.kickoff.

        Includes chat_history key to opt into automatic history injection.
        """
        return {
            "topic": user_prompt_content,
            "chat_history": "",
        }
