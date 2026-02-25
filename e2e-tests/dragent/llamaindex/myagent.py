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


class MyAgent(LlamaIndexAgent):
    """LlamaIndex planner/writer agent with NAT."""

    def __init__(
        self,
        llm: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._llm = llm

    def build_workflow(self) -> AgentWorkflow:
        """Build a planner -> writer agent workflow."""
        planner = FunctionAgent(
            name="planner",
            description="Plans content outlines for blog articles",
            system_prompt=make_system_prompt(
                "You are a content planner. You create brief, structured outlines for blog "
                "articles. You identify the most important points and cite relevant sources. "
                "Keep it simple and to the point - this is just an outline for the writer.\n\n"
                "Create a simple outline with:\n"
                "1. 10-15 key points or facts (bullet points only, no paragraphs)\n"
                "2. 2-3 relevant sources or references\n"
                "3. A brief suggested structure (intro, 2-3 sections, conclusion)\n\n"
                "Do NOT write paragraphs or detailed explanations. "
                "Just provide a focused list.\n"
                "When done, hand off to the writer agent."
            ),
            llm=self._llm,
            tools=self.mcp_tools,
            can_handoff_to=["writer"],
        )

        writer = FunctionAgent(
            name="writer",
            description="Writes blog posts based on content outlines",
            system_prompt=make_system_prompt(
                "You are a content writer working with a planner colleague. "
                "You write opinion pieces based on the planner's outline and context. "
                "You provide objective and impartial insights backed by the planner's "
                "information. You acknowledge when your statements are opinions versus "
                "objective facts.\n\n"
                "1. Use the content plan to craft a compelling blog post.\n"
                "2. Structure with an engaging introduction, insightful body, and "
                "summarizing conclusion.\n"
                "3. Sections/Subtitles are properly named in an engaging manner.\n"
                "4. CRITICAL: Keep the total output under 500 words. Each section "
                "should have 1-2 brief paragraphs.\n\n"
                "Write in markdown format, ready for publication."
            ),
            llm=self._llm,
            tools=self.mcp_tools,
        )

        return AgentWorkflow(agents=[planner, writer], root_agent="planner")

    def extract_response_text(self, result_state: Any, events: list[Any]) -> str:
        """Extract final response text from workflow state and/or events."""
        for event in reversed(events):
            resp = getattr(event, "response", None)
            if resp is not None:
                content = getattr(resp, "content", None)
                if content:
                    return str(content)
        return ""
