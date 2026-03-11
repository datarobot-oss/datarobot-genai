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
from typing import Any

from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.langgraph.agent import LangGraphAgent
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph


class MyAgent(LangGraphAgent):
    """LangGraph planner/writer agent with NAT."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._llm = llm

    @property
    def workflow(self) -> StateGraph[MessagesState]:
        """Planner -> Writer sequential workflow."""
        graph = StateGraph(MessagesState)
        graph.add_node("planner_node", self.agent_planner)
        graph.add_node("writer_node", self.agent_writer)
        graph.add_edge(START, "planner_node")
        graph.add_edge("planner_node", "writer_node")
        graph.add_edge("writer_node", END)
        return graph

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that plans and writes content based on the "
                    "user's topic. Chat history is provided via {chat_history} (it may be empty). "
                    "Use it when helpful to stay consistent across turns.",
                ),
                (
                    "user",
                    f"The topic is {{topic}}. Make sure you find any interesting and "
                    f"relevant information given the current year is {datetime.now().year}.",
                ),
            ]
        )

    @property
    def agent_planner(self) -> Any:
        return create_agent(
            self._llm,
            tools=self.mcp_tools,
            system_prompt=make_system_prompt(
                "You are a content planner. You create brief, structured outlines for blog "
                "articles. You identify the most important points and cite relevant sources. "
                "Keep it simple and to the point - this is just an outline for the writer.\n\n"
                "Create a simple outline with:\n"
                "1. 10-15 key points or facts (bullet points only, no paragraphs)\n"
                "2. 2-3 relevant sources or references\n"
                "3. A brief suggested structure (intro, 2-3 sections, conclusion)\n\n"
                "Do NOT write paragraphs or detailed explanations. Just provide a focused list."
            ),
            name="planner",
        )

    @property
    def agent_writer(self) -> Any:
        return create_agent(
            self._llm,
            tools=self.mcp_tools,
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
                "4. CRITICAL: Keep the total output under 500 words. Each section should "
                "have 1-2 brief paragraphs.\n\n"
                "Write in markdown format, ready for publication."
            ),
            name="writer",
        )
