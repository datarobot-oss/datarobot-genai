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
from datarobot_genai.langgraph.agent import LangGraphAgent
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph

from dragent.tool import generate_objectid

generate_objectid_tool = tool(generate_objectid)


class MyAgent(LangGraphAgent):
    """Planner -> Writer LangGraph agent with calculator tool for e2e testing."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._llm = llm

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("user", "{topic}"),
            ]
        )

    @property
    def workflow(self) -> StateGraph[MessagesState]:
        graph = StateGraph(MessagesState)
        graph.add_node("planner_node", self.agent_planner)
        graph.add_node("writer_node", self.agent_writer)
        graph.add_edge(START, "planner_node")
        graph.add_edge("planner_node", "writer_node")
        graph.add_edge("writer_node", END)
        return graph

    @property
    def agent_planner(self) -> Any:
        return create_agent(
            self._llm,
            tools=[generate_objectid_tool] + self.tools,
            system_prompt=make_system_prompt(
                "You are a content planner. Given a topic, produce a short bullet-point "
                "outline with 3-5 key points. No paragraphs, no explanations — just the list. "
                "Use the generate_objectid tool when asked to generate an object ID for a "
                "deployment."
            ),
            name="planner",
        )

    @property
    def agent_writer(self) -> Any:
        return create_agent(
            self._llm,
            tools=[generate_objectid_tool] + self.tools,
            system_prompt=make_system_prompt(
                "You are a concise writer. Using the planner's outline, write a short response "
                "in 2-3 sentences. Use the generate_objectid tool when asked to "
                "generate an object ID for a deployment."
            ),
            name="writer",
        )
