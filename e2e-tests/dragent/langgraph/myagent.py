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

from dragent.common import calculator as _calculator_fn


@tool
def calculator(expression: str) -> str:
    """Calculate a math expression, e.g. '15 * 7'."""
    return _calculator_fn(expression)


class MyAgent(LangGraphAgent):
    """Single-node LangGraph agent with calculator tool for e2e testing."""

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
                ("system", "You are a helpful assistant. {chat_history}"),
                ("user", "{topic}"),
            ]
        )

    @property
    def workflow(self) -> StateGraph[MessagesState]:
        graph = StateGraph(MessagesState)
        graph.add_node("agent_node", self.agent)
        graph.add_edge(START, "agent_node")
        graph.add_edge("agent_node", END)
        return graph

    @property
    def agent(self) -> Any:
        return create_agent(
            self._llm,
            tools=[calculator] + self.mcp_tools,
            system_prompt=make_system_prompt(
                "You are a helpful assistant. Answer questions concisely. "
                "Use the calculator tool when asked to compute math expressions."
            ),
            name="assistant",
        )
