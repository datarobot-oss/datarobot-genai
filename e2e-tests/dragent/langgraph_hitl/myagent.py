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

"""Minimal LangGraph workflow for E2E interrupt / resume (no LLM)."""

from datarobot_genai.langgraph.agent import datarobot_agent_class_from_langgraph
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer
from langgraph.types import interrupt

# Emitted in assistant text so HTTP e2e can assert without parsing tool payloads.
E2E_INTERRUPT_CANCELLED = "E2E_INTERRUPT_CANCELLED"
E2E_INTERRUPT_CONTINUING = "E2E_INTERRUPT_CONTINUING"

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("user", "{topic}"),
    ]
)


def graph_factory(
    _llm: BaseChatModel,
    _tools: list[BaseTool],
    _verbose: bool = False,
) -> StateGraph[MessagesState]:
    """Single node that interrupts once, then branches on the resume payload."""

    def approval(state: MessagesState) -> MessagesState:
        decision = interrupt({"e2e_prompt": "approve?", "kind": "e2e_hitl"})
        prior = list(state.get("messages", []))
        if decision == "no":
            reply = E2E_INTERRUPT_CANCELLED
        else:
            reply = E2E_INTERRUPT_CONTINUING
        return {"messages": prior + [AIMessage(content=reply)]}

    graph = StateGraph(MessagesState)
    graph.add_node("approval", approval)
    graph.add_edge(START, "approval")
    graph.add_edge("approval", END)
    return graph


# NAT constructs a new agent per HTTP request; a fresh InMemorySaver per instance would
# drop checkpoint state between the interrupt request and the resume request. One
# shared saver keeps thread state for this E2E workflow only (not for production).
_HITL_E2E_CHECKPOINTER = InMemorySaver()

_BaseHitlAgent = datarobot_agent_class_from_langgraph(graph_factory, prompt_template)


class HitlMyAgent(_BaseHitlAgent):
    """Same graph as the factory-built agent, with a process-wide checkpointer for E2E."""

    @property
    def langgraph_checkpointer(self) -> Checkpointer | None:
        return _HITL_E2E_CHECKPOINTER
