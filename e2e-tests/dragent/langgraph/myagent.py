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

from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.langgraph.agent import datarobot_agent_class_from_langgraph
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer
from langgraph.types import interrupt

from dragent.tool import generate_objectid

generate_objectid_tool = tool(generate_objectid)

# Emitted in assistant text so HTTP e2e can assert without parsing tool payloads.
E2E_INTERRUPT_CANCELLED = "E2E_INTERRUPT_CANCELLED"
E2E_INTERRUPT_CONTINUING = "E2E_INTERRUPT_CONTINUING"

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. {chat_history}"),
        ("user", "{topic}"),
    ]
)


def graph_factory(
    llm: BaseChatModel, tools: list[BaseTool], verbose: bool = False
) -> StateGraph[MessagesState]:
    agent_planner = create_agent(
        llm,
        tools=[generate_objectid_tool] + tools,
        system_prompt=make_system_prompt(
            "You are a content planner. Given a topic, produce a short bullet-point "
            "outline with 3-5 key points. No paragraphs, no explanations — just the list. "
            "Use the generate_objectid tool when asked to generate an object ID for a "
            "deployment."
        ),
        name="planner",
        debug=verbose,
    )

    agent_writer = create_agent(
        llm,
        tools=[generate_objectid_tool] + tools,
        system_prompt=make_system_prompt(
            "You are a concise writer. Using the planner's outline, write a short response "
            "in 2-3 sentences. Use the generate_objectid tool when asked to "
            "generate an object ID for a deployment."
        ),
        name="writer",
        debug=verbose,
    )

    def human_review(state: MessagesState) -> MessagesState:
        decision = interrupt({"e2e_prompt": "approve?", "kind": "e2e_hitl"})
        prior = list(state.get("messages", []))
        if decision == "no":
            reply = E2E_INTERRUPT_CANCELLED
        else:
            reply = E2E_INTERRUPT_CONTINUING
        return {"messages": prior + [AIMessage(content=reply)]}

    def route_after_human_review(state: MessagesState) -> str:
        """Return the next node: END or writer_node.

        Use graph node names / END directly (no path_map) so the branch does not
        treat a state dict or other value as a routing key (avoids unhashable dict).
        """
        messages = state.get("messages", [])
        if not messages:
            return "writer_node"
        last = messages[-1]
        text = getattr(last, "content", None)

        if text == E2E_INTERRUPT_CANCELLED:
            return END
        return "writer_node"

    graph = StateGraph(MessagesState)
    graph.add_node("planner_node", agent_planner)
    graph.add_node("writer_node", agent_writer)
    graph.add_node("human_review", human_review)
    graph.add_edge(START, "planner_node")
    graph.add_edge("planner_node", "human_review")
    graph.add_conditional_edges("human_review", route_after_human_review)
    graph.add_edge("writer_node", END)
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
