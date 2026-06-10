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

from datarobot_genai.langgraph.agent import datarobot_agent_class_from_langgraph
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.types import interrupt

from dragent.tool import generate_objectid

generate_objectid_tool = tool(generate_objectid)

# Emitted in assistant text so HTTP e2e can assert without parsing tool payloads.
E2E_INTERRUPT_CANCELLED = "E2E_INTERRUPT_CANCELLED"
E2E_INTERRUPT_CONTINUING = "E2E_INTERRUPT_CONTINUING"

prompt_template = ChatPromptTemplate.from_messages(
    [
        # No {chat_history} placeholder, so BUZZOK-31124 structured_history is active:
        # prior turns (incl. tool calls) replay as native messages instead of a flattened
        # text summary. This is what lets the multi-turn e2e exercise the structured path.
        ("system", "Use any available tools to answer the user's request."),
        ("user", "{topic}"),
    ]
)


def graph_factory(
    llm: BaseChatModel, tools: list[BaseTool], verbose: bool = False
) -> StateGraph[MessagesState]:
    agent_planner = create_agent(
        llm,
        tools=[generate_objectid_tool] + tools,
        system_prompt=(
            "Call any required tool. Reply with only the tool's result, "
            "or 1 brief line if no tool is needed."
        ),
        name="planner",
        debug=verbose,
    )

    agent_writer = create_agent(
        llm,
        tools=[generate_objectid_tool] + tools,
        system_prompt="Reply with only the tool's result, or 1 brief line.",
        name="writer",
        debug=verbose,
    )

    def human_review(state: MessagesState) -> MessagesState:
        decision = interrupt({"e2e_prompt": "approve?", "kind": "e2e_hitl"})
        if decision == "no":
            reply = E2E_INTERRUPT_CANCELLED
        else:
            reply = E2E_INTERRUPT_CONTINUING
        return {"messages": [AIMessageChunk(content=reply)]}

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

    def route_after_planner(state: MessagesState) -> str:
        """Send only the interrupt/resume scenario (topic ``start``) through the
        HITL interrupt; every other run goes straight to the writer so the agent
        completes in a single turn. The multi-turn structured-history e2e relies
        on this (it must not pause on an interrupt).
        """
        for message in state.get("messages", []):
            content = getattr(message, "content", "")
            if isinstance(content, str) and (
                '"topic": "start"' in content or "'topic': 'start'" in content
            ):
                return "human_review"
        return "writer_node"

    graph = StateGraph(MessagesState)
    graph.add_node("planner_node", agent_planner)
    graph.add_node("writer_node", agent_writer)
    graph.add_node("human_review", human_review)
    graph.add_edge(START, "planner_node")
    graph.add_conditional_edges("planner_node", route_after_planner)
    graph.add_conditional_edges("human_review", route_after_human_review)
    graph.add_edge("writer_node", END)
    return graph


# NAT constructs a new agent per HTTP request; a fresh InMemorySaver per instance would
# drop checkpoint state between the interrupt request and the resume request. One
# shared saver keeps thread state for this E2E workflow only (not for production).
HITL_E2E_CHECKPOINTER = InMemorySaver()

MyAgent = datarobot_agent_class_from_langgraph(graph_factory, prompt_template)
