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
"""Smoke test: LangGraph agent against a live LLM.

Verifies that a minimal LangGraphAgent can complete a round-trip with the
model specified by LLM_DEFAULT_MODEL, without spinning up an HTTP server.
"""
from __future__ import annotations

import uuid

import pytest
from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import UserMessage
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph

from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.langgraph.agent import datarobot_agent_class_from_langgraph
from datarobot_genai.langgraph.llm import get_llm


def _make_workflow(llm, tools, verbose):
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=make_system_prompt("You are a helpful assistant."),
        name="assistant",
        debug=verbose,
    )
    graph = StateGraph(MessagesState)
    graph.add_node("assistant", agent)
    graph.add_edge(START, "assistant")
    graph.add_edge("assistant", END)
    return graph


_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
    ]
)

SmokeAgent = datarobot_agent_class_from_langgraph(_make_workflow, _prompt)


def _make_input(content: str) -> RunAgentInput:
    uid = uuid.uuid4().hex[:8]
    return RunAgentInput(
        thread_id=f"smoke-{uid}",
        run_id=f"run-{uid}",
        messages=[UserMessage(id=f"msg-{uid}", role="user", content=content)],
        tools=[],
        context=[],
        forwarded_props={},
        state={},
    )


@pytest.mark.asyncio
async def test_basic_response() -> None:
    agent = SmokeAgent(llm=get_llm())
    text_parts: list[str] = []
    async for event, _, _ in agent.invoke(_make_input("Reply with exactly one word: OK.")):
        if event.type in (EventType.TEXT_MESSAGE_CONTENT, EventType.TEXT_MESSAGE_CHUNK):
            text_parts.append(event.delta)
    assert "".join(text_parts).strip(), "Expected non-empty text response from LangGraph agent"
