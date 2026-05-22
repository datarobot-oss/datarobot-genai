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
"""Smoke test: CrewAI agent against a live LLM.

Verifies that a minimal CrewAIAgent can complete a round-trip with the
model specified by LLM_DEFAULT_MODEL, without spinning up an HTTP server.
"""
from __future__ import annotations

import uuid

import pytest
from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import UserMessage
from crewai import Agent
from crewai import Crew
from crewai import Task

from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.crewai.agent import datarobot_agent_class_from_crew
from datarobot_genai.crewai.llm import get_llm

_llm = get_llm()

_agent = Agent(
    role="Assistant",
    goal="Answer the user's question directly: {topic}",
    backstory=make_system_prompt("You are a helpful assistant."),
    llm=_llm,
)

_task = Task(
    description="Answer the following question: {topic}. Context: {chat_history}",
    expected_output="A direct, concise answer.",
    agent=_agent,
)

_crew = Crew(agents=[_agent], tasks=[_task], stream=True)

SmokeAgent = datarobot_agent_class_from_crew(
    _crew,
    [_agent],
    [_task],
    lambda user_prompt: {"topic": user_prompt, "chat_history": ""},
)


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
    agent = SmokeAgent(llm=_llm)
    text_parts: list[str] = []
    async for event, _, _ in agent.invoke(_make_input("Reply with exactly one word: OK.")):
        if event.type in (EventType.TEXT_MESSAGE_CONTENT, EventType.TEXT_MESSAGE_CHUNK):
            text_parts.append(event.delta)
    assert "".join(text_parts).strip(), "Expected non-empty text response from CrewAI agent"
