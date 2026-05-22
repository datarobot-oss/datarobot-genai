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
"""Smoke test: LlamaIndex agent against a live LLM.

Verifies that a minimal LlamaIndexAgent can complete a round-trip with the
model specified by LLM_DEFAULT_MODEL, without spinning up an HTTP server.
"""
from __future__ import annotations

import uuid
from typing import Any

import pytest
from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import UserMessage
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import FunctionAgent

from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.llama_index.agent import datarobot_agent_class_from_llamaindex
from datarobot_genai.llama_index.llm import get_llm

_llm = get_llm()

_agent = FunctionAgent(
    name="assistant",
    description="A helpful assistant that answers questions directly.",
    system_prompt=make_system_prompt("You are a helpful assistant."),
    llm=_llm,
    tools=[],
)

_workflow = AgentWorkflow(agents=[_agent], root_agent="assistant")


def _extract_response_text(result_state: Any, events: list[Any]) -> str:
    for event in reversed(events):
        resp = getattr(event, "response", None)
        if resp is not None:
            content = getattr(resp, "content", None)
            if content:
                return str(content)
    return ""


SmokeAgent = datarobot_agent_class_from_llamaindex(_workflow, [_agent], _extract_response_text)


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
    assert "".join(text_parts).strip(), "Expected non-empty text response from LlamaIndex agent"
