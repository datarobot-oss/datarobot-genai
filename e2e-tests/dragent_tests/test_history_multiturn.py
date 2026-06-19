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
"""Multi-turn end-to-end test for structured tool-call history.

Turn 1 forces a deterministic tool call (and, with thinking enabled, reasoning).
``events_to_messages`` folds that streamed run back into
``AssistantMessage(tool_calls=...)`` / ``ToolMessage`` / ``ReasoningMessage``,
which are replayed as conversation history on turn 2. The follow-up turn must
recall the tool result -- exercising native tool-call history end to end.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from ag_ui.core import EventType
from datarobot_genai.core.agents.events import events_to_messages
from datarobot_genai.core.agents.verify import validate_sequence

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import AGENT_SUPPORTS_TOOL_CALLS_STREAMING
from dragent_tests.helpers import collect_ag_ui_events
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import make_generate_payload
from dragent_tests.helpers import should_run_reasoning_test
from dragent_tests.helpers import stream_sse_responses

# Runs for every framework that streams tool-call events, so the fold can rebuild
# the tool call from the stream: langgraph, nat, llamaindex. crewai (supports tools
# but does not stream tool-call events) and base (no tools) are skipped. For
# langgraph and llama_index the replayed history is structured (their e2e agents
# have no ``{chat_history}`` prompt); nat replays it as the flattened text summary.
if not AGENT_SUPPORTS_TOOL_CALLS_STREAMING:
    pytest.skip(
        f"Multi-turn tool-call history needs streaming tool-call events ({AGENT} does not "
        "emit them).",
        allow_module_level=True,
    )

EXPECTED_OBJECT_ID = "69cbb73789723b6936c6c9e1"  # from tool.py


def _turn_one_prompt() -> str:
    """Nudge a brief reasoning step, force the deterministic tool, pin the answer.

    The trailing nonce defeats response caching across runs.
    """
    return (
        "Think briefly about the request. You MUST use the generate_objectid tool to generate "
        "an object ID for a deployment; do NOT make one up. Call generate_objectid with this "
        "exact input: deployment. Then reply with ONLY the exact object ID the tool returned, "
        f"nothing else. Here is an ID you SHOULD NOT USE: {uuid.uuid4().hex}."
    )


def test_multiturn_replays_tool_call_and_reasoning_history(http_client: httpx.Client) -> None:
    """Fold a tool+reasoning turn into history and recall its result on the next turn."""
    # GIVEN: turn 1 forces the deterministic generate_objectid tool
    turn_one_prompt = _turn_one_prompt()
    turn_one_responses = stream_sse_responses(http_client, make_generate_payload(turn_one_prompt))
    turn_one_events = collect_ag_ui_events(turn_one_responses)

    # THEN: turn 1 is a valid stream that called the tool
    validate_sequence(turn_one_events)
    event_types = {e.type for e in turn_one_events}
    assert EventType.TOOL_CALL_START in event_types, (
        f"turn 1 did not call a tool; got {sorted(event_types)}"
    )
    tool_names = {e.tool_call_name for e in turn_one_events if e.type == EventType.TOOL_CALL_START}
    assert "generate_objectid" in tool_names, f"expected generate_objectid, got {tool_names}"

    # WHEN: the streamed run is folded into replayable history messages
    history = events_to_messages(turn_one_events)
    roles = [m.role for m in history]

    # THEN: the assistant tool-call turn, its paired result, and reasoning are rebuilt
    assistant_with_calls = next(
        (m for m in history if m.role == "assistant" and getattr(m, "tool_calls", None)),
        None,
    )
    assert assistant_with_calls is not None, f"no assistant tool-call message in {roles}"
    assert any(
        call.function.name == "generate_objectid" for call in assistant_with_calls.tool_calls
    )
    tool_result = next((m for m in history if m.role == "tool"), None)
    assert tool_result is not None and EXPECTED_OBJECT_ID in tool_result.content, (
        f"tool result {EXPECTED_OBJECT_ID} not captured in history: "
        f"{getattr(tool_result, 'content', None)!r}"
    )

    if should_run_reasoning_test():
        # Reasoning is only emitted by langgraph and llama_index (with thinking enabled);
        # for those the fold must also carry it as a reasoning message.
        # And not models support it.
        assert "reasoning" in roles, (
            "expected the run to emit reasoning (thinking enabled) and the helper to "
            f"capture it; history roles were {roles}"
        )

    # WHEN: the history is replayed before a follow-up question
    uid = uuid.uuid4().hex[:8]
    history_dicts = [m.model_dump(by_alias=True, exclude_none=True) for m in history]
    turn_two_payload = {
        "threadId": f"test-multiturn-{uid}",
        "runId": f"run-{uid}",
        "messages": [
            {"role": "user", "content": turn_one_prompt, "id": f"u1-{uid}"},
            *history_dicts,
            {
                "role": "user",
                "content": (
                    "What object ID did the generate_objectid tool return earlier in this "
                    "conversation? Reply with ONLY that ID and nothing else."
                ),
                "id": f"u2-{uid}",
            },
        ],
        "tools": [],
        "context": [],
        "forwardedProps": {},
        "state": {},
    }
    turn_two_events = collect_ag_ui_events(stream_sse_responses(http_client, turn_two_payload))

    # THEN: the follow-up recalls the tool result from the replayed history
    validate_sequence(turn_two_events)
    answer = collect_text(turn_two_events)
    assert EXPECTED_OBJECT_ID in answer, (
        "follow-up turn did not recall the tool result from replayed history. "
        f"Got: {answer[:500]!r}"
    )
