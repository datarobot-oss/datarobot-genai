# Copyright 2025 DataRobot, Inc. and its affiliates.
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

import pytest
from ag_ui.core import AssistantMessage
from ag_ui.core import RunAgentInput
from ag_ui.core import SystemMessage
from ag_ui.core import UserMessage
from ag_ui.core.types import FunctionCall
from ag_ui.core.types import ToolCall
from ag_ui.core.types import ToolMessage
from ragas.messages import AIMessage
from ragas.messages import HumanMessage

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import extract_history_messages
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.agents.base import make_system_prompt


def _make_run_agent_input_from_dicts(messages: list[dict[str, Any]]) -> RunAgentInput:
    """Convert dict-style messages to RunAgentInput for testing history functions."""
    from ag_ui.core.types import DeveloperMessage

    ag_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        tool_calls_raw = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")

        if role == "user":
            ag_messages.append(UserMessage(id=f"msg_{len(ag_messages)}", content=content or ""))
        elif role == "assistant":
            tool_calls = None
            if tool_calls_raw:
                tool_calls = []
                for tc in tool_calls_raw:
                    fn = tc.get("function", {})
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", f"call_{len(tool_calls)}"),
                            function=FunctionCall(
                                name=fn.get("name", ""),
                                arguments=fn.get("arguments", "{}"),
                            ),
                        )
                    )
            ag_messages.append(
                AssistantMessage(
                    id=f"msg_{len(ag_messages)}",
                    content=content or "",
                    toolCalls=tool_calls,
                )
            )
        elif role == "system":
            ag_messages.append(SystemMessage(id=f"msg_{len(ag_messages)}", content=content or ""))
        elif role == "tool":
            # Use proper ToolMessage type
            ag_messages.append(
                ToolMessage(
                    id=f"msg_{len(ag_messages)}",
                    content=content or "",
                    toolCallId=tool_call_id or "unknown",
                )
            )
        elif role == "developer":
            ag_messages.append(
                DeveloperMessage(id=f"msg_{len(ag_messages)}", content=content or "")
            )

    return RunAgentInput(
        messages=ag_messages,
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )


class SimpleAgent(BaseAgent):
    async def invoke(
        self, run_agent_input: RunAgentInput
    ) -> tuple[str, Any | None, dict[str, int]]:
        return "ok", None, {}


def test_base_agent_env_defaults_and_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "env-token")
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2")

    agent = SimpleAgent(verbose="false")

    assert agent.api_key == "env-token"
    # Stored as provided; normalization happens in litellm_api_base
    assert agent.api_base == "https://app.datarobot.com/api/v2"
    assert agent.verbose is False


def test_base_agent_litellm_api_base_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "token")
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2")

    agent = SimpleAgent()
    api = agent.litellm_api_base("dep-123")
    assert api == "https://app.datarobot.com/api/v2/deployments/dep-123/chat/completions"


@pytest.fixture
def run_agent_input() -> RunAgentInput:
    return RunAgentInput(
        messages=[],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )


def test_extract_user_prompt_content_no_user_messages(run_agent_input: RunAgentInput) -> None:
    # GIVEN a completion create params with no user messages
    # WHEN extracting the user prompt content
    user_prompt = extract_user_prompt_content(run_agent_input)
    # THEN the user prompt is empty
    assert user_prompt == ""


def test_extract_user_prompt_content_the_last_user_message(run_agent_input: RunAgentInput) -> None:
    # GIVEN a run agent input with multiple user messages
    run_agent_input.messages = [
        SystemMessage(content="x", id="message_0"),
        UserMessage(content="ignored", id="message_1"),
        UserMessage(content="something", id="message_2"),
    ]
    # WHEN extracting the user prompt content
    user_prompt = extract_user_prompt_content(run_agent_input)
    # THEN the user prompt is the last user message
    assert user_prompt == "something"


def test_extract_user_prompt_content_the_last_user_message_is_a_json_string(
    run_agent_input: RunAgentInput,
) -> None:
    # GIVEN a run agent input with a user message that is a json string
    run_agent_input.messages = [
        SystemMessage(content="x", id="message_0"),
        UserMessage(content="ignored", id="message_1"),
        UserMessage(content='{"foo": "bar"}', id="message_2"),
    ]
    # WHEN extracting the user prompt content
    user_prompt = extract_user_prompt_content(run_agent_input)
    # THEN the user prompt is the last user message
    assert user_prompt == {"foo": "bar"}


def test_extract_history_messages_excludes_final_user_turn_only_when_last_message_is_user() -> None:
    # GIVEN a RunAgentInput with multiple messages ending with user
    run_agent_input = _make_run_agent_input_from_dicts(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
    )
    # WHEN extracting history
    history = extract_history_messages(run_agent_input, max_history=20)
    # THEN the final user message is excluded from history
    assert [(m["role"], m["content"]) for m in history] == [("user", "u1"), ("assistant", "a1")]


def test_extract_history_messages_keeps_trailing_assistant_or_tool_messages() -> None:
    # GIVEN a RunAgentInput that ends with assistant output
    run_agent_input = _make_run_agent_input_from_dicts(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
    )
    # WHEN extracting history
    history = extract_history_messages(run_agent_input, max_history=20)
    # THEN all messages are included since last message is not user
    assert [(m["role"], m["content"]) for m in history] == [("user", "u1"), ("assistant", "a1")]


def test_extract_history_messages_preserves_tool_call_only_assistant_messages() -> None:
    # GIVEN a RunAgentInput with tool calls
    run_agent_input = _make_run_agent_input_from_dicts(
        [
            {"role": "user", "content": "u1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "search"}}],
            },
            {"role": "user", "content": "u2"},
        ]
    )
    # WHEN extracting history
    history = extract_history_messages(run_agent_input, max_history=20)
    # THEN tool call info is preserved
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "u1"
    assert history[1]["role"] == "assistant"
    assert "search" in history[1]["content"]


def test_extract_history_messages_preserves_tool_messages_even_with_empty_content() -> None:
    # GIVEN a RunAgentInput with a tool message
    run_agent_input = _make_run_agent_input_from_dicts(
        [
            {"role": "user", "content": "u1"},
            {"role": "tool", "tool_call_id": "call_1", "content": None},
            {"role": "user", "content": "u2"},
        ]
    )
    # WHEN extracting history
    history = extract_history_messages(run_agent_input, max_history=20)
    # THEN tool message is preserved
    assert len(history) == 2
    assert history[1]["role"] == "tool"


def test_build_history_summary_includes_tool_call_summaries() -> None:
    # GIVEN a RunAgentInput with tool calls
    run_agent_input = _make_run_agent_input_from_dicts(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "t"}}]},
            {"role": "user", "content": "u2"},
        ]
    )
    # WHEN building history summary via the public API
    agent = SimpleAgent()
    summary = agent.build_history_summary(run_agent_input)
    # THEN tool call is included in summary
    assert "user: u1" in summary
    assert "assistant:" in summary
    assert "t" in summary


def test_make_system_prompt() -> None:
    msg = make_system_prompt("TAIL")
    assert msg.endswith("\nTAIL")
    assert "helpful AI assistant" in msg


def test_create_pipeline_interactions_from_events_simple() -> None:
    assert BaseAgent.create_pipeline_interactions_from_events(None) is None
    msgs: list[Any] = [HumanMessage(content="hi"), AIMessage(content="ok")]
    sample = BaseAgent.create_pipeline_interactions_from_events(msgs)
    assert sample is not None
    assert sample.user_input == msgs


def test_base_agent_forwarded_headers_none() -> None:
    """Test BaseAgent with no forwarded headers."""
    agent = SimpleAgent(forwarded_headers=None)
    assert agent.forwarded_headers == {}


def test_base_agent_forwarded_headers_empty() -> None:
    """Test BaseAgent with empty forwarded headers."""
    agent = SimpleAgent(forwarded_headers={})
    assert agent.forwarded_headers == {}


def test_base_agent_forwarded_headers_with_scoped_token() -> None:
    """Test BaseAgent with forwarded headers including scoped token."""
    headers = {
        "x-datarobot-api-key": "scoped-token-123",
        "x-custom-header": "custom-value",
    }
    agent = SimpleAgent(forwarded_headers=headers)
    assert agent.forwarded_headers == headers
    assert agent.forwarded_headers["x-datarobot-api-key"] == "scoped-token-123"
    assert agent.forwarded_headers["x-custom-header"] == "custom-value"


def test_base_agent_forwarded_headers_with_bearer_token() -> None:
    """Test BaseAgent with forwarded headers containing Bearer token format."""
    headers = {
        "x-datarobot-api-key": "Bearer scoped-token-456",
        "authorization": "Bearer main-token",
    }
    agent = SimpleAgent(forwarded_headers=headers)
    assert agent.forwarded_headers == headers
    assert agent.forwarded_headers["x-datarobot-api-key"] == "Bearer scoped-token-456"
