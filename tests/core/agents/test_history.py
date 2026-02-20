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


from ag_ui.core import AssistantMessage
from ag_ui.core import RunAgentInput
from ag_ui.core import UserMessage

from datarobot_genai.core.agents.history import _summarize_tool_calls
from datarobot_genai.core.agents.history import extract_history_messages


def _make_run_agent_input(messages: list) -> RunAgentInput:
    return RunAgentInput(
        messages=messages,
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )


# ---------------------------------------------------------------------------
# _summarize_tool_calls
# ---------------------------------------------------------------------------


class TestSummarizeToolCalls:
    def test_none_returns_placeholder(self) -> None:
        assert _summarize_tool_calls(None) == "[tool_calls]"

    def test_empty_list_returns_placeholder(self) -> None:
        assert _summarize_tool_calls([]) == "[tool_calls]"

    def test_falsy_value_returns_placeholder(self) -> None:
        assert _summarize_tool_calls(0) == "[tool_calls]"
        assert _summarize_tool_calls("") == "[tool_calls]"

    def test_single_dict_with_function_name(self) -> None:
        tool_calls = [{"function": {"name": "search"}}]
        assert _summarize_tool_calls(tool_calls) == "[tool_calls] search"

    def test_multiple_dicts_with_function_names(self) -> None:
        tool_calls = [
            {"function": {"name": "search"}},
            {"function": {"name": "calculate"}},
        ]
        assert _summarize_tool_calls(tool_calls) == "[tool_calls] search, calculate"

    def test_dict_with_name_key_fallback(self) -> None:
        """When no 'function' key exists, falls back to 'name' key."""
        tool_calls = [{"name": "lookup"}]
        assert _summarize_tool_calls(tool_calls) == "[tool_calls] lookup"

    def test_dict_with_tool_name_key_fallback(self) -> None:
        """When no 'function' or 'name' key exists, falls back to 'tool_name'."""
        tool_calls = [{"tool_name": "fetch_data"}]
        assert _summarize_tool_calls(tool_calls) == "[tool_calls] fetch_data"

    def test_dict_function_name_takes_priority_over_name(self) -> None:
        tool_calls = [{"function": {"name": "fn_name"}, "name": "top_name"}]
        assert _summarize_tool_calls(tool_calls) == "[tool_calls] fn_name"

    def test_dict_name_takes_priority_over_tool_name(self) -> None:
        tool_calls = [{"name": "n", "tool_name": "tn"}]
        assert _summarize_tool_calls(tool_calls) == "[tool_calls] n"

    def test_dict_with_no_name_anywhere(self) -> None:
        """When no name can be found, the tool call is skipped."""
        tool_calls = [{"id": "call_1"}]
        assert _summarize_tool_calls(tool_calls) == "[tool_calls]"

    def test_dict_with_function_key_but_no_name(self) -> None:
        tool_calls = [{"function": {"arguments": "{}"}}]
        assert _summarize_tool_calls(tool_calls) == "[tool_calls]"

    def test_dict_function_is_not_mapping(self) -> None:
        """When 'function' exists but is not a Mapping, falls back to name/tool_name."""
        tool_calls = [{"function": "not_a_dict", "name": "fallback"}]
        assert _summarize_tool_calls(tool_calls) == "[tool_calls] fallback"

    def test_pydantic_like_objects_with_function_attr(self) -> None:
        """Pydantic-style objects with .function.name attribute."""

        class FnObj:
            name = "pydantic_search"

        class TcObj:
            function = FnObj()

        assert _summarize_tool_calls([TcObj()]) == "[tool_calls] pydantic_search"

    def test_pydantic_like_objects_name_attr_fallback(self) -> None:
        class TcObj:
            function = None
            name = "direct_name"

        assert _summarize_tool_calls([TcObj()]) == "[tool_calls] direct_name"

    def test_pydantic_like_objects_tool_name_attr_fallback(self) -> None:
        class TcObj:
            function = None
            name = None
            tool_name = "alt_name"

        assert _summarize_tool_calls([TcObj()]) == "[tool_calls] alt_name"

    def test_mixed_named_and_unnamed(self) -> None:
        """Only tool calls with extractable names appear in the summary."""
        tool_calls = [
            {"function": {"name": "found"}},
            {"id": "no_name"},
            {"name": "also_found"},
        ]
        assert _summarize_tool_calls(tool_calls) == "[tool_calls] found, also_found"

    def test_non_list_truthy_value(self) -> None:
        """A non-list truthy value that is not iterable returns the fallback."""
        assert _summarize_tool_calls(42) == "[tool_calls]"


# ---------------------------------------------------------------------------
# extract_history_messages — max_history truncation
# ---------------------------------------------------------------------------


class TestExtractHistoryMessagesTruncation:
    def test_truncates_to_max_history(self) -> None:
        """When history exceeds max_history, only the most recent N are kept."""
        messages = [
            UserMessage(id="u1", content="msg1"),
            AssistantMessage(id="a1", content="reply1"),
            UserMessage(id="u2", content="msg2"),
            AssistantMessage(id="a2", content="reply2"),
            UserMessage(id="u3", content="msg3"),
            AssistantMessage(id="a3", content="reply3"),
            UserMessage(id="u4", content="current"),  # final user msg, excluded from history
        ]
        rai = _make_run_agent_input(messages)
        # 6 history messages before current, ask for only 2
        history = extract_history_messages(rai, max_history=2)
        assert len(history) == 2
        # Should keep the 2 most recent: u3, a3
        assert history[0]["content"] == "msg3"
        assert history[1]["content"] == "reply3"

    def test_truncation_keeps_exact_count(self) -> None:
        messages = [
            UserMessage(id="u1", content="a"),
            AssistantMessage(id="a1", content="b"),
            UserMessage(id="u2", content="c"),
            AssistantMessage(id="a2", content="d"),
            UserMessage(id="u3", content="current"),
        ]
        rai = _make_run_agent_input(messages)
        history = extract_history_messages(rai, max_history=3)
        assert len(history) == 3
        assert [m["content"] for m in history] == ["b", "c", "d"]

    def test_max_history_equal_to_message_count_no_truncation(self) -> None:
        messages = [
            UserMessage(id="u1", content="a"),
            AssistantMessage(id="a1", content="b"),
            UserMessage(id="u2", content="current"),
        ]
        rai = _make_run_agent_input(messages)
        history = extract_history_messages(rai, max_history=2)
        assert len(history) == 2
        assert [m["content"] for m in history] == ["a", "b"]

    def test_max_history_larger_than_messages_returns_all(self) -> None:
        messages = [
            UserMessage(id="u1", content="a"),
            AssistantMessage(id="a1", content="b"),
            UserMessage(id="u2", content="current"),
        ]
        rai = _make_run_agent_input(messages)
        history = extract_history_messages(rai, max_history=100)
        assert len(history) == 2

    def test_max_history_one(self) -> None:
        messages = [
            UserMessage(id="u1", content="old"),
            AssistantMessage(id="a1", content="older_reply"),
            UserMessage(id="u2", content="recent"),
            AssistantMessage(id="a2", content="recent_reply"),
            UserMessage(id="u3", content="current"),
        ]
        rai = _make_run_agent_input(messages)
        history = extract_history_messages(rai, max_history=1)
        assert len(history) == 1
        assert history[0]["content"] == "recent_reply"

    def test_max_history_zero_disables(self) -> None:
        messages = [
            UserMessage(id="u1", content="a"),
            AssistantMessage(id="a1", content="b"),
            UserMessage(id="u2", content="current"),
        ]
        rai = _make_run_agent_input(messages)
        assert extract_history_messages(rai, max_history=0) == []

    def test_max_history_negative_disables(self) -> None:
        messages = [
            UserMessage(id="u1", content="a"),
            AssistantMessage(id="a1", content="b"),
            UserMessage(id="u2", content="current"),
        ]
        rai = _make_run_agent_input(messages)
        assert extract_history_messages(rai, max_history=-5) == []


# ---------------------------------------------------------------------------
# extract_history_messages — single user message edge case
# ---------------------------------------------------------------------------


class TestExtractHistoryMessagesSingleUserMessage:
    def test_single_user_message_returns_empty_history(self) -> None:
        """When RunAgentInput has exactly one user message, history is empty."""
        messages = [UserMessage(id="u1", content="hello")]
        rai = _make_run_agent_input(messages)
        assert extract_history_messages(rai, max_history=20) == []

    def test_empty_messages_returns_empty_history(self) -> None:
        rai = _make_run_agent_input([])
        assert extract_history_messages(rai, max_history=20) == []
