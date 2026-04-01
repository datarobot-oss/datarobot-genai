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

from ag_ui.core import AssistantMessage
from ag_ui.core import SystemMessage
from ag_ui.core import UserMessage
from ag_ui.core.types import FunctionCall
from ag_ui.core.types import ToolCall
from ag_ui.core.types import ToolMessage

from datarobot_genai.core.agents.message_converters import to_crewai_chat_messages
from datarobot_genai.core.agents.message_converters import to_langchain_messages
from datarobot_genai.core.agents.message_converters import truncate_messages


class TestTruncateMessages:
    def test_empty_messages(self) -> None:
        assert truncate_messages([], max_history=10) == []

    def test_single_user_message_no_history(self) -> None:
        msgs = [UserMessage(id="u1", content="hello")]
        result = truncate_messages(msgs, max_history=10)
        assert len(result) == 1
        assert result[0].content == "hello"

    def test_zero_max_history_returns_only_last_user(self) -> None:
        msgs = [
            UserMessage(id="u1", content="first"),
            AssistantMessage(id="a1", content="reply"),
            UserMessage(id="u2", content="second"),
        ]
        result = truncate_messages(msgs, max_history=0)
        assert len(result) == 1
        assert result[0].content == "second"

    def test_negative_max_history_returns_only_last_user(self) -> None:
        msgs = [
            UserMessage(id="u1", content="first"),
            UserMessage(id="u2", content="second"),
        ]
        result = truncate_messages(msgs, max_history=-5)
        assert len(result) == 1
        assert result[0].content == "second"

    def test_truncates_to_max_history(self) -> None:
        msgs = [
            UserMessage(id="u1", content="msg1"),
            AssistantMessage(id="a1", content="msg2"),
            UserMessage(id="u2", content="msg3"),
            AssistantMessage(id="a2", content="msg4"),
            UserMessage(id="u3", content="msg5"),
        ]
        result = truncate_messages(msgs, max_history=2)
        # 2 history messages + current (last user + anything after)
        assert len(result) == 3
        assert result[0].content == "msg3"
        assert result[1].content == "msg4"
        assert result[2].content == "msg5"

    def test_keeps_all_when_under_max(self) -> None:
        msgs = [
            UserMessage(id="u1", content="first"),
            AssistantMessage(id="a1", content="reply"),
            UserMessage(id="u2", content="second"),
        ]
        result = truncate_messages(msgs, max_history=10)
        assert len(result) == 3

    def test_drops_orphan_tool_messages_at_start(self) -> None:
        msgs = [
            ToolMessage(id="t1", content="orphan result", tool_call_id="call_1"),
            UserMessage(id="u1", content="question"),
            AssistantMessage(id="a1", content="answer"),
            UserMessage(id="u2", content="follow-up"),
        ]
        result = truncate_messages(msgs, max_history=10)
        # Orphan tool message at start should be dropped
        assert result[0].role == "user"
        assert result[0].content == "question"

    def test_does_not_drop_tool_after_assistant(self) -> None:
        msgs = [
            UserMessage(id="u1", content="search"),
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(name="search", arguments='{"q": "test"}'),
                    )
                ],
            ),
            ToolMessage(id="t1", content="results", tool_call_id="call_1"),
            UserMessage(id="u2", content="tell me more"),
        ]
        result = truncate_messages(msgs, max_history=10)
        assert len(result) == 4
        roles = [getattr(m, "role", None) for m in result]
        assert roles == ["user", "assistant", "tool", "user"]

    def test_truncation_creates_orphan_then_drops_it(self) -> None:
        msgs = [
            UserMessage(id="u1", content="old"),
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(name="f", arguments="{}"),
                    )
                ],
            ),
            ToolMessage(id="t1", content="result", tool_call_id="call_1"),
            UserMessage(id="u2", content="recent"),
            AssistantMessage(id="a2", content="reply"),
            UserMessage(id="u3", content="latest"),
        ]
        # max_history=2 truncates history to last 2 before last user,
        # which would be [tool, user] — tool is orphaned, gets dropped
        result = truncate_messages(msgs, max_history=2)
        assert result[0].role == "user"
        assert result[0].content == "recent"

    def test_no_user_messages(self) -> None:
        msgs = [
            SystemMessage(id="s1", content="system"),
            AssistantMessage(id="a1", content="hello"),
        ]
        result = truncate_messages(msgs, max_history=10)
        assert len(result) == 2

    def test_no_user_messages_zero_max(self) -> None:
        msgs = [
            SystemMessage(id="s1", content="system"),
            AssistantMessage(id="a1", content="hello"),
        ]
        result = truncate_messages(msgs, max_history=0)
        assert result == []

    def test_messages_after_last_user_are_included(self) -> None:
        msgs = [
            UserMessage(id="u1", content="search"),
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(name="search", arguments="{}"),
                    )
                ],
            ),
            ToolMessage(id="t1", content="results", tool_call_id="call_1"),
        ]
        # Last user is u1; assistant+tool after it are "current"
        result = truncate_messages(msgs, max_history=0)
        assert len(result) == 3


class TestToLangchainMessages:
    def test_user_message(self) -> None:
        msgs = [UserMessage(id="u1", content="hello")]
        result = to_langchain_messages(msgs)
        assert len(result) == 1
        assert result[0].type == "human"
        assert result[0].content == "hello"

    def test_assistant_message_with_content(self) -> None:
        msgs = [AssistantMessage(id="a1", content="response")]
        result = to_langchain_messages(msgs)
        assert len(result) == 1
        assert result[0].type == "ai"
        assert result[0].content == "response"

    def test_assistant_message_with_tool_calls(self) -> None:
        msgs = [
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(name="search", arguments='{"query": "test"}'),
                    )
                ],
            )
        ]
        result = to_langchain_messages(msgs)
        assert len(result) == 1
        ai_msg = result[0]
        assert ai_msg.type == "ai"
        assert len(ai_msg.tool_calls) == 1
        assert ai_msg.tool_calls[0]["name"] == "search"
        assert ai_msg.tool_calls[0]["args"] == {"query": "test"}
        assert ai_msg.tool_calls[0]["id"] == "call_1"

    def test_tool_message(self) -> None:
        msgs = [ToolMessage(id="t1", content="result", tool_call_id="call_1")]
        result = to_langchain_messages(msgs)
        assert len(result) == 1
        assert result[0].type == "tool"
        assert result[0].content == "result"
        assert result[0].tool_call_id == "call_1"

    def test_system_message(self) -> None:
        msgs = [SystemMessage(id="s1", content="you are helpful")]
        result = to_langchain_messages(msgs)
        assert len(result) == 1
        assert result[0].type == "system"
        assert result[0].content == "you are helpful"

    def test_full_conversation(self) -> None:
        msgs = [
            SystemMessage(id="s1", content="system"),
            UserMessage(id="u1", content="search for cats"),
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(name="search", arguments='{"q": "cats"}'),
                    )
                ],
            ),
            ToolMessage(id="t1", content="found cats", tool_call_id="call_1"),
            AssistantMessage(id="a2", content="Here are the results about cats."),
            UserMessage(id="u2", content="tell me more"),
        ]
        result = to_langchain_messages(msgs)
        assert len(result) == 6
        types = [m.type for m in result]
        assert types == ["system", "human", "ai", "tool", "ai", "human"]

    def test_empty_messages(self) -> None:
        assert to_langchain_messages([]) == []

    def test_assistant_with_invalid_json_arguments(self) -> None:
        msgs = [
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(name="f", arguments="not json"),
                    )
                ],
            )
        ]
        result = to_langchain_messages(msgs)
        assert result[0].tool_calls[0]["args"] == {"_raw": "not json"}


class TestToCrewaiChatMessages:
    def test_user_message(self) -> None:
        msgs = [UserMessage(id="u1", content="hello")]
        result = to_crewai_chat_messages(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_assistant_message(self) -> None:
        msgs = [AssistantMessage(id="a1", content="response")]
        result = to_crewai_chat_messages(msgs)
        assert result == [{"role": "assistant", "content": "response"}]

    def test_assistant_with_tool_calls(self) -> None:
        msgs = [
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(name="search", arguments='{"q": "test"}'),
                    )
                ],
            )
        ]
        result = to_crewai_chat_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "search" in result[0]["content"]

    def test_tool_message(self) -> None:
        msgs = [ToolMessage(id="t1", content="result", tool_call_id="call_1")]
        result = to_crewai_chat_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert "result" in result[0]["content"]

    def test_system_message(self) -> None:
        msgs = [SystemMessage(id="s1", content="system")]
        result = to_crewai_chat_messages(msgs)
        assert result == [{"role": "system", "content": "system"}]

    def test_empty_messages(self) -> None:
        assert to_crewai_chat_messages([]) == []

    def test_full_conversation(self) -> None:
        msgs = [
            SystemMessage(id="s1", content="system"),
            UserMessage(id="u1", content="search cats"),
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function=FunctionCall(name="search", arguments='{"q":"cats"}'),
                    )
                ],
            ),
            ToolMessage(id="t1", content="found cats", tool_call_id="c1"),
            AssistantMessage(id="a2", content="Here are cats."),
        ]
        result = to_crewai_chat_messages(msgs)
        assert len(result) == 5
        roles = [m["role"] for m in result]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]
