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
