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

from datarobot_genai.core.agents.message_converters import to_nat_messages


class TestToNatMessages:
    def test_user_message(self) -> None:
        msgs = [UserMessage(id="u1", content="hello")]
        result = to_nat_messages(msgs)
        assert len(result) == 1
        assert result[0].role.value == "user"
        assert result[0].content == "hello"

    def test_assistant_message(self) -> None:
        msgs = [AssistantMessage(id="a1", content="response")]
        result = to_nat_messages(msgs)
        assert len(result) == 1
        assert result[0].role.value == "assistant"
        assert result[0].content == "response"

    def test_assistant_with_tool_calls_serialized_as_text(self) -> None:
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
        result = to_nat_messages(msgs)
        assert len(result) == 1
        assert result[0].role.value == "assistant"
        assert "search" in result[0].content

    def test_system_message(self) -> None:
        msgs = [SystemMessage(id="s1", content="system prompt")]
        result = to_nat_messages(msgs)
        assert len(result) == 1
        assert result[0].role.value == "system"

    def test_tool_message_mapped_as_system(self) -> None:
        """NAT has no tool role — tool results become system messages."""
        msgs = [ToolMessage(id="t1", content="search results", tool_call_id="call_1")]
        result = to_nat_messages(msgs)
        assert len(result) == 1
        assert result[0].role.value == "system"
        assert "call_1" in result[0].content
        assert "search results" in result[0].content

    def test_empty_messages(self) -> None:
        assert to_nat_messages([]) == []
