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
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage

from datarobot_genai.langgraph.history import ag_ui_history_to_langchain


def test_converts_roles_and_preserves_tool_calls() -> None:
    history = [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "content": "18C, sunny", "tool_call_id": "c1"},
        {"role": "system", "content": "sys"},
    ]

    msgs = ag_ui_history_to_langchain(history)

    assert [type(m) for m in msgs] == [HumanMessage, AIMessage, ToolMessage, SystemMessage]
    assert msgs[0].content == "weather in Paris?"
    # Tool call mapped to LangChain shape: name top-level, args parsed to a dict.
    assert msgs[1].content == "Let me check."
    assert msgs[1].tool_calls == [
        {"name": "get_weather", "args": {"city": "Paris"}, "id": "c1", "type": "tool_call"}
    ]
    # Tool result paired by id.
    assert msgs[2].content == "18C, sunny"
    assert msgs[2].tool_call_id == "c1"


def test_tool_call_only_assistant_turn_has_empty_content() -> None:
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "search", "arguments": "{}"}}],
        }
    ]
    msgs = ag_ui_history_to_langchain(history)
    assert isinstance(msgs[0], AIMessage)
    assert msgs[0].content == ""  # must be "" not None for tool-call-only turns
    assert msgs[0].tool_calls[0]["name"] == "search"


def test_malformed_or_missing_arguments_default_to_empty_dict() -> None:
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "bad", "function": {"name": "x", "arguments": "not-json"}},
                {"id": "none", "function": {"name": "y"}},
                # valid JSON but not an object -> coerced to {}
                {"id": "arr", "function": {"name": "z", "arguments": "[1, 2]"}},
            ],
        }
    ]
    msgs = ag_ui_history_to_langchain(history)
    assert msgs[0].tool_calls[0]["args"] == {}
    assert msgs[0].tool_calls[1]["args"] == {}
    assert msgs[0].tool_calls[2]["args"] == {}  # non-dict JSON coerced to {}


def test_parallel_tool_calls_in_one_turn_all_mapped() -> None:
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
                {"id": "c2", "function": {"name": "log_event", "arguments": '{"e": "x"}'}},
            ],
        }
    ]
    tool_calls = ag_ui_history_to_langchain(history)[0].tool_calls
    assert [tc["id"] for tc in tool_calls] == ["c1", "c2"]
    assert [tc["name"] for tc in tool_calls] == ["get_weather", "log_event"]
    assert tool_calls[0]["args"] == {"city": "Paris"}


def test_carries_folded_reasoning_in_assistant_content() -> None:
    # extract_history_messages folds standalone reasoning into the assistant content;
    # the converter must carry that <reasoning> text through to the model (it would
    # otherwise drop a standalone reasoning-role message).
    history = [{"role": "assistant", "content": "<reasoning>\nadded 2+2\n</reasoning>\n4"}]
    msgs = ag_ui_history_to_langchain(history)
    assert isinstance(msgs[0], AIMessage)
    assert "<reasoning>" in msgs[0].content
    assert "added 2+2" in msgs[0].content
