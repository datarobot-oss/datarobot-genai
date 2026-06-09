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
from llama_index.llms.litellm.utils import to_openai_message_dicts

from datarobot_genai.llama_index.history import ag_ui_history_to_chat_messages


def test_converts_roles_and_preserves_tool_calls() -> None:
    history = [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
            ],
        },
        {"role": "tool", "content": "18C, sunny", "tool_call_id": "c1"},
        {"role": "system", "content": "sys"},
    ]

    msgs = ag_ui_history_to_chat_messages(history)

    assert [m.role for m in msgs] == ["user", "assistant", "tool", "system"]
    # Tool calls ride in additional_kwargs as OpenAI wire shape (arguments stays a string).
    assert msgs[1].content == "Let me check."
    assert msgs[1].additional_kwargs["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ]
    # Tool result paired by id.
    assert msgs[2].additional_kwargs["tool_call_id"] == "c1"
    assert "18C" in msgs[2].content


def test_tool_call_only_assistant_turn_has_empty_content() -> None:
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "search", "arguments": "{}"}}],
        }
    ]
    msgs = ag_ui_history_to_chat_messages(history)
    assert msgs[0].content == ""
    assert msgs[0].additional_kwargs["tool_calls"][0]["function"]["name"] == "search"


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
    tool_calls = ag_ui_history_to_chat_messages(history)[0].additional_kwargs["tool_calls"]
    assert [tc["id"] for tc in tool_calls] == ["c1", "c2"]
    assert [tc["function"]["name"] for tc in tool_calls] == ["get_weather", "log_event"]


def test_missing_or_dict_arguments_normalized() -> None:
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "function": {"name": "x"}},  # missing arguments -> "{}"
                {"id": "b", "function": {"name": "y", "arguments": {"k": 1}}},  # dict -> JSON str
            ],
        }
    ]
    tool_calls = ag_ui_history_to_chat_messages(history)[0].additional_kwargs["tool_calls"]
    assert tool_calls[0]["function"]["arguments"] == "{}"
    assert tool_calls[1]["function"]["arguments"] == '{"k": 1}'


def test_converted_history_serializes_to_openai_without_crashing() -> None:
    """Regression for the ToolCallBlock crash: the converted history must serialize
    through the LiteLLM message adapter into valid OpenAI tool_calls.
    """
    history = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
            ],
        },
        {"role": "tool", "content": "18C, sunny", "tool_call_id": "c1"},
    ]

    dicts = to_openai_message_dicts(ag_ui_history_to_chat_messages(history))

    assistant = next(d for d in dicts if d["role"] == "assistant")
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    tool = next(d for d in dicts if d["role"] == "tool")
    assert tool["tool_call_id"] == "c1"


def test_carries_folded_reasoning_in_assistant_content() -> None:
    # extract_history_messages folds standalone reasoning into the assistant content;
    # the converter must carry that <reasoning> text through to the model (it would
    # otherwise drop a standalone reasoning-role message).
    history = [{"role": "assistant", "content": "<reasoning>\nadded 2+2\n</reasoning>\n4"}]
    msgs = ag_ui_history_to_chat_messages(history)
    assert msgs[0].role == "assistant"
    assert "<reasoning>" in msgs[0].content
    assert "added 2+2" in msgs[0].content
