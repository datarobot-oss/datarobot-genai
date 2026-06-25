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

"""Tests for :mod:`datarobot_genai.crewai.llm`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from crewai import LLM

from datarobot_genai.core.config import DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM
from datarobot_genai.core.config import LLMType
from datarobot_genai.crewai import llm as crewai_llm
from datarobot_genai.crewai.llm import LitellmStopWordLLM


@pytest.fixture(autouse=True)
def patched_crewai_llm_defaults() -> None:
    with (
        patch.object(crewai_llm, "default_api_key", return_value="sk-test-key"),
        patch.object(crewai_llm, "default_model_name", return_value="default-model"),
        patch.object(
            crewai_llm,
            "default_datarobot_llm_gateway_url",
            return_value="https://example.test/genai/llmgw",
        ),
        patch.object(
            crewai_llm,
            "default_deployment_url",
            side_effect=lambda deployment_id: (
                f"https://example.test/deployments/{deployment_id}/chat/completions"
            ),
        ),
    ):
        yield


def test_module_exports_llm_factory_functions() -> None:
    assert callable(crewai_llm.get_datarobot_gateway_llm)
    assert callable(crewai_llm.get_datarobot_deployment_llm)
    assert callable(crewai_llm.get_datarobot_nim_llm)


def test_get_datarobot_gateway_llm_returns_crewai_llm() -> None:
    llm = crewai_llm.get_datarobot_gateway_llm()
    assert isinstance(llm, LLM)
    assert llm.model == "datarobot/default-model"
    assert llm.api_base == "https://example.test/genai/llmgw"
    assert llm.api_key == "sk-test-key"
    assert llm.is_litellm is True
    assert llm.additional_params == {"stream_options": {"include_usage": True}}


def test_get_datarobot_gateway_llm_adds_datarobot_model_prefix_when_missing() -> None:
    llm = crewai_llm.get_datarobot_gateway_llm("azure/gpt-4")
    assert llm.model == "datarobot/azure/gpt-4"


def test_get_datarobot_gateway_llm_preserves_existing_datarobot_prefix() -> None:
    llm = crewai_llm.get_datarobot_gateway_llm("datarobot/azure/gpt-4")
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.model == "datarobot/azure/gpt-4"


def test_get_datarobot_gateway_llm_merges_parameters() -> None:
    llm = crewai_llm.get_datarobot_gateway_llm(parameters={"temperature": 0.25})
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.temperature == 0.25


def test_get_datarobot_deployment_llm_appends_chat_completions_to_api_base() -> None:
    llm = crewai_llm.get_datarobot_deployment_llm("dep-abc-123")
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.api_base == ("https://example.test/deployments/dep-abc-123/chat/completions")
    assert llm.additional_params == {
        "stream_options": {"include_usage": True},
    }


def test_get_datarobot_deployment_llm_merges_parameters() -> None:
    llm = crewai_llm.get_datarobot_deployment_llm(
        "dep-abc-123",
        parameters={"temperature": 0.4},
    )
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.temperature == 0.4


def test_get_datarobot_deployment_llm_uses_deployed_placeholder_when_default_model_unset() -> None:
    with patch.object(crewai_llm, "default_model_name", return_value=None):
        llm = crewai_llm.get_datarobot_deployment_llm("dep-abc-123")
    assert llm.model == DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM


def test_get_datarobot_gateway_llm_raises_when_model_unset() -> None:
    with patch.object(crewai_llm, "default_model_name", return_value=None):
        with pytest.raises(ValueError, match="Model name is required"):
            crewai_llm.get_datarobot_gateway_llm()


def test_get_datarobot_nim_llm_builds_nim_endpoint_and_model() -> None:
    llm = crewai_llm.get_datarobot_nim_llm("nim-1", "m", {"max_tokens": 10})
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.api_base == "https://example.test/deployments/nim-1/chat/completions"
    assert llm.model == "datarobot/m"


def test_get_datarobot_nim_llm_raises_when_model_unset() -> None:
    with patch.object(crewai_llm, "default_model_name", return_value=None):
        with pytest.raises(ValueError, match="Model name is required"):
            crewai_llm.get_datarobot_nim_llm("nim-1")


def test_get_external_llm_raises_when_model_unset() -> None:
    with patch.object(crewai_llm, "default_model_name", return_value=None):
        with pytest.raises(ValueError, match="Model name is required"):
            crewai_llm.get_external_llm()


def test_get_external_llm_returns_crewai_llm() -> None:
    llm = crewai_llm.get_external_llm()
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.api_base is None
    assert llm.api_key is None
    assert llm.is_litellm is True
    assert llm.additional_params == {
        "stream_options": {"include_usage": True},
    }
    assert llm.model == "default-model"


def test_get_external_llm_strips_datarobot_prefix() -> None:
    llm = crewai_llm.get_external_llm("datarobot/azure/gpt-4")
    assert llm.model == "azure/gpt-4"


def test_get_external_llm_preserves_model_without_prefix() -> None:
    llm = crewai_llm.get_external_llm("azure/gpt-4")
    assert llm.model == "azure/gpt-4"


def test_get_external_llm_merges_parameters() -> None:
    llm = crewai_llm.get_external_llm(parameters={"temperature": 0.7})
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.temperature == 0.7


def test_get_llm_routes_to_gateway() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.GATEWAY
    with patch.object(crewai_llm, "Config", return_value=config):
        llm = crewai_llm.get_llm()
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.api_base == "https://example.test/genai/llmgw"


def test_get_llm_routes_to_deployment() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.DEPLOYMENT
    config.llm_deployment_id = "dep-123"
    with patch.object(crewai_llm, "Config", return_value=config):
        llm = crewai_llm.get_llm()
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.api_base == "https://example.test/deployments/dep-123/chat/completions"


def test_get_llm_routes_to_nim() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.NIM
    config.nim_deployment_id = "nim-456"
    with patch.object(crewai_llm, "Config", return_value=config):
        llm = crewai_llm.get_llm()
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.api_base == "https://example.test/deployments/nim-456/chat/completions"


def test_get_llm_routes_to_external() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.EXTERNAL
    with patch.object(crewai_llm, "Config", return_value=config):
        llm = crewai_llm.get_llm()
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.model == "default-model"


def test_get_llm_raises_on_unknown_type() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = "unknown"
    with patch.object(crewai_llm, "Config", return_value=config):
        with pytest.raises(ValueError, match="Invalid LLM type"):
            crewai_llm.get_llm()


def test_get_llm_forwards_model_name_and_parameters() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.GATEWAY
    with patch.object(crewai_llm, "Config", return_value=config):
        llm = crewai_llm.get_llm(model_name="azure/gpt-4", parameters={"temperature": 0.5})
    assert llm.model == "datarobot/azure/gpt-4"
    assert llm.is_litellm is True
    assert llm.temperature == 0.5


# ---------------------------------------------------------------------------
# LitellmStopWordLLM – isolated unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def stop_word_llm() -> LitellmStopWordLLM:
    """Return a LitellmStopWordLLM instance with a single stop word configured."""
    return LitellmStopWordLLM(model="openai/gpt-4o", stop=["\nObservation:"])


def test_litellm_stop_word_llm_is_litellm_subclass() -> None:
    llm = LitellmStopWordLLM(model="openai/gpt-4o")
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True


def test_litellm_stop_word_llm_call_applies_stop_words(
    stop_word_llm: LitellmStopWordLLM,
) -> None:
    """Hallucinated content after the stop word is truncated."""
    hallucinated = (
        "Thought: I need to search.\n"
        "Action: search\n"
        "Action Input: query\n"
        "Observation: fake result\n"
        "Final Answer: hallucinated"
    )
    with patch.object(LLM, "call", return_value=hallucinated):
        result = stop_word_llm.call("test message")
    assert result == "Thought: I need to search.\nAction: search\nAction Input: query"
    assert "Observation:" not in result
    assert "Final Answer:" not in result


async def test_litellm_stop_word_llm_acall_applies_stop_words(
    stop_word_llm: LitellmStopWordLLM,
) -> None:
    """Async kickoff uses ``acall``; stop words must truncate there too."""
    hallucinated = (
        "Thought: I need to search.\n"
        "Action: search\n"
        "Action Input: query\n"
        "Observation: fake result\n"
        "Final Answer: hallucinated"
    )
    with patch.object(LLM, "acall", return_value=hallucinated):
        result = await stop_word_llm.acall("test message")
    assert result == "Thought: I need to search.\nAction: search\nAction Input: query"
    assert "Observation:" not in result
    assert "Final Answer:" not in result


def test_litellm_stop_word_llm_truncates_inline_react_after_action_input(
    stop_word_llm: LitellmStopWordLLM,
) -> None:
    """Models may hallucinate a second ``Thought`` without an ``Observation:`` label."""
    hallucinated = (
        "Thought: you should always think about what to do\n"
        "Action: generate_objectid\n"
        'Action Input: {"type":"deployment"}\n'
        "dff6ff5bc0f04cf69bf4c020cff634c0Thought: you should always think about what to do\n"
        "Action: generate_objectid\n"
    )
    with patch.object(LLM, "call", return_value=hallucinated):
        result = stop_word_llm.call("test message")
    assert result == (
        "Thought: you should always think about what to do\n"
        "Action: generate_objectid\n"
        'Action Input: {"type":"deployment"}'
    )


def test_litellm_stop_word_llm_call_no_stop_words_returns_unchanged() -> None:
    """Without stop words configured, responses pass through unchanged."""
    llm = LitellmStopWordLLM(model="openai/gpt-4o")
    response = "Some text\nObservation: data\nFinal Answer: done"
    with patch.object(LLM, "call", return_value=response):
        result = llm.call("test message")
    assert result == response


def test_litellm_stop_word_llm_call_non_string_result_passes_through(
    stop_word_llm: LitellmStopWordLLM,
) -> None:
    """Non-string return values (e.g. tool call results) are not truncated."""
    tool_call_result = [{"function": {"name": "search", "arguments": "{}"}}]
    with patch.object(LLM, "call", return_value=tool_call_result):
        result = stop_word_llm.call("test message")
    assert result == tool_call_result


def _delta(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_litellm_stop_word_llm_call_streams_and_returns_native_tool_calls(
    stop_word_llm: LitellmStopWordLLM,
) -> None:
    """A native call (tools + available_functions=None) assembles streamed tool-call
    deltas into the OpenAI-format list CrewAI's native loop executes.
    """
    tc = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(name="generate_objectid", arguments='{"type":"deployment"}'),
    )
    chunks = [_delta(content=None, tool_calls=[tc])]
    with patch("litellm.completion", return_value=iter(chunks)):
        result = stop_word_llm.call("m", tools=[{"type": "function"}], available_functions=None)
    assert result == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "generate_objectid", "arguments": '{"type":"deployment"}'},
        }
    ]


def test_litellm_stop_word_llm_call_native_without_tool_calls_returns_truncated_text(
    stop_word_llm: LitellmStopWordLLM,
) -> None:
    """A native call that streams only content returns stop-word-truncated text."""
    chunks = [_delta(content="Final Answer: hi\nObservation: leaked")]
    with patch("litellm.completion", return_value=iter(chunks)):
        result = stop_word_llm.call("m", tools=[{"type": "function"}], available_functions=None)
    assert result == "Final Answer: hi"


def test_recover_text_tool_calls_child_tag_format() -> None:
    """The `<invoke><tool_name>` child-tag form — exactly what the primary-test gateway
    leak emitted (bedrock sonnet under thinking).
    """
    text = (
        "<function_calls>\n<invoke>\n<tool_name>generate_objectid</tool_name>\n"
        "<parameters>\n<object_type>deployment</object_type>\n</parameters>\n</invoke>\n</function_calls>"
    )
    assert crewai_llm._recover_text_tool_calls(text) == [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "generate_objectid", "arguments": '{"object_type": "deployment"}'},
        }
    ]


def test_recover_text_tool_calls_anthropic_attribute_format() -> None:
    """Anthropic's canonical form: name as `<invoke>` attr, `<parameter name=>` children."""
    text = (
        '<function_calls><invoke name="get_weather">'
        '<parameter name="city">Paris</parameter><parameter name="unit">C</parameter>'
        "</invoke></function_calls>"
    )
    assert crewai_llm._recover_text_tool_calls(text) == [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris", "unit": "C"}'},
        }
    ]


def test_recover_text_tool_calls_tool_name_as_tag() -> None:
    """Model drops the wrapper and uses the tool name itself as the tag.

    The dominant primary-test leak (e.g. `<generate_objectid><object_type>x</object_type>
    </generate_objectid>`); recovered only when anchored on the request's tool names.
    """
    text = (
        "<generate_objectid> <object_type>deployment</object_type> </generate_objectid>\n"
        "<search_datarobot_agentic_docs> <query>MCP server</query> "
        "<max_results>1</max_results> </search_datarobot_agentic_docs>"
    )
    names = ["generate_objectid", "search_datarobot_agentic_docs"]
    calls = crewai_llm._recover_text_tool_calls(text, names)
    assert [c["function"]["name"] for c in calls] == names
    assert calls[0]["function"]["arguments"] == '{"object_type": "deployment"}'
    assert calls[1]["function"]["arguments"] == '{"query": "MCP server", "max_results": "1"}'


def test_recover_text_tool_calls_tool_name_as_tag_needs_known_names() -> None:
    """The tool-name-as-tag form is anchored on real tool names, so prose markup is ignored."""
    text = "<generate_objectid><object_type>deployment</object_type></generate_objectid>"
    assert crewai_llm._recover_text_tool_calls(text) == []


def test_sanitize_tool_schema_strips_invalid_placeholders() -> None:
    """Strip the `anyOf: []` / `enum: null` / `items: null` placeholders mcpadapt emits.

    Bedrock rejects them as not draft-2020-12 compliant, so they must go before sending.
    """
    tool = {
        "type": "function",
        "function": {
            "name": "jira_get_issue",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "The key.",
                        "anyOf": [],
                        "enum": None,
                        "items": None,
                    },
                },
                "required": ["issue_key"],
            },
        },
    }
    cleaned = crewai_llm._sanitize_tool_schema([tool])
    params = cleaned[0]["function"]["parameters"]
    assert params["properties"]["issue_key"] == {"type": "string", "description": "The key."}
    assert params["required"] == ["issue_key"]


def test_sanitize_tool_schema_keeps_valid_schema() -> None:
    """A populated `anyOf` (and other valid keywords) must be preserved untouched."""
    schema = {
        "type": "object",
        "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
    }
    assert crewai_llm._sanitize_tool_schema(schema) == schema


def test_recover_text_tool_calls_use_tool_format() -> None:
    """Bare `<use_tool>` wrapper with `<tool_name>` + `<parameters>` children.

    The format the gateway model leaked in primary-test run 1 (no `function_calls`/
    `use_mcp_tool` wrapper).
    """
    text = (
        "<use_tool> <tool_name>generate_objectid</tool_name> "
        "<parameters> <object_type>deployment</object_type> </parameters> </use_tool>"
    )
    calls = crewai_llm._recover_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"] == {
        "name": "generate_objectid",
        "arguments": '{"object_type": "deployment"}',
    }


def test_recover_text_tool_calls_namespaced_wrapper() -> None:
    """Namespaced `<budget:function_calls>` wrapper + attribute-form invoke, emitted twice.

    Exactly the primary-test gateway leak; the namespace-agnostic guard recovers it.
    """
    text = (
        '<budget:function_calls>\n<invoke name="generate_objectid">\n'
        '<parameter name="object_type">deployment</parameter>\n</invoke>\n</budget:function_calls>'
    ) * 2
    calls = crewai_llm._recover_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"] == {
        "name": "generate_objectid",
        "arguments": '{"object_type": "deployment"}',
    }


def test_recover_text_tool_calls_bare_invoke_in_prose_ignored() -> None:
    """Prose quoting a bare <invoke> (no <function_calls> wrapper) is not a real call."""
    text = "To call a tool, emit <invoke><tool_name>x</tool_name></invoke> in your reply."
    assert crewai_llm._recover_text_tool_calls(text) == []


def test_recover_text_tool_calls_mcp_format_and_dedup() -> None:
    """Same call repeated as Anthropic + MCP markup → one deduped tool call."""
    text = (
        "<function_calls><invoke><tool_name>generate_objectid</tool_name>"
        "<parameters><object_type>deployment</object_type></parameters></invoke></function_calls>"
        "<use_mcp_tool><server_name>testing</server_name><tool_name>generate_objectid</tool_name>"
        "<arguments><object_type>deployment</object_type></arguments></use_mcp_tool>"
    )
    calls = crewai_llm._recover_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "generate_objectid"
    assert calls[0]["function"]["arguments"] == '{"object_type": "deployment"}'


def test_recover_text_tool_calls_plain_text_returns_empty() -> None:
    assert crewai_llm._recover_text_tool_calls("A normal answer, no tools here.") == []


def test_litellm_stop_word_llm_call_recovers_text_emitted_tool_call(
    stop_word_llm: LitellmStopWordLLM,
) -> None:
    """A native call whose content leaks a text-encoded call is recovered + run."""
    leaked = (
        "<function_calls><invoke><tool_name>word_counter</tool_name>"
        "<parameters><text>hi there</text></parameters></invoke></function_calls>"
    )
    chunks = [_delta(content=leaked)]
    with patch("litellm.completion", return_value=iter(chunks)):
        result = stop_word_llm.call("m", tools=[{"type": "function"}], available_functions=None)
    assert result == [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "word_counter", "arguments": '{"text": "hi there"}'},
        }
    ]


def test_litellm_stop_word_llm_call_stop_word_absent_returns_unchanged(
    stop_word_llm: LitellmStopWordLLM,
) -> None:
    """Stop words configured but not present in the response — unchanged."""
    clean_response = "Thought: I know the answer.\nFinal Answer: 42"
    with patch.object(LLM, "call", return_value=clean_response):
        result = stop_word_llm.call("test message")
    assert result == clean_response


def test_litellm_stop_word_llm_call_multiple_stop_words_truncates_at_earliest() -> None:
    """Multiple stop words: truncation happens at the earliest occurrence."""
    llm = LitellmStopWordLLM(model="openai/gpt-4o", stop=["\nObservation:", "\nFinal Answer:"])
    response = "Action: search\nObservation: found\nFinal Answer: done"
    with patch.object(LLM, "call", return_value=response):
        result = llm.call("test message")
    assert result == "Action: search"


# ---------------------------------------------------------------------------
# _format_messages_for_provider – assistant prefill handling
# ---------------------------------------------------------------------------


class TestFormatMessagesForProvider:
    """Tests for LitellmStopWordLLM._format_messages_for_provider."""

    @pytest.fixture
    def llm(self) -> LitellmStopWordLLM:
        return LitellmStopWordLLM(model="datarobot/anthropic/claude-sonnet-4-6")

    def test_appends_user_message_when_trailing_assistant(self, llm: LitellmStopWordLLM) -> None:
        """A trailing assistant message gets followed by a 'Please continue.' user message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "I'll help you with that."},
        ]
        result = llm._format_messages_for_provider(messages)
        assert result[-1] == {"role": "user", "content": "Please continue."}
        assert result[-2] == {"role": "assistant", "content": "I'll help you with that."}

    def test_no_change_when_trailing_user(self, llm: LitellmStopWordLLM) -> None:
        """Messages ending with a user message are not modified."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
        ]
        result = llm._format_messages_for_provider(messages)
        assert result[-1] == {"role": "user", "content": "What is 2+2?"}
        assert len(result) == 1

    def test_empty_messages_no_trailing_assistant(self, llm: LitellmStopWordLLM) -> None:
        """Empty message list does not end with an assistant message."""
        result = llm._format_messages_for_provider([])
        assert not result or result[-1].get("role") != "assistant"

    def test_multi_turn_conversation_with_trailing_assistant(self, llm: LitellmStopWordLLM) -> None:
        """Multi-turn conversation ending with assistant gets user message appended."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Use the tool"},
            {"role": "assistant", "content": "I'll use the tool now."},
        ]
        result = llm._format_messages_for_provider(messages)
        assert result[-1] == {"role": "user", "content": "Please continue."}
        assert result[-2] == {"role": "assistant", "content": "I'll use the tool now."}


def test_gateway_llm_derives_function_calling_from_tool_choice() -> None:
    """Gateway models report tool-calling support even when litellm omits
    ``supports_function_calling`` but sets ``supports_tool_choice`` (e.g. the
    vertex llama-3.1 maas entry), since tool_choice implies function calling.
    """
    llm = crewai_llm.get_datarobot_gateway_llm("vertex_ai/meta/llama-3.1-70b-instruct-maas")
    assert llm.model == "datarobot/vertex_ai/meta/llama-3.1-70b-instruct-maas"
    assert llm.supports_function_calling() is True


def test_external_llm_defers_function_calling_to_litellm() -> None:
    """Non-DataRobot (external) models defer to litellm's verdict."""
    llm = crewai_llm.get_external_llm("gpt-4o")
    assert llm.model == "gpt-4o"
    assert llm.supports_function_calling() is True
