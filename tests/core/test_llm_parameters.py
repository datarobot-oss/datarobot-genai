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

from unittest.mock import patch

import pytest

from datarobot_genai.core.llm_parameters import apply_reasoning_to_parameters
from datarobot_genai.core.llm_parameters import default_reasoning_extra_body
from datarobot_genai.core.llm_parameters import supports_parallel_tool_calls
from datarobot_genai.langgraph.llm import get_datarobot_gateway_llm


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("azure/o3", False),
        ("openai/o1", False),
        ("o1-mini", False),
        ("datarobot/azure/o4-mini", False),
        ("azure/gpt-5", True),
        ("azure/gpt-4o", True),
        ("gpt-4o-mini", True),
        ("anthropic/claude-sonnet-4-6", True),
        ("vertex_ai/gemini-2.5-pro", True),
        ("some-custom-deployment", True),
        (None, True),
    ],
)
def test_supports_parallel_tool_calls(model_name: str | None, expected: bool) -> None:
    assert supports_parallel_tool_calls(model_name) is expected


@pytest.mark.parametrize(
    ("model_name", "expected_extra_body"),
    [
        pytest.param(
            "anthropic/claude-sonnet-4-6",
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            id="anthropic-sonnet",
        ),
        pytest.param(
            "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            id="bedrock-sonnet",
        ),
        pytest.param(
            "vertex_ai/claude-opus-4-7",
            {"thinking": {"type": "adaptive"}},
            id="anthropic-opus",
        ),
        pytest.param(
            "vertex_ai/gemini-3.5-flash",
            {"thinking_config": {"thinking_budget": 1024}},
            id="gemini",
        ),
        pytest.param(
            "azure/gpt-5-4-2026-03-05",
            {"reasoning_effort": "low"},
            id="azure-gpt-5",
        ),
        pytest.param(
            "openai/o3-mini",
            {"reasoning_effort": "low"},
            id="openai-o-series",
        ),
    ],
)
def test_default_reasoning_extra_body_for_provider_models(
    model_name: str,
    expected_extra_body: dict,
) -> None:
    assert default_reasoning_extra_body(model_name) == expected_extra_body


def test_apply_reasoning_to_parameters_enables_thinking() -> None:
    params = apply_reasoning_to_parameters(
        {"temperature": 0},
        reasoning=True,
        model_name="anthropic/claude-sonnet-4-6",
    )
    assert "temperature" not in params
    assert params["extra_body"] == {
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }


def test_apply_reasoning_to_parameters_preserves_explicit_extra_body() -> None:
    params = apply_reasoning_to_parameters(
        {"temperature": 0, "extra_body": {"reasoning_effort": "low"}},
        reasoning=True,
        model_name="openai/o3-mini",
    )
    assert "temperature" not in params
    assert params["extra_body"] == {"reasoning_effort": "low"}


def test_apply_reasoning_to_parameters_noop_when_disabled() -> None:
    params = apply_reasoning_to_parameters(
        {"temperature": 0},
        reasoning=False,
        model_name="anthropic/claude-sonnet-4-6",
    )
    assert params == {"temperature": 0}


@patch("datarobot_genai.langgraph.llm._create_datarobot_chat_litellm")
def test_get_datarobot_gateway_llm_applies_reasoning(mock_create) -> None:
    mock_create.return_value = object()
    get_datarobot_gateway_llm(
        "anthropic/claude-sonnet-4-6",
        parameters={"temperature": 0},
        reasoning=True,
        streaming=False,
    )
    config = mock_create.call_args[0][0]
    assert "temperature" not in config
    assert config["extra_body"] == {
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }


@patch("datarobot_genai.langgraph.llm._create_datarobot_chat_litellm")
def test_get_datarobot_gateway_llm_skips_reasoning_when_extra_body_set(mock_create) -> None:
    mock_create.return_value = object()
    get_datarobot_gateway_llm(
        "anthropic/claude-sonnet-4-6",
        parameters={"temperature": 0, "extra_body": {"reasoning_effort": "low"}},
        reasoning=True,
        streaming=False,
    )
    config = mock_create.call_args[0][0]
    assert "temperature" not in config
    assert config["extra_body"] == {"reasoning_effort": "low"}
