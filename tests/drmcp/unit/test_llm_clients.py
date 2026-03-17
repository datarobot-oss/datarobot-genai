# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for LLM client temperature support."""

import asyncio
import os
from unittest.mock import MagicMock
from unittest.mock import patch

from datarobot_genai.drmcp.test_utils.clients.openai import OpenAILLMMCPClient
from datarobot_genai.drmcp.test_utils.mcp_utils_ete import get_dr_llm_gateway_client_config
from datarobot_genai.drmcp.test_utils.mcp_utils_ete import get_openai_llm_client_config


def _make_openai_client(extra_config: dict | None = None) -> OpenAILLMMCPClient:
    """Create an OpenAILLMMCPClient with a mocked openai.OpenAI."""
    config = {"openai_api_key": "test-key", "model": "gpt-4o"}
    if extra_config:
        config.update(extra_config)
    with patch("openai.OpenAI"):
        return OpenAILLMMCPClient(config)


class TestBaseLLMMCPClientTemperature:
    """Tests for temperature reading in BaseLLMMCPClient.__init__."""

    def test_temperature_stored_as_float(self) -> None:
        """Temperature from config dict is stored as float."""
        client = _make_openai_client({"temperature": 0.2})
        assert client.temperature == 0.2

    def test_temperature_string_converted_to_float(self) -> None:
        """Temperature provided as a string (e.g. from env var) is converted to float."""
        client = _make_openai_client({"temperature": "0.5"})
        assert client.temperature == 0.5

    def test_temperature_zero_stored(self) -> None:
        """Temperature of 0 (falsy) is stored correctly, not treated as None."""
        client = _make_openai_client({"temperature": 0})
        assert client.temperature == 0.0

    def test_temperature_zero_string_stored(self) -> None:
        """Temperature '0' as string is stored as 0.0."""
        client = _make_openai_client({"temperature": "0"})
        assert client.temperature == 0.0

    def test_no_temperature_defaults_to_none(self) -> None:
        """When temperature is not in config, client.temperature is None."""
        client = _make_openai_client()
        assert client.temperature is None

    def test_temperature_forwarded_to_api_call(self) -> None:
        """Temperature is passed to chat.completions.create when set."""
        client = _make_openai_client({"temperature": 0.2})
        mock_response = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        client.openai_client.chat.completions.create.return_value = mock_response

        asyncio.run(client._get_llm_response([{"role": "user", "content": "hi"}]))

        call_kwargs = client.openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.2

    def test_temperature_not_forwarded_when_none(self) -> None:
        """Temperature is NOT passed to chat.completions.create when not set."""
        client = _make_openai_client()
        mock_response = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        client.openai_client.chat.completions.create.return_value = mock_response

        asyncio.run(client._get_llm_response([{"role": "user", "content": "hi"}]))

        call_kwargs = client.openai_client.chat.completions.create.call_args[1]
        assert "temperature" not in call_kwargs


class TestMcpUtilsEteTemperature:
    """Tests for LLM_TEMPERATURE env var in config helpers."""

    def test_get_dr_llm_gateway_config_includes_temperature_when_env_set(self) -> None:
        """get_dr_llm_gateway_client_config includes temperature when LLM_TEMPERATURE is set."""
        env = {
            "DATAROBOT_API_TOKEN": "test-token",
            "DR_LLM_GATEWAY_MODEL": "claude-3",
            "LLM_TEMPERATURE": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            config = get_dr_llm_gateway_client_config()

        assert config.get("temperature") == "0"

    def test_get_dr_llm_gateway_config_no_temperature_key_when_env_unset(self) -> None:
        """get_dr_llm_gateway_client_config omits temperature when LLM_TEMPERATURE not set."""
        env = {
            "DATAROBOT_API_TOKEN": "test-token",
            "DR_LLM_GATEWAY_MODEL": "claude-3",
        }
        with patch.dict(os.environ, env, clear=False):
            # Ensure LLM_TEMPERATURE is absent
            os.environ.pop("LLM_TEMPERATURE", None)
            config = get_dr_llm_gateway_client_config()

        assert "temperature" not in config

    def test_get_openai_config_includes_temperature_when_env_set(self) -> None:
        """get_openai_llm_client_config includes temperature when LLM_TEMPERATURE is set."""
        env = {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "gpt-4o",
            "LLM_TEMPERATURE": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            config = get_openai_llm_client_config()

        assert config.get("temperature") == "0"
