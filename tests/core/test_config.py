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

from datarobot_genai.core import config as config_mod
from datarobot_genai.core.config import DEFAULT_MAX_HISTORY_MESSAGES
from datarobot_genai.core.config import Config
from datarobot_genai.core.config import LLMType
from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.config import default_base_url
from datarobot_genai.core.config import default_datarobot_llm_gateway_url
from datarobot_genai.core.config import default_deployment_url
from datarobot_genai.core.config import default_llm_deployment_id
from datarobot_genai.core.config import default_model_name
from datarobot_genai.core.config import default_nim_deployment_id
from datarobot_genai.core.config import default_use_datarobot_llm_gateway
from datarobot_genai.core.config import deployment_url
from datarobot_genai.core.config import get_max_history_messages_default
from datarobot_genai.core.config import llm_gateway_url


def _make_config(**overrides: object) -> Config:
    """Build a Config with explicit defaults, immune to .env files."""
    defaults = {
        "datarobot_endpoint": "https://app.datarobot.com/api/v2",
        "datarobot_api_token": None,
        "llm_deployment_id": None,
        "nim_deployment_id": None,
        "use_datarobot_llm_gateway": True,
        "llm_default_model": None,
    }
    defaults.update(overrides)
    return Config.model_construct(**defaults)


# --- get_max_history_messages_default (existing tests, preserved) ---


def test_get_max_history_messages_default_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATAROBOT_GENAI_MAX_HISTORY_MESSAGES", raising=False)
    assert get_max_history_messages_default() == DEFAULT_MAX_HISTORY_MESSAGES


def test_get_max_history_messages_default_env_zero_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAROBOT_GENAI_MAX_HISTORY_MESSAGES", "0")
    assert get_max_history_messages_default() == 0


def test_get_max_history_messages_default_env_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAROBOT_GENAI_MAX_HISTORY_MESSAGES", "7")
    assert get_max_history_messages_default() == 7


# --- Config.get_llm_type ---


def test_get_llm_type_returns_gateway_when_use_gateway_true() -> None:
    cfg = _make_config(use_datarobot_llm_gateway=True)
    assert cfg.get_llm_type() == LLMType.GATEWAY


def test_get_llm_type_returns_deployment_when_gateway_false_and_deployment_id_set() -> None:
    cfg = _make_config(use_datarobot_llm_gateway=False, llm_deployment_id="dep-123")
    assert cfg.get_llm_type() == LLMType.DEPLOYMENT


def test_get_llm_type_returns_nim_when_gateway_false_and_nim_id_set() -> None:
    cfg = _make_config(use_datarobot_llm_gateway=False, nim_deployment_id="nim-456")
    assert cfg.get_llm_type() == LLMType.NIM


def test_get_llm_type_returns_external_when_nothing_else_set() -> None:
    cfg = _make_config(use_datarobot_llm_gateway=False)
    assert cfg.get_llm_type() == LLMType.EXTERNAL


def test_get_llm_type_deployment_takes_priority_over_nim() -> None:
    cfg = _make_config(
        use_datarobot_llm_gateway=False,
        llm_deployment_id="dep-123",
        nim_deployment_id="nim-456",
    )
    assert cfg.get_llm_type() == LLMType.DEPLOYMENT


# --- default_base_url ---


def test_default_base_url_strips_trailing_slash_for_gateway() -> None:
    cfg = _make_config(
        datarobot_endpoint="https://app.datarobot.com/api/v2/",
        use_datarobot_llm_gateway=True,
    )
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_base_url() == "https://app.datarobot.com/api/v2"


def test_default_base_url_appends_deployment_path_when_not_gateway() -> None:
    cfg = _make_config(
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        use_datarobot_llm_gateway=False,
        llm_deployment_id="dep-xyz",
    )
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_base_url() == "https://app.datarobot.com/api/v2/deployments/dep-xyz"


# --- default_api_key ---


def test_default_api_key_returns_token_when_set() -> None:
    cfg = _make_config(datarobot_api_token="my-secret-token")
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_api_key() == "my-secret-token"


def test_default_api_key_returns_none_when_unset() -> None:
    cfg = _make_config(datarobot_api_token=None)
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_api_key() is None


# --- default_model_name ---


def test_default_model_name_returns_configured_value() -> None:
    cfg = _make_config(llm_default_model="azure/gpt-4o")
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_model_name() == "azure/gpt-4o"


def test_default_model_name_falls_back_to_builtin() -> None:
    cfg = _make_config(llm_default_model=None)
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_model_name() == "datarobot-deployed-llm"


# --- default_use_datarobot_llm_gateway ---


def test_default_use_datarobot_llm_gateway_true_by_default() -> None:
    cfg = _make_config(use_datarobot_llm_gateway=True)
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_use_datarobot_llm_gateway() is True


def test_default_use_datarobot_llm_gateway_respects_config() -> None:
    cfg = _make_config(use_datarobot_llm_gateway=False)
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_use_datarobot_llm_gateway() is False


# --- deployment_url (pure function) ---


def test_deployment_url_builds_correct_path() -> None:
    result = deployment_url("dep-abc", "https://app.datarobot.com/api/v2")
    assert result == "https://app.datarobot.com/api/v2/deployments/dep-abc/chat/completions"


# --- default_deployment_url ---


def test_default_deployment_url_uses_explicit_id() -> None:
    cfg = _make_config(datarobot_endpoint="https://app.datarobot.com/api/v2")
    with patch.object(config_mod, "Config", return_value=cfg):
        result = default_deployment_url("dep-explicit")
    assert result == ("https://app.datarobot.com/api/v2/deployments/dep-explicit/chat/completions")


def test_default_deployment_url_falls_back_to_config_id() -> None:
    cfg = _make_config(
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        llm_deployment_id="dep-from-config",
    )
    with patch.object(config_mod, "Config", return_value=cfg):
        result = default_deployment_url()
    assert result == (
        "https://app.datarobot.com/api/v2/deployments/dep-from-config/chat/completions"
    )


def test_default_deployment_url_raises_when_no_id() -> None:
    cfg = _make_config(llm_deployment_id=None)
    with patch.object(config_mod, "Config", return_value=cfg):
        with pytest.raises(
            ValueError, match="Neither deployment ID nor default deployment ID is set"
        ):
            default_deployment_url()


# --- llm_gateway_url (pure function) ---


def test_llm_gateway_url_strips_api_v2_suffix() -> None:
    assert llm_gateway_url("https://app.datarobot.com/api/v2") == "https://app.datarobot.com"


def test_llm_gateway_url_no_op_when_suffix_absent() -> None:
    assert llm_gateway_url("https://custom.endpoint.com") == "https://custom.endpoint.com"


# --- default_datarobot_llm_gateway_url ---


def test_default_datarobot_llm_gateway_url() -> None:
    cfg = _make_config(datarobot_endpoint="https://app.datarobot.com/api/v2")
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_datarobot_llm_gateway_url() == "https://app.datarobot.com"


# --- default_llm_deployment_id ---


def test_default_llm_deployment_id_returns_configured_value() -> None:
    cfg = _make_config(llm_deployment_id="dep-999")
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_llm_deployment_id() == "dep-999"


def test_default_llm_deployment_id_returns_none_when_unset() -> None:
    cfg = _make_config(llm_deployment_id=None)
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_llm_deployment_id() is None


# --- default_nim_deployment_id ---


def test_default_nim_deployment_id_returns_configured_value() -> None:
    cfg = _make_config(nim_deployment_id="nim-999")
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_nim_deployment_id() == "nim-999"


def test_default_nim_deployment_id_returns_none_when_unset() -> None:
    cfg = _make_config(nim_deployment_id=None)
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_nim_deployment_id() is None
