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

import logging
from unittest.mock import patch

import pytest

from datarobot_genai.core import config as config_mod
from datarobot_genai.core.config import DEFAULT_MAX_HISTORY_MESSAGES
from datarobot_genai.core.config import Config
from datarobot_genai.core.config import LLMConfig
from datarobot_genai.core.config import LLMType
from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.config import default_datarobot_llm_gateway_url
from datarobot_genai.core.config import default_deployment_url
from datarobot_genai.core.config import default_llm_deployment_id
from datarobot_genai.core.config import default_model_name
from datarobot_genai.core.config import default_nim_deployment_id
from datarobot_genai.core.config import default_response_model
from datarobot_genai.core.config import default_use_datarobot_llm_gateway
from datarobot_genai.core.config import deployment_url
from datarobot_genai.core.config import get_max_history_messages_default
from datarobot_genai.core.config import llm_gateway_url
from datarobot_genai.core.config import register_config_provider
from datarobot_genai.core.config import resolve_config
from datarobot_genai.core.config import resolve_llm_config


def _make_config(**overrides: object) -> Config:
    """Build a Config with explicit defaults, immune to .env files."""
    defaults = {
        "datarobot_endpoint": "https://app.datarobot.com/api/v2",
        "datarobot_api_token": None,
        "llm_deployment_id": None,
        "llm_nim_deployment_id": None,
        "llm_use_datarobot_llm_gateway": True,
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


# --- LLMConfig.get_llm_type (routing lives on the per-LLM value object) ---


def test_get_llm_type_returns_gateway_when_use_gateway_true() -> None:
    cfg = LLMConfig(llm_use_datarobot_llm_gateway=True)
    assert cfg.get_llm_type() == LLMType.GATEWAY


def test_get_llm_type_returns_deployment_when_gateway_false_and_deployment_id_set() -> None:
    cfg = LLMConfig(llm_use_datarobot_llm_gateway=False, llm_deployment_id="dep-123")
    assert cfg.get_llm_type() == LLMType.DEPLOYMENT


def test_get_llm_type_returns_nim_when_gateway_false_and_nim_id_set() -> None:
    cfg = LLMConfig(llm_use_datarobot_llm_gateway=False, llm_nim_deployment_id="nim-456")
    assert cfg.get_llm_type() == LLMType.NIM


def test_get_llm_type_returns_external_when_nothing_else_set() -> None:
    cfg = LLMConfig(llm_use_datarobot_llm_gateway=False)
    assert cfg.get_llm_type() == LLMType.EXTERNAL


def test_get_llm_type_deployment_takes_priority_over_nim() -> None:
    cfg = LLMConfig(
        llm_use_datarobot_llm_gateway=False,
        llm_deployment_id="dep-123",
        llm_nim_deployment_id="nim-456",
    )
    assert cfg.get_llm_type() == LLMType.DEPLOYMENT


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


def test_default_model_name_returns_none_when_unset() -> None:
    cfg = _make_config(llm_default_model=None)
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_model_name() is None


# --- default_response_model (configured model reported in chat/completions responses) ---


def test_default_response_model_prefixes_configured_value() -> None:
    cfg = _make_config(llm_default_model="azure/gpt-4o")
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_response_model() == "datarobot/azure/gpt-4o"


def test_default_response_model_keeps_existing_datarobot_prefix() -> None:
    cfg = _make_config(llm_default_model="datarobot/anthropic/claude-sonnet-4-20250514")
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_response_model() == "datarobot/anthropic/claude-sonnet-4-20250514"


def test_default_response_model_falls_back_to_deployed_llm_when_unset() -> None:
    # Never returns None — must terminate in the same literal the LLM client uses,
    # otherwise response.model could regress to "unknown-model".
    cfg = _make_config(llm_default_model=None)
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_response_model() == "datarobot/datarobot-deployed-llm"


# --- default_use_datarobot_llm_gateway ---


def test_default_use_datarobot_llm_gateway_true_by_default() -> None:
    cfg = _make_config(llm_use_datarobot_llm_gateway=True)
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_use_datarobot_llm_gateway() is True


def test_default_use_datarobot_llm_gateway_respects_config() -> None:
    cfg = _make_config(llm_use_datarobot_llm_gateway=False)
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
    cfg = _make_config(llm_nim_deployment_id="nim-999")
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_nim_deployment_id() == "nim-999"


def test_default_nim_deployment_id_returns_none_when_unset() -> None:
    cfg = _make_config(llm_nim_deployment_id=None)
    with patch.object(config_mod, "Config", return_value=cfg):
        assert default_nim_deployment_id() is None


# --- App config injection seam ---------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config_provider() -> object:
    """Ensure the injection provider never leaks between tests."""
    register_config_provider(None)
    yield
    register_config_provider(None)


def test_resolve_config_falls_back_to_env_config_when_no_provider() -> None:
    sentinel = _make_config(llm_default_model="from-env-config")
    with patch.object(config_mod, "Config", return_value=sentinel):
        # No provider registered -> genai's own Config() is used.
        assert resolve_config() is sentinel


def test_resolve_config_falls_back_when_provider_returns_none() -> None:
    env_cfg = _make_config(llm_default_model="from-env-config")
    register_config_provider(lambda: None)
    with patch.object(config_mod, "Config", return_value=env_cfg):
        # A provider that yields nothing -> genai's own Config() is used.
        assert resolve_config() is env_cfg


def test_resolve_config_uses_injected_provider_when_registered() -> None:
    injected = _make_config(llm_default_model="from-app-config")
    register_config_provider(lambda: injected)
    # Even if genai builds its own Config, the injected provider takes precedence.
    with patch.object(config_mod, "Config", return_value=_make_config()):
        assert resolve_config() is injected


def test_injected_config_overrides_env_for_user_intent_fields() -> None:
    """Verify the hammer case.

    A user hardcodes values in the app config and sets NO env var. genai must
    read the app's values, not its own env-only defaults.
    """
    # App config: gateway off, a deployment target, and a specific model, all set
    # as plain values, exactly as a user would hardcode them in config.py.
    app_config = LLMConfig(
        llm_use_datarobot_llm_gateway=False,
        llm_deployment_id="dep-from-app",
        llm_default_model="anthropic/claude-sonnet-4-20250514",
    )
    register_config_provider(lambda: app_config)

    # genai's own env-only Config would say the opposite (gateway on, no model).
    env_only = _make_config(llm_use_datarobot_llm_gateway=True, llm_default_model=None)
    with patch.object(config_mod, "Config", return_value=env_only):
        assert default_use_datarobot_llm_gateway() is False
        assert default_llm_deployment_id() == "dep-from-app"
        assert default_model_name() == "anthropic/claude-sonnet-4-20250514"
        assert resolve_config().get_llm_type() == LLMType.DEPLOYMENT


def test_provider_is_called_each_resolve_for_dynamic_values() -> None:
    """Provider is a factory: re-read picks up changed values (dynamic env vars)."""
    calls = {"n": 0}

    def _provider() -> LLMConfig:
        calls["n"] += 1
        return _make_config(llm_default_model=f"model-{calls['n']}")

    register_config_provider(_provider)
    with patch.object(config_mod, "Config", return_value=_make_config()):
        assert default_model_name() == "model-1"
        assert default_model_name() == "model-2"


def test_injected_config_drives_endpoint_helpers() -> None:
    """The endpoint (a true global) is authoritative from the app config too.

    A provider supplies a custom endpoint and deployment id; genai's own env-only
    Config would say the defaults. The URL builders must use the injected values.
    """
    app_config = LLMConfig(
        datarobot_endpoint="https://custom.datarobot.example/api/v2",
        llm_deployment_id="dep-injected",
    )
    register_config_provider(lambda: app_config)
    env_only = _make_config(
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        llm_deployment_id="dep-env",
    )
    with patch.object(config_mod, "Config", return_value=env_only):
        assert default_deployment_url() == (
            "https://custom.datarobot.example/api/v2"
            "/deployments/dep-injected/chat/completions"
        )
        assert default_datarobot_llm_gateway_url() == "https://custom.datarobot.example"


# --- Deprecated LLM param bridge (REMOVE WITH THE BRIDGE IN A FUTURE RELEASE) --
#
# resolve_llm_config falls back to the pre-rename bare runtime-parameter names
# (nim_deployment_id / use_datarobot_llm_gateway) when the namespaced field was
# not explicitly provided, warning loudly. These tests pin the four cases per
# param: new-only (silent), old-only (fallback + warning), both (new wins,
# silent), and neither (default, silent).

_NIM_BANNER = "DEPRECATED LLM CONFIG PARAMETER IN USE"


def _config_fields_set(fields_set: set[str], **values: object) -> Config:
    """Build a Config while controlling which fields count as explicitly set.

    model_fields_set is the signal the deprecation bridge keys off. Config()
    normally derives it from its settings sources; here we set it directly so a
    test can model "the namespaced field was (not) provided" precisely.
    """
    base = {
        "datarobot_endpoint": "https://app.datarobot.com/api/v2",
        "datarobot_api_token": None,
        "llm_deployment_id": None,
        "llm_nim_deployment_id": None,
        "llm_use_datarobot_llm_gateway": True,
        "llm_default_model": None,
    }
    base.update(values)
    return Config.model_construct(_fields_set=set(fields_set), **base)


@pytest.fixture
def clear_legacy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear both the runtime-param and bare forms of the deprecated names."""
    for name in ("NIM_DEPLOYMENT_ID", "USE_DATAROBOT_LLM_GATEWAY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(f"MLOPS_RUNTIME_PARAM_{name}", raising=False)


# nim_deployment_id


def test_bridge_nim_new_only_is_used_silently(
    clear_legacy_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config_fields_set({"llm_nim_deployment_id"}, llm_nim_deployment_id="new-nim")
    with patch.object(config_mod, "Config", return_value=cfg):
        with caplog.at_level(logging.WARNING):
            result = resolve_llm_config()
    assert result.llm_nim_deployment_id == "new-nim"
    assert _NIM_BANNER not in caplog.text


def test_bridge_nim_falls_back_to_old_name_with_warning(
    clear_legacy_env: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("MLOPS_RUNTIME_PARAM_NIM_DEPLOYMENT_ID", "legacy-nim")
    cfg = _config_fields_set(set())  # new name not explicitly provided
    with patch.object(config_mod, "Config", return_value=cfg):
        with caplog.at_level(logging.WARNING):
            result = resolve_llm_config()
    assert result.llm_nim_deployment_id == "legacy-nim"
    assert _NIM_BANNER in caplog.text
    assert "NIM_DEPLOYMENT_ID" in caplog.text
    assert "LLM_NIM_DEPLOYMENT_ID" in caplog.text


def test_bridge_nim_new_wins_over_old_silently(
    clear_legacy_env: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("MLOPS_RUNTIME_PARAM_NIM_DEPLOYMENT_ID", "legacy-nim")
    cfg = _config_fields_set({"llm_nim_deployment_id"}, llm_nim_deployment_id="new-nim")
    with patch.object(config_mod, "Config", return_value=cfg):
        with caplog.at_level(logging.WARNING):
            result = resolve_llm_config()
    assert result.llm_nim_deployment_id == "new-nim"
    assert _NIM_BANNER not in caplog.text


def test_bridge_nim_neither_set_defaults_to_none_silently(
    clear_legacy_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config_fields_set(set())
    with patch.object(config_mod, "Config", return_value=cfg):
        with caplog.at_level(logging.WARNING):
            result = resolve_llm_config()
    assert result.llm_nim_deployment_id is None
    assert _NIM_BANNER not in caplog.text


# use_datarobot_llm_gateway (bool; default True, so model_fields_set is the signal)


def test_bridge_gateway_new_only_is_used_silently(
    clear_legacy_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config_fields_set(
        {"llm_use_datarobot_llm_gateway"}, llm_use_datarobot_llm_gateway=False
    )
    with patch.object(config_mod, "Config", return_value=cfg):
        with caplog.at_level(logging.WARNING):
            result = resolve_llm_config()
    assert result.llm_use_datarobot_llm_gateway is False
    assert _NIM_BANNER not in caplog.text


def test_bridge_gateway_falls_back_to_old_name_with_warning(
    clear_legacy_env: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Old value "0" must override the truthy default AND be coerced to a real bool.
    monkeypatch.setenv("MLOPS_RUNTIME_PARAM_USE_DATAROBOT_LLM_GATEWAY", "0")
    cfg = _config_fields_set(set())  # new name not explicitly provided (default True)
    with patch.object(config_mod, "Config", return_value=cfg):
        with caplog.at_level(logging.WARNING):
            result = resolve_llm_config()
    assert result.llm_use_datarobot_llm_gateway is False
    assert _NIM_BANNER in caplog.text
    assert "USE_DATAROBOT_LLM_GATEWAY" in caplog.text


def test_bridge_gateway_new_wins_over_old_silently(
    clear_legacy_env: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("MLOPS_RUNTIME_PARAM_USE_DATAROBOT_LLM_GATEWAY", "1")
    cfg = _config_fields_set(
        {"llm_use_datarobot_llm_gateway"}, llm_use_datarobot_llm_gateway=False
    )
    with patch.object(config_mod, "Config", return_value=cfg):
        with caplog.at_level(logging.WARNING):
            result = resolve_llm_config()
    assert result.llm_use_datarobot_llm_gateway is False
    assert _NIM_BANNER not in caplog.text


def test_bridge_gateway_neither_set_defaults_to_true_silently(
    clear_legacy_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config_fields_set(set())
    with patch.object(config_mod, "Config", return_value=cfg):
        with caplog.at_level(logging.WARNING):
            result = resolve_llm_config()
    assert result.llm_use_datarobot_llm_gateway is True
    assert _NIM_BANNER not in caplog.text


def test_bridge_uses_instance_namespace_but_bare_legacy_name(
    clear_legacy_env: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-default instance name still falls back to the bare legacy param.

    The namespaced field is ``{instance}_...`` but the deprecated param was always
    bare, so the fallback and the warning's target name reflect the instance.
    """
    monkeypatch.setenv("MLOPS_RUNTIME_PARAM_NIM_DEPLOYMENT_ID", "legacy-nim")
    # The provider takes precedence over Config(); its object has an empty
    # model_fields_set, so the namespaced field reads as "not provided".
    register_config_provider(LLMConfig, default_llm_name="myagent")
    with caplog.at_level(logging.WARNING):
        result = resolve_llm_config()
    assert result.llm_nim_deployment_id == "legacy-nim"
    assert "MYAGENT_NIM_DEPLOYMENT_ID" in caplog.text
