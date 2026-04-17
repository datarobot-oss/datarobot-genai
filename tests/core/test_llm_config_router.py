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

"""Tests for LLMConfig.to_litellm_params() and build_litellm_router."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.core import config as config_mod
from datarobot_genai.core.config import LLMConfig
from datarobot_genai.core.config import LLMType


def _make_env_config(**overrides: object) -> config_mod.Config:
    defaults = {
        "datarobot_endpoint": "https://app.datarobot.com/api/v2",
        "datarobot_api_token": "env-token",
        "llm_deployment_id": None,
        "nim_deployment_id": None,
        "use_datarobot_llm_gateway": True,
        "llm_default_model": None,
    }
    defaults.update(overrides)
    return config_mod.Config.model_construct(**defaults)


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch Config() calls inside to_litellm_params to use a deterministic env config."""
    env = _make_env_config()
    monkeypatch.setattr(config_mod, "Config", lambda: env)


# ---------------------------------------------------------------------------
# LLMConfig.get_llm_type — same logic as Config, now inherited
# ---------------------------------------------------------------------------


def test_llm_config_get_llm_type_gateway() -> None:
    assert LLMConfig(use_datarobot_llm_gateway=True).get_llm_type() == LLMType.GATEWAY


def test_llm_config_get_llm_type_deployment() -> None:
    cfg = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
    assert cfg.get_llm_type() == LLMType.DEPLOYMENT


def test_llm_config_get_llm_type_nim() -> None:
    cfg = LLMConfig(use_datarobot_llm_gateway=False, nim_deployment_id="nim-1")
    assert cfg.get_llm_type() == LLMType.NIM


def test_llm_config_get_llm_type_external() -> None:
    cfg = LLMConfig(use_datarobot_llm_gateway=False)
    assert cfg.get_llm_type() == LLMType.EXTERNAL


# ---------------------------------------------------------------------------
# LLMConfig.to_litellm_params — all four LLMTypes
# ---------------------------------------------------------------------------


def test_to_litellm_params_gateway() -> None:
    cfg = LLMConfig(use_datarobot_llm_gateway=True)
    params = cfg.to_litellm_params()
    assert params["model"].startswith("datarobot/")
    assert params["api_base"] == "https://app.datarobot.com"
    assert params["api_key"] == "env-token"


def test_to_litellm_params_deployment() -> None:
    cfg = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-abc")
    params = cfg.to_litellm_params()
    assert params["model"].startswith("datarobot/")
    assert "dep-abc" in params["api_base"]
    assert params["api_key"] == "env-token"


def test_to_litellm_params_nim() -> None:
    cfg = LLMConfig(use_datarobot_llm_gateway=False, nim_deployment_id="nim-xyz")
    params = cfg.to_litellm_params()
    assert "nim-xyz" in params["api_base"]
    assert params["api_key"] == "env-token"


def test_to_litellm_params_external() -> None:
    cfg = LLMConfig(use_datarobot_llm_gateway=False, llm_default_model="gpt-4")
    params = cfg.to_litellm_params()
    assert params["model"] == "gpt-4"
    assert "api_base" not in params


def test_to_litellm_params_external_strips_datarobot_prefix() -> None:
    cfg = LLMConfig(use_datarobot_llm_gateway=False, llm_default_model="datarobot/azure/gpt-4")
    params = cfg.to_litellm_params()
    assert params["model"] == "azure/gpt-4"


def test_to_litellm_params_uses_explicit_api_key_over_env() -> None:
    cfg = LLMConfig(use_datarobot_llm_gateway=True, datarobot_api_token="explicit-key")
    params = cfg.to_litellm_params()
    assert params["api_key"] == "explicit-key"


def test_to_litellm_params_uses_explicit_endpoint() -> None:
    cfg = LLMConfig(
        use_datarobot_llm_gateway=True,
        datarobot_endpoint="https://custom.host/api/v2",
    )
    params = cfg.to_litellm_params()
    assert params["api_base"] == "https://custom.host"


def test_to_litellm_params_gateway_adds_datarobot_prefix_when_missing() -> None:
    cfg = LLMConfig(use_datarobot_llm_gateway=True, llm_default_model="azure/gpt-4o")
    params = cfg.to_litellm_params()
    assert params["model"] == "datarobot/azure/gpt-4o"


def test_to_litellm_params_gateway_does_not_double_prefix() -> None:
    cfg = LLMConfig(use_datarobot_llm_gateway=True, llm_default_model="datarobot/azure/gpt-4o")
    params = cfg.to_litellm_params()
    assert params["model"] == "datarobot/azure/gpt-4o"


# ---------------------------------------------------------------------------
# build_litellm_router
# ---------------------------------------------------------------------------


def test_build_litellm_router_model_list_structure() -> None:
    from datarobot_genai.core.router import build_litellm_router

    primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
    fallback = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")

    with patch("litellm.Router") as mock_router_cls:
        mock_router_cls.return_value = MagicMock()
        build_litellm_router(primary, [fallback])

    call_kwargs = mock_router_cls.call_args.kwargs
    model_list = call_kwargs["model_list"]
    assert len(model_list) == 2
    assert model_list[0]["model_name"] == "primary"
    assert model_list[1]["model_name"] == "fallback_0"
    assert "dep-1" in model_list[0]["litellm_params"]["api_base"]
    assert "dep-2" in model_list[1]["litellm_params"]["api_base"]


def test_build_litellm_router_fallbacks_chain() -> None:
    from datarobot_genai.core.router import build_litellm_router

    primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
    fb0 = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")
    fb1 = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-3")

    with patch("litellm.Router") as mock_router_cls:
        mock_router_cls.return_value = MagicMock()
        build_litellm_router(primary, [fb0, fb1])

    call_kwargs = mock_router_cls.call_args.kwargs
    assert call_kwargs["fallbacks"] == [{"primary": ["fallback_0", "fallback_1"]}]


def test_build_litellm_router_passes_settings() -> None:
    from datarobot_genai.core.router import build_litellm_router

    primary = LLMConfig(use_datarobot_llm_gateway=True)
    fb = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")

    with patch("litellm.Router") as mock_router_cls:
        mock_router_cls.return_value = MagicMock()
        build_litellm_router(
            primary, [fb], {"allowed_fails": 5, "cooldown_time": 60.0}
        )

    call_kwargs = mock_router_cls.call_args.kwargs
    assert call_kwargs["allowed_fails"] == 5
    assert call_kwargs["cooldown_time"] == 60.0


# ---------------------------------------------------------------------------
# merge_streaming_tool_calls
# ---------------------------------------------------------------------------


def test_merge_streaming_tool_calls_single_fragment() -> None:
    from types import SimpleNamespace

    from datarobot_genai.core.router import merge_streaming_tool_calls

    tc = SimpleNamespace(
        index=0,
        id="call-abc",
        function=SimpleNamespace(name="my_tool", arguments='{"x": 1}'),
    )
    result = merge_streaming_tool_calls([tc])
    assert result == [
        {"id": "call-abc", "type": "function", "function": {"name": "my_tool", "arguments": '{"x": 1}'}}
    ]


def test_merge_streaming_tool_calls_merges_argument_fragments() -> None:
    from types import SimpleNamespace

    from datarobot_genai.core.router import merge_streaming_tool_calls

    frag1 = SimpleNamespace(
        index=0, id="call-1", function=SimpleNamespace(name="tool_a", arguments='{"x":')
    )
    frag2 = SimpleNamespace(
        index=0, id=None, function=SimpleNamespace(name=None, arguments=' 1}')
    )
    result = merge_streaming_tool_calls([frag1, frag2])
    assert len(result) == 1
    assert result[0]["function"]["arguments"] == '{"x": 1}'
    assert result[0]["id"] == "call-1"


def test_merge_streaming_tool_calls_multiple_calls() -> None:
    from types import SimpleNamespace

    from datarobot_genai.core.router import merge_streaming_tool_calls

    tc0 = SimpleNamespace(index=0, id="id-0", function=SimpleNamespace(name="tool_a", arguments="{}"))
    tc1 = SimpleNamespace(index=1, id="id-1", function=SimpleNamespace(name="tool_b", arguments='{"y": 2}'))
    result = merge_streaming_tool_calls([tc0, tc1])
    names = {r["function"]["name"] for r in result}
    assert names == {"tool_a", "tool_b"}


# ---------------------------------------------------------------------------
# Failover / error-propagation scenario
# ---------------------------------------------------------------------------


def test_router_propagates_exception_when_all_models_fail() -> None:
    """When litellm.Router exhausts all fallbacks it raises; wrappers must not swallow it."""
    from datarobot_genai.core.router import build_litellm_router

    primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-bad")
    fb = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-also-bad")

    mock_router_instance = MagicMock()
    mock_router_instance.completion.side_effect = RuntimeError("all models failed")

    with patch("litellm.Router", return_value=mock_router_instance):
        router = build_litellm_router(primary, [fb])

    with pytest.raises(RuntimeError, match="all models failed"):
        router.completion("primary", messages=[{"role": "user", "content": "hi"}])
