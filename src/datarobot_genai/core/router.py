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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import litellm

    from datarobot_genai.nat.datarobot_llm_providers import DataRobotLLMComponentModelConfig


def _config_to_litellm_params(config: DataRobotLLMComponentModelConfig) -> dict:
    """Convert a DataRobotLLMComponentModelConfig to a litellm_params dict.

    Returns a dict with keys ``model``, ``api_base``, and ``api_key`` suitable
    for use as an entry in ``litellm.Router``'s ``model_list``.
    """
    from datarobot_genai.core.config import Config
    from datarobot_genai.core.config import LLMType
    from datarobot_genai.core.config import deployment_url
    from datarobot_genai.core.config import llm_gateway_url

    dr_config = Config()
    _raw_key = getattr(config, "api_key", None)
    # Pydantic v2 may store api_key as SecretStr; unwrap it if so.
    if hasattr(_raw_key, "get_secret_value"):
        _raw_key = _raw_key.get_secret_value()
    api_key = _raw_key or dr_config.datarobot_api_token
    llm_type = config.get_llm_type()
    model_name = config.model_name

    if llm_type == LLMType.GATEWAY:
        if not model_name.startswith("datarobot/"):
            model_name = "datarobot/" + model_name
        return {
            "model": model_name,
            "api_base": llm_gateway_url(dr_config.datarobot_endpoint),
            "api_key": api_key,
        }
    elif llm_type == LLMType.DEPLOYMENT:
        if not model_name.startswith("datarobot/"):
            model_name = "datarobot/" + model_name
        return {
            "model": model_name,
            "api_base": deployment_url(
                config.llm_deployment_id,  # type: ignore[arg-type]
                dr_config.datarobot_endpoint,
            ),
            "api_key": api_key,
        }
    elif llm_type == LLMType.NIM:
        if not model_name.startswith("datarobot/"):
            model_name = "datarobot/" + model_name
        return {
            "model": model_name,
            "api_base": deployment_url(
                config.nim_deployment_id,  # type: ignore[arg-type]
                dr_config.datarobot_endpoint,
            ),
            "api_key": api_key,
        }
    else:  # EXTERNAL
        model_name = model_name.removeprefix("datarobot/")
        return {
            "model": model_name,
            "api_key": api_key,
        }


def build_litellm_router(
    primary_params: dict,
    fallback_params: list[dict],
    router_settings: dict | None = None,
) -> litellm.Router:
    """Build a ``litellm.Router`` with automatic failover.

    Args:
        primary_params: litellm_params dict for the primary model
            (keys: ``model``, ``api_base``, ``api_key``, …).
        fallback_params: litellm_params dicts for each fallback model, in
            priority order.
        router_settings: Extra keyword arguments forwarded verbatim to
            ``litellm.Router`` (e.g. ``allowed_fails``, ``cooldown_time``,
            ``retry_policy``).

    Returns
    -------
        A configured ``litellm.Router`` that tries ``primary`` first and
        cascades through ``fallback_0``, ``fallback_1``, … on failure.
    """
    import litellm

    model_list = [
        {"model_name": "primary", "litellm_params": primary_params},
        *[
            {"model_name": f"fallback_{i}", "litellm_params": p}
            for i, p in enumerate(fallback_params)
        ],
    ]
    fallbacks_cfg = [{"primary": [f"fallback_{i}" for i in range(len(fallback_params))]}]
    settings = router_settings or {}
    return litellm.Router(
        model_list=model_list,
        fallbacks=fallbacks_cfg,
        **settings,
    )
