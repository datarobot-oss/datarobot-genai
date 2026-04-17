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

    from datarobot_genai.core.config import LLMConfig


def build_litellm_router(
    primary: LLMConfig,
    fallbacks: list[LLMConfig],
    router_settings: dict | None = None,
) -> litellm.Router:
    """Build a ``litellm.Router`` with automatic failover.

    Args:
        primary: ``LLMConfig`` for the primary model.
        fallbacks: Ordered list of ``LLMConfig`` fallback models.
        router_settings: Extra keyword arguments forwarded to ``litellm.Router``
            (e.g. ``allowed_fails``, ``cooldown_time``, ``retry_policy``).

    Returns:
        A configured ``litellm.Router`` that tries ``primary`` first and
        cascades through ``fallback_0``, ``fallback_1``, … on failure.
    """
    import litellm

    model_list = [
        {"model_name": "primary", "litellm_params": primary.to_litellm_params()},
        *[
            {"model_name": f"fallback_{i}", "litellm_params": c.to_litellm_params()}
            for i, c in enumerate(fallbacks)
        ],
    ]
    fallbacks_cfg = [{"primary": [f"fallback_{i}" for i in range(len(fallbacks))]}]
    settings = router_settings or {}
    return litellm.Router(
        model_list=model_list,
        fallbacks=fallbacks_cfg,
        **settings,
    )
