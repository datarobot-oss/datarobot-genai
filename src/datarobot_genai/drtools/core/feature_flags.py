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
"""DataRobot entitlements-backed feature flag evaluation.

Lives in ``drtools`` so consumers (tools, global-mcp, any drtools-only
package) can evaluate per-user entitlements without depending on the
heavier ``drmcp``/``drmcpbase`` packages or their module-load side effects.

This evaluator is per-user: it requires a request-scoped
:class:`~datarobot.rest.RESTClientObject` (e.g. from
:func:`datarobot_genai.drtools.core.rest_client.get_rest_client`) and caches
results per ``(flag, principal)``. It is the building block for per-user,
live tool gating (e.g. hiding the sandbox tool unless an entitlement is set).

It is intentionally separate from
:mod:`datarobot_genai.drmcp.core.feature_flags`, which evaluates the
application-static MCP-container account against ``drmcpbase`` for dynamic
tool/prompt registration. The two may be consolidated later.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from datarobot.rest import RESTClientObject

_DEFAULT_TTL_SECONDS = 300.0
_MAX_CACHE_ENTRIES = 1024
_eval_cache: dict[tuple[str, str], tuple[float, bool]] = {}


def _principal_key(client: RESTClientObject) -> str:
    """Stable, non-reversible cache key derived from the API token.

    Hashing avoids holding raw bearer tokens in process memory longer than
    the request that already needed them.
    """
    token = getattr(client, "token", None) or ""
    if not token:
        return "__no_token__"
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def _prune_expired(now: float, ttl_seconds: float) -> None:
    if len(_eval_cache) < _MAX_CACHE_ENTRIES:
        return
    expired = [k for k, (ts, _) in _eval_cache.items() if (now - ts) >= ttl_seconds]
    for k in expired:
        _eval_cache.pop(k, None)
    if len(_eval_cache) >= _MAX_CACHE_ENTRIES:
        # Still over the cap — drop the oldest entries to make room.
        oldest = sorted(_eval_cache.items(), key=lambda kv: kv[1][0])
        for k, _ in oldest[: len(_eval_cache) - _MAX_CACHE_ENTRIES + 1]:
            _eval_cache.pop(k, None)


@dataclass
class FeatureFlag:
    name: str
    enabled: bool

    @classmethod
    def create(
        cls,
        feature_flag_name: str,
        *,
        client: RESTClientObject,
    ) -> FeatureFlag:
        """Evaluate a DR entitlement against the principal owning ``client``.

        ``client`` is required at this layer. If you need the historical
        "use the application-static client by default" behavior, use
        ``datarobot_genai.drmcp.core.feature_flags.FeatureFlag`` instead.
        """
        response = client.post(
            "entitlements/evaluate/",
            json={"entitlements": [{"name": feature_flag_name}]},
        )
        feature_flag_info = response.json()["entitlements"][0]
        return cls(
            name=feature_flag_info["name"],
            enabled=bool(feature_flag_info["value"]),
        )

    @classmethod
    def is_enabled(
        cls,
        feature_flag_name: str,
        *,
        client: RESTClientObject,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> bool:
        """Return the cached entitlement state, keyed by ``(flag, principal)``.

        Pass ``ttl_seconds=0`` to bypass the cache. The principal portion of
        the key is a hash of the client's token, so different users do not
        share cache entries.
        """
        cache_key = (feature_flag_name, _principal_key(client))
        now = time.monotonic()
        if ttl_seconds > 0:
            cached = _eval_cache.get(cache_key)
            if cached is not None and (now - cached[0]) < ttl_seconds:
                return cached[1]
        enabled = cls.create(feature_flag_name, client=client).enabled
        if ttl_seconds > 0:
            _prune_expired(now, ttl_seconds)
            _eval_cache[cache_key] = (now, enabled)
        return enabled

    @classmethod
    def is_mcp_tools_gallery_support_enabled(cls, *, client: RESTClientObject) -> bool:
        return cls.is_enabled("ENABLE_MCP_TOOLS_GALLERY_SUPPORT", client=client)
