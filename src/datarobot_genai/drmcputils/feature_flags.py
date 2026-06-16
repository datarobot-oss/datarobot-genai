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
:func:`datarobot_genai.drmcputils.clients.datarobot.request_user_dr_client`)
and caches results per ``(flag, principal)``. It is the building block for per-user,
live tool gating (e.g. hiding the sandbox tool unless an entitlement is set).

It is intentionally separate from
:mod:`datarobot_genai.drmcp.core.feature_flags`, which evaluates the
application-static MCP-container account against ``drmcpbase`` for dynamic
tool/prompt registration. The two may be consolidated later.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from datarobot.rest import RESTClientObject

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 300.0
_MAX_CACHE_ENTRIES = 1024
_eval_cache: dict[tuple[str, str], tuple[float, bool]] = {}


def is_tool_feature_enabled(
    feature_flag_name: str | None,
    *,
    evaluator: Callable[[str], bool],
) -> bool:
    """Decide whether a feature-flag-gated tool should be exposed.

    Shared gating policy for every MCP tool registry: ``drmcp``'s static-account
    registry and ``global-mcp``'s per-user registry both call this, so the
    "no flag → expose; flag set → evaluate; lookup error → fail closed" decision
    lives once in the drtools layer instead of being duplicated per server.

    ``evaluator`` performs the actual entitlement check for one flag name. Only
    the calling registry knows which principal/client to evaluate against (the
    static container account for ``drmcp``, the requesting user for global-mcp),
    so it supplies that via the closure rather than this layer reaching for a
    client it cannot choose correctly.

    Args:
        feature_flag_name: The entitlement gating the tool, or ``None`` if ungated.
        evaluator: Callable returning whether ``feature_flag_name`` is enabled.

    Returns
    -------
        ``True`` if the tool should be registered. ``False`` if the flag is
        disabled or its evaluation raised (fail-closed, so a lookup failure
        never accidentally exposes a gated tool).
    """
    if feature_flag_name is None:
        return True
    try:
        return evaluator(feature_flag_name)
    except Exception:
        logger.debug(
            "feature flag %s evaluation failed; gating tool off (fail-closed)",
            feature_flag_name,
            exc_info=True,
        )
        return False


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
