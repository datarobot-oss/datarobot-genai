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

"""Okta XAA token for the MCP connection that ``MCPConfig`` builds (``MCP_USE_XAA``)."""

from __future__ import annotations

import asyncio
import functools
import threading
from typing import TYPE_CHECKING
from typing import Any

import jwt
from cachetools import TLRUCache
from cachetools import TTLCache

from datarobot_genai.dragent.cross_app_access_config import TokenEndpointAuthMethod
from datarobot_genai.dragent.inbound_token import OAUTH_ACCESS_TOKEN_HEADER
from datarobot_genai.dragent.inbound_token import find_idp_token
from datarobot_genai.dragent.plugins.datarobot_user_mcp_xaa_client import (
    build_mcp_auth_server_metadata_url,
)
from datarobot_genai.dragent.plugins.datarobot_user_mcp_xaa_client import fetch_xaa_params
from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessAuthProviderConfig,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import XAATokenExchange
from datarobot_genai.dragent.plugins.okta_a2a_auth import XAATokenExchangeCacheManager
from datarobot_genai.dragent.plugins.okta_a2a_auth import _CrossAppFlowParams
from datarobot_genai.dragent.plugins.okta_a2a_auth import get_token_exchange

if TYPE_CHECKING:
    from datarobot_genai.core.mcp import MCPConfig

#: Seconds the parameters an MCP server publishes are reused before asking it again.
DISCOVERY_TTL_SECONDS = 300
#: Discovery runs inside a request, so its retries share this one deadline.
DISCOVERY_TIMEOUT_SECONDS = 10
#: The token is sent for a whole agent run. A cached one is dropped this long before it
#: expires, or at half its life if that is sooner.
TOKEN_REFRESH_MARGIN_SECONDS = 300

# Inline runs use one event loop per thread, so threads share both caches.
_LOCK = threading.RLock()
_DISCOVERED_PARAMS: TTLCache[str, _CrossAppFlowParams] = TTLCache(
    maxsize=32, ttl=DISCOVERY_TTL_SECONDS
)


class _TokenCache(TLRUCache):
    """Token cache that locks the get and set ``XAATokenExchange`` uses."""

    def __getitem__(self, key: Any) -> Any:
        with _LOCK:
            return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        with _LOCK:
            super().__setitem__(key, value)


def clear_caches() -> None:
    """Forget the discovered parameters and the token exchange (used by tests)."""
    _DISCOVERED_PARAMS.clear()
    _token_exchange.cache_clear()


def _reuse_seconds(ttl: float) -> float:
    """Return how long a token with ``ttl`` seconds left is served from the cache."""
    return ttl - min(TOKEN_REFRESH_MARGIN_SECONDS, ttl / 2)


@functools.lru_cache(maxsize=1)
def _token_exchange() -> XAATokenExchange:
    """Return the one exchange of this process, so requests share its token cache."""
    config = OAuth2CrossApplicationAccessAuthProviderConfig()
    if not (config.principal_id and config.private_jwk):
        raise RuntimeError(
            "MCP_USE_XAA needs the agent's Okta identity. "
            "Set IDP_AGENT_ID and IDP_AGENT_PRIVATE_KEY_JWK."
        )
    cache_manager = XAATokenExchangeCacheManager(config)

    def expires_at(_key: str, token: str, now: float) -> float:
        return now + _reuse_seconds(cache_manager.get_ttl_or_fallback(token))

    return get_token_exchange(config, _TokenCache(config.cache_max_size, ttu=expires_at))


def _xaa_params_from_settings(mcp_config: MCPConfig) -> _CrossAppFlowParams | None:
    """Return the XAA parameters from the ``MCP_XAA_*`` settings, or None when none is set."""
    trusted_issuer = mcp_config.mcp_xaa_trusted_issuer
    exchange_audience = mcp_config.mcp_xaa_exchange_audience
    token_url = mcp_config.mcp_xaa_token_url
    token_audience = mcp_config.mcp_xaa_token_audience
    scopes = [scope.strip() for scope in (mcp_config.mcp_xaa_scopes or "").split(",")]
    scopes = [scope for scope in scopes if scope]

    # The same four the MCP server needs before it publishes its XAA parameters.
    required = {
        "MCP_XAA_TRUSTED_ISSUER": trusted_issuer,
        "MCP_XAA_EXCHANGE_AUDIENCE": exchange_audience,
        "MCP_XAA_TOKEN_URL": token_url,
        "MCP_XAA_SCOPES": scopes,
    }
    if not any([*required.values(), token_audience]):
        return None
    if missing := [name for name, value in required.items() if not value]:
        raise RuntimeError(f"Incomplete MCP_XAA_* settings. Missing: {', '.join(missing)}.")
    assert trusted_issuer and exchange_audience and token_url  # no name is missing
    return _CrossAppFlowParams(
        trusted_issuer=trusted_issuer,
        exchange_audience=exchange_audience,
        token_url=token_url,
        target_audience=token_audience or None,
        id_jag_scopes=scopes,
        token_endpoint_auth_method=TokenEndpointAuthMethod.PRIVATE_KEY_JWT.value,
    )


async def _xaa_params(mcp_config: MCPConfig, mcp_server_url: str) -> _CrossAppFlowParams:
    """Return the XAA parameters: the ``MCP_XAA_*`` settings, else what the server publishes."""
    if params := _xaa_params_from_settings(mcp_config):
        return params

    metadata_url = build_mcp_auth_server_metadata_url(mcp_server_url)
    with _LOCK:
        params = _DISCOVERED_PARAMS.get(metadata_url)
    if params is None:
        try:
            async with asyncio.timeout(DISCOVERY_TIMEOUT_SECONDS):
                params = await fetch_xaa_params(metadata_url)
        except Exception as exc:
            raise RuntimeError(
                f"MCP_USE_XAA could not read the MCP server's XAA parameters from {metadata_url} "
                f"({type(exc).__name__}: {exc}). Set the MCP_XAA_* settings on the agent to "
                "skip this lookup."
            ) from exc
        with _LOCK:
            _DISCOVERED_PARAMS[metadata_url] = params
    return params


async def exchange_caller_token(mcp_config: MCPConfig, mcp_server_url: str) -> str:
    """Return the Okta token for the MCP server, exchanged from the caller's token."""
    headers = {name.lower(): value for name, value in (mcp_config.forwarded_headers or {}).items()}
    subject_token = find_idp_token(headers)
    if subject_token is None:
        raise RuntimeError(
            f"MCP_USE_XAA is on, but the request has no caller token in "
            f"'{OAUTH_ACCESS_TOKEN_HEADER}'. The MCP connection needs the caller's Okta "
            "access token."
        )
    try:
        XAATokenExchangeCacheManager.get_jwt_json_without_validating(subject_token)
    except jwt.PyJWTError:
        raise RuntimeError(
            f"MCP_USE_XAA cannot exchange the caller token in '{OAUTH_ACCESS_TOKEN_HEADER}': "
            "it is not a JWT."
        ) from None

    token_exchange = _token_exchange()
    params = await _xaa_params(mcp_config, mcp_server_url)
    return await token_exchange.exchange_token(params, subject_token)
