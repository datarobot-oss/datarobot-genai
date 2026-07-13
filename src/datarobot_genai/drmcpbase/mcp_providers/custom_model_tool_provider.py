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
"""Request-time MCP tool provider for ``tool=tool``-tagged deployments.

This is the zero-restart counterpart to the startup batch registration in
``drmcp.core.dynamic_tools.deployment.register``: instead of materialising
tools once at boot, :class:`CustomModelToolProvider` re-discovers the caller's
tool-tagged deployments on every ``tools/list`` — tag a deployment (see the
``datarobot-register-mcp-tool`` agent skill) and it appears on the next
listing, untag it and it disappears. Mirrors :class:`UserMCPProvider`, which
does the same for whole user MCP servers (``targetType == "MCP"``); this
provider covers individual custom-model deployments.

Tools are assembled from the same pure seams the startup path uses
(``build_mcp_tool_metadata`` → ``assemble_deployment_tool_config`` →
``_external_tool_callable_factory``), so both paths produce byte-identical
tool definitions. Because the caller's bearer token is baked into each tool's
outbound auth headers, built tools are cached per (deployment, token) — one
user's tools are never served to another.

Hot-path cost: static tools are unaffected (providers are consulted only on a
static miss). tools/list and deployment-tool calls resolve against per-user
TTL caches — the deployment listing (60s) and the built tools (10 min) — so
steady-state adds no DR API round trip. A newly tagged deployment appears
when the listing entry expires or immediately after ``invalidate_for_user``
(exposed as the ``refresh_deployment_tools`` MCP tool in drmcp).
"""

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any

from aiohttp import ClientResponseError
from cachetools import TTLCache
from fastmcp.server.providers import Provider
from fastmcp.tools.base import Tool
from mcp.types import ToolAnnotations

from datarobot_genai.drmcpbase.auth.enums import DataRobotBearerHeaderEnum
from datarobot_genai.drmcpbase.auth.exceptions import InvalidBearerTokenError
from datarobot_genai.drmcpbase.auth.exceptions import (
    NoDataRobotBearerTokenFoundInRequestContextError,
)
from datarobot_genai.drmcpbase.auth.exceptions import NoHeadersFoundInRequestContextError
from datarobot_genai.drmcpbase.datarobot_services.client import DataRobotClientWithAsyncAPI
from datarobot_genai.drmcpbase.datarobot_services.client import TimeMeasurement
from datarobot_genai.drmcpbase.dynamic_tools.deployment.config import (
    assemble_deployment_tool_config,
)
from datarobot_genai.drmcpbase.dynamic_tools.deployment.config import build_deployment_auth_headers
from datarobot_genai.drmcpbase.dynamic_tools.deployment.config import get_deployment_base_url
from datarobot_genai.drmcpbase.dynamic_tools.deployment.metadata import (
    _is_datarobot_structured_prediction,
)
from datarobot_genai.drmcpbase.dynamic_tools.deployment.metadata import build_mcp_tool_metadata
from datarobot_genai.drmcpbase.dynamic_tools.enums import DataRobotMCPToolCategory
from datarobot_genai.drmcpbase.dynamic_tools.external_tool import _external_tool_callable_factory
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import is_category_disabled_for_request
from datarobot_genai.drmcpbase.feature_flags import check_mcp_tools_gallery_support

logger = logging.getLogger(__name__)

TOOL_CACHE_TTL_IN_SECOND = 10 * TimeMeasurement.MINUTE.to_numeric_value_in_second()
MAX_DEPLOYMENT_TOOLS_TO_CACHE = 256
# The per-user deployment-id listing is the only per-request DR API call on
# the hot path; cache it briefly so repeated tools/list and tool calls don't
# each pay a listing round trip. Newly tagged deployments appear when the
# entry expires or on an explicit refresh (see invalidate_for_user).
LISTING_CACHE_TTL_IN_SECOND = 60
MAX_LISTINGS_TO_CACHE = 512

# Exceptions that mean "this request has no usable per-user credentials" — the
# provider contributes no tools rather than failing the whole tools/list.
_TOKEN_RESOLUTION_ERRORS = (
    NoHeadersFoundInRequestContextError,
    NoDataRobotBearerTokenFoundInRequestContextError,
    RuntimeError,
    ClientResponseError,
    InvalidBearerTokenError,
)


def _token_digest(datarobot_token: str) -> str:
    """Short digest identifying a user's token without retaining it."""
    return hashlib.sha256(datarobot_token.encode("utf-8")).hexdigest()[:16]


def _cache_key(deployment_id: str, datarobot_token: str) -> tuple[str, str]:
    """Cache key isolating built tools per user without retaining the raw token."""
    return (deployment_id, _token_digest(datarobot_token))


class CustomModelToolProvider(Provider):
    """Serve tool-tagged custom-model deployments as MCP tools at request time.

    Listing flow per ``tools/list`` request: per-request category gate →
    tools-gallery entitlement gate → list the caller's ``tool=tool``-tagged
    deployments → build (or reuse from the per-user TTL cache) one Tool per
    deployment. Tool calls resolve through :meth:`Provider._get_tool`, which
    re-enters ``_list_tools`` under the calling request's credentials, so the
    outbound auth always belongs to the caller.
    """

    def __init__(
        self,
        datarobot_api_endpoint: str,
        allow_empty_schema: bool = False,
        listing_cache_ttl_s: float = LISTING_CACHE_TTL_IN_SECOND,
    ) -> None:
        super().__init__()
        self.datarobot_api_client: DataRobotClientWithAsyncAPI | None = None
        self.datarobot_api_endpoint = datarobot_api_endpoint
        self.allow_empty_schema = allow_empty_schema
        self._tool_cache: TTLCache = TTLCache(
            maxsize=MAX_DEPLOYMENT_TOOLS_TO_CACHE, ttl=TOOL_CACHE_TTL_IN_SECOND
        )
        # user digest -> list of tool-tagged deployment ids
        self._ids_cache: TTLCache = TTLCache(maxsize=MAX_LISTINGS_TO_CACHE, ttl=listing_cache_ttl_s)

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        async with DataRobotClientWithAsyncAPI(self.datarobot_api_endpoint) as client:
            self.datarobot_api_client = client
            yield

    def is_datarobot_api_client_initialized(self) -> bool:
        return self.datarobot_api_client is not None

    async def get_tool_deployment_ids(self, datarobot_token: str) -> Sequence[str]:
        if not self.is_datarobot_api_client_initialized():
            logger.warning(
                "Failed to list tool-tagged deployments. "
                "Because it executed before the MCP provider entered lifespan."
            )
            return []
        digest = _token_digest(datarobot_token)
        ids = self._ids_cache.get(digest)
        if ids is None:
            ids = await self.datarobot_api_client._list_mcp_tool_custom_model_deployment_ids(  # type: ignore[union-attr]
                datarobot_token
            )
            self._ids_cache[digest] = ids
        return ids

    def invalidate_for_user(self, datarobot_token: str) -> int:
        """Drop the caller's cached deployment listing and built tools.

        The next listing (or tool call) re-discovers tool-tagged deployments
        immediately instead of waiting out the listing TTL. Returns the number
        of cache entries dropped. Only the calling user's entries are touched.
        """
        digest = _token_digest(datarobot_token)
        dropped = 0
        if self._ids_cache.pop(digest, None) is not None:
            dropped += 1
        stale_tool_keys = [key for key in self._tool_cache if key[1] == digest]
        for key in stale_tool_keys:
            self._tool_cache.pop(key, None)
            dropped += 1
        return dropped

    async def _build_tool(self, deployment_id: str, datarobot_token: str) -> Tool:
        """Build a Tool for one deployment with the caller's credentials.

        Uses the same pure assembly seams as the startup batch registration
        (``register_tool_of_datarobot_deployment``) so both paths produce the
        same tool for a given deployment.
        """
        client = self.datarobot_api_client
        assert client is not None  # guarded by get_tool_deployment_ids
        deployment = await client._get_datarobot_deployment(deployment_id, datarobot_token)

        if _is_datarobot_structured_prediction(deployment):
            metadata = build_mcp_tool_metadata(deployment, None, False)
        else:
            info_payload, supports_chat_api = await asyncio.gather(
                client._get_deployment_directaccess_info(deployment_id, datarobot_token),
                client._get_deployment_supports_chat_api(deployment_id, datarobot_token),
            )
            metadata = build_mcp_tool_metadata(deployment, info_payload, supports_chat_api)

        config = assemble_deployment_tool_config(
            deployment=deployment,
            metadata=metadata,
            base_url=get_deployment_base_url(deployment, self.datarobot_api_endpoint),
            auth_headers=build_deployment_auth_headers(deployment, datarobot_token),
        )
        annotations = ToolAnnotations()  # type: ignore[call-arg]
        annotations.deployment_id = deployment_id  # type: ignore[attr-defined]
        return Tool.from_function(
            fn=_external_tool_callable_factory(config, allow_empty=self.allow_empty_schema),
            name=config.name,
            title=config.title,
            description=config.description,
            annotations=annotations,
            tags=config.tags,
            meta={"tool_category": DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT.name},
        )

    async def _get_or_build_tool(self, deployment_id: str, datarobot_token: str) -> Tool:
        key = _cache_key(deployment_id, datarobot_token)
        tool = self._tool_cache.get(key)
        if tool is None:
            tool = await self._build_tool(deployment_id, datarobot_token)
            self._tool_cache[key] = tool
        return tool

    async def _list_tools(self) -> Sequence[Tool]:
        tools: list[Tool] = []

        if is_category_disabled_for_request(DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT.name):
            # Per-request gate: skip the entitlement check, the deployment
            # listing and the per-deployment fan-out — not just the final listing.
            logger.debug("Deployment tools are disabled for this request; skipping fan-out.")
            return tools

        if not await check_mcp_tools_gallery_support(self.datarobot_api_client):
            return tools

        try:
            datarobot_token = (
                DataRobotBearerHeaderEnum.X_DATAROBOT_AUTHORIZATION.get_from_mcp_request()
            )
            deployment_ids = await self.get_tool_deployment_ids(datarobot_token)
        except _TOKEN_RESOLUTION_ERRORS as ex:
            logger.warning("Failed to list tool-tagged deployments: %s.", ex)
            return tools

        results = await asyncio.gather(
            *[
                self._get_or_build_tool(deployment_id, datarobot_token)
                for deployment_id in deployment_ids
            ],
            return_exceptions=True,
        )
        for deployment_id, result in zip(deployment_ids, results):
            if isinstance(result, BaseException):
                # One misconfigured deployment (e.g. missing inputSchema) must
                # not hide the caller's other tools.
                logger.warning("Failed to build tool for deployment %s: %s", deployment_id, result)
            else:
                tools.append(result)
        return tools

    async def _get_tool(self, name: str, version: Any = None) -> Tool | None:
        """Resolve a tool call without a DR API round trip when possible.

        Tool calls land here (static tools take precedence, so only
        deployment tools reach this provider). If the calling user already has
        a built tool of this name in the cache, serve it directly; otherwise
        fall back to the default listing-based resolution (which itself hits
        the cached deployment listing). A cached tool only exists because a
        prior fully-gated listing built it for this same user.
        """
        if version is None and not is_category_disabled_for_request(
            DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT.name
        ):
            try:
                datarobot_token = (
                    DataRobotBearerHeaderEnum.X_DATAROBOT_AUTHORIZATION.get_from_mcp_request()
                )
            except _TOKEN_RESOLUTION_ERRORS:
                return None
            digest = _token_digest(datarobot_token)
            for key in list(self._tool_cache):
                if key[1] != digest:
                    continue
                tool = self._tool_cache.get(key)
                if tool is not None and tool.name == name:
                    return tool
        return await super()._get_tool(name, version)
