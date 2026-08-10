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

"""Read OAuth scope settings from this server's configuration.

The mechanism lives in :mod:`datarobot_genai.drmcpbase.oauth_scopes`, which
knows nothing about where settings come from. This module is the adapter that
reads them off :class:`MCPServerConfig` and the environment.

Tag requirements are **one environment variable per tag**::

    MCP_OAUTH_TAG_SCOPES_DATABASE=mcp:tools:execute,mcp:tools:database:write
    MCP_OAUTH_TAG_SCOPES_READONLY=mcp:tools:read,mcp:resources:read

rather than one packed variable holding every tag. Each tag is then independent:
readable in a diff, greppable, overridable on its own per environment, and
comma-separated like every other list setting here — no second delimiter to
learn and no long line where one misplaced separator silently reshapes the
rules. The suffix is the component's tag, matched case-insensitively with ``-``
and ``_`` treated alike, so ``MCP_OAUTH_TAG_SCOPES_READ_ONLY`` guards a
component tagged ``read-only``.

Values may also arrive as DataRobot runtime parameters, which the platform
exposes with a ``MLOPS_RUNTIME_PARAM_`` prefix and a JSON envelope around the
value, so both spellings are accepted and the envelope is unwrapped.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from datarobot_genai.drmcpbase.oauth_scopes import ScopeSettings
from datarobot_genai.drmcpbase.oauth_scopes import ScopeSource
from datarobot_genai.drmcpbase.oauth_scopes import probe_verification_keys
from datarobot_genai.drmcpbase.oauth_scopes import wire_scopes as _wire_scopes
from datarobot_genai.drmcputils.constants import RUNTIME_PARAM_ENV_VAR_NAME_PREFIX

from .config import MCPServerConfig
from .config import get_config
from .runtime_identity import resolve_self_url

logger = logging.getLogger(__name__)

#: Prefix of the per-tag scope requirement variables.
TAG_SCOPES_ENV_PREFIX = "MCP_OAUTH_TAG_SCOPES_"

#: ``type`` values a runtime-parameter envelope may declare for a plain payload,
#: matching what ``datarobot.core.config.getenv`` accepts.
_RUNTIME_PARAM_TYPES = ("string", "boolean", "numeric", "deployment")


def split_setting(value: str | None) -> list[str]:
    """Read a comma-separated setting, treating blank and unset alike."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _runtime_param_payload(value: str) -> str:
    """Unwrap the JSON envelope a DataRobot runtime parameter arrives in.

    The platform does not deliver a runtime parameter's bare value: the env var
    holds ``{"type": "string", "payload": "<value>"}``. Declared settings fields
    get this unwrapped by ``datarobot.core.config.getenv``; it is restated here
    because the per-tag variables are discovered by prefix rather than declared,
    so no field ever maps to them. A value that is not the envelope is returned
    as it came — the platform has no contract that it will never deliver one
    bare, and comma-separated scope names cannot be mistaken for JSON.
    """
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(decoded, dict) and decoded.get("type") in _RUNTIME_PARAM_TYPES:
        return str(decoded.get("payload") or "")
    return value


def read_tag_scopes(environ: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Collect ``MCP_OAUTH_TAG_SCOPES_<TAG>`` variables into ``{tag: scopes}``.

    A variable with no scopes in it is skipped rather than registered as an
    empty requirement, so blanking one out turns the guard off instead of
    guarding with nothing.
    """
    environ = os.environ.copy() if environ is None else environ
    collected: dict[str, list[str]] = {}
    for name, value in environ.items():
        bare = name
        is_runtime_param = bare.startswith(RUNTIME_PARAM_ENV_VAR_NAME_PREFIX)
        if is_runtime_param:
            bare = bare[len(RUNTIME_PARAM_ENV_VAR_NAME_PREFIX) :]
        if not bare.upper().startswith(TAG_SCOPES_ENV_PREFIX):
            continue
        tag = bare[len(TAG_SCOPES_ENV_PREFIX) :]
        if is_runtime_param:
            # The serverless path delivers a JSON envelope, not the bare value.
            scopes = split_setting(_runtime_param_payload(value))
        else:
            scopes = split_setting(value)
        if tag and scopes:
            collected[tag] = scopes
    return collected


def build_scope_settings(config: MCPServerConfig | None = None) -> ScopeSettings:
    """Assemble the scope settings this server is configured with.

    There is no enforcement setting to read: whether a caller with no verifiable
    token loses a guarded component follows from these values — see
    :attr:`ScopeSettings.enforced`.

    The audience falls back through the same chain the published document's
    ``resource`` does — ``MCP_OAUTH_AUDIENCE``, then ``MCP_OAUTH_RESOURCE``,
    then the runtime-resolved URL — so the identity a discovering client will
    mint its token for is the identity this server checks ``aud`` against.
    Without the last step, a deployment that lets ``resource`` resolve at
    runtime would publish an audience it never verifies: full OAuth dance on
    the client, no enforcement on the server, and nothing to say so. On a
    DataRobot deployment this also means setting the authorization server
    alone is what turns verification on.
    """
    config = config or get_config()
    issuers = split_setting(config.mcp_oauth_authorization_servers)
    if len(issuers) > 1:
        logger.warning(
            "MCP_OAUTH_AUTHORIZATION_SERVERS lists %d servers; bearer tokens are "
            "verified against the first (%s) only. One JWKS URI can serve one "
            "issuer, so a token minted by any of the others will not verify.",
            len(issuers),
            issuers[0],
        )
    return ScopeSettings(
        source=ScopeSource.parse(config.mcp_oauth_scope_source),
        tag_scopes=read_tag_scopes(),
        issuer=issuers[0] if issuers else None,
        audience=config.mcp_oauth_audience or config.mcp_oauth_resource or resolve_self_url(),
        jwks_uri=config.mcp_oauth_jwks_uri,
    )


async def wire_scopes(mcp: Any, config: MCPServerConfig | None = None) -> None:
    """Install this server's scope settings and apply them to every component.

    Also probes the JWKS once, so an unreachable IdP shows up in the startup
    log rather than as a server that silently serves nothing to anyone.
    """
    await _wire_scopes(mcp, build_scope_settings(config))
    await probe_verification_keys()
