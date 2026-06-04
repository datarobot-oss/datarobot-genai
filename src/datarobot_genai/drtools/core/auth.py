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


import contextvars
import logging
from typing import Any

from datarobot.auth.datarobot.exceptions import OAuthServiceClientErr
from datarobot.auth.session import AuthCtx
from datarobot.models.genai.agent.auth import ToolAuth

from datarobot_genai.core.utils.auth import AsyncOAuthTokenProvider
from datarobot_genai.core.utils.auth import AuthContextHeaderHandler
from datarobot_genai.core.utils.auth import DRAppCtx
from datarobot_genai.drtools.core.constants import HEADER_TOKEN_CANDIDATE_NAMES
from datarobot_genai.drtools.core.credentials import AuthResolutionStrategy
from datarobot_genai.drtools.core.credentials import get_credentials
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)

# Context variable for request headers. Set by RequestHeadersMiddleware for every
# HTTP request so drtools clients can resolve the API token in custom routes and tools.
_request_headers_ctx: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "request_headers", default=None
)
_auth_context_ctx: contextvars.ContextVar[AuthCtx | dict[str, Any] | None] = contextvars.ContextVar(
    "auth_context", default=None
)


def set_request_headers(headers: dict[str, str]) -> None:
    """Inject request headers from any runtime adapter (FastMCP, Starlette, LangChain, tests)."""
    _request_headers_ctx.set({str(k).lower(): v for k, v in headers.items() if isinstance(v, str)})


def get_request_headers() -> dict[str, str]:
    """Return headers injected for the current request context."""
    return _request_headers_ctx.get() or {}


def set_auth_context(auth_context: AuthCtx | dict[str, Any] | None) -> None:
    """Inject authorization context from a runtime adapter."""
    _auth_context_ctx.set(auth_context)


def get_auth_context() -> AuthCtx | None:
    """Return authorization context injected for the current request."""
    auth_ctx = _auth_context_ctx.get()
    if auth_ctx is None:
        return None
    if isinstance(auth_ctx, AuthCtx):
        return auth_ctx
    if isinstance(auth_ctx, dict):
        try:
            return AuthCtx(**auth_ctx)
        except Exception:
            return None
    return None


async def must_get_auth_context() -> AuthCtx:
    """Retrieve the AuthCtx from the current request context or raise error."""
    auth_ctx = get_auth_context()
    if not auth_ctx:
        raise RuntimeError("Could not retrieve authorization context.")
    return auth_ctx


def extract_auth_context_from_headers(headers: dict[str, str]) -> AuthCtx | None:
    """Parse ``x-datarobot-authorization-context`` into an :class:`AuthCtx`."""
    auth_handler = AuthContextHeaderHandler()
    return auth_handler.get_context(headers)


async def get_access_token(provider_type: str | None = None) -> str:
    """Retrieve access token from the DataRobot OAuth Provider Service."""
    auth_ctx = await must_get_auth_context()
    logger.debug("Retrieved authorization context")

    oauth_token_provider = AsyncOAuthTokenProvider(auth_ctx)
    oauth_access_token = await oauth_token_provider.get_token(
        auth_type=ToolAuth.OBO,
        provider_type=provider_type,
    )
    return oauth_access_token


def oauth_access_token_header_name(header_segment: str) -> str:
    """Build ``x-datarobot-<segment>-access-token`` from *header_segment* (normalized)."""
    segment = header_segment.strip().lower().replace("_", "-")
    return f"x-datarobot-{segment}-access-token"


def _parse_optional_bearer_token(raw: str) -> str | None:
    value = str(raw).strip()
    if not value:
        return None
    bearer_prefix = "bearer "
    if value.lower().startswith(bearer_prefix):
        value = value[len(bearer_prefix) :].strip()
    return value or None


def _safe_request_headers() -> dict[str, str]:
    try:
        return get_request_headers()
    except Exception:
        return {}


def _resolve_by_strategy(
    *,
    strategy: AuthResolutionStrategy,
    http_only: bool,
    header_value: str | None,
    config_value: str | None,
) -> str | None:
    """Return header or config value per ``auth_resolution_strategy``."""
    if http_only or strategy == AuthResolutionStrategy.HTTP:
        return header_value
    return config_value


def resolve_datarobot_token(*, http_only: bool = False) -> str | None:
    """Resolve the DataRobot API token according to ``auth_resolution_strategy``."""
    creds = get_credentials()
    strategy = creds.auth_resolution_strategy

    use_headers = http_only or strategy != AuthResolutionStrategy.CONFIG
    header_token = (
        _extract_token_from_headers_with_fallback(_safe_request_headers()) if use_headers else None
    )

    use_config = not http_only and strategy != AuthResolutionStrategy.HTTP
    config_token = (creds.datarobot.datarobot_api_token or None) if use_config else None

    return _resolve_by_strategy(
        strategy=strategy,
        http_only=http_only,
        header_value=header_token,
        config_value=config_token,
    )


def resolve_secret(header_name: str, config_value: str) -> str | None:
    """Resolve a secret from request headers and/or config per ``auth_resolution_strategy``."""
    creds = get_credentials()
    strategy = creds.auth_resolution_strategy

    use_headers = strategy != AuthResolutionStrategy.CONFIG
    header_value = (
        _extract_header_value(_safe_request_headers(), header_name) if use_headers else None
    )

    use_config = strategy != AuthResolutionStrategy.HTTP
    config_secret = (config_value or None) if use_config else None

    return _resolve_by_strategy(
        strategy=strategy,
        http_only=False,
        header_value=header_value,
        config_value=config_secret,
    )


def _extract_header_value(headers: dict[str, str], header_name: str) -> str | None:
    lowered = {
        str(k).lower(): v for k, v in headers.items() if isinstance(v, str) and str(v).strip()
    }

    candidates = [header_name]
    if header_name.startswith("x-") and not header_name.startswith("x-datarobot-"):
        candidates.append(f"x-datarobot-{header_name[2:]}")
    elif not header_name.startswith("x-datarobot-"):
        candidates.append(f"x-datarobot-{header_name}")

    for name in candidates:
        key = str(name).lower()
        if value := lowered.get(key):
            return value
    return None


def resolve_oauth_access_token_from_headers(header_segment: str) -> str | None:
    """Read ``x-datarobot-<segment>-access-token`` for *header_segment* (raw or Bearer)."""
    raw = _extract_header_value(
        _safe_request_headers(), oauth_access_token_header_name(header_segment)
    )
    return _parse_optional_bearer_token(raw) if raw else None


async def _try_get_oauth_access_token(
    provider_type: str,
) -> tuple[str | None, BaseException | None]:
    oauth_exc: BaseException | None = None
    try:
        access_token = await get_access_token(provider_type)
        if access_token:
            return access_token, None
        logger.debug("OAuth returned empty token; checking header fallback for %s.", provider_type)
    except OAuthServiceClientErr as e:
        oauth_exc = e
        logger.info(
            "OAuth token not available for %s (%s); checking header fallback.",
            provider_type,
            e,
            exc_info=True,
        )
    except RuntimeError as e:
        oauth_exc = e
        logger.info(
            "No OAuth auth context for %s (%s); checking header fallback.", provider_type, e
        )
    except Exception as e:
        oauth_exc = e
        logger.warning(
            "Unexpected error obtaining OAuth token for %s; checking header fallback: %s",
            provider_type,
            e,
            exc_info=True,
        )
    return None, oauth_exc


def _oauth_access_token_failure_error(
    *,
    display_name: str,
    provider_type: str,
    header: str,
    oauth_exc: BaseException | None,
) -> ToolError:
    if isinstance(oauth_exc, OAuthServiceClientErr):
        logger.error("OAuth client error (no header fallback): %s", oauth_exc, exc_info=True)
        return ToolError(
            f"Could not obtain access token for {display_name}. Complete the OAuth flow or pass "
            f"an access token via the {header} header.",
            kind=ToolErrorKind.AUTHENTICATION,
        )
    if isinstance(oauth_exc, RuntimeError):
        return ToolError(
            f"No OAuth context for {display_name} and no access token in request headers. "
            f"Complete the OAuth flow or pass a token via {header}.",
            kind=ToolErrorKind.AUTHENTICATION,
        )
    if oauth_exc is not None:
        logger.error(
            "Unexpected error obtaining access token for %s: %s",
            provider_type,
            oauth_exc,
            exc_info=True,
        )
        return ToolError(
            f"An unexpected error occurred while obtaining access token for {display_name}.",
            kind=ToolErrorKind.INTERNAL,
        )
    return ToolError(
        f"Received empty access token for {display_name}. Complete the OAuth flow or pass an "
        f"access token via the {header} header.",
        kind=ToolErrorKind.AUTHENTICATION,
    )


async def get_oauth_access_token_with_header_fallback(
    provider_type: str,
    *,
    display_name: str,
    access_token_header_segment: str,
) -> str | ToolError:
    """Resolve an OAuth access token via OBO or request header (``http`` strategy only)."""
    creds = get_credentials()
    strategy = creds.auth_resolution_strategy
    pt = provider_type.strip()

    if strategy == AuthResolutionStrategy.CONFIG:
        header = oauth_access_token_header_name(access_token_header_segment)
        return ToolError(
            f"{display_name} does not support auth_resolution_strategy=config. "
            f"Use auth_resolution_strategy=http with OAuth OBO or pass an access token "
            f"via the {header} header.",
            kind=ToolErrorKind.AUTHENTICATION,
        )

    access_token, oauth_exc = await _try_get_oauth_access_token(pt)
    if access_token:
        return access_token

    if header_token := resolve_oauth_access_token_from_headers(access_token_header_segment):
        return header_token

    header = oauth_access_token_header_name(access_token_header_segment)
    return _oauth_access_token_failure_error(
        display_name=display_name,
        provider_type=pt,
        header=header,
        oauth_exc=oauth_exc,
    )


def _extract_token_from_headers(headers: dict[str, str]) -> str | None:
    for candidate_name in HEADER_TOKEN_CANDIDATE_NAMES:
        auth_header = headers.get(candidate_name)
        if not auth_header or not isinstance(auth_header, str):
            continue

        bearer_prefix = "bearer "
        if auth_header.lower().startswith(bearer_prefix):
            token = auth_header[len(bearer_prefix) :].strip()
        else:
            token = auth_header.strip()

        if token:
            return token
    return None


def _extract_token_from_auth_context(headers: dict[str, str]) -> str | None:
    try:
        auth_handler = AuthContextHeaderHandler()
        auth_ctx = auth_handler.get_context(headers)
        if not auth_ctx or not auth_ctx.metadata:
            return None

        metadata = auth_ctx.metadata
        if not isinstance(metadata, dict):
            return None

        dr_ctx: DRAppCtx = DRAppCtx(**metadata.get("dr_ctx", {}))
        if dr_ctx.api_key:
            logger.debug("Extracted token from auth context")
            return dr_ctx.api_key
        return None
    except Exception as e:
        logger.debug(f"Failed to get token from auth context: {e}")
        return None


def _extract_token_from_headers_with_fallback(headers: dict[str, str]) -> str | None:
    if token := _extract_token_from_headers(headers):
        return token
    if token := _extract_token_from_auth_context(headers):
        return token
    return None
