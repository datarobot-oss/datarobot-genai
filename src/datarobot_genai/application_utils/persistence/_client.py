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

"""Async HTTP transport for the Memory Service.

``DRMemoryServiceClient`` is the single entry point for all HTTP requests.
It resolves configuration, sets bearer auth headers, builds absolute URLs,
and maps HTTP status codes to typed exceptions.

All routes include trailing slashes — the Memory Service runs with
``redirect_slashes=False`` so a missing trailing slash returns 404.
"""

from __future__ import annotations

import math
from datetime import UTC
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from datarobot_genai.application_utils.persistence._config import build_base_url
from datarobot_genai.application_utils.persistence._config import resolve_api_token
from datarobot_genai.application_utils.persistence._config import resolve_endpoint
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryBadRequestError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryConflictError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryNotFoundError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryRateLimitError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryServiceError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryUnavailableError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryValidationError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryVersionConflictError

# The detail string the service returns for stale-event PATCH (HTTP 422).
_EVENT_VERSION_DETAIL: str = "Patch of incorrect version of event"


class DRMemoryServiceClient:
    """Async HTTP client for the DataRobot Agentic Memory Service.

    Resolves ``DATAROBOT_ENDPOINT`` and ``DATAROBOT_API_TOKEN`` from the
    environment (overridable via constructor arguments).  Owns an
    ``httpx.AsyncClient`` unless one is injected.

    The instance itself is lightweight — the resolved base URL plus a headers
    dict.  Applications that act on behalf of many principals (a different
    API token per end user) are supported by constructing one
    ``DRMemoryServiceClient`` per principal over a single shared
    ``http_client``: the shared pool carries the connections, each instance
    carries only an identity.

    Parameters
    ----------
    endpoint : str | None
        DataRobot API endpoint (e.g. ``https://app.datarobot.com/api/v2``).
        Defaults to the ``DATAROBOT_ENDPOINT`` environment variable.
    api_token : str | None
        DataRobot API token.  Defaults to the ``DATAROBOT_API_TOKEN``
        environment variable.  Multi-principal applications must pass this
        explicitly — the environment fallback is the *application's* own
        credential, not the requesting user's.
    base_path : str
        Sub-path appended to the endpoint.  Defaults to ``"memory"`` — the
        Tyk gateway mount path for the Memory Service (``/api/v2/memory``).
    http_client : httpx.AsyncClient | None
        Injected async client.  When supplied, the ``DRMemoryServiceClient``
        will **not** close it on ``aclose()`` — the caller owns its lifetime.
        This is a supported production pattern (share one connection pool
        across many per-principal client instances) as well as the hook for
        testing with ``respx``.  When omitted, the client creates and owns a
        pool of its own.
    timeout : float
        Default per-request timeout in seconds.  Ignored when ``http_client``
        is injected — configure the timeout on the injected client instead.

    Examples
    --------
    One client, one credential (scripts, single-principal apps):

    .. code-block:: python

        import asyncio
        from datarobot_genai.application_utils.persistence import (
            DRMemorySpace,
            DRMemoryServiceClient,
        )

        async def main() -> None:
            async with DRMemoryServiceClient() as client:
                space = await DRMemorySpace.post(client, description="my-space")
                print(space.id)

        asyncio.run(main())

    One shared pool, one thin client per principal (multi-user web apps):

    .. code-block:: python

        import httpx

        # App startup.  Do not set default auth headers on the shared pool —
        # identity belongs on each DRMemoryServiceClient, not the transport.
        shared_http = httpx.AsyncClient(timeout=30.0)

        def client_for(user_token: str) -> DRMemoryServiceClient:
            # Cheap: per-request construction, no new connections.
            return DRMemoryServiceClient(
                api_token=user_token,
                http_client=shared_http,
            )

        async def shutdown() -> None:
            await shared_http.aclose()  # close the pool once, at app shutdown
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_token: str | None = None,
        base_path: str = "memory",
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        resolved_endpoint = resolve_endpoint(endpoint)
        resolved_token = resolve_api_token(api_token)
        self._base_url = build_base_url(resolved_endpoint, base_path)
        self._auth_headers: dict[str, str] = {
            "Authorization": f"Bearer {resolved_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._owns_client = http_client is None
        self._http_client: httpx.AsyncClient = http_client or httpx.AsyncClient(timeout=timeout)

    @property
    def base_url(self) -> str:
        """The resolved Memory Service base URL."""
        return self._base_url

    def _url(self, path: str) -> str:
        """Build an absolute URL from a path relative to ``base_url``."""
        return self._base_url.rstrip("/") + "/" + path.lstrip("/")

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Send a request and return the response, mapping errors to typed exceptions.

        Parameters
        ----------
        method : str
            HTTP method (``"GET"``, ``"POST"``, ``"PATCH"``, ``"DELETE"``).
        path : str
            Path relative to ``base_url``.  Must end with ``"/"``.
        params : dict | None
            Query parameters.
        json : Any
            JSON-serializable request body.
        extra_headers : dict | None
            Additional headers (e.g. ``{"If-Match": "3"}``).

        Returns
        -------
        httpx.Response
            On 2xx status codes.

        Raises
        ------
        DRMemoryBadRequestError
            HTTP 400 — bad request (e.g. emitter not a participant).
        DRMemoryNotFoundError
            HTTP 404 — resource not found.
        DRMemoryConflictError
            HTTP 409 — deduplication conflict on create.
        DRMemoryVersionConflictError
            HTTP 409 — stale ``If-Match`` on session ``patch()``, or HTTP 422
            with the event-version detail.
        DRMemoryValidationError
            HTTP 422 — schema validation error.
        DRMemoryRateLimitError
            HTTP 429 — quota or rate limit exceeded; carries ``retry_after``
            seconds parsed from the ``Retry-After`` response header.
        DRMemoryUnavailableError
            No response received — request timeout or transport failure
            (connection refused, DNS, TLS).  The original ``httpx`` exception
            is preserved as ``__cause__``.
        DRMemoryServiceError
            Any other 4xx/5xx error.
        """
        headers = {**self._auth_headers, **(extra_headers or {})}
        url = self._url(path)

        try:
            resp = await self._http_client.request(
                method,
                url,
                params=params,
                json=json,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise DRMemoryUnavailableError(
                f"Memory Service request timed out ({method} {url}): {exc!r}"
            ) from exc
        except httpx.TransportError as exc:
            raise DRMemoryUnavailableError(
                f"Memory Service unreachable ({method} {url}): {exc!r}"
            ) from exc

        if resp.status_code < 400:
            return resp

        # ── Parse the error body ──────────────────────────────────────────
        try:
            parsed = resp.json()
        except Exception:
            parsed = None
        # The service should return a JSON object, but gateways/proxies may emit a JSON
        # array, scalar, or non-JSON body. Coerce anything that is not a dict to ``{}`` so
        # the ``.get(...)`` lookups below cannot raise AttributeError and mask the real error.
        body: dict[str, Any] = parsed if isinstance(parsed, dict) else {}

        detail = _extract_detail(body, resp.text)

        if resp.status_code == 400:
            raise DRMemoryBadRequestError(detail, status_code=400, payload=body)

        if resp.status_code == 404:
            raise DRMemoryNotFoundError(detail, status_code=404, payload=body)

        if resp.status_code == 409:
            error_name = str(body.get("errorName", ""))
            existing_id = body.get("existingSessionId") or body.get("existingMemorySpaceId")
            location = (
                body.get("existingSessionUrl")
                or body.get("existingMemorySpaceUrl")
                or resp.headers.get("Location")
            )
            # Treat it as a deduplication conflict when the service says so *or*
            # when it returns the id of the existing resource — do not rely on the
            # errorName string alone, which can drift and would otherwise misroute
            # a real dedup conflict (dropping existing_id and breaking 409-adopt).
            if "DeduplicationConflict" in error_name or existing_id is not None:
                raise DRMemoryConflictError(
                    detail,
                    status_code=409,
                    payload=body,
                    existing_id=str(existing_id) if existing_id else None,
                    location=str(location) if location else None,
                )
            # 409 without dedup markers = session version mismatch
            raise DRMemoryVersionConflictError(detail, status_code=409, payload=body)

        if resp.status_code == 422:
            if _EVENT_VERSION_DETAIL.lower() in detail.lower():
                raise DRMemoryVersionConflictError(detail, status_code=422, payload=body)
            raise DRMemoryValidationError(detail, status_code=422, payload=body)

        if resp.status_code == 429:
            raise DRMemoryRateLimitError(
                detail,
                status_code=429,
                payload=body,
                retry_after=_parse_retry_after(resp.headers.get("Retry-After")),
            )

        raise DRMemoryServiceError(detail, status_code=resp.status_code, payload=body)

    async def aclose(self) -> None:
        """Close the underlying HTTP client (only if owned by this instance)."""
        if self._owns_client:
            await self._http_client.aclose()

    async def __aenter__(self) -> DRMemoryServiceClient:
        """Return self for use as an async context manager."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Close the client on context exit."""
        await self.aclose()


def _parse_retry_after(value: str | None) -> int | None:
    """Parse a ``Retry-After`` header into whole seconds.

    Accepts both RFC 9110 forms: delta-seconds (``"30"``) and HTTP-date
    (``"Wed, 22 Jul 2026 07:28:00 GMT"``, rounded up).  Whole seconds so the
    value can be propagated verbatim into another ``Retry-After`` header —
    RFC 9110 delay-seconds and the service's OpenAPI contract are integers,
    and ``"30.0"`` would be rejected by digit-gated parsers.  Returns ``None``
    when the header is absent or unparseable; never returns a negative number.
    """
    if not value:
        return None
    try:
        return max(0, int(value.strip()))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0, math.ceil((when - datetime.now(UTC)).total_seconds()))


def _extract_detail(body: dict[str, Any], fallback: str) -> str:
    """Extract a human-readable error detail from a response body."""
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict):
                return str(first.get("msg", fallback))
        if "message" in body:
            return str(body["message"])
    return fallback or "Unknown error"
