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

from typing import Any

import httpx

from datarobot_genai.application_utils.persistence._config import build_base_url
from datarobot_genai.application_utils.persistence._config import resolve_api_token
from datarobot_genai.application_utils.persistence._config import resolve_endpoint
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryBadRequestError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryConflictError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryNotFoundError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryServiceError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryValidationError
from datarobot_genai.application_utils.persistence.exceptions import DRMemoryVersionConflictError

# The detail string the service returns for stale-event PATCH (HTTP 422).
_EVENT_VERSION_DETAIL: str = "Patch of incorrect version of event"


class DRMemoryServiceClient:
    """Async HTTP client for the DataRobot Agentic Memory Service.

    Resolves ``DATAROBOT_ENDPOINT`` and ``DATAROBOT_API_TOKEN`` from the
    environment (overridable via constructor arguments).  Owns an
    ``httpx.AsyncClient`` unless one is injected (for testing).

    Parameters
    ----------
    endpoint : str | None
        DataRobot API endpoint (e.g. ``https://app.datarobot.com/api/v2``).
        Defaults to the ``DATAROBOT_ENDPOINT`` environment variable.
    api_token : str | None
        DataRobot API token.  Defaults to the ``DATAROBOT_API_TOKEN``
        environment variable.
    base_path : str
        Sub-path appended to the endpoint.  Defaults to ``"memory"`` — the
        Tyk gateway mount path for the Memory Service (``/api/v2/memory``).
    http_client : httpx.AsyncClient | None
        Injected async client (for testing with ``respx``).  When supplied,
        the ``DRMemoryServiceClient`` will **not** close it on ``aclose()``.
    timeout : float
        Default per-request timeout in seconds.

    Examples
    --------
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
        DRMemoryServiceError
            Any other 4xx/5xx error.
        """
        headers = {**self._auth_headers, **(extra_headers or {})}
        url = self._url(path)

        resp = await self._http_client.request(
            method,
            url,
            params=params,
            json=json,
            headers=headers,
        )

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
