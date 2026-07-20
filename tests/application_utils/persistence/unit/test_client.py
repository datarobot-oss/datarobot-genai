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

"""Unit tests for DRMemoryServiceClient (_client.py)."""

from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from email.utils import format_datetime

import httpx
import pytest
import respx

from datarobot_genai.application_utils.persistence import DRMemoryBadRequestError
from datarobot_genai.application_utils.persistence import DRMemoryConflictError
from datarobot_genai.application_utils.persistence import DRMemoryNotFoundError
from datarobot_genai.application_utils.persistence import DRMemoryRateLimitError
from datarobot_genai.application_utils.persistence import DRMemoryServiceError
from datarobot_genai.application_utils.persistence import DRMemoryUnavailableError
from datarobot_genai.application_utils.persistence import DRMemoryValidationError
from datarobot_genai.application_utils.persistence import DRMemoryVersionConflictError
from datarobot_genai.application_utils.persistence._client import DRMemoryServiceClient
from datarobot_genai.application_utils.persistence._config import build_base_url

BASE = "https://app.datarobot.com/api/v2"
MEMORY_BASE = f"{BASE}/memory"


# ── Configuration resolution ──────────────────────────────────────────────────


def test_base_url_is_endpoint_plus_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """GIVEN DATAROBOT_ENDPOINT env var WHEN client is created THEN base_url is correct."""
    monkeypatch.setenv("DATAROBOT_ENDPOINT", BASE)
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok")
    client = DRMemoryServiceClient()
    assert client.base_url == MEMORY_BASE
    del client


def test_base_url_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    """GIVEN endpoint with trailing slash WHEN client THEN base_url has no double slash."""
    monkeypatch.setenv("DATAROBOT_ENDPOINT", f"{BASE}/")
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok")
    client = DRMemoryServiceClient()
    assert not client.base_url.endswith("//")
    assert client.base_url == MEMORY_BASE
    del client


def test_missing_endpoint_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """GIVEN no endpoint configured WHEN client is created THEN raises ValueError."""
    monkeypatch.delenv("DATAROBOT_ENDPOINT", raising=False)
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok")
    with pytest.raises(ValueError, match="DATAROBOT_ENDPOINT"):
        DRMemoryServiceClient()


def test_missing_api_token_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """GIVEN no API token configured WHEN client is created THEN raises ValueError."""
    monkeypatch.setenv("DATAROBOT_ENDPOINT", BASE)
    monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="DATAROBOT_API_TOKEN"):
        DRMemoryServiceClient()


def test_build_base_url_helper() -> None:
    """GIVEN endpoint and base_path WHEN build_base_url THEN correct URL."""
    assert build_base_url(BASE) == MEMORY_BASE
    assert build_base_url(f"{BASE}/", "memory") == MEMORY_BASE
    assert build_base_url(BASE, "custom-path") == f"{BASE}/custom-path"


# ── Bearer auth header ────────────────────────────────────────────────────────


@respx.mock
async def test_bearer_auth_header_is_sent() -> None:
    """GIVEN a DRMemoryServiceClient WHEN a request THEN Authorization: Bearer header is sent."""
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"items": []})

    respx.get(f"{MEMORY_BASE}/new-route/").mock(side_effect=_capture)

    client = DRMemoryServiceClient(
        endpoint=BASE,
        api_token="my-secret-token",
        http_client=httpx.AsyncClient(),
    )
    await client.request("GET", "new-route/")
    assert captured["auth"] == "Bearer my-secret-token"


# ── DI client ownership ───────────────────────────────────────────────────────


async def test_injected_client_is_not_closed_on_aclose() -> None:
    """GIVEN an injected httpx client WHEN aclose() THEN the injected client is NOT closed."""
    inner = httpx.AsyncClient()
    client = DRMemoryServiceClient(
        endpoint=BASE,
        api_token="tok",
        http_client=inner,
    )
    await client.aclose()
    assert not inner.is_closed


async def test_owned_client_is_closed_on_aclose(monkeypatch: pytest.MonkeyPatch) -> None:
    """GIVEN no injected client WHEN aclose() THEN the owned client IS closed."""
    monkeypatch.setenv("DATAROBOT_ENDPOINT", BASE)
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok")
    async with DRMemoryServiceClient() as client:
        inner = client._http_client
    # After __aexit__ the owned client should be closed
    assert inner.is_closed


# ── Status → exception mapping ────────────────────────────────────────────────


@respx.mock
async def test_400_raises_memory_bad_request_error() -> None:
    """GIVEN service returns 400 WHEN request THEN DRMemoryBadRequestError."""
    respx.post(f"{MEMORY_BASE}/resource/").mock(
        return_value=httpx.Response(400, json={"detail": "Bad input"})
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryBadRequestError) as exc_info:
        await client.request("POST", "resource/")
    assert exc_info.value.status_code == 400
    assert "Bad input" in exc_info.value.detail


@respx.mock
async def test_404_raises_memory_not_found_error() -> None:
    """GIVEN service returns 404 WHEN request THEN DRMemoryNotFoundError."""
    respx.get(f"{MEMORY_BASE}/missing/").mock(
        return_value=httpx.Response(404, json={"detail": "Not found"})
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryNotFoundError) as exc_info:
        await client.request("GET", "missing/")
    assert exc_info.value.status_code == 404


@respx.mock
async def test_409_dedup_raises_memory_conflict_error() -> None:
    """GIVEN service returns 409 with DeduplicationConflict THEN DRMemoryConflictError."""
    respx.post(f"{MEMORY_BASE}/new/").mock(
        return_value=httpx.Response(
            409,
            json={
                "errorName": "SessionDeduplicationConflict",
                "detail": "Duplicate key",
                "deduplicationKey": "key1",
                "existingSessionId": "sess-123",
                "existingSessionUrl": "/memory/space-1/sessions/sess-123/",
            },
        )
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryConflictError) as exc_info:
        await client.request("POST", "new/")
    err = exc_info.value
    assert err.existing_id == "sess-123"
    assert err.location is not None


@respx.mock
async def test_409_version_mismatch_raises_version_conflict_error() -> None:
    """GIVEN service returns 409 without DeduplicationConflict THEN DRMemoryVersionConflictError."""
    respx.patch(f"{MEMORY_BASE}/space/sessions/sess/").mock(
        return_value=httpx.Response(
            409,
            json={"detail": "Session version mismatch: expected 1, current 3"},
        )
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryVersionConflictError) as exc_info:
        await client.request("PATCH", "space/sessions/sess/")
    assert exc_info.value.status_code == 409


@respx.mock
async def test_422_event_version_raises_version_conflict_error() -> None:
    """GIVEN service returns 422 with event-version detail THEN DRMemoryVersionConflictError."""
    respx.patch(f"{MEMORY_BASE}/space/sessions/sess/events/0/").mock(
        return_value=httpx.Response(
            422,
            json={"detail": "Patch of incorrect version of event"},
        )
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryVersionConflictError) as exc_info:
        await client.request("PATCH", "space/sessions/sess/events/0/")
    assert exc_info.value.status_code == 422


@respx.mock
async def test_422_schema_validation_raises_validation_error() -> None:
    """GIVEN service returns 422 with schema error THEN DRMemoryValidationError."""
    respx.post(f"{MEMORY_BASE}/space/sessions/").mock(
        return_value=httpx.Response(
            422,
            json={"detail": [{"msg": "field required", "loc": ["body", "participants"]}]},
        )
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryValidationError):
        await client.request("POST", "space/sessions/")


@respx.mock
async def test_500_raises_base_memory_service_error() -> None:
    """GIVEN service returns 500 WHEN request THEN DRMemoryServiceError."""
    respx.get(f"{MEMORY_BASE}/anything/").mock(
        return_value=httpx.Response(500, json={"detail": "Internal error"})
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryServiceError) as exc_info:
        await client.request("GET", "anything/")
    assert exc_info.value.status_code == 500


@respx.mock
async def test_429_quota_raises_rate_limit_error_with_retry_after_seconds() -> None:
    """GIVEN the trial-quota 429 (Retry-After + X-RateLimit-*) THEN DRMemoryRateLimitError."""
    # Payload shape mirrors memoryservice middleware/trial_limits.py (quota branch).
    respx.post(f"{MEMORY_BASE}/space/sessions/").mock(
        return_value=httpx.Response(
            429,
            json={
                "detail": (
                    "Trial write request limit reached for this month."
                    " Upgrade your plan to continue."
                )
            },
            headers={
                "Retry-After": "30",
                "X-RateLimit-Limit": "1000",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1785542400",
            },
        )
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryRateLimitError) as exc_info:
        await client.request("POST", "space/sessions/")
    err = exc_info.value
    assert err.status_code == 429
    # Whole seconds: "30.0" would be rejected by digit-gated Retry-After parsers.
    assert err.retry_after == 30
    assert isinstance(err.retry_after, int)
    assert "limit reached" in err.detail.lower()


@respx.mock
async def test_429_parses_past_http_date_retry_after_as_zero() -> None:
    """GIVEN a 429 with a past HTTP-date Retry-After THEN retry_after clamps to 0."""
    respx.get(f"{MEMORY_BASE}/space/").mock(
        return_value=httpx.Response(
            429,
            json={"detail": "Rate limited"},
            headers={"Retry-After": "Wed, 01 Jan 2020 00:00:00 GMT"},  # in the past
        )
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryRateLimitError) as exc_info:
        await client.request("GET", "space/")
    assert exc_info.value.retry_after == 0


@respx.mock
async def test_429_parses_future_http_date_retry_after_as_positive_delta() -> None:
    """GIVEN a 429 with a future HTTP-date Retry-After THEN retry_after is the ceil'd delta."""
    future = datetime.now(UTC) + timedelta(seconds=3600)
    respx.get(f"{MEMORY_BASE}/space/").mock(
        return_value=httpx.Response(
            429,
            json={"detail": "Rate limited"},
            headers={"Retry-After": format_datetime(future, usegmt=True)},
        )
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryRateLimitError) as exc_info:
        await client.request("GET", "space/")
    retry_after = exc_info.value.retry_after
    assert retry_after is not None
    assert 3590 <= retry_after <= 3601
    assert isinstance(retry_after, int)


@respx.mock
async def test_429_storage_cap_without_retry_after_header() -> None:
    """GIVEN the trial storage-cap 429 (no Retry-After by design) THEN retry_after is None."""
    # Payload shape mirrors memoryservice middleware/trial_limits.py (storage branch):
    # storage is a level, not a windowed quota, so the service sends no Retry-After.
    respx.post(f"{MEMORY_BASE}/space/sessions/").mock(
        return_value=httpx.Response(
            429,
            json={"detail": "Trial storage limit reached. Delete data or upgrade your plan."},
            headers={"X-RateLimit-Limit": "1073741824", "X-RateLimit-Remaining": "0"},
        )
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryRateLimitError) as exc_info:
        await client.request("POST", "space/sessions/")
    assert exc_info.value.retry_after is None
    assert "storage limit" in exc_info.value.detail.lower()


@respx.mock
async def test_429_unparseable_retry_after_is_none() -> None:
    """GIVEN a 429 with a garbage Retry-After value THEN retry_after is None (no crash)."""
    respx.get(f"{MEMORY_BASE}/space/").mock(
        return_value=httpx.Response(
            429,
            json={"detail": "Rate limited"},
            headers={"Retry-After": "soon"},
        )
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryRateLimitError) as exc_info:
        await client.request("GET", "space/")
    assert exc_info.value.retry_after is None


@respx.mock
async def test_429_is_catchable_as_base_service_error() -> None:
    """GIVEN a 429 THEN except DRMemoryServiceError still catches it (hierarchy intact)."""
    respx.get(f"{MEMORY_BASE}/space/").mock(
        return_value=httpx.Response(429, json={"detail": "Rate limited"})
    )
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryServiceError):
        await client.request("GET", "space/")


# ── Transport failures → DRMemoryUnavailableError ─────────────────────────────


@respx.mock
async def test_timeout_raises_unavailable_error() -> None:
    """GIVEN the request times out THEN DRMemoryUnavailableError with httpx cause."""
    respx.get(f"{MEMORY_BASE}/slow/").mock(side_effect=httpx.ReadTimeout("read timed out"))
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryUnavailableError) as exc_info:
        await client.request("GET", "slow/")
    err = exc_info.value
    assert err.status_code is None
    assert "timed out" in err.detail
    assert isinstance(err.__cause__, httpx.ReadTimeout)


@respx.mock
async def test_connect_error_raises_unavailable_error() -> None:
    """GIVEN the connection fails THEN DRMemoryUnavailableError with httpx cause."""
    respx.get(f"{MEMORY_BASE}/down/").mock(side_effect=httpx.ConnectError("connection refused"))
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryUnavailableError) as exc_info:
        await client.request("GET", "down/")
    err = exc_info.value
    assert err.status_code is None
    assert isinstance(err.__cause__, httpx.ConnectError)


@respx.mock
async def test_transport_failure_is_catchable_as_base_service_error() -> None:
    """GIVEN a transport failure THEN except DRMemoryServiceError covers it, no httpx import."""
    respx.get(f"{MEMORY_BASE}/down/").mock(side_effect=httpx.ConnectError("connection refused"))
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    with pytest.raises(DRMemoryServiceError):
        await client.request("GET", "down/")


@respx.mock
async def test_200_returns_response_object() -> None:
    """GIVEN service returns 200 WHEN request THEN returns httpx.Response."""
    respx.get(f"{MEMORY_BASE}/ok/").mock(return_value=httpx.Response(200, json={"result": "ok"}))
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    resp = await client.request("GET", "ok/")
    assert resp.status_code == 200
    assert resp.json() == {"result": "ok"}


# ── Request body and params passthrough ───────────────────────────────────────


@respx.mock
async def test_json_body_is_forwarded_correctly() -> None:
    """GIVEN a JSON body WHEN request THEN the body reaches the mock."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(201, json={"id": "1"})

    respx.post(f"{MEMORY_BASE}/items/").mock(side_effect=_capture)
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    await client.request("POST", "items/", json={"key": "value"})
    assert captured["body"] == {"key": "value"}


@respx.mock
async def test_extra_headers_are_merged() -> None:
    """GIVEN extra_headers WHEN request THEN extra headers are present alongside auth."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["if_match"] = req.headers.get("if-match")
        captured["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={})

    respx.patch(f"{MEMORY_BASE}/sessions/x/").mock(side_effect=_capture)
    client = DRMemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    await client.request("PATCH", "sessions/x/", extra_headers={"If-Match": "5"})
    assert captured["if_match"] == "5"
    assert captured["auth"] == "Bearer tok"
