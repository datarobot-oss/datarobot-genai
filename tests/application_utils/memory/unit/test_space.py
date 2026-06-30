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

"""Unit tests for DRMemorySpace (space.py)."""

from __future__ import annotations

import json

import httpx
import respx

from datarobot_genai.application_utils.memory import DRMemorySpace
from datarobot_genai.application_utils.memory import MemoryServiceClient

BASE = "https://app.datarobot.com/api/v2"
MEMORY_BASE = f"{BASE}/memory"
SPACE_ID = "aaaaaaaa-0000-0000-0000-000000000001"


def _client() -> MemoryServiceClient:
    return MemoryServiceClient(
        endpoint=BASE,
        api_token="test-token",
        http_client=httpx.AsyncClient(),
    )


def _space_wire(
    *,
    space_id: str = SPACE_ID,
    description: str | None = "Test space",
    dedup_key: str | None = "key1",
) -> dict:
    return {
        "memorySpaceId": space_id,
        "userId": "u1",
        "tenantId": "t1",
        "description": description,
        "deduplicationKey": dedup_key,
        "llmModelName": None,
        "llmBaseUrl": None,
        "customInstructions": None,
        "createdAt": "2026-06-30T00:00:00Z",
    }


# ── DRMemorySpace.post ────────────────────────────────────────────────────────


@respx.mock
async def test_post_creates_and_returns_space() -> None:
    """GIVEN a POST to /new/ returns 201 WHEN DRMemorySpace.post THEN space with correct id."""
    respx.post(f"{MEMORY_BASE}/new/").mock(return_value=httpx.Response(201, json=_space_wire()))
    space = await DRMemorySpace.post(_client(), description="Test space", deduplication_key="key1")
    assert space.id == SPACE_ID
    assert space.description == "Test space"
    assert space.deduplication_key == "key1"


@respx.mock
async def test_post_sends_correct_payload() -> None:
    """GIVEN post() is called with kwargs WHEN POST /new/ THEN camelCase payload sent."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(201, json=_space_wire())

    respx.post(f"{MEMORY_BASE}/new/").mock(side_effect=_capture)
    await DRMemorySpace.post(
        _client(),
        description="desc",
        deduplication_key="k",
        llm_model_name="gpt-4",
        custom_instructions="Be helpful",
    )
    assert captured["body"]["description"] == "desc"
    assert captured["body"]["deduplicationKey"] == "k"
    assert captured["body"]["llmModelName"] == "gpt-4"
    assert captured["body"]["customInstructions"] == "Be helpful"


@respx.mock
async def test_post_adopts_existing_on_409_conflict() -> None:
    """GIVEN POST returns 409 dedup WHEN DRMemorySpace.post THEN fetches and returns existing."""
    existing_id = "eeeeeeee-0000-0000-0000-000000000001"
    respx.post(f"{MEMORY_BASE}/new/").mock(
        return_value=httpx.Response(
            409,
            json={
                "errorName": "MemorySpaceDeduplicationConflict",
                "detail": "Conflict",
                "deduplicationKey": "key1",
                "existingMemorySpaceId": existing_id,
                "existingMemorySpaceUrl": f"/memory/{existing_id}/",
            },
        )
    )
    respx.get(f"{MEMORY_BASE}/{existing_id}/").mock(
        return_value=httpx.Response(200, json=_space_wire(space_id=existing_id))
    )
    space = await DRMemorySpace.post(_client(), deduplication_key="key1")
    assert space.id == existing_id


# ── DRMemorySpace.get ─────────────────────────────────────────────────────────


@respx.mock
async def test_get_returns_space_by_id() -> None:
    """GIVEN GET /{id}/ returns 200 WHEN DRMemorySpace.get THEN correct space."""
    respx.get(f"{MEMORY_BASE}/{SPACE_ID}/").mock(
        return_value=httpx.Response(200, json=_space_wire())
    )
    space = await DRMemorySpace.get(_client(), SPACE_ID)
    assert space.id == SPACE_ID
    assert space.created_at == "2026-06-30T00:00:00Z"


# ── DRMemorySpace.list ────────────────────────────────────────────────────────


@respx.mock
async def test_list_returns_spaces() -> None:
    """GIVEN GET / returns paginated response WHEN list THEN list of spaces."""
    respx.get(f"{MEMORY_BASE}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [_space_wire(), _space_wire(space_id="bbbb-0001")],
                "offset": 0,
                "limit": 100,
                "total": 2,
            },
        )
    )
    spaces = await DRMemorySpace.list(_client())
    assert len(spaces) == 2
    assert spaces[0].id == SPACE_ID


@respx.mock
async def test_list_with_dedup_key_filter_sends_param() -> None:
    """GIVEN list(deduplication_key='k') WHEN GET THEN deduplicationKey query param sent."""
    captured: dict = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json={"items": [], "offset": 0, "limit": 100, "total": 0})

    respx.get(f"{MEMORY_BASE}/").mock(side_effect=_capture)
    await DRMemorySpace.list(_client(), deduplication_key="my-key")
    assert captured["params"]["deduplicationKey"] == "my-key"


# ── DRMemorySpace.patch ───────────────────────────────────────────────────────


@respx.mock
async def test_patch_updates_space_in_place() -> None:
    """GIVEN patch() is called WHEN PATCH /{id}/ THEN space attributes updated."""
    respx.get(f"{MEMORY_BASE}/{SPACE_ID}/").mock(
        return_value=httpx.Response(200, json=_space_wire())
    )
    respx.patch(f"{MEMORY_BASE}/{SPACE_ID}/").mock(
        return_value=httpx.Response(200, json=_space_wire(description="Updated description"))
    )
    space = await DRMemorySpace.get(_client(), SPACE_ID)
    await space.patch(description="Updated description")
    assert space.description == "Updated description"


@respx.mock
async def test_patch_sends_camel_case_payload() -> None:
    """GIVEN patch(llm_model_name=...) WHEN PATCH THEN camelCase key in payload."""
    captured: dict = {}

    respx.get(f"{MEMORY_BASE}/{SPACE_ID}/").mock(
        return_value=httpx.Response(200, json=_space_wire())
    )

    def _capture(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=_space_wire())

    respx.patch(f"{MEMORY_BASE}/{SPACE_ID}/").mock(side_effect=_capture)
    space = await DRMemorySpace.get(_client(), SPACE_ID)
    await space.patch(llm_model_name="gpt-4")
    assert "llmModelName" in captured["body"]
    assert captured["body"]["llmModelName"] == "gpt-4"


# ── DRMemorySpace.delete ──────────────────────────────────────────────────────


@respx.mock
async def test_delete_sends_delete_request() -> None:
    """GIVEN a space WHEN delete() THEN DELETE /{id}/ is sent."""
    respx.get(f"{MEMORY_BASE}/{SPACE_ID}/").mock(
        return_value=httpx.Response(200, json=_space_wire())
    )
    delete_route = respx.delete(f"{MEMORY_BASE}/{SPACE_ID}/").mock(return_value=httpx.Response(204))
    space = await DRMemorySpace.get(_client(), SPACE_ID)
    await space.delete()
    assert delete_route.called


# ── _from_wire field mapping ──────────────────────────────────────────────────


def test_from_wire_maps_memory_space_id_to_id() -> None:
    """GIVEN wire dict with memorySpaceId WHEN _from_wire THEN space.id is set."""
    client = MemoryServiceClient(endpoint=BASE, api_token="tok", http_client=httpx.AsyncClient())
    space = DRMemorySpace._from_wire(client, _space_wire())
    assert space.id == SPACE_ID
    assert space.user_id == "u1"
    assert space.tenant_id == "t1"
