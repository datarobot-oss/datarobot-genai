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

"""Shared fixtures, constants and subclass models for the chat-history unit tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic import Field

from datarobot_genai.application_utils.chat_history.models import Chat
from datarobot_genai.application_utils.chat_history.models import Message
from datarobot_genai.application_utils.persistence import DRMemoryServiceClient
from datarobot_genai.application_utils.persistence import DRMemorySpace

# ── Constants used across tests ───────────────────────────────────────────────

BASE = "https://app.datarobot.com/api/v2"
MEMORY_BASE = f"{BASE}/memory"
SPACE_ID = "aaaaaaaa-0000-0000-0000-000000000001"
SESSION_ID = "bbbbbbbb-0000-0000-0000-000000000001"
PARTICIPANT = "aabbccddeeff001122334455"  # 24-hex ObjectId

SESSIONS_URL = f"{MEMORY_BASE}/{SPACE_ID}/sessions/"
SESSION_URL = f"{SESSIONS_URL}{SESSION_ID}/"
EVENTS_URL = f"{SESSION_URL}events/"


# ── Subclass models exercising the extensibility contract ──────────────────────


class Attachment(BaseModel):
    """A nested model used as an extra field on chat-model subclasses."""

    filename: str
    size: int = 0


class ProjectChat(Chat):
    """A ``Chat`` subclass adding a scalar and a nested metadata field."""

    project_code: str = ""
    labels: list[Attachment] = Field(default_factory=list)


class RichMessage(Message):
    """A ``Message`` subclass adding a scalar and a nested body field."""

    priority: int = 0
    attachments: list[Attachment] = Field(default_factory=list)


# ── Wire builders / fakes ──────────────────────────────────────────────────────


def make_client() -> DRMemoryServiceClient:
    """Return a client pinned at the test endpoint (no network without respx)."""
    return DRMemoryServiceClient(endpoint=BASE, api_token="t", http_client=httpx.AsyncClient())


def make_space() -> DRMemorySpace:
    """Return a minimally-hydrated memory space for the test endpoint."""
    return DRMemorySpace._from_wire(
        make_client(),
        {"memorySpaceId": SPACE_ID, "userId": "u", "tenantId": "t", "createdAt": ""},
    )


def make_chat() -> Chat:
    """Return a hydrated ``Chat`` session that events can be posted under."""
    wire = {
        "id": SESSION_ID,
        "participants": [PARTICIPANT],
        "description": "//thread/thread-1/",
        "deduplicationKey": "dedup-1",
        "metadata": {"name": "Test chat"},
        "version": 1,
        "createdAt": "2026-06-30T00:00:00Z",
    }
    return Chat._from_wire(make_space(), wire)  # type: ignore[return-value]


def echo_event(captured: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    """Return a respx side-effect that echoes the posted event body back verbatim."""

    def _handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        captured["body"] = payload
        return httpx.Response(
            201,
            json={
                "sequenceId": 7,
                "createdAt": "2026-06-30T00:00:01Z",
                "eventType": payload["type"],
                "emitterType": payload["emitter"]["type"],
                "emitterId": payload["emitter"].get("id"),
                "body": payload["body"],
            },
        )

    return _handler


def echo_session(captured: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    """Return a respx side-effect that echoes the posted session back verbatim."""

    def _handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        captured["body"] = payload
        return httpx.Response(
            201,
            json={
                "id": SESSION_ID,
                "participants": payload.get("participants", [PARTICIPANT]),
                "description": payload.get("description"),
                "deduplicationKey": payload.get("deduplicationKey"),
                "metadata": payload.get("metadata", {}),
                "version": 1,
                "createdAt": "2026-06-30T00:00:00Z",
            },
        )

    return _handler
