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

"""Shared fixtures and model definitions for the memory ORM unit tests."""

from __future__ import annotations

from typing import Annotated

import httpx
import pytest

from datarobot_genai.application_utils.memory import DRConcurrencyField
from datarobot_genai.application_utils.memory import DRDeduplicationKey
from datarobot_genai.application_utils.memory import DREvent
from datarobot_genai.application_utils.memory import DRRangeKey
from datarobot_genai.application_utils.memory import DRSession
from datarobot_genai.application_utils.memory import MemoryServiceClient

# ── Constants used across tests ───────────────────────────────────────────────

BASE_URL = "https://app.datarobot.com/api/v2/memory"
SPACE_ID = "aaaaaaaa-0000-0000-0000-000000000001"
SESSION_ID = "bbbbbbbb-0000-0000-0000-000000000001"
PARTICIPANT = "aabbccddeeff001122334455"  # 24-hex ObjectId
SYSTEM_OID = "000000000000000000000000"


# ── Domain model definitions used across tests ────────────────────────────────


class ChatSession(DRSession):
    """Two-level range key + dedup key + concurrency field + metadata."""

    __description_prefix__ = "chat"

    tenant: Annotated[str, DRRangeKey]
    topic: Annotated[str, DRRangeKey]
    chat_id: Annotated[str, DRDeduplicationKey]
    rev: Annotated[int, DRConcurrencyField]
    title: str


class MinimalSession(DRSession):
    """Session with no special markers — all fields go to metadata."""

    label: str = ""


class DedupeOnlySession(DRSession):
    """Session with only a dedup key and no range keys."""

    __description_prefix__ = "dedup-only"

    key: Annotated[str, DRDeduplicationKey]
    notes: str = ""


class ChatMessage(DREvent, session=ChatSession):
    """Event bound to ChatSession; one extra body field."""

    __event_type__ = "message"

    score: float = 0.0


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def memory_client() -> MemoryServiceClient:
    """Return a ``MemoryServiceClient`` with an injected httpx client (no real network)."""
    # The injected client is replaced per-test by respx; we create a shared instance
    # here so fixtures can reference the same base URL.
    return MemoryServiceClient(
        endpoint="https://app.datarobot.com/api/v2",
        api_token="test-token",
        http_client=httpx.AsyncClient(),
    )


@pytest.fixture
def space_wire() -> dict:
    """Minimal valid MemorySpaceResponse wire dict."""
    return {
        "memorySpaceId": SPACE_ID,
        "userId": "user-001",
        "tenantId": "tenant-001",
        "description": "Test space",
        "deduplicationKey": "test-space-key",
        "llmModelName": None,
        "llmBaseUrl": None,
        "customInstructions": None,
        "createdAt": "2026-06-30T00:00:00Z",
    }


@pytest.fixture
def session_wire() -> dict:
    """Minimal valid SessionResponse wire dict for a ChatSession."""
    return {
        "id": SESSION_ID,
        "participants": [PARTICIPANT],
        "description": "//chat/acme/billing/",
        "deduplicationKey": "chat-001",
        "metadata": {"title": "Billing enquiry"},
        "lifecycleStrategies": [],
        "version": 1,
        "createdAt": "2026-06-30T00:00:00Z",
    }


@pytest.fixture
def event_wire() -> dict:
    """Minimal valid EventResponse wire dict for a ChatMessage."""
    return {
        "sequenceId": 0,
        "createdAt": "2026-06-30T00:00:01Z",
        "eventType": "message",
        "emitterType": "user",
        "emitterId": PARTICIPANT,
        "body": {"content": "Hello!", "score": 0.9},
    }
