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

"""Live integration tests for the unified DR/Mem0 memory provider.

Two backends are exercised against their real services:

* **DataRobot Memory Service** — a real ``MemorySpace`` is created via the
  DataRobot SDK (``datarobot.models.memory.MemorySpace``, see
  ``public_api_client``'s ``datarobot/models/memory.py``), the provider is
  built with ``memory_space_id=<that id>``, and add/search/remove round-trip
  through the path-prefixed mem0-compatible endpoint at
  ``{DATAROBOT_ENDPOINT}/memory/{memory_space_id}/``. The space is deleted in
  a finalizer so we never leak resources across runs.
* **Mem0 SaaS** — same provider, no ``memory_space_id``, just a ``MEM0_API_KEY``.
  Confirms the existing SaaS path still works after the routing changes.

All tests skip when the required credentials aren't in the environment so
this module is safe to run in CI without secrets.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from nat.memory.models import MemoryItem

from datarobot_genai.nat.datarobot_mem0_memory import DRMem0Editor
from datarobot_genai.nat.datarobot_mem0_memory import DRMem0MemoryClientConfig
from datarobot_genai.nat.datarobot_mem0_memory import dr_mem0_memory_client


# --------------------------------------------------------------------------- #
# Skip conditions                                                             #
# --------------------------------------------------------------------------- #
def _dr_memory_sdk_available() -> bool:
    try:
        importlib.import_module("datarobot.models.memory")
    except ImportError:
        return False
    return True


_HAS_DR_CREDS = bool(os.getenv("DATAROBOT_ENDPOINT") and os.getenv("DATAROBOT_API_TOKEN"))
_HAS_DR_SDK = _dr_memory_sdk_available()
_HAS_MEM0_KEY = bool(os.getenv("MEM0_API_KEY"))

skip_unless_dr = pytest.mark.skipif(
    not (_HAS_DR_CREDS and _HAS_DR_SDK),
    reason=(
        "DR live tests require DATAROBOT_ENDPOINT + DATAROBOT_API_TOKEN and a "
        "DataRobot SDK that ships datarobot.models.memory (v3.15+)."
    ),
)
skip_unless_mem0 = pytest.mark.skipif(
    not _HAS_MEM0_KEY, reason="Mem0 live tests require MEM0_API_KEY."
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _unique_user_id() -> str:
    """Return a 24-hex-char ObjectId-shaped string.

    The DR memory service rejects ``participants`` and ``emitter.id`` values
    that aren't valid MongoDB ObjectIds with HTTP 422. For mem0 SaaS this is
    just an arbitrary user identifier.
    """
    return uuid.uuid4().hex[:24]


async def _build_editor(
    config: DRMem0MemoryClientConfig,
) -> AsyncGenerator[DRMem0Editor]:
    """Yield a real editor from the registered NAT factory.

    We use the registered factory (not just ``DRMem0Editor(...)`` directly)
    so the routing logic in :func:`dr_mem0_memory_client` is on the hot
    path — that's the integration we want to validate.
    """
    async with dr_mem0_memory_client(config, object()) as editor:
        assert isinstance(editor, DRMem0Editor)
        yield editor


async def _search_until_visible(
    editor: DRMem0Editor,
    *,
    query: str,
    user_id: str,
    needle: str,
    timeout_s: float = 60.0,
    interval_s: float = 1.5,
) -> list[Any]:
    """Poll ``editor.search`` until ``needle`` appears or the deadline passes.

    Mem0 SaaS defaults to ``async_mode=True`` so ``add`` returns before the
    fact-extraction worker has ingested the memory. The DR endpoint behaves
    similarly. A short poll absorbs that ingest lag without making the test
    rely on fixed sleeps that flake under load.
    """
    deadline = asyncio.get_event_loop().time() + timeout_s
    last: list[Any] = []
    while asyncio.get_event_loop().time() < deadline:
        last = await editor.search(query, top_k=10, user_id=user_id)
        if any(needle in (m.memory or "") for m in last):
            return last
        await asyncio.sleep(interval_s)
    return last


# --------------------------------------------------------------------------- #
# DataRobot Memory Service (live)                                             #
# --------------------------------------------------------------------------- #
@pytest.fixture
def _dr_client() -> Any:
    """Configure the process-global DR SDK client against the test endpoint."""
    import datarobot as dr  # type: ignore[import-not-found]

    dr.Client(
        endpoint=os.environ["DATAROBOT_ENDPOINT"],
        token=os.environ["DATAROBOT_API_TOKEN"],
    )
    return dr


@pytest.fixture
def dr_memory_space(_dr_client: Any) -> Any:
    """Create and tear down a real DataRobot MemorySpace.

    The space is named per-test (uuid suffix) so concurrent runs don't
    collide, and is deleted in the finalizer even when the test fails.
    """
    from datarobot.models.memory import MemorySpace  # type: ignore[import-not-found]

    test_id = uuid.uuid4().hex
    space = MemorySpace.create(description=f"NAT dr-mem0 integration test {test_id}")
    try:
        yield space
    finally:
        space.delete()


@skip_unless_dr
async def test_dr_mem0_endpoint_add_search_round_trip(
    dr_memory_space: Any,
) -> None:
    # GIVEN a real DR memory space and the unified provider configured to
    # route through the path-prefixed DR mem0 endpoint (memory_space_id set,
    # DR token used as the auth credential).
    config = DRMem0MemoryClientConfig(
        api_key=None,
        memory_space_id=dr_memory_space.id,
        datarobot_endpoint=os.environ["DATAROBOT_ENDPOINT"],
        datarobot_api_token=os.environ["DATAROBOT_API_TOKEN"],
    )
    user_id = _unique_user_id()
    secret = f"DR-MEM0-INT-{uuid.uuid4().hex[:12]}"

    # WHEN we add a memory item, then search for it.
    async for editor in _build_editor(config):
        await editor.add_items(
            [
                MemoryItem(
                    conversation=[
                        {"role": "user", "content": f"Please remember my code: {secret}."},
                        {"role": "assistant", "content": "Got it."},
                    ],
                    user_id=user_id,
                    tags=["preference"],
                    metadata={"thread_id": "t-1"},
                )
            ]
        )

        # THEN the stored secret is recoverable through the DR endpoint —
        # proves the editor → Mem0Client → DR mem0 endpoint → MemorySpace
        # pipeline is wired correctly end to end. Poll because mem0 ingests
        # facts asynchronously by default.
        memories = await _search_until_visible(
            editor, query="my code", user_id=user_id, needle=secret
        )
        joined = " ".join(m.memory or "" for m in memories)
        assert secret in joined, (
            f"Expected secret {secret!r} in DR-memory search results within timeout, got: "
            f"{[m.memory for m in memories]!r}"
        )

        # AND deleting by user_id wipes only this user's memories.
        await editor.remove_items(user_id=user_id)
        post_delete = await editor.search("my code", top_k=5, user_id=user_id)
        assert all(secret not in (m.memory or "") for m in post_delete), (
            f"After remove_items(user_id=...), expected no traces of {secret!r}; "
            f"got: {[m.memory for m in post_delete]!r}"
        )


@skip_unless_dr
async def test_dr_mem0_endpoint_isolates_memories_per_user(
    dr_memory_space: Any,
) -> None:
    # GIVEN the provider pointed at the same DR memory space, and two
    # distinct users.
    config = DRMem0MemoryClientConfig(
        api_key=None,
        memory_space_id=dr_memory_space.id,
        datarobot_endpoint=os.environ["DATAROBOT_ENDPOINT"],
        datarobot_api_token=os.environ["DATAROBOT_API_TOKEN"],
    )
    user_a, user_b = _unique_user_id(), _unique_user_id()
    secret_a = f"A-{uuid.uuid4().hex[:10]}"
    secret_b = f"B-{uuid.uuid4().hex[:10]}"

    async for editor in _build_editor(config):
        await editor.add_items(
            [
                MemoryItem(
                    conversation=[{"role": "user", "content": f"My code is {secret_a}."}],
                    user_id=user_a,
                ),
                MemoryItem(
                    conversation=[{"role": "user", "content": f"My code is {secret_b}."}],
                    user_id=user_b,
                ),
            ]
        )

        # WHEN each user searches their own scope (after async ingest lands).
        results_a = await _search_until_visible(
            editor, query="code", user_id=user_a, needle=secret_a
        )
        results_b = await _search_until_visible(
            editor, query="code", user_id=user_b, needle=secret_b
        )

        # THEN each user only sees their own memory — the mem0 v2 filters
        # ({"AND": [{"user_id": ...}]}) are correctly forwarded through the
        # DR endpoint, not collapsed into a global scope.
        text_a = " ".join(m.memory or "" for m in results_a)
        text_b = " ".join(m.memory or "" for m in results_b)
        assert secret_a in text_a and secret_b not in text_a, (
            f"user_a results bled across users: {text_a!r}"
        )
        assert secret_b in text_b and secret_a not in text_b, (
            f"user_b results bled across users: {text_b!r}"
        )

        await editor.remove_items(user_id=user_a)
        await editor.remove_items(user_id=user_b)


# --------------------------------------------------------------------------- #
# Mem0 SaaS (live) — regression after routing changes                         #
# --------------------------------------------------------------------------- #
@skip_unless_mem0
async def test_mem0_saas_add_search_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN the unified provider configured *without* memory_space_id — it
    # must fall through to the Mem0 SaaS endpoint and use MEM0_API_KEY.
    # We also ensure no stale DR creds steer the router elsewhere.
    monkeypatch.delenv("DATAROBOT_ENDPOINT", raising=False)
    monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)

    config = DRMem0MemoryClientConfig(
        api_key=os.environ["MEM0_API_KEY"],
        memory_space_id=None,
    )
    user_id = f"nat-int-{uuid.uuid4().hex[:12]}"
    secret = f"MEM0-SAAS-INT-{uuid.uuid4().hex[:12]}"

    async for editor in _build_editor(config):
        await editor.add_items(
            [
                MemoryItem(
                    conversation=[
                        {"role": "user", "content": f"Remember: my code is {secret}."},
                        {"role": "assistant", "content": "Acknowledged."},
                    ],
                    user_id=user_id,
                    tags=["integration"],
                )
            ]
        )

        try:
            # THEN Mem0 SaaS returns the stored fact — proves the routing
            # changes didn't regress the original SaaS path. Poll because
            # mem0 ingests facts asynchronously by default.
            memories = await _search_until_visible(
                editor, query="code", user_id=user_id, needle=secret
            )
            joined = " ".join(m.memory or "" for m in memories)
            assert secret in joined, (
                f"Expected secret {secret!r} in Mem0 SaaS search results within timeout, got: "
                f"{[m.memory for m in memories]!r}"
            )
        finally:
            await editor.remove_items(user_id=user_id)


@skip_unless_mem0
async def test_mem0_saas_validates_credentials_on_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN a deliberately invalid Mem0 API key.
    monkeypatch.delenv("DATAROBOT_ENDPOINT", raising=False)
    monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)
    config = DRMem0MemoryClientConfig(api_key="m0-this-key-does-not-exist", memory_space_id=None)

    # WHEN we try to build the client, THEN Mem0's _validate_api_key surfaces
    # the auth failure as ValueError (it wraps the HTTPError). This proves
    # the SaaS path is actually contacting Mem0, not silently returning a
    # broken client.
    with pytest.raises((ValueError, RuntimeError)):
        async for _editor in _build_editor(config):
            pass
