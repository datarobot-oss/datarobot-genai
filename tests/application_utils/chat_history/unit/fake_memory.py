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

"""A stateful, respx-backed in-memory fake of the Memory Service REST API.

Registers a single catch-all route (under the test memory space) and services
session / event CRUD against in-process dicts.  It reproduces the wire
behaviours the repositories rely on: idempotent create by ``deduplicationKey``
(409 → adopt), the case-insensitive ``description`` substring filter, participant
scoping, monotonic ``sequenceId`` addressing, and the optimistic-concurrency
tokens (session ``If-Match`` version, event ``createdAt``).  Conflicts can be
injected to drive the bounded-retry paths.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import httpx
import respx

from datarobot_genai.application_utils.persistence import DRMemoryServiceClient
from datarobot_genai.application_utils.persistence import DRMemorySpace

BASE = "https://app.datarobot.com/api/v2"
MEMORY_BASE = f"{BASE}/memory"
SPACE_ID = "aaaaaaaa-0000-0000-0000-000000000001"


class FakeMemoryService:
    """In-memory Memory Service fake wired into the active ``respx`` router."""

    def __init__(self, space_id: str = SPACE_ID) -> None:
        self.space_id = space_id
        self.sessions: dict[str, dict[str, Any]] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}
        self._sid_counter = 0
        self._seq_counter = 0
        self._clock = 0
        #: Number of subsequent event PATCHes to fail with a stale-token 422.
        self.fail_event_patch_times = 0
        #: Number of subsequent session PATCHes to fail with a version-conflict 409.
        self.fail_session_patch_times = 0
        #: When True, fail every ``EntityLocator`` (``//loc/``) session create with a
        #: 500 — simulates a replica that cannot write the best-effort index.
        self.fail_locator_write = False
        #: Simple request-count telemetry keyed by ``"METHOD path"``.
        self.calls: list[str] = []

    # ── Wiring ─────────────────────────────────────────────────────────────

    def install(self) -> None:
        """Register the catch-all route on the currently-active respx router."""
        pattern = rf"{re.escape(MEMORY_BASE)}/{re.escape(self.space_id)}/.*"
        respx.route(url__regex=pattern).mock(side_effect=self._dispatch)

    def client(self) -> DRMemoryServiceClient:
        """Return a client pinned at the fake endpoint."""
        return DRMemoryServiceClient(
            endpoint=BASE, api_token="test-token", http_client=httpx.AsyncClient()
        )

    def space(self) -> DRMemorySpace:
        """Return a minimally-hydrated space bound to this fake."""
        return DRMemorySpace._from_wire(
            self.client(),
            {"memorySpaceId": self.space_id, "userId": "u", "tenantId": "t", "createdAt": ""},
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _now(self) -> str:
        self._clock += 1
        return f"2026-06-30T00:00:00.{self._clock:06d}Z"

    def event_count(self, session_id: str) -> int:
        """Return the number of events stored under a session."""
        return len(self.events.get(session_id, []))

    # ── Dispatch ───────────────────────────────────────────────────────────

    def _dispatch(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(f"{request.method} {request.url.path}")
        marker = f"/memory/{self.space_id}/"
        rel = request.url.path.split(marker, 1)[1] if marker in request.url.path else ""
        parts = [p for p in rel.split("/") if p]

        handler = self._route(request.method, parts[1:]) if parts[:1] == ["sessions"] else None
        if handler is None:
            return httpx.Response(
                404, json={"detail": f"unhandled {request.method} {request.url.path}"}
            )
        return handler(request)

    def _route(
        self, method: str, rest: list[str]
    ) -> Callable[[httpx.Request], httpx.Response] | None:
        """Resolve a ``(method, path-tail)`` pair to a bound handler, or ``None``."""
        if len(rest) == 0:
            return {"POST": self._create_session, "GET": self._list_sessions}.get(method)
        if len(rest) == 1:
            sid = rest[0]
            return {
                "GET": lambda _req: self._get_session(sid),
                "PATCH": lambda req: self._patch_session(sid, req),
                "DELETE": lambda _req: self._delete_session(sid),
            }.get(method)
        if len(rest) == 2 and rest[1] == "events":
            sid = rest[0]
            return {
                "POST": lambda req: self._create_event(sid, req),
                "GET": lambda req: self._list_events(sid, req),
            }.get(method)
        if len(rest) == 3 and rest[1] == "events":
            sid, seq = rest[0], int(rest[2])
            return {
                "PATCH": lambda req: self._patch_event(sid, seq, req),
                "DELETE": lambda _req: self._delete_event(sid, seq),
            }.get(method)
        return None

    # ── Sessions ───────────────────────────────────────────────────────────

    def _create_session(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if self.fail_locator_write and (payload.get("description") or "") == "//loc/":
            return httpx.Response(500, json={"detail": "injected locator write failure"})
        dedup = payload.get("deduplicationKey")
        if dedup is not None:
            for sid, stored in self.sessions.items():
                if stored.get("deduplicationKey") == dedup:
                    return httpx.Response(
                        409,
                        json={"errorName": "DeduplicationConflict", "existingSessionId": sid},
                    )
        self._sid_counter += 1
        sid = f"session-{self._sid_counter:04d}"
        wire = {
            "id": sid,
            "participants": payload.get("participants", []),
            "description": payload.get("description"),
            "deduplicationKey": dedup,
            "metadata": payload.get("metadata", {}),
            "version": 1,
            "createdAt": self._now(),
        }
        self.sessions[sid] = wire
        self.events[sid] = []
        return httpx.Response(201, json=wire)

    def _get_session(self, sid: str) -> httpx.Response:
        stored = self.sessions.get(sid)
        if stored is None:
            return httpx.Response(404, json={"detail": "session not found"})
        return httpx.Response(200, json=stored)

    def _list_sessions(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        dedup = params.get("deduplicationKey")
        description = params.get("description")
        participant = params.get("participants")
        offset = int(params.get("offset") or 0)
        limit = int(params.get("limit") or 100)

        matches: list[dict[str, Any]] = []
        for stored in self.sessions.values():
            if dedup is not None and stored.get("deduplicationKey") != dedup:
                continue
            if description is not None:
                if description.lower() not in (stored.get("description") or "").lower():
                    continue
            if participant is not None and participant not in (stored.get("participants") or []):
                continue
            matches.append(stored)

        matches.sort(key=lambda s: s["id"])
        page = matches[offset : offset + limit]
        return httpx.Response(200, json={"items": page, "total": len(matches)})

    def _patch_session(self, sid: str, request: httpx.Request) -> httpx.Response:
        stored = self.sessions.get(sid)
        if stored is None:
            return httpx.Response(404, json={"detail": "session not found"})
        if self.fail_session_patch_times > 0:
            self.fail_session_patch_times -= 1
            return httpx.Response(409, json={"detail": "session version conflict"})
        if_match = request.headers.get("If-Match")
        if if_match is not None and str(stored["version"]) != str(if_match):
            return httpx.Response(409, json={"detail": "stale If-Match"})
        payload = json.loads(request.content)
        if "metadata" in payload:
            stored["metadata"] = payload["metadata"]
        if "description" in payload:
            stored["description"] = payload["description"]
        stored["version"] = int(stored["version"]) + 1
        return httpx.Response(200, json=stored)

    def _delete_session(self, sid: str) -> httpx.Response:
        self.sessions.pop(sid, None)
        self.events.pop(sid, None)
        return httpx.Response(204)

    # ── Events ─────────────────────────────────────────────────────────────

    def _create_event(self, sid: str, request: httpx.Request) -> httpx.Response:
        if sid not in self.sessions:
            return httpx.Response(404, json={"detail": "session not found"})
        payload = json.loads(request.content)
        emitter = payload.get("emitter", {})
        self._seq_counter += 1
        wire = {
            "sequenceId": self._seq_counter,
            "createdAt": self._now(),
            "eventType": payload.get("type", "message"),
            "emitterType": emitter.get("type", "agent"),
            "emitterId": emitter.get("id"),
            "body": payload.get("body", {}),
        }
        self.events[sid].append(wire)
        return httpx.Response(201, json=wire)

    def _list_events(self, sid: str, request: httpx.Request) -> httpx.Response:
        if sid not in self.sessions:
            return httpx.Response(404, json={"detail": "session not found"})
        params = request.url.params
        event_type = params.get("eventType")
        events = list(self.events.get(sid, []))
        if event_type is not None:
            events = [e for e in events if e["eventType"] == event_type]

        last_n = params.get("lastN")
        if last_n is not None:
            n = int(last_n)
            page = events[-n:] if n > 0 else []
            return httpx.Response(200, json={"items": page})

        offset = int(params.get("offset") or 0)
        limit = int(params.get("limit") or 100)
        page = events[offset : offset + limit]
        return httpx.Response(200, json={"items": page})

    def _find_event(self, sid: str, seq: int) -> dict[str, Any] | None:
        for event in self.events.get(sid, []):
            if event["sequenceId"] == seq:
                return event
        return None

    def _patch_event(self, sid: str, seq: int, request: httpx.Request) -> httpx.Response:
        event = self._find_event(sid, seq)
        if event is None:
            return httpx.Response(404, json={"detail": "event not found"})
        if self.fail_event_patch_times > 0:
            self.fail_event_patch_times -= 1
            return httpx.Response(422, json={"detail": "Patch of incorrect version of event"})
        token = request.url.params.get("createdAt")
        if token is not None and token != event["createdAt"]:
            return httpx.Response(422, json={"detail": "Patch of incorrect version of event"})
        payload = json.loads(request.content)
        if "body" in payload:
            event["body"] = payload["body"]
        if "emitter" in payload:
            event["emitterType"] = payload["emitter"].get("type", event["emitterType"])
            event["emitterId"] = payload["emitter"].get("id")
        event["createdAt"] = self._now()
        return httpx.Response(200, json=event)

    def _delete_event(self, sid: str, seq: int) -> httpx.Response:
        events = self.events.get(sid, [])
        self.events[sid] = [e for e in events if e["sequenceId"] != seq]
        return httpx.Response(204)
