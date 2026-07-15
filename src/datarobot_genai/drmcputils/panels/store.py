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

"""Server-side panel store.

Persists panels through a :class:`~datarobot_genai.drmcputils.files.store.BlobStore`:
each panel is a small JSON *manifest* blob (the panel metadata) tagged for
discovery, plus an optional separate *payload* blob for bulky content (a
Dataset's Parquet, a Chart's spec). The manifest blob's id is the panel id, so
no separate id allocation is needed and ids are globally unique.

**Conversation scoping.** A store may be scoped to a conversation
(``PanelStore(blobs, conversation_id=...)``). Scoped panels are stored under a
folder-style name prefix ``panels/<conversation_id>/`` (the Files registry has
no real directories, so the structured name is the hierarchy) and tagged
``dr_panel_conversation:<conversation_id>`` so ``list`` returns only the
current conversation's panels. Unscoped stores keep the legacy behavior: blobs
under ``panels/`` and a global list view, so consumers without conversations
(and panels created before scoping existed) keep working. ``get``/``delete``
are id-based and never scoped. Staging vs main separation stays tag-based
(``dr_panel_source:<source>``), and ``move`` promotes a panel between sources
by retagging in place, preserving the panel id.

The store depends only on the ``BlobStore`` Protocol, so it is backed by the
DataRobot Files API in production and by an in-memory fake in tests.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re

from datarobot_genai.drmcputils.files.store import BlobStore
from datarobot_genai.drmcputils.panels.models import BasePanel
from datarobot_genai.drmcputils.panels.models import Panel
from datarobot_genai.drmcputils.panels.models import panel_from_manifest

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = "main"
DEFAULT_LIST_LIMIT = 100

# HTTP header carrying the conversation id (same header the previous
# panel-library MCP server used, so existing clients keep working unchanged).
CONVERSATION_ID_HEADER = "x-datarobot-conversation-id"

# NOTE: the DataRobot Files API rejects tags containing "-" (422), so panel tags
# use "_" as the word separator. ":" is permitted and used for key:value tags.
# Conversation ids are normalized to [0-9A-Za-z_] for the same reason.
_MANIFEST_TAG = "dr_panel"
_PAYLOAD_TAG = "dr_panel_payload"

# Folder-style name prefix: the Files registry renders no hierarchy of its own,
# so panel blobs are named "panels/<conversation_id>/<file>" (scoped) or
# "panels/<file>" (unscoped) to keep the registry page structured.
_NAME_ROOT = "panels"

_CONVERSATION_ID_MAX_LENGTH = 128
_CONVERSATION_ID_SANITIZE = re.compile(r"[^0-9A-Za-z_]")


def normalize_conversation_id(conversation_id: str | None) -> str | None:
    """Normalize a raw conversation id to a Files-API-safe token.

    Every character outside ``[0-9A-Za-z_]`` becomes ``_`` (the Files API
    rejects tags containing ``-``), and the result is capped at 128 characters.
    Returns ``None`` for missing/blank ids (an unscoped store).
    """
    if not conversation_id:
        return None
    cleaned = _CONVERSATION_ID_SANITIZE.sub("_", conversation_id.strip())
    cleaned = cleaned[:_CONVERSATION_ID_MAX_LENGTH]
    return cleaned or None


def _source_tag(source: str) -> str:
    return f"dr_panel_source:{source}"


def _conversation_tag(conversation_id: str) -> str:
    return f"dr_panel_conversation:{conversation_id}"


def _type_tag(panel: BasePanel) -> str:
    return f"dr_panel_type:{panel.type.value}"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


class PanelStore:
    """CRUD + listing for panels over a :class:`BlobStore`.

    ``conversation_id`` (optional) scopes the store to one conversation: blobs
    it creates are placed under ``panels/<conversation_id>/`` and ``list``
    returns only that conversation's panels. ``None`` keeps the legacy
    unscoped behavior (global list view).
    """

    def __init__(self, blob_store: BlobStore, *, conversation_id: str | None = None) -> None:
        self._blobs = blob_store
        self._conversation_id = normalize_conversation_id(conversation_id)

    @property
    def conversation_id(self) -> str | None:
        """The normalized conversation id this store is scoped to (None = unscoped)."""
        return self._conversation_id

    def _name(self, filename: str, *, conversation_id: str | None = None) -> str:
        conversation_id = conversation_id or self._conversation_id
        if conversation_id:
            return f"{_NAME_ROOT}/{conversation_id}/{filename}"
        return f"{_NAME_ROOT}/{filename}"

    def _manifest_tags(self, panel: BasePanel, source: str) -> list[str]:
        tags = [_MANIFEST_TAG, _source_tag(source), _type_tag(panel)]
        if panel.conversation_id:
            tags.append(_conversation_tag(panel.conversation_id))
        return tags

    def _payload_tags(self, panel: BasePanel, source: str) -> list[str]:
        tags = [_PAYLOAD_TAG, _source_tag(source)]
        if panel.conversation_id:
            tags.append(_conversation_tag(panel.conversation_id))
        return tags

    async def create(
        self,
        panel: BasePanel,
        *,
        source: str = DEFAULT_SOURCE,
        payload: bytes | None = None,
        payload_name: str | None = None,
        content_type: str | None = None,
    ) -> Panel:
        """Persist ``panel`` (and an optional ``payload`` blob); returns it with ``id`` set."""
        panel.updated_at = _now_iso()
        panel.conversation_id = self._conversation_id
        if payload is not None:
            payload_ref = await self._blobs.put(
                payload,
                name=self._name(payload_name or f"{panel.type.value}-payload"),
                content_type=content_type,
                tags=self._payload_tags(panel, source),
            )
            panel.payload_files_id = payload_ref.files_id
            panel.payload_name = payload_ref.name

        manifest = json.dumps(panel.model_dump(mode="json", exclude={"id"})).encode("utf-8")
        try:
            manifest_ref = await self._blobs.put(
                manifest,
                name=self._name(f"panel-{panel.type.value}.json"),
                content_type="application/json",
                tags=self._manifest_tags(panel, source),
            )
        except Exception:
            # The panel was not created; don't leave the payload blob orphaned.
            if panel.payload_files_id:
                try:
                    await self._blobs.delete(panel.payload_files_id)
                except Exception:  # noqa: BLE001 - cleanup is best-effort; surface the original error
                    logger.warning(
                        "Failed to clean up payload blob %s after manifest write failed",
                        panel.payload_files_id,
                    )
                panel.payload_files_id = None
                panel.payload_name = None
            raise
        panel.id = manifest_ref.files_id
        return panel  # type: ignore[return-value]

    async def get(self, panel_id: str) -> Panel:
        """Load a panel by id (its manifest blob id). Payload is not hydrated here.

        Id-based access is intentionally not conversation-scoped: panel ids are
        globally unique blob ids, and panels created before conversation
        scoping existed must stay reachable.
        """
        raw = await self._blobs.get(panel_id)
        manifest = json.loads(raw.decode("utf-8"))
        panel = panel_from_manifest(manifest)
        panel.id = panel_id
        return panel

    async def list(
        self,
        *,
        source: str = DEFAULT_SOURCE,
        limit: int = DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> list[Panel]:
        """List panels in ``source`` (metadata only); page with ``limit``/``offset``.

        A conversation-scoped store lists only its conversation's panels; an
        unscoped store keeps the legacy global view across conversations.
        """
        tags = [_MANIFEST_TAG, _source_tag(source)]
        if self._conversation_id:
            tags.append(_conversation_tag(self._conversation_id))
        refs = await self._blobs.list(tags=tags, limit=limit, offset=offset)
        panels: list[Panel] = []
        for ref in refs:
            raw = await self._blobs.get(ref.files_id)
            try:
                panel = panel_from_manifest(json.loads(raw.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError) as exc:
                # One corrupt manifest must not break listing the healthy ones.
                logger.warning("Skipping unreadable panel manifest %s: %s", ref.files_id, exc)
                continue
            panel.id = ref.files_id
            panels.append(panel)
        return panels

    async def move(self, panel_id: str, *, to_source: str) -> Panel:
        """Move a panel to ``to_source`` (e.g. promote staging→main), preserving its id.

        The move is an in-place retag of the manifest (and payload) blobs — no
        copy, no delete — so the panel id and any external references to it
        stay valid. The panel's own conversation scope (recorded on create) is
        kept regardless of the scope of the store performing the move.
        """
        panel = await self.get(panel_id)
        await self._blobs.set_tags(panel_id, self._manifest_tags(panel, to_source))
        if panel.payload_files_id:
            await self._blobs.set_tags(panel.payload_files_id, self._payload_tags(panel, to_source))
        return panel

    async def get_payload(self, panel: Panel | str) -> bytes | None:
        """Fetch a panel's payload blob bytes (by id or loaded panel); None if it has none."""
        if isinstance(panel, str):
            panel = await self.get(panel)
        if not panel.payload_files_id:
            return None
        return await self._blobs.get(panel.payload_files_id)

    async def delete(self, panel_id: str) -> None:
        """Delete a panel's manifest and its payload blob (if any).

        Transient fetch errors propagate (so the caller can retry without
        orphaning the payload); only an unreadable/corrupt manifest falls back
        to deleting the manifest alone, since the payload id is unknowable.
        """
        payload_files_id: str | None = None
        try:
            payload_files_id = (await self.get(panel_id)).payload_files_id
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError) as exc:
            logger.warning(
                "Panel %s manifest is unreadable (%s); deleting the manifest only", panel_id, exc
            )
        if payload_files_id:
            await self._blobs.delete(payload_files_id)
        await self._blobs.delete(panel_id)
