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
each panel is a small JSON *manifest* blob (the panel metadata) plus an
optional separate *payload* blob for bulky content (a Dataset's Parquet, a
Chart's spec). Panel ids are client-generated (``uuid4().hex``) and embedded in
the blob path, so a panel keeps its id across moves.

**Layout.** All panels live in one shared Files container under
``<source>/<scope>/`` paths, where *scope* is the normalized conversation id
(``_shared`` for unscoped consumers)::

    panels                            <- the shared container (one registry row)
    ├── staging/
    │   └── <conversation_id>/
    │       ├── <panel_id>.json       (manifest)
    │       └── <panel_id>.payload    (optional payload blob)
    └── main/
        └── _shared/
            └── <panel_id>.json

Source-first ordering keeps every listing a single server-side prefix query: a
conversation-scoped store lists ``<source>/<conversation_id>/`` and an unscoped
store keeps the legacy global view via ``<source>/``.

**Conversation scoping.** A store may be scoped to a conversation
(``PanelStore(blobs, conversation_id=...)``); the shared store factory resolves
the id per request from the ``x-datarobot-conversation-id`` header (the same
header the previous panel-library MCP server used). Conversation ids are
normalized to ``[0-9A-Za-z_]`` and capped at 128 chars so they are safe as path
segments. ``get``/``delete`` are id-based and never scoped.

**Moves.** ``move`` promotes a panel between sources (staging→main) by renaming
its blobs' paths in place — no copy, no delete — so the panel id and external
references stay valid. The panel's own conversation scope (recorded on create)
is kept regardless of the mover's scope.

**Legacy panels.** Panels stored before the shared-container layout (one
standalone Files container per blob, tag-based discovery) stay reachable —
``get``/``get_payload``/``delete`` fall back to the blob id when the panel is
not found in the shared container. They no longer appear in listings and
cannot be moved (recreate them instead).

The store depends only on the ``BlobStore`` Protocol, so it is backed by the
DataRobot Files API in production and by an in-memory fake in tests.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import uuid

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
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

# Path segment for panels created without a conversation ("_" keeps it out of
# the normalized-conversation-id alphabet's typical values while staying
# path- and tag-safe).
_SHARED_SCOPE = "_shared"

_MANIFEST_SUFFIX = ".json"
_PAYLOAD_SUFFIX = ".payload"

_CONVERSATION_ID_MAX_LENGTH = 128
_CONVERSATION_ID_SANITIZE = re.compile(r"[^0-9A-Za-z_]")

# Sources and panel ids become path segments; keep them to a safe alphabet so
# no caller-controlled value can smuggle separators into a blob path.
_SOURCE_RE = re.compile(r"^[0-9A-Za-z_]{1,64}$")
_PANEL_ID_RE = re.compile(r"^[0-9A-Za-z_-]{1,128}$")


def normalize_conversation_id(conversation_id: str | None) -> str | None:
    """Normalize a raw conversation id to a path- and tag-safe token.

    Every character outside ``[0-9A-Za-z_]`` becomes ``_`` and the result is
    capped at 128 characters. Returns ``None`` for missing/blank ids (an
    unscoped store).
    """
    if not conversation_id:
        return None
    cleaned = _CONVERSATION_ID_SANITIZE.sub("_", conversation_id.strip())
    cleaned = cleaned[:_CONVERSATION_ID_MAX_LENGTH]
    return cleaned or None


def _validate_source(source: str) -> None:
    if not _SOURCE_RE.fullmatch(source or ""):
        raise ToolError(
            "source must be 1-64 characters of [0-9A-Za-z_] (e.g. 'main' or 'staging')",
            kind=ToolErrorKind.VALIDATION,
        )


def _validate_panel_id(panel_id: str) -> None:
    if not _PANEL_ID_RE.fullmatch(panel_id or ""):
        raise ToolError(
            "panel_id must be 1-128 characters of [0-9A-Za-z_-]",
            kind=ToolErrorKind.VALIDATION,
        )


def _payload_path(manifest_path: str) -> str:
    return manifest_path[: -len(_MANIFEST_SUFFIX)] + _PAYLOAD_SUFFIX


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


class PanelStore:
    """CRUD + listing for panels over a :class:`BlobStore`.

    ``conversation_id`` (optional) scopes the store to one conversation: blobs
    it creates are placed under ``<source>/<conversation_id>/`` and ``list``
    returns only that conversation's panels. ``None`` keeps the legacy
    unscoped behavior (blobs under ``<source>/_shared/``, global list view).
    """

    def __init__(self, blob_store: BlobStore, *, conversation_id: str | None = None) -> None:
        self._blobs = blob_store
        self._conversation_id = normalize_conversation_id(conversation_id)

    @property
    def conversation_id(self) -> str | None:
        """The normalized conversation id this store is scoped to (None = unscoped)."""
        return self._conversation_id

    def _scope_segment(self) -> str:
        return self._conversation_id or _SHARED_SCOPE

    def _manifest_path(self, source: str, panel_id: str) -> str:
        return f"{source}/{self._scope_segment()}/{panel_id}{_MANIFEST_SUFFIX}"

    async def _locate(self, panel_id: str) -> str | None:
        """Resolve a panel id to its manifest path in the shared container.

        Ids are embedded in blob paths, not derivable from them alone (the
        source/scope segments are unknown for a bare id), so resolution is one
        container-wide listing plus a suffix match. Returns ``None`` when the
        id is not in the shared container (unknown, or a legacy panel).
        """
        refs = await self._blobs.list(limit=0)
        suffix = f"/{panel_id}{_MANIFEST_SUFFIX}"
        for ref in refs:
            if ref.path.endswith(suffix):
                return ref.path
        return None

    @staticmethod
    def _panel_from_raw(raw: bytes, panel_id: str, manifest_path: str | None) -> Panel:
        panel = panel_from_manifest(json.loads(raw.decode("utf-8")))
        panel.id = panel_id
        if manifest_path and panel.payload_files_id:
            panel.payload_path = _payload_path(manifest_path)
        return panel

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
        _validate_source(source)
        panel.updated_at = _now_iso()
        panel.conversation_id = self._conversation_id
        panel_id = uuid.uuid4().hex
        manifest_path = self._manifest_path(source, panel_id)

        payload_ref = None
        if payload is not None:
            payload_ref = await self._blobs.put(
                payload,
                path=_payload_path(manifest_path),
                content_type=content_type,
            )
            panel.payload_files_id = payload_ref.container_id
            panel.payload_name = payload_name or f"{panel.type.value}-payload"
            panel.payload_path = payload_ref.path

        # id and payload_path are excluded: both are derived from the blob
        # paths on load, so a move (path rename) never has to rewrite the
        # manifest.
        manifest = json.dumps(panel.model_dump(mode="json", exclude={"id", "payload_path"})).encode(
            "utf-8"
        )
        try:
            await self._blobs.put(
                manifest,
                path=manifest_path,
                content_type="application/json",
            )
        except Exception:
            # The panel was not created; don't leave the payload blob orphaned.
            if payload_ref is not None:
                try:
                    await self._blobs.delete(payload_ref.path)
                except Exception:  # noqa: BLE001 - cleanup is best-effort; surface the original error
                    logger.warning(
                        "Failed to clean up payload blob %s after manifest write failed",
                        payload_ref.path,
                    )
                panel.payload_files_id = None
                panel.payload_name = None
                panel.payload_path = None
            raise
        panel.id = panel_id
        return panel  # type: ignore[return-value]

    async def get(self, panel_id: str) -> Panel:
        """Load a panel by id. Payload is not hydrated here.

        Id-based access is intentionally not conversation-scoped: panel ids are
        globally unique, and panels created before conversation scoping existed
        (legacy standalone blobs) must stay reachable.
        """
        _validate_panel_id(panel_id)
        manifest_path = await self._locate(panel_id)
        raw = await self._blobs.get(manifest_path or panel_id)
        return self._panel_from_raw(raw, panel_id, manifest_path)

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
        _validate_source(source)
        if self._conversation_id:
            prefix = f"{source}/{self._conversation_id}/"
        else:
            prefix = f"{source}/"
        # One prefix query, then page over manifests only (payload blobs share
        # the prefix), so limit/offset always count panels.
        refs = await self._blobs.list(prefix=prefix, limit=0)
        manifest_refs = sorted(
            (ref for ref in refs if ref.path.endswith(_MANIFEST_SUFFIX)),
            key=lambda ref: ref.path,
        )
        page = manifest_refs[offset : offset + limit] if limit else manifest_refs[offset:]
        panels: list[Panel] = []
        for ref in page:
            panel_id = ref.path.rsplit("/", 1)[-1][: -len(_MANIFEST_SUFFIX)]
            raw = await self._blobs.get(ref.path)
            try:
                panel = self._panel_from_raw(raw, panel_id, ref.path)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError) as exc:
                # One corrupt manifest must not break listing the healthy ones.
                logger.warning("Skipping unreadable panel manifest %s: %s", ref.path, exc)
                continue
            panels.append(panel)
        return panels

    async def move(self, panel_id: str, *, to_source: str) -> Panel:
        """Move a panel to ``to_source`` (e.g. promote staging→main), preserving its id.

        The move renames the manifest (and payload) blob paths in place — no
        copy, no delete — so the panel id and any external references to it
        stay valid. The panel's own conversation scope (recorded on create) is
        kept regardless of the scope of the store performing the move.
        """
        _validate_panel_id(panel_id)
        _validate_source(to_source)
        manifest_path = await self._locate(panel_id)
        if manifest_path is None:
            raise ToolError(
                f"Panel '{panel_id}' was not found in the shared panels container. "
                "Panels stored before the shared-container layout cannot be moved; "
                "recreate the panel instead.",
                kind=ToolErrorKind.NOT_FOUND,
            )
        raw = await self._blobs.get(manifest_path)
        panel = self._panel_from_raw(raw, panel_id, manifest_path)
        # The scope segment is the panel's own (from its current path), not the
        # mover's: an unscoped/admin mover must not re-home the panel.
        scope = manifest_path.split("/", 2)[1]
        target_path = f"{to_source}/{scope}/{panel_id}{_MANIFEST_SUFFIX}"
        if target_path == manifest_path:
            return panel
        if panel.payload_files_id:
            await self._blobs.move(_payload_path(manifest_path), _payload_path(target_path))
        try:
            await self._blobs.move(manifest_path, target_path)
        except Exception:
            # Keep payload and manifest co-located so payload-path derivation
            # stays valid for the (unmoved) panel.
            if panel.payload_files_id:
                try:
                    await self._blobs.move(_payload_path(target_path), _payload_path(manifest_path))
                except Exception:  # noqa: BLE001 - rollback is best-effort; surface the original error
                    logger.warning(
                        "Failed to move payload back for panel %s after manifest move failed",
                        panel_id,
                    )
            raise
        if panel.payload_files_id:
            panel.payload_path = _payload_path(target_path)
        return panel

    async def get_payload(self, panel: Panel | str) -> bytes | None:
        """Fetch a panel's payload blob bytes (by id or loaded panel); None if it has none."""
        if isinstance(panel, str):
            panel = await self.get(panel)
        if panel.payload_path:
            return await self._blobs.get(panel.payload_path)
        if panel.payload_files_id:
            # Legacy panel: the payload is its own standalone container.
            return await self._blobs.get(panel.payload_files_id)
        return None

    async def delete(self, panel_id: str) -> None:
        """Delete a panel's manifest and its payload blob (if any)."""
        _validate_panel_id(panel_id)
        manifest_path = await self._locate(panel_id)
        if manifest_path is not None:
            # One batch delete; a missing payload path is silently ignored.
            await self._blobs.delete([manifest_path, _payload_path(manifest_path)])
            return
        # Legacy panel: standalone containers for manifest and payload. Read
        # the manifest for the payload id first; transient fetch errors
        # propagate (so the caller can retry without orphaning the payload),
        # while an unreadable/corrupt manifest falls back to deleting the
        # manifest alone, since the payload id is unknowable.
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
