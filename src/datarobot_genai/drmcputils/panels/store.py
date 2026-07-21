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
segments. Scoping is enforced on every id-based operation, not just listings:
a scoped store resolves ids only against its own conversation and ``_shared``
(other conversations' panels are invisible — not-found, no existence leak), it
can read ``_shared`` panels but not delete or move them, and resolution probes
exact manifest paths (own scope, then ``_shared``, across the hinted + known
sources) so the hot path never lists the whole container. An unscoped store
keeps the global view and may modify anything.

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

# The sources panels are created in and promoted between. Scoped id resolution
# probes these (plus any caller-supplied source hint) with exact-path checks;
# exotic sources are still reachable through the listing fallback.
_KNOWN_SOURCES = (DEFAULT_SOURCE, "staging")

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

    async def _dir_contains(self, directory: str, path: str) -> bool:
        """Report whether ``path`` exists under ``directory`` (one bounded prefix query).

        The Files API only accepts *directory* prefixes (must end with ``/``;
        an exact file path is rejected with a 400), so existence probes list
        the parent directory — bounded by one scope of one source — and match
        the exact path client-side.
        """
        refs = await self._blobs.list(prefix=directory, limit=0)
        return any(ref.path == path for ref in refs)

    async def _locate(self, panel_id: str, *, source: str | None = None) -> str | None:
        """Resolve a panel id to its manifest path in the shared container.

        A scoped store resolves only against its own conversation and
        ``_shared`` — other conversations' panels are invisible. Resolution
        probes the ``<source>/<scope>/`` directories under the hinted + known
        sources (each one bounded prefix listing); a container-wide listing
        runs only for exotic sources, filtered to the readable scopes.
        ``source`` is a hint, not a filter: a stale hint (e.g. after a
        promote) still resolves. An unscoped store keeps the global view
        (prefix listing per source, full listing without one).

        Returns ``None`` when the id is not in the shared container (unknown,
        other conversation, or a legacy panel).
        """
        suffix = f"/{panel_id}{_MANIFEST_SUFFIX}"
        if self._conversation_id:
            scopes = (self._conversation_id, _SHARED_SCOPE)
            sources = dict.fromkeys((source, *_KNOWN_SOURCES) if source else _KNOWN_SOURCES)
            for src in sources:
                for scope in scopes:
                    directory = f"{src}/{scope}/"
                    path = f"{directory}{panel_id}{_MANIFEST_SUFFIX}"
                    if await self._dir_contains(directory, path):
                        return path
            refs = await self._blobs.list(limit=0)
            return next(
                (
                    ref.path
                    for ref in refs
                    if ref.path.endswith(suffix) and ref.path.split("/", 2)[1] in scopes
                ),
                None,
            )
        prefix = f"{source}/" if source else None
        refs = await self._blobs.list(prefix=prefix, limit=0)
        located = next((ref.path for ref in refs if ref.path.endswith(suffix)), None)
        if located is None and source:
            # The hint missed (e.g. the panel was promoted since): global view.
            refs = await self._blobs.list(limit=0)
            located = next((ref.path for ref in refs if ref.path.endswith(suffix)), None)
        return located

    def _ensure_writable(self, manifest_path: str, panel_id: str) -> None:
        """Reject writes to panels outside the store's own conversation.

        Only ``_shared`` panels can be located outside the own scope (foreign
        conversations are invisible to ``_locate``), so this is specifically
        the scoped-consumer-touching-shared-panels guard.
        """
        if not self._conversation_id:
            return
        if manifest_path.split("/", 2)[1] == self._conversation_id:
            return
        raise ToolError(
            f"Panel '{panel_id}' is shared, not owned by this conversation; "
            "shared panels can only be modified by an unscoped consumer.",
            kind=ToolErrorKind.VALIDATION,
        )

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

    async def get(self, panel_id: str, *, source: str | None = None) -> Panel:
        """Load a panel by id. Payload is not hydrated here.

        A scoped store reads its own conversation's panels plus ``_shared``
        ones; other conversations' panels are not found. ``source`` is an
        optional resolution hint (O(1) direct probe when it matches). Panels
        created before conversation scoping existed (legacy standalone blobs)
        stay reachable by id.
        """
        _validate_panel_id(panel_id)
        if source is not None:
            _validate_source(source)
        manifest_path = await self._locate(panel_id, source=source)
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

    async def move(self, panel_id: str, *, to_source: str, source: str | None = None) -> Panel:
        """Move a panel to ``to_source`` (e.g. promote staging→main), preserving its id.

        The move renames the manifest (and payload) blob paths in place — no
        copy, no delete — so the panel id and any external references to it
        stay valid. The panel's own conversation scope (recorded on create) is
        kept regardless of the scope of the store performing the move. A
        scoped store may only move its own conversation's panels; ``source``
        is an optional resolution hint.
        """
        _validate_panel_id(panel_id)
        _validate_source(to_source)
        if source is not None:
            _validate_source(source)
        manifest_path = await self._locate(panel_id, source=source)
        if manifest_path is None:
            raise ToolError(
                f"Panel '{panel_id}' was not found in the shared panels container. "
                "Panels stored before the shared-container layout cannot be moved; "
                "recreate the panel instead.",
                kind=ToolErrorKind.NOT_FOUND,
            )
        self._ensure_writable(manifest_path, panel_id)
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

    async def get_payload(self, panel: Panel | str, *, source: str | None = None) -> bytes | None:
        """Fetch a panel's payload blob bytes (by id or loaded panel); None if it has none."""
        if isinstance(panel, str):
            panel = await self.get(panel, source=source)
        if panel.payload_path:
            return await self._blobs.get(panel.payload_path)
        if panel.payload_files_id:
            # Legacy panel: the payload is its own standalone container.
            return await self._blobs.get(panel.payload_files_id)
        return None

    async def delete(self, panel_id: str, *, source: str | None = None) -> None:
        """Delete a panel's manifest and its payload blob (if any).

        A scoped store may only delete its own conversation's panels: shared
        panels are rejected, other conversations' panels are not found.
        ``source`` is an optional resolution hint.
        """
        _validate_panel_id(panel_id)
        if source is not None:
            _validate_source(source)
        manifest_path = await self._locate(panel_id, source=source)
        if manifest_path is not None:
            # Reject before the legacy fallback below: for shared-layout
            # panels payload_files_id is the shared container itself, which
            # the fallback would delete wholesale.
            self._ensure_writable(manifest_path, panel_id)
            # One batch delete; a missing payload path is silently ignored.
            await self._blobs.delete([manifest_path, _payload_path(manifest_path)])
            return
        # Legacy panel: standalone containers for manifest and payload. Read
        # the manifest for the payload id first; fetch errors — including
        # not-found, so an unknown or other-conversation id never reports a
        # successful delete — propagate (the caller can retry without
        # orphaning the payload), while an unreadable/corrupt manifest falls
        # back to deleting the manifest alone, since the payload id is
        # unknowable.
        raw = await self._blobs.get(panel_id)
        payload_files_id: str | None = None
        try:
            payload_files_id = self._panel_from_raw(raw, panel_id, None).payload_files_id
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError) as exc:
            logger.warning(
                "Panel %s manifest is unreadable (%s); deleting the manifest only", panel_id, exc
            )
        if payload_files_id:
            await self._blobs.delete(payload_files_id)
        await self._blobs.delete(panel_id)
