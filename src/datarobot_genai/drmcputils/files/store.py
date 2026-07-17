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

"""Blob storage for drtools, backed by the DataRobot Files API (v3.10+).

Defines a minimal path-based :class:`BlobStore` Protocol and a Files-API-backed
implementation. Higher-level domains (e.g. panels) depend on the Protocol, not
the concrete backend, so storage stays swappable (a local/in-memory backend can
satisfy the same contract for tests) and the Files API is the production
default.

**Layout.** All blobs live inside one shared Files *container* (a catalog item
created via ``Files.create_empty_catalog_item_dir`` and discovered by the
``dr_panel_root`` marker tag), addressed by ``/``-separated paths. This is the
Files API's sanctioned folder mechanism: ``upload_file(prefix=...)`` joins the
folder path server-side *after* filename sanitization, so paths survive intact
— unlike a path embedded in an uploaded *filename*, which the server strips to
its basename as a disk-traversal defense (and logs a security warning for).
One container also renders as a single folder row on the registry page instead
of one row per blob, and prefix listing is a single server-side query (the
per-blob-container + tag design this replaces needed client-side AND filtering
over the whole catalog, because catalog tag search matches with OR semantics).

**Legacy blobs.** Blobs stored before the shared-container layout are
standalone single-file containers addressed by their Files id. :meth:`get` and
:meth:`delete` accept a bare id (no ``/``) and fall back to that layout, so
pre-existing panels stay reachable by id.

The DataRobot Files SDK (``datarobot.models.Files``) is synchronous; its calls
are dispatched to a worker thread via :func:`asyncio.to_thread`, which copies
the current context so the per-request client configured by
:func:`request_user_dr_sdk` remains in effect inside the thread.
"""

from __future__ import annotations

import asyncio
import io
import logging
import posixpath
from dataclasses import dataclass
from typing import Protocol
from typing import runtime_checkable

import datarobot as dr
from datarobot.enums import FilesOverwriteStrategy
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_sdk

logger = logging.getLogger(__name__)

DEFAULT_LIST_LIMIT = 100
DEFAULT_PUT_TIMEOUT_SECONDS = 600  # matches the Files SDK default for read_timeout/max_wait

# The shared container's registry name and its discovery marker tag. Discovery
# goes by the tag (names are not unique); duplicates from a create race resolve
# deterministically to the oldest container (see _find_container).
CONTAINER_NAME = "panels"
CONTAINER_TAG = "dr_panel_root"


@dataclass(frozen=True, slots=True)
class BlobRef:
    """A lightweight, serializable handle to a stored blob.

    - ``path`` — the blob's ``/``-separated path inside the shared container;
      unique per blob and stable until the blob is moved.
    - ``container_id`` — the shared container's Files id. Together with
      ``path`` this is exactly what the Files download API needs
      (``POST files/<container_id>/downloads/`` with ``fileName=<path>``).
    - ``size`` — byte size (recorded on put, returned by the listing API).

    MIME content-type is intentionally *not* a field: the Files API does not
    persist one, so it could not be populated symmetrically by ``list``.
    Callers that need a content-type should record it themselves (e.g. a panel
    manifest); see ``content_type`` on :meth:`BlobStore.put`.
    """

    path: str
    container_id: str
    size: int | None = None


@runtime_checkable
class BlobStore(Protocol):
    """Storage seam for opaque byte payloads. Implementations must be async-safe."""

    async def put(
        self,
        data: bytes,
        *,
        path: str,
        content_type: str | None = None,
        timeout: int = DEFAULT_PUT_TIMEOUT_SECONDS,
    ) -> BlobRef:
        """Store ``data`` at ``path``, replacing any existing blob at that path.

        ``content_type`` is advisory: the Files backend does not persist it, so
        it is not reflected on the returned :class:`BlobRef`. Callers that need
        it should record it alongside their own metadata.

        ``timeout`` bounds the upload (seconds); raise it for very large blobs.
        """
        ...

    async def get(self, path: str) -> bytes:
        """Fetch the bytes at ``path``.

        A bare token without ``/`` is treated as a *legacy* blob id (a
        standalone pre-shared-container Files container) and fetched from that
        layout instead.
        """
        ...

    async def delete(self, paths: str | list[str]) -> None:
        """Delete the blobs at ``paths``; missing paths are silently ignored.

        Like :meth:`get`, a bare token without ``/`` addresses a legacy
        standalone container.
        """
        ...

    async def move(self, from_path: str, to_path: str) -> None:
        """Rename/move a blob within the container, replacing any blob at ``to_path``."""
        ...

    async def list(
        self,
        *,
        prefix: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> list[BlobRef]:
        """List blobs under ``prefix`` (server-side filter; ``None`` = all).

        ``limit=0`` returns every match. Returns ``[]`` when the shared
        container does not exist yet (nothing was ever stored).
        """
        ...


class DataRobotFilesBlobStore:
    """:class:`BlobStore` backed by one shared DataRobot Files container.

    The container is found by the ``dr_panel_root`` marker tag (created on
    first write; read paths never create it). The resolved SDK object is cached
    per store instance — instances are built per request, so the cache never
    outlives the requesting user's credentials.
    """

    def __init__(self, *, headers_auth_only: bool = True) -> None:
        # When False, fall back to the application API token outside HTTP contexts.
        self._headers_auth_only = headers_auth_only
        self._container: dr.models.Files | None = None

    # -- container resolution -------------------------------------------------

    def _find_container(self) -> dr.models.Files | None:
        """Find the shared container by marker tag; oldest wins on duplicates."""
        results = dr.models.Files.search_catalog(tags=[CONTAINER_TAG], limit=DEFAULT_LIST_LIMIT)
        # Catalog tag search is OR/loose — require the exact marker tag.
        candidates = [r for r in results if CONTAINER_TAG in (getattr(r, "tags", None) or ())]
        if not candidates:
            return None
        # Files ids are ObjectIds (time-ordered), so min(id) is the oldest —
        # the deterministic winner when a create race left duplicates behind.
        oldest = min(candidates, key=lambda r: str(getattr(r, "id")))
        return dr.models.Files.get(str(getattr(oldest, "id")))

    def _container_for_read(self) -> dr.models.Files | None:
        if self._container is None:
            self._container = self._find_container()
        return self._container

    def _container_for_write(self) -> dr.models.Files:
        container = self._container_for_read()
        if container is not None:
            return container
        created = dr.models.Files.create_empty_catalog_item_dir()
        created.modify(name=CONTAINER_NAME, tags=[CONTAINER_TAG])
        # If a concurrent request also created one, converge on the shared
        # winner (oldest id); the loser stays behind as an empty orphan. The
        # re-find may miss the just-created container on catalog indexing lag,
        # in which case ours is used for this request and the next request
        # converges.
        self._container = self._find_container() or created
        return self._container

    # -- BlobStore -------------------------------------------------------------

    async def put(
        self,
        data: bytes,
        *,
        path: str,
        content_type: str | None = None,
        timeout: int = DEFAULT_PUT_TIMEOUT_SECONDS,
    ) -> BlobRef:
        # content_type is advisory only — the Files API has no field for it.
        folder, filename = posixpath.split(path)

        def _upload() -> str:
            container = self._container_for_write()
            # The uploaded *filename* is basename-sanitized server-side, so the
            # folder part must travel via ``prefix`` (joined after sanitization).
            container.upload_file(
                io.BytesIO(data),
                prefix=f"{folder}/" if folder else None,
                file_name=filename,
                use_archive_contents=False,
                overwrite=FilesOverwriteStrategy.REPLACE,
                read_timeout=timeout,
                max_wait=timeout,
            )
            return str(container.id)

        with request_user_dr_sdk(headers_auth_only=self._headers_auth_only):
            try:
                container_id = await asyncio.to_thread(_upload)
            except ClientError as exc:
                raise_tool_error_for_client_error(exc)
        logger.debug("Stored blob %s (%d bytes) in container %s", path, len(data), container_id)
        return BlobRef(path=path, container_id=container_id, size=len(data))

    async def get(self, path: str) -> bytes:
        def _download() -> bytes:
            buffer = io.BytesIO()
            if "/" in path:
                container = self._container_for_read()
                if container is None:
                    raise ClientError("shared blob container does not exist", 404)
                container.download(file_name=path, filelike=buffer)
            else:
                # Legacy layout: the token is a standalone single-file
                # container's Files id.
                dr.models.Files.get(path).download(filelike=buffer)
            return buffer.getvalue()

        with request_user_dr_sdk(headers_auth_only=self._headers_auth_only):
            try:
                return await asyncio.to_thread(_download)
            except ClientError as exc:
                raise_tool_error_for_client_error(exc)

    async def delete(self, paths: str | list[str]) -> None:
        path_list = [paths] if isinstance(paths, str) else list(paths)
        container_paths = [p for p in path_list if "/" in p]
        legacy_ids = [p for p in path_list if "/" not in p]

        def _delete() -> None:
            if container_paths:
                container = self._container_for_read()
                if container is not None:
                    container.delete_files(container_paths)
            for files_id in legacy_ids:
                dr.models.Files.delete(files_id)

        with request_user_dr_sdk(headers_auth_only=self._headers_auth_only):
            try:
                await asyncio.to_thread(_delete)
            except ClientError as exc:
                raise_tool_error_for_client_error(exc)

    async def move(self, from_path: str, to_path: str) -> None:
        def _rename() -> None:
            container = self._container_for_read()
            if container is None:
                raise ClientError("shared blob container does not exist", 404)
            container.rename_files(from_path, to_path, overwrite=FilesOverwriteStrategy.REPLACE)

        with request_user_dr_sdk(headers_auth_only=self._headers_auth_only):
            try:
                await asyncio.to_thread(_rename)
            except ClientError as exc:
                raise_tool_error_for_client_error(exc)

    async def list(
        self,
        *,
        prefix: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> list[BlobRef]:
        def _list() -> list[BlobRef]:
            container = self._container_for_read()
            if container is None:
                return []
            files = container.list_contained_files(
                prefix=prefix,
                limit=limit,
                offset=offset,
                recursive=True,
            )
            container_id = str(container.id)
            return [
                BlobRef(
                    path=str(getattr(f, "name")),
                    container_id=container_id,
                    size=getattr(f, "size", None),
                )
                for f in files
            ]

        with request_user_dr_sdk(headers_auth_only=self._headers_auth_only):
            try:
                return await asyncio.to_thread(_list)
            except ClientError as exc:
                raise_tool_error_for_client_error(exc)
