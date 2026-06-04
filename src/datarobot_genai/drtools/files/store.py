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

"""Block storage for drtools, backed by the DataRobot Files API (v3.10+).

Defines a minimal :class:`BlobStore` Protocol and a Files-API-backed
implementation. Higher-level domains (e.g. panels) depend on the Protocol, not
the concrete backend, so storage stays swappable (a local/in-memory backend can
satisfy the same contract for tests) and the Files API is the production
default.

The DataRobot Files SDK (``datarobot.models.Files``) is synchronous; its calls
are dispatched to a worker thread via :func:`asyncio.to_thread`, which copies
the current context so the per-request client configured by
:func:`request_user_dr_sdk` remains in effect inside the thread.
"""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from typing import Protocol
from typing import runtime_checkable

import datarobot as dr
from datarobot.errors import ClientError

from datarobot_genai.drtools.core.clients.datarobot import request_user_dr_sdk
from datarobot_genai.drtools.predictive.client_exceptions import raise_tool_error_for_client_error

logger = logging.getLogger(__name__)

DEFAULT_LIST_LIMIT = 100


class _NamedBytesIO(io.BytesIO):
    """``BytesIO`` carrying a ``name`` so the Files SDK records a filename.

    ``datarobot.models.Files.create_from_file`` reads ``filelike.name`` to label
    the stored file; a bare :class:`io.BytesIO` cannot hold one (no ``__dict__``),
    so this subclass adds it.
    """

    def __init__(self, data: bytes, name: str) -> None:
        super().__init__(data)
        self.name = name


@dataclass(frozen=True, slots=True)
class BlobRef:
    """A lightweight, serializable handle to a stored blob.

    ``files_id`` is the durable handle (the DataRobot Files container id). The
    remaining fields are advisory metadata for callers/UX.
    """

    files_id: str
    name: str
    content_type: str | None = None
    size: int | None = None


@runtime_checkable
class BlobStore(Protocol):
    """Storage seam for opaque byte payloads. Implementations must be async-safe."""

    async def put(
        self,
        data: bytes,
        *,
        name: str,
        content_type: str | None = None,
        tags: list[str] | None = None,
    ) -> BlobRef:
        """Store ``data`` and return a handle to it."""
        ...

    async def get(self, ref: BlobRef | str) -> bytes:
        """Fetch the bytes for ``ref`` (a :class:`BlobRef` or raw files id)."""
        ...

    async def delete(self, ref: BlobRef | str) -> None:
        """Delete the blob referenced by ``ref``."""
        ...

    async def list(
        self,
        *,
        search: str | None = None,
        tags: list[str] | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> list[BlobRef]:
        """List stored blobs the requesting user can access."""
        ...


def _files_id(ref: BlobRef | str) -> str:
    return ref.files_id if isinstance(ref, BlobRef) else ref


class DataRobotFilesBlobStore:
    """:class:`BlobStore` backed by the DataRobot Files API (``datarobot.models.Files``).

    Each blob is stored as its own single-file Files container; the container id
    is the durable handle (:attr:`BlobRef.files_id`). Scoped to the requesting
    user's DataRobot token; blocking SDK calls run in a worker thread.
    """

    def __init__(self, *, headers_auth_only: bool = True) -> None:
        # When False, fall back to the application API token outside HTTP contexts.
        self._headers_auth_only = headers_auth_only

    async def put(
        self,
        data: bytes,
        *,
        name: str,
        content_type: str | None = None,
        tags: list[str] | None = None,
    ) -> BlobRef:
        def _upload() -> dr.models.Files:
            buf = _NamedBytesIO(data, name)
            # use_archive_contents=False: store the payload as-is, never auto-extract.
            return dr.models.Files.create_from_file(
                filelike=buf,
                tags=tags,
                use_archive_contents=False,
            )

        with request_user_dr_sdk(headers_auth_only=self._headers_auth_only):
            try:
                files = await asyncio.to_thread(_upload)
            except ClientError as exc:
                raise_tool_error_for_client_error(exc)
        logger.debug("Stored blob %s (name=%s, %d bytes)", files.id, name, len(data))
        return BlobRef(
            files_id=files.id,
            name=getattr(files, "name", name),
            content_type=content_type,
            size=len(data),
        )

    async def get(self, ref: BlobRef | str) -> bytes:
        files_id = _files_id(ref)

        def _download() -> bytes:
            files = dr.models.Files.get(files_id)
            buffer = io.BytesIO()
            files.download(filelike=buffer)
            return buffer.getvalue()

        with request_user_dr_sdk(headers_auth_only=self._headers_auth_only):
            try:
                return await asyncio.to_thread(_download)
            except ClientError as exc:
                raise_tool_error_for_client_error(exc)

    async def delete(self, ref: BlobRef | str) -> None:
        files_id = _files_id(ref)
        with request_user_dr_sdk(headers_auth_only=self._headers_auth_only):
            try:
                await asyncio.to_thread(dr.models.Files.delete, files_id)
            except ClientError as exc:
                raise_tool_error_for_client_error(exc)

    async def list(
        self,
        *,
        search: str | None = None,
        tags: list[str] | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> list[BlobRef]:
        def _search() -> list:
            return dr.models.Files.search_catalog(
                search=search,
                tags=tags,
                limit=limit,
                offset=offset,
            )

        with request_user_dr_sdk(headers_auth_only=self._headers_auth_only):
            try:
                results = await asyncio.to_thread(_search)
            except ClientError as exc:
                raise_tool_error_for_client_error(exc)
        return [
            BlobRef(
                files_id=item.id,
                name=getattr(item, "name", None) or getattr(item, "catalog_name", "") or "",
            )
            for item in results
        ]
