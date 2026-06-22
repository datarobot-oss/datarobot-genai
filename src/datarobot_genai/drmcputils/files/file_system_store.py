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

"""Hierarchical filesystem seam for drtools, backed by the DataRobot Files API.

Unlike the flat :class:`~datarobot_genai.drmcputils.files.store.BlobStore` (one
blob == one Files container), this store exposes DataRobot's *hierarchical*
filesystem: files live inside a catalog item addressed as
``dr://<catalog_item_id>/path/to/file``. It is a thin async wrapper around
:class:`datarobot.fs.DataRobotFileSystem` (an fsspec implementation).

Design mirrors :mod:`datarobot_genai.drmcputils.files.store`:

- A minimal :class:`FileSystemStore` Protocol so higher layers depend on the
  contract, not the concrete backend (an in-memory backend can satisfy it in
  tests).
- The fsspec filesystem is synchronous; its calls are dispatched to a worker
  thread via :func:`asyncio.to_thread`, which copies the current context so the
  per-request client configured by :func:`request_user_dr_sdk` stays in effect
  inside the thread.
- DataRobot/fsspec errors are normalised to :class:`ToolError` so tool code
  never leaks SDK or OS exception types.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import Literal
from typing import NoReturn
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable
from urllib.parse import urlparse

import datarobot as dr
from datarobot.enums import FilesOverwriteStrategy
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_sdk
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)

T = TypeVar("T")

DR_PROTOCOL = "dr://"
DEFAULT_LIST_LIMIT = 100
DEFAULT_SIGN_EXPIRATION_SECONDS = 100
FILE_IMPORT_COMPLETED_STATUS = "completed"
FILE_IMPORT_TERMINAL_FAILURE_PREFIXES = ("error", "abort")

OverwriteStrategyName = Literal["rename", "replace", "skip", "error"]


def normalize_status_location(status_id: str) -> str:
    """Return a relative API route for ``status/<id>/`` from a bare id, route, or URL."""
    cleaned = status_id.strip()
    if not cleaned:
        raise ToolError(
            "Argument validation error: 'status_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    bare_id = extract_status_id(cleaned)
    return f"status/{bare_id}/"


def extract_status_id(async_location: str) -> str:
    """Extract the bare status id from an async Location header or route."""
    location = async_location.strip().rstrip("/")
    if location.startswith("http://") or location.startswith("https://"):
        path = urlparse(location).path.rstrip("/")
        location = path.split("/api/v2/", 1)[-1] if "/api/v2/" in path else path.lstrip("/")
    if location.startswith("status/"):
        return location.split("/", 1)[1].split("/", 1)[0]
    return location.split("/")[-1]


def overwrite_strategy_from_name(name: OverwriteStrategyName) -> FilesOverwriteStrategy:
    return FilesOverwriteStrategy(name)


def is_terminal_import_failure_status(status: str | None) -> bool:
    if not status:
        return False
    lowered = status.lower()
    return lowered[:5] in FILE_IMPORT_TERMINAL_FAILURE_PREFIXES


@dataclass(frozen=True, slots=True)
class FileEntry:
    """A serializable description of a file or directory in the DataRobot filesystem.

    Built from the fsspec ``FileInfo`` mapping the backend returns. ``name`` is
    the path *without* the ``dr://`` protocol prefix (e.g.
    ``<catalog_id>/data/file.csv``); directories carry ``type == "directory"``
    and ``size == 0``.
    """

    name: str
    type: str
    size: int = 0
    format: str | None = None
    created_at: str | None = None

    @classmethod
    def from_info(cls, info: Mapping[str, Any]) -> FileEntry:
        """Build a :class:`FileEntry` from an fsspec ``FileInfo`` mapping."""
        created = info.get("created_at")
        if isinstance(created, datetime):
            created = created.isoformat()
        return cls(
            name=str(info.get("name", "")),
            type=str(info.get("type", "")),
            size=int(info.get("size") or 0),
            format=info.get("format"),
            created_at=created,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serializable dict for tool responses."""
        return {
            "name": self.name,
            "type": self.type,
            "size": self.size,
            "format": self.format,
            "created_at": self.created_at,
        }


@runtime_checkable
class FileSystemStore(Protocol):
    """Read seam over the DataRobot hierarchical filesystem. Implementations must be async-safe."""

    async def ls(self, path: str, *, detail: bool = True) -> list[FileEntry]:
        """List the immediate children of ``path`` (catalog items at ``dr://`` root)."""
        ...

    async def find(
        self, path: str, *, maxdepth: int | None = None, withdirs: bool = False
    ) -> list[FileEntry]:
        """Recursively list everything under ``path`` (posix ``find`` semantics, no globs)."""
        ...

    async def glob(self, pattern: str, *, maxdepth: int | None = None) -> list[FileEntry]:
        """List entries matching a glob ``pattern`` (e.g. ``dr://<id>/**/*.csv``)."""
        ...

    async def tree(
        self,
        path: str,
        *,
        recursion_limit: int = 2,
        max_display: int = 25,
        display_size: bool = False,
    ) -> str:
        """Return a compact, indented tree view of ``path``."""
        ...

    async def info(self, path: str) -> FileEntry:
        """Return metadata for a single file or directory ``path``."""
        ...

    async def read(self, path: str, *, start: int | None = None, end: int | None = None) -> bytes:
        """Read (a byte range of) the file at ``path``."""
        ...

    async def sign(self, path: str, *, expiration: int = DEFAULT_SIGN_EXPIRATION_SECONDS) -> str:
        """Return a temporary signed download URL for the file at ``path``."""
        ...

    async def write(self, path: str, data: bytes, *, mode: str = "overwrite") -> None:
        """Write ``data`` to the file at ``path`` ('overwrite' or 'create' if absent)."""
        ...

    async def upload(
        self,
        local_path: str | list[str],
        dest: str | list[str],
        *,
        recursive: bool = False,
        maxdepth: int | None = None,
        overwrite: OverwriteStrategyName = "rename",
    ) -> None:
        """Upload local file(s)/director(ies) into the filesystem.

        Content is streamed in chunks (no inline size cap) and multiple files are
        batched into a single optimized request when possible.
        """
        ...

    async def create_dir(self) -> str:
        """Create an empty catalog item directory and return its id."""
        ...

    async def delete(
        self, path: str, *, recursive: bool = False, maxdepth: int | None = None
    ) -> None:
        """Delete file(s) or director(ies) at ``path`` (silent if absent)."""
        ...

    async def copy(
        self, source: str, dest: str, *, recursive: bool = False, maxdepth: int | None = None
    ) -> None:
        """Copy ``source`` to ``dest`` within the filesystem."""
        ...

    async def move(
        self, source: str, dest: str, *, recursive: bool = False, maxdepth: int | None = None
    ) -> None:
        """Move/rename ``source`` to ``dest`` within the filesystem."""
        ...

    async def clone(self, path_or_id: str, *, files_to_omit: list[str] | None = None) -> str:
        """Clone a catalog item directory and return the new catalog item id."""
        ...

    async def import_from_url(
        self,
        path: str,
        url: str,
        *,
        unpack_archive: bool = True,
        overwrite: OverwriteStrategyName = "rename",
    ) -> str:
        """Start a URL import and return the async status id."""
        ...

    async def import_from_data_source(
        self,
        path: str,
        data_source_id: str,
        *,
        credential_id: str | None = None,
        unpack_archive: bool = True,
        overwrite: OverwriteStrategyName = "rename",
    ) -> str:
        """Start a data-source import and return the async status id."""
        ...

    async def get_status(self, status_id: str) -> dict[str, Any]:
        """Fetch the current async status payload for a single import job."""
        ...


def _raise_tool_error_for_fs_error(exc: Exception) -> NoReturn:
    """Normalise fsspec/OS exceptions to :class:`ToolError`.

    :class:`datarobot.fs.DataRobotFileSystem` converts SDK ``ClientError`` into
    builtin OS exceptions (404/410 -> ``FileNotFoundError``, 403 ->
    ``PermissionError``, 409 -> ``FileExistsError``, 400/422 -> ``ValueError``),
    so map those back to semantic tool errors.
    """
    if isinstance(exc, FileNotFoundError):
        raise ToolError(f"Path not found: {exc}", kind=ToolErrorKind.NOT_FOUND) from exc
    if isinstance(exc, FileExistsError):
        raise ToolError(f"Path already exists: {exc}", kind=ToolErrorKind.VALIDATION) from exc
    if isinstance(exc, PermissionError):
        raise ToolError(f"Permission denied: {exc}", kind=ToolErrorKind.UPSTREAM) from exc
    if isinstance(exc, (ValueError, NotADirectoryError, IsADirectoryError)):
        raise ToolError(
            f"Invalid file path or argument: {exc}", kind=ToolErrorKind.VALIDATION
        ) from exc
    raise ToolError(f"File system error: {exc}", kind=ToolErrorKind.UPSTREAM) from exc


class DataRobotFileSystemStore:
    """:class:`FileSystemStore` backed by :class:`datarobot.fs.DataRobotFileSystem`.

    Scoped to the requesting user's DataRobot token; blocking fsspec calls run in
    a worker thread. A fresh filesystem is built per call inside the thread — it
    holds no client state (the ``Files`` SDK resolves the request-scoped client
    from the context configured by :func:`request_user_dr_sdk`).
    """

    def __init__(self, *, headers_auth_only: bool = True) -> None:
        # When False, fall back to the application API token outside HTTP contexts.
        self._headers_auth_only = headers_auth_only

    async def _run(self, fn: Callable[[Any], T]) -> T:
        """Run ``fn(fs)`` in a worker thread under the request-scoped DR client."""
        from datarobot.fs import DataRobotFileSystem

        def _call() -> T:
            return fn(DataRobotFileSystem())

        with request_user_dr_sdk(headers_auth_only=self._headers_auth_only):
            try:
                return await asyncio.to_thread(_call)
            except ClientError as exc:
                raise_tool_error_for_client_error(exc)
            except OSError as exc:
                _raise_tool_error_for_fs_error(exc)
            except ValueError as exc:
                _raise_tool_error_for_fs_error(exc)

    async def ls(self, path: str, *, detail: bool = True) -> list[FileEntry]:
        results: list[Mapping[str, Any]] = await self._run(lambda fs: fs.ls(path, detail=True))
        return [FileEntry.from_info(item) for item in results]

    async def find(
        self, path: str, *, maxdepth: int | None = None, withdirs: bool = False
    ) -> list[FileEntry]:
        results: Mapping[str, Mapping[str, Any]] = await self._run(
            lambda fs: fs.find(path, maxdepth=maxdepth, withdirs=withdirs, detail=True)
        )
        return [FileEntry.from_info(info) for info in results.values()]

    async def glob(self, pattern: str, *, maxdepth: int | None = None) -> list[FileEntry]:
        results: Mapping[str, Mapping[str, Any]] = await self._run(
            lambda fs: fs.glob(pattern, maxdepth=maxdepth, detail=True)
        )
        return [FileEntry.from_info(info) for info in results.values()]

    async def tree(
        self,
        path: str,
        *,
        recursion_limit: int = 2,
        max_display: int = 25,
        display_size: bool = False,
    ) -> str:
        return await self._run(
            lambda fs: fs.tree(
                path,
                recursion_limit=recursion_limit,
                max_display=max_display,
                display_size=display_size,
            )
        )

    async def info(self, path: str) -> FileEntry:
        info: Mapping[str, Any] = await self._run(lambda fs: fs.info(path))
        return FileEntry.from_info(info)

    async def read(self, path: str, *, start: int | None = None, end: int | None = None) -> bytes:
        return await self._run(lambda fs: fs.cat_file(path, start=start, end=end))

    async def sign(self, path: str, *, expiration: int = DEFAULT_SIGN_EXPIRATION_SECONDS) -> str:
        return await self._run(lambda fs: fs.sign(path, expiration=expiration))

    async def write(self, path: str, data: bytes, *, mode: str = "overwrite") -> None:
        await self._run(lambda fs: fs.pipe_file(path, value=data, mode=mode))

    async def upload(
        self,
        local_path: str | list[str],
        dest: str | list[str],
        *,
        recursive: bool = False,
        maxdepth: int | None = None,
        overwrite: OverwriteStrategyName = "rename",
    ) -> None:
        strategy = overwrite_strategy_from_name(overwrite)
        await self._run(
            lambda fs: fs.put(
                local_path,
                dest,
                recursive=recursive,
                maxdepth=maxdepth,
                overwrite_strategy=strategy,
            )
        )

    async def create_dir(self) -> str:
        return await self._run(lambda fs: fs.create_catalog_item_dir())

    async def delete(
        self, path: str, *, recursive: bool = False, maxdepth: int | None = None
    ) -> None:
        await self._run(lambda fs: fs.rm(path, recursive=recursive, maxdepth=maxdepth))

    async def copy(
        self, source: str, dest: str, *, recursive: bool = False, maxdepth: int | None = None
    ) -> None:
        await self._run(lambda fs: fs.copy(source, dest, recursive=recursive, maxdepth=maxdepth))

    async def move(
        self, source: str, dest: str, *, recursive: bool = False, maxdepth: int | None = None
    ) -> None:
        await self._run(lambda fs: fs.mv(source, dest, recursive=recursive, maxdepth=maxdepth))

    async def clone(self, path_or_id: str, *, files_to_omit: list[str] | None = None) -> str:
        return await self._run(
            lambda fs: fs.clone_catalog_item_dir(path_or_id, files_to_omit=files_to_omit)
        )

    async def import_from_url(
        self,
        path: str,
        url: str,
        *,
        unpack_archive: bool = True,
        overwrite: OverwriteStrategyName = "rename",
    ) -> str:
        strategy = overwrite_strategy_from_name(overwrite)

        def _call(fs: Any) -> str:
            catalog_id, internal_path = fs._split_path(path)
            prefix = f"{internal_path.rstrip('/')}/" if internal_path else None
            entity = fs._get_files_wrapper_for_folder_id(catalog_id).upload_from_url(
                url=url,
                use_archive_contents=unpack_archive,
                overwrite=strategy,
                wait_for_completion=False,
                prefix=prefix,
            )
            location = entity._async_status_location
            if not location:
                raise ToolError(
                    "Import did not return an async status location.",
                    kind=ToolErrorKind.UPSTREAM,
                )
            return extract_status_id(location)

        return await self._run(_call)

    async def import_from_data_source(
        self,
        path: str,
        data_source_id: str,
        *,
        credential_id: str | None = None,
        unpack_archive: bool = True,
        overwrite: OverwriteStrategyName = "rename",
    ) -> str:
        strategy = overwrite_strategy_from_name(overwrite)

        def _call(fs: Any) -> str:
            catalog_id, internal_path = fs._split_path(path)
            prefix = f"{internal_path.rstrip('/')}/" if internal_path else None
            entity = fs._get_files_wrapper_for_folder_id(catalog_id).upload_from_data_source(
                data_source_id=data_source_id,
                credential_id=credential_id,
                prefix=prefix,
                use_archive_contents=unpack_archive,
                overwrite=strategy,
                wait_for_completion=False,
            )
            location = entity._async_status_location
            if not location:
                raise ToolError(
                    "Import did not return an async status location.",
                    kind=ToolErrorKind.UPSTREAM,
                )
            return extract_status_id(location)

        return await self._run(_call)

    async def get_status(self, status_id: str) -> dict[str, Any]:
        location = normalize_status_location(status_id)

        def _fetch() -> dict[str, Any]:
            client = dr.client.get_client()
            response = client.get(location, allow_redirects=False)
            if response.status_code == 307:
                redirect = response.headers.get("Location")
                if not redirect:
                    raise ToolError(
                        "Status redirect (307) missing Location header.",
                        kind=ToolErrorKind.UPSTREAM,
                    )
                response = client.get(redirect, allow_redirects=False)
            if response.status_code == 303:
                return {
                    "status": FILE_IMPORT_COMPLETED_STATUS,
                    "redirect_location": response.headers.get("Location"),
                }
            return response.json()

        with request_user_dr_sdk(headers_auth_only=self._headers_auth_only):
            try:
                return await asyncio.to_thread(_fetch)
            except ClientError as exc:
                raise_tool_error_for_client_error(exc)
