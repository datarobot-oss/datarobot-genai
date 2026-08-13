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

"""In-memory Files API store for MCP integration tests."""

from __future__ import annotations

import fnmatch
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.files.file_system_store import DR_PROTOCOL
from datarobot_genai.drmcputils.files.file_system_store import FileEntry
from datarobot_genai.drmcputils.files.file_system_store import OverwriteStrategyName

STUB_CATALOG_ID = "stub_catalog_id"
STUB_CATALOG_ID_2 = "stub_catalog_id_2"
STUB_IMPORT_STATUS_ID = "stub_import_status_id"
STUB_IMPORT_STATUS_INPROGRESS = "INPROGRESS"

_DEFAULT_FILES: dict[str, bytes] = {
    f"{STUB_CATALOG_ID}/data/employees.csv": b"id,name\n1,Alice\n2,Bob\n",
    f"{STUB_CATALOG_ID}/notes.txt": b"hello world\n",
    f"{STUB_CATALOG_ID}/data/report.pdf": b"%PDF-1.4 \xff\xfe binary stub",
    f"{STUB_CATALOG_ID_2}/readme.txt": b"catalog two\n",
}

_DEFAULT_CATALOGS: set[str] = {STUB_CATALOG_ID, STUB_CATALOG_ID_2}


def _strip_protocol(path: str) -> str:
    cleaned = path.strip()
    if cleaned.startswith(DR_PROTOCOL):
        cleaned = cleaned[len(DR_PROTOCOL) :]
    return cleaned.strip("/")


def _entry_for_file(internal_path: str, data: bytes) -> FileEntry:
    ext = internal_path.rsplit(".", 1)[-1] if "." in internal_path.rsplit("/", 1)[-1] else None
    return FileEntry(
        name=internal_path,
        type="file",
        size=len(data),
        format=ext,
        created_at="2026-01-01T00:00:00Z",
    )


def _entry_for_dir(internal_path: str) -> FileEntry:
    name = internal_path if internal_path.endswith("/") else f"{internal_path}/"
    return FileEntry(
        name=name,
        type="directory",
        size=0,
        format=None,
        created_at="2026-01-01T00:00:00Z",
    )


class InMemoryFileSystemStore:
    """Async in-memory :class:`FileSystemStore` with seeded catalog data."""

    def __init__(self) -> None:
        self._load_defaults()

    def _load_defaults(self) -> None:
        self._files = deepcopy(_DEFAULT_FILES)
        self._catalogs = set(_DEFAULT_CATALOGS)
        self._import_statuses = {
            STUB_IMPORT_STATUS_ID: {"status": STUB_IMPORT_STATUS_INPROGRESS},
        }

    def reset(self) -> None:
        """Restore seeded catalogs and files (for integration test server startup)."""
        self._load_defaults()

    def _catalog_for_path(self, internal: str) -> str | None:
        if not internal:
            return None
        return internal.split("/", 1)[0]

    def _ensure_catalog(self, internal: str) -> None:
        catalog = self._catalog_for_path(internal)
        if catalog is None:
            raise ToolError(
                "Path must be under a catalog item.",
                kind=ToolErrorKind.VALIDATION,
            )
        if catalog not in self._catalogs:
            raise ToolError(f"Catalog not found: {catalog}", kind=ToolErrorKind.NOT_FOUND)

    def _resolve_file(self, path: str) -> str:
        internal = _strip_protocol(path)
        if internal in self._files:
            return internal
        raise ToolError(f"Path not found: {path}", kind=ToolErrorKind.NOT_FOUND)

    def _resolve_info_path(self, path: str) -> str:
        internal = _strip_protocol(path)
        if internal in self._files:
            return internal
        if internal.rstrip("/") in self._catalogs:
            return internal.rstrip("/")
        prefix = internal.rstrip("/") + "/"
        if any(fp.startswith(prefix) or fp == internal.rstrip("/") for fp in self._files):
            return internal.rstrip("/")
        raise ToolError(f"Path not found: {path}", kind=ToolErrorKind.NOT_FOUND)

    def _list_immediate_children(self, parent: str) -> list[FileEntry]:
        if parent == "":
            return [_entry_for_dir(catalog) for catalog in sorted(self._catalogs)]

        prefix = parent.rstrip("/") + "/"
        child_names: dict[str, FileEntry] = {}
        for file_path, data in self._files.items():
            if not file_path.startswith(prefix):
                continue
            remainder = file_path[len(prefix) :]
            if not remainder:
                continue
            segment = remainder.split("/", 1)[0]
            child_internal = f"{prefix}{segment}"
            if "/" in remainder:
                child_names[child_internal] = _entry_for_dir(child_internal.rstrip("/"))
            else:
                child_names[child_internal] = _entry_for_file(child_internal, data)
        return sorted(child_names.values(), key=lambda e: e.name)

    async def ls(self, path: str, *, detail: bool = True) -> list[FileEntry]:
        del detail
        parent = _strip_protocol(path)
        if parent and parent not in self._files:
            self._ensure_catalog(parent.split("/", 1)[0])
        return self._list_immediate_children(parent)

    async def find(
        self, path: str, *, maxdepth: int | None = None, withdirs: bool = False
    ) -> list[FileEntry]:
        del withdirs
        base = _strip_protocol(path).rstrip("/")
        if base:
            self._ensure_catalog(base.split("/", 1)[0])
        base_depth = 0 if not base else base.count("/") + 1
        results: list[FileEntry] = []
        for file_path, data in sorted(self._files.items()):
            if base and not (file_path == base or file_path.startswith(base + "/")):
                continue
            depth = file_path.count("/") + 1 - base_depth
            if maxdepth is not None and depth > maxdepth:
                continue
            results.append(_entry_for_file(file_path, data))
        return results

    async def glob(self, pattern: str, *, maxdepth: int | None = None) -> list[FileEntry]:
        internal_pattern = _strip_protocol(pattern)
        results: list[FileEntry] = []
        for file_path, data in sorted(self._files.items()):
            if not fnmatch.fnmatch(file_path, internal_pattern):
                continue
            if maxdepth is not None:
                wildcard_prefix = internal_pattern.split("*", 1)[0].rstrip("/")
                rel_depth = (
                    file_path[len(wildcard_prefix) :].count("/")
                    if wildcard_prefix
                    else file_path.count("/")
                )
                if rel_depth >= maxdepth:
                    continue
            results.append(_entry_for_file(file_path, data))
        return results

    async def tree(
        self,
        path: str,
        *,
        recursion_limit: int = 2,
        max_display: int = 25,
        display_size: bool = False,
    ) -> str:
        del display_size
        base = _strip_protocol(path).rstrip("/") or STUB_CATALOG_ID
        lines = [f"{base}/"]
        shown = 0
        for file_path in sorted(self._files):
            if not file_path.startswith(base + "/"):
                continue
            rel = file_path[len(base) + 1 :]
            depth = rel.count("/") + 1
            if depth > recursion_limit:
                continue
            if shown >= max_display:
                lines.append("  ...")
                break
            indent = "  " * depth
            lines.append(f"{indent}{rel.split('/')[-1]}")
            shown += 1
        return "\n".join(lines)

    async def info(self, path: str) -> FileEntry:
        resolved = self._resolve_info_path(path)
        if resolved in self._files:
            return _entry_for_file(resolved, self._files[resolved])
        return _entry_for_dir(resolved)

    async def read(self, path: str, *, start: int | None = None, end: int | None = None) -> bytes:
        internal = self._resolve_file(path)
        data = self._files[internal]
        start_idx = start or 0
        end_idx = end if end is not None else len(data)
        return data[start_idx:end_idx]

    async def sign(self, path: str, *, expiration: int = 100) -> str:
        internal = self._resolve_file(path)
        return f"https://stub.app.example/files/{internal}?expires={expiration}"

    async def write(self, path: str, data: bytes, *, mode: str = "overwrite") -> None:
        internal = _strip_protocol(path)
        self._ensure_catalog(internal)
        if mode == "create" and internal in self._files:
            raise ToolError(f"Path already exists: {path}", kind=ToolErrorKind.VALIDATION)
        self._files[internal] = data

    async def upload(
        self,
        local_path: str | list[str],
        dest: str | list[str],
        *,
        recursive: bool = False,
        maxdepth: int | None = None,
        overwrite: OverwriteStrategyName = "rename",
    ) -> None:
        del recursive, maxdepth, overwrite
        local = local_path if isinstance(local_path, str) else local_path[0]
        destination = dest if isinstance(dest, str) else dest[0]
        data = Path(local).read_bytes()
        dest_internal = _strip_protocol(destination)
        if dest_internal.endswith("/") or destination.endswith("/"):
            filename = Path(local).name
            dest_internal = f"{dest_internal.rstrip('/')}/{filename}"
        await self.write(f"{DR_PROTOCOL}{dest_internal}", data)

    async def create_dir(self) -> str:
        catalog_id = f"stub_catalog_{uuid.uuid4().hex[:8]}"
        self._catalogs.add(catalog_id)
        return catalog_id

    async def delete(
        self, path: str, *, recursive: bool = False, maxdepth: int | None = None
    ) -> None:
        del maxdepth
        internal = _strip_protocol(path).rstrip("/")
        if internal in self._catalogs:
            if not recursive:
                prefix = internal + "/"
                if any(fp.startswith(prefix) for fp in self._files):
                    raise ToolError(
                        "Directory is not empty; set recursive=True.",
                        kind=ToolErrorKind.VALIDATION,
                    )
            self._catalogs.discard(internal)
            self._files = {
                fp: data for fp, data in self._files.items() if not fp.startswith(internal + "/")
            }
            return
        prefix = internal + "/"
        if internal in self._files:
            del self._files[internal]
            return
        if recursive:
            self._files = {
                fp: data
                for fp, data in self._files.items()
                if not (fp == internal or fp.startswith(prefix))
            }
            return
        if any(fp.startswith(prefix) for fp in self._files):
            raise ToolError(
                "Directory is not empty; set recursive=True.",
                kind=ToolErrorKind.VALIDATION,
            )

    async def copy(
        self,
        source: str,
        dest: str,
        *,
        recursive: bool = False,
        maxdepth: int | None = None,
        overwrite: str = "rename",
    ) -> None:
        del maxdepth, overwrite
        src = _strip_protocol(source)
        dst = _strip_protocol(dest)
        if src in self._files:
            self._files[dst] = self._files[src]
            return
        if recursive:
            prefix = src.rstrip("/") + "/"
            for fp, data in list(self._files.items()):
                if fp.startswith(prefix):
                    rel = fp[len(prefix) :]
                    self._files[f"{dst.rstrip('/')}/{rel}"] = data

    async def move(
        self,
        source: str,
        dest: str,
        *,
        recursive: bool = False,
        maxdepth: int | None = None,
        overwrite: str = "rename",
    ) -> None:
        await self.copy(source, dest, recursive=recursive, maxdepth=maxdepth, overwrite=overwrite)
        await self.delete(source, recursive=recursive)

    async def clone(self, path_or_id: str, *, files_to_omit: list[str] | None = None) -> str:
        omit = set(files_to_omit or [])
        source_catalog = _strip_protocol(path_or_id).split("/", 1)[0]
        new_id = await self.create_dir()
        prefix = source_catalog + "/"
        for fp, data in list(self._files.items()):
            if not fp.startswith(prefix):
                continue
            rel = fp[len(prefix) :]
            if rel in omit:
                continue
            self._files[f"{new_id}/{rel}"] = data
        return new_id

    async def import_from_url(
        self,
        path: str,
        url: str,
        *,
        unpack_archive: bool = True,
        overwrite: OverwriteStrategyName = "rename",
    ) -> str:
        del path, url, unpack_archive, overwrite
        status_id = f"import_{uuid.uuid4().hex[:8]}"
        self._import_statuses[status_id] = {"status": STUB_IMPORT_STATUS_INPROGRESS}
        return status_id

    async def import_from_data_source(
        self,
        path: str,
        data_source_id: str,
        *,
        credential_id: str | None = None,
        unpack_archive: bool = True,
        overwrite: OverwriteStrategyName = "rename",
    ) -> str:
        del path, data_source_id, credential_id, unpack_archive, overwrite
        status_id = f"import_{uuid.uuid4().hex[:8]}"
        self._import_statuses[status_id] = {"status": STUB_IMPORT_STATUS_INPROGRESS}
        return status_id

    async def get_status(self, status_id: str) -> dict[str, Any]:
        bare = status_id.strip().split("/")[-1]
        if bare not in self._import_statuses:
            raise ToolError(f"Unknown status id: {status_id}", kind=ToolErrorKind.NOT_FOUND)
        return dict(self._import_statuses[bare])

    def set_import_status(self, status_id: str, status: str) -> None:
        """Test helper: update an import job status in the stub store."""
        bare = status_id.strip().split("/")[-1]
        self._import_statuses[bare] = {"status": status}


_STUB_STORE = InMemoryFileSystemStore()


def get_stub_files_store() -> InMemoryFileSystemStore:
    """Return the shared in-memory Files API store for integration tests."""
    return _STUB_STORE


def reset_stub_files_store() -> InMemoryFileSystemStore:
    """Reset stub filesystem state (fresh catalogs and seed files)."""
    _STUB_STORE.reset()
    return _STUB_STORE


def apply_files_api_stubs() -> None:
    """Patch Files API tools to use the in-memory stub store."""
    from datarobot_genai.drtools.files_api import common_utils

    reset_stub_files_store()
    common_utils.get_store = get_stub_files_store  # type: ignore[method-assign]
