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

"""Shared helpers for the Files API tool modules."""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
from pathlib import Path

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from fsspec.implementations.local import LocalFileSystem
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.files.file_system_store import DR_PROTOCOL
from datarobot_genai.drmcputils.files.file_system_store import DataRobotFileSystemStore
from datarobot_genai.drmcputils.files.file_system_store import FileSystemStore

ROOT_PATH = DR_PROTOCOL


class FilesApiLocalSettings(DataRobotAppFrameworkBaseSettings):
    """Settings governing local-disk access for the Files API tools.

    Resolves from env vars, ``.env``, file secrets, and ``MLOPS_RUNTIME_PARAM_``
    runtime parameters by field name (``files_api_local_allowed_roots`` reads
    ``FILES_API_LOCAL_ALLOWED_ROOTS``). Defined here, rather than in the drmcp
    config, so the drtools layer stays independent of the server package.
    """

    files_api_local_allowed_roots: str = Field(
        default="",
        description=(
            "Comma-separated absolute directories that file_upload and "
            "file_write(local_path=...) may read from. Empty (default) disables all "
            "local-disk access."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_ignore_empty=True,
    )


def _allowed_local_roots() -> list[Path]:
    """Return the configured allowed local roots as resolved absolute paths."""
    raw = FilesApiLocalSettings().files_api_local_allowed_roots or ""
    roots: list[Path] = []
    for part in raw.split(","):
        candidate = part.strip()
        if candidate:
            roots.append(Path(candidate).expanduser().resolve())
    return roots


def _is_within_roots(resolved: Path, roots: list[Path]) -> bool:
    return any(resolved == root or root in resolved.parents for root in roots)


def ensure_local_path_allowed(local_path: str) -> Path:
    """Validate ``local_path`` against the allowlist and return its resolved real path.

    Resolves ``..`` and symlinks before checking so a path cannot escape an
    allowed root. Raises :class:`ToolError` when local access is disabled (no
    roots configured) or the path falls outside every allowed root.
    """
    roots = _allowed_local_roots()
    if not roots:
        raise ToolError(
            "Local filesystem access is disabled. Set FILES_API_LOCAL_ALLOWED_ROOTS to a "
            "comma-separated list of allowed base directories to enable file_upload and "
            "file_write(local_path=...).",
            kind=ToolErrorKind.VALIDATION,
        )
    resolved = Path(local_path).expanduser().resolve()
    if not _is_within_roots(resolved, roots):
        allowed = ", ".join(str(root) for root in roots)
        raise ToolError(
            f"Local path {local_path!r} is outside the allowed directories ({allowed}).",
            kind=ToolErrorKind.VALIDATION,
        )
    return resolved


def get_store() -> FileSystemStore:
    """Return the filesystem store. Indirection keeps tools easy to patch in tests."""
    return DataRobotFileSystemStore()


def require_path(path: str | None, name: str = "path") -> str:
    if not path or not path.strip():
        raise ToolError(
            f"Argument validation error: '{name}' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    return path.strip()


def is_root(path: str) -> bool:
    return path.rstrip("/").lower() in (DR_PROTOCOL.rstrip("/"), "dr:", "")


def require_file_path(path: str | None, name: str = "path") -> str:
    """Validate a path that must reference something under a catalog item, not the root."""
    cleaned = require_path(path, name)
    if is_root(cleaned):
        raise ToolError(
            f"Argument validation error: '{name}' must reference a path under a catalog item "
            f"(e.g. dr://<catalog_id>/file.txt), not the filesystem root.",
            kind=ToolErrorKind.VALIDATION,
        )
    return cleaned


async def read_local_file(local_path: str, *, max_bytes: int | None = None) -> bytes:
    """Read a local file's bytes off the event loop.

    Reads run in a worker thread so a large file never blocks the loop. When
    ``max_bytes`` is set, at most ``max_bytes + 1`` bytes are read so the caller
    can detect (and reject) oversized files without loading the whole thing.
    """
    cleaned = require_path(local_path, "local_path")
    resolved = str(ensure_local_path_allowed(cleaned))

    def _read() -> bytes:
        if os.path.isdir(resolved):
            raise ToolError(
                f"Local path {cleaned!r} is a directory; use file_upload(recursive=True) "
                "to upload a directory tree.",
                kind=ToolErrorKind.VALIDATION,
            )
        try:
            with open(resolved, "rb") as handle:
                return handle.read(max_bytes + 1) if max_bytes is not None else handle.read()
        except FileNotFoundError as exc:
            raise ToolError(
                f"Local file not found: {cleaned}", kind=ToolErrorKind.NOT_FOUND
            ) from exc
        except OSError as exc:
            raise ToolError(
                f"Could not read local file {cleaned!r}: {exc}", kind=ToolErrorKind.UPSTREAM
            ) from exc

    return await asyncio.to_thread(_read)


async def resolve_local_sources(
    local_path: str, *, recursive: bool, maxdepth: int | None
) -> list[tuple[str, int]]:
    """Expand a local file/dir/glob to a manifest of (path, size) tuples.

    Mirrors the expansion the upload backend performs, so callers can validate
    that something will actually be uploaded and report an accurate file count
    and byte total. Runs in a worker thread.
    """
    cleaned = require_path(local_path, "local_path")
    ensure_local_path_allowed(cleaned)
    roots = _allowed_local_roots()

    def _expand() -> list[tuple[str, int]]:
        if os.path.isdir(cleaned) and not recursive:
            raise ToolError(
                f"Local path {cleaned!r} is a directory; set recursive=True "
                "to upload its contents.",
                kind=ToolErrorKind.VALIDATION,
            )
        lfs = LocalFileSystem()
        try:
            candidates = lfs.expand_path(cleaned, recursive=recursive, maxdepth=maxdepth)
        except FileNotFoundError:
            # No file or glob match — report uniformly via the caller's empty-manifest path.
            return []

        manifest: list[tuple[str, int]] = []
        for path in candidates:
            if not os.path.isfile(path):
                continue
            # Re-check each expanded path so a symlink cannot escape the allowlist.
            if not _is_within_roots(Path(path).resolve(), roots):
                raise ToolError(
                    f"Local path {path!r} resolves outside the allowed directories.",
                    kind=ToolErrorKind.VALIDATION,
                )
            manifest.append((path, os.path.getsize(path)))
        return manifest

    return await asyncio.to_thread(_expand)


def decode_content(content: str, encoding: str) -> bytes:
    """Decode tool-supplied ``content`` to bytes per ``encoding`` ('utf-8' or 'base64')."""
    if encoding == "utf-8":
        return content.encode("utf-8")
    if encoding == "base64":
        try:
            return base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ToolError(
                f"Argument validation error: 'content' is not valid base64: {exc}",
                kind=ToolErrorKind.VALIDATION,
            ) from exc
    raise ToolError(
        "Argument validation error: 'encoding' must be 'utf-8' or 'base64'.",
        kind=ToolErrorKind.VALIDATION,
    )
