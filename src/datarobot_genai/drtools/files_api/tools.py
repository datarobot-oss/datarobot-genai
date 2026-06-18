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

"""Read-only DataRobot Files API tools.

The DataRobot filesystem is hierarchical: files live inside a catalog item
addressed as ``dr://<catalog_item_id>/path/to/file``. ``dr://`` (root) lists
catalog items; each catalog item behaves like a top-level directory.

All tools issue a single, non-blocking filesystem call and return immediately.
File content is exchanged inline and bounded by ``MAX_INLINE_SIZE``; for larger
payloads, read a byte range (``file_read`` offset/length) or hand off a signed
URL (``file_sign``).
"""

from __future__ import annotations

import base64
import logging
from typing import Annotated
from typing import Any

from datarobot_genai.drmcputils.constants import MAX_INLINE_SIZE
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.files.file_system_store import DEFAULT_SIGN_EXPIRATION_SECONDS
from datarobot_genai.drmcputils.files.file_system_store import DR_PROTOCOL
from datarobot_genai.drmcputils.files.file_system_store import DataRobotFileSystemStore
from datarobot_genai.drmcputils.files.file_system_store import FileSystemStore
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata

logger = logging.getLogger(__name__)

ROOT_PATH = DR_PROTOCOL


def _get_store() -> FileSystemStore:
    """Return the filesystem store. Indirection keeps tools easy to patch in tests."""
    return DataRobotFileSystemStore()


def _require_path(path: str, name: str = "path") -> str:
    if not path or not path.strip():
        raise ToolError(
            f"Argument validation error: '{name}' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    return path.strip()


def _is_root(path: str) -> bool:
    return path.rstrip("/").lower() in (DR_PROTOCOL.rstrip("/"), "dr:", "")


def _require_file_path(path: str, name: str = "path") -> str:
    """Validate a path that must reference a file under a catalog item, not the root."""
    cleaned = _require_path(path, name)
    if _is_root(cleaned):
        raise ToolError(
            f"Argument validation error: '{name}' must reference a file under a catalog item "
            f"(e.g. dr://<catalog_id>/file.txt), not the filesystem root.",
            kind=ToolErrorKind.VALIDATION,
        )
    return cleaned


# ------------------------------------------------------------------ #
# file_list                                                            #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"file", "datarobot", "list", "search", "glob", "tree"},
    description=(
        "[Files—list] Browse the DataRobot filesystem. Paths look like "
        "dr://<catalog_id>/path. Call with path='dr://' to list catalog items "
        "(top-level directories); call with a catalog path to list its contents.\n"
        "  - pattern: glob match (e.g. dr://<catalog_id>/**/*.csv). Overrides recursive.\n"
        "  - recursive: list everything beneath path (no globbing).\n"
        "  - as_tree: return a compact indented tree string instead of entries.\n"
        "Returns entries with name, type ('file'|'directory'), size, format, created_at.\n\n"
        "Example: file_list(path='dr://')\n"
        "Example: file_list(path='dr://abc123/', recursive=True)\n"
        "Example: file_list(pattern='dr://abc123/**/*.pdf')"
    ),
)
async def file_list(
    *,
    path: Annotated[
        str,
        "Directory path. 'dr://' lists catalog items; 'dr://<catalog_id>/sub/' lists contents.",
    ] = ROOT_PATH,
    pattern: Annotated[
        str | None,
        "Optional glob pattern (e.g. 'dr://<catalog_id>/**/*.csv'). When set, overrides recursive.",
    ] = None,
    recursive: Annotated[
        bool,
        "List everything beneath path (posix find semantics, no globbing). Default False.",
    ] = False,
    as_tree: Annotated[
        bool,
        "Return a compact indented tree string instead of a list of entries. Default False.",
    ] = False,
    maxdepth: Annotated[
        int | None,
        "Maximum depth for recursive/glob listing. None means unlimited.",
    ] = None,
    limit: Annotated[int, "Maximum entries to return (1-100). Default 100."] = 100,
    offset: Annotated[int, "Number of entries to skip for pagination. Default 0."] = 0,
) -> dict[str, Any]:
    store = _get_store()

    if pattern is not None and not pattern.strip():
        raise ToolError(
            "Argument validation error: 'pattern' cannot be empty when provided.",
            kind=ToolErrorKind.VALIDATION,
        )
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if maxdepth is not None and maxdepth < 1:
        raise ToolError(
            "Argument validation error: 'maxdepth' must be >= 1 when provided.",
            kind=ToolErrorKind.VALIDATION,
        )

    cleaned_path = _require_path(path)

    if as_tree:
        tree = await store.tree(cleaned_path, recursion_limit=maxdepth or 2)
        return {"path": cleaned_path, "tree": tree}

    if pattern is not None:
        entries = await store.glob(pattern.strip(), maxdepth=maxdepth)
        listed = pattern.strip()
    elif recursive:
        entries = await store.find(cleaned_path, maxdepth=maxdepth)
        listed = cleaned_path
    else:
        entries = await store.ls(cleaned_path)
        listed = cleaned_path

    clamped_limit, note = clamp_limit(limit)
    page = entries[offset : offset + clamped_limit]
    results: dict[str, Any] = {
        "path": listed,
        "entries": [entry.to_dict() for entry in page],
        "count": len(page),
        "total_count": len(entries),
    }
    return merge_pagination_metadata(results, {}, note, offset=offset, limit=clamped_limit)


# ------------------------------------------------------------------ #
# file_info                                                            #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"file", "datarobot", "info", "metadata"},
    description=(
        "[Files—info] Fetch metadata for a single file or directory: name, type "
        "('file'|'directory'), size in bytes, format, and created_at. Append a "
        "trailing '/' to disambiguate a directory from a same-named file.\n\n"
        "Example: file_info(path='dr://abc123/data/employees.csv')"
    ),
)
async def file_info(
    *,
    path: Annotated[str, "Path to the file or directory to inspect (dr://<catalog_id>/...)."],
) -> dict[str, Any]:
    cleaned = _require_file_path(path)
    entry = await _get_store().info(cleaned)
    return entry.to_dict()


# ------------------------------------------------------------------ #
# file_read                                                            #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"file", "datarobot", "read", "download"},
    description=(
        "[Files—read] Read a file's content. Content is returned as UTF-8 text "
        "when valid, otherwise base64 (check the 'encoding' field). Reads are "
        f"capped at {MAX_INLINE_SIZE} bytes per call; for larger files, read a "
        "byte range with offset/length, or use file_sign to get a download URL.\n\n"
        "Example: file_read(path='dr://abc123/notes.txt')\n"
        "Example (range): file_read(path='dr://abc123/big.bin', offset=0, length=65536)"
    ),
)
async def file_read(
    *,
    path: Annotated[str, "File path to read (dr://<catalog_id>/...)."],
    offset: Annotated[int, "Start byte for partial reads. Default 0."] = 0,
    length: Annotated[
        int | None,
        "Max bytes to read from offset. None reads to end of file (subject to the inline cap).",
    ] = None,
) -> dict[str, Any]:
    cleaned = _require_file_path(path)
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if length is not None and length <= 0:
        raise ToolError(
            "Argument validation error: 'length' must be positive when set.",
            kind=ToolErrorKind.VALIDATION,
        )

    store = _get_store()
    info = await store.info(cleaned)
    if info.type == "directory":
        raise ToolError(
            f"Path '{cleaned}' is a directory; use file_list to browse it.",
            kind=ToolErrorKind.VALIDATION,
        )
    total_size = info.size

    requested = length if length is not None else max(total_size - offset, 0)
    if requested > MAX_INLINE_SIZE:
        raise ToolError(
            f"Requested read of {requested} bytes exceeds the inline limit of "
            f"{MAX_INLINE_SIZE} bytes. Read a smaller byte range with offset/length, "
            "or call file_sign to obtain a download URL.",
            kind=ToolErrorKind.VALIDATION,
        )

    end = offset + length if length is not None else None
    data = await store.read(cleaned, start=offset, end=end)

    if len(data) > MAX_INLINE_SIZE:
        raise ToolError(
            f"Read returned {len(data)} bytes, exceeding the inline limit of "
            f"{MAX_INLINE_SIZE} bytes. Use a smaller range or file_sign.",
            kind=ToolErrorKind.VALIDATION,
        )

    try:
        content = data.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = base64.b64encode(data).decode("ascii")
        encoding = "base64"

    return {
        "path": cleaned,
        "content": content,
        "encoding": encoding,
        "bytes_read": len(data),
        "total_size": total_size,
    }


# ------------------------------------------------------------------ #
# file_sign                                                            #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"file", "datarobot", "sign", "url", "download"},
    description=(
        "[Files—sign] Create a temporary signed URL granting direct download "
        "access to a file. Use this for large files (over the inline read cap) "
        "or to hand a download link to a user. The URL expires after 'expiration' "
        "seconds.\n\n"
        "Example: file_sign(path='dr://abc123/report.pdf', expiration=300)"
    ),
)
async def file_sign(
    *,
    path: Annotated[str, "File path to sign (dr://<catalog_id>/...)."],
    expiration: Annotated[
        int,
        f"Seconds until the signed URL expires. Default {DEFAULT_SIGN_EXPIRATION_SECONDS}.",
    ] = DEFAULT_SIGN_EXPIRATION_SECONDS,
) -> dict[str, Any]:
    cleaned = _require_file_path(path)
    if expiration <= 0:
        raise ToolError(
            "Argument validation error: 'expiration' must be positive.",
            kind=ToolErrorKind.VALIDATION,
        )
    url = await _get_store().sign(cleaned, expiration=expiration)
    return {"path": cleaned, "url": url, "expiration": expiration}
