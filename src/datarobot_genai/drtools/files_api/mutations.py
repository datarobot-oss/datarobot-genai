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

"""Write and structural DataRobot Files API tools.

These tools mutate the filesystem: writing file content and the structural
lifecycle actions (create directory, delete, copy, move, clone). Each issues a
single filesystem call and returns immediately. Content is supplied inline and
bounded by ``MAX_INLINE_SIZE``; to ingest large or remote files, use the import
tools instead.
"""

from __future__ import annotations

import logging
from typing import Annotated
from typing import Any
from typing import Literal

from datarobot_genai.drmcputils.constants import MAX_INLINE_SIZE
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.files_api.common_utils import decode_content
from datarobot_genai.drtools.files_api.common_utils import get_store as _get_store
from datarobot_genai.drtools.files_api.common_utils import require_file_path as _require_file_path

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# file_write                                                           #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"file", "datarobot", "write", "create", "upload"},
    description=(
        "[Files—write] Write content to a file at dr://<catalog_id>/path, creating "
        "parent folders implicitly (DataRobot has no empty directories). The path "
        "must be under a catalog item; create one first with "
        "file_manage(action='create_dir') if needed.\n"
        "  - encoding: 'utf-8' for text (default) or 'base64' for binary content.\n"
        "  - mode: 'overwrite' (default) or 'create' (fails if the file exists).\n"
        f"Content is capped at {MAX_INLINE_SIZE} bytes; use file_import for larger "
        "or remote files.\n\n"
        "Example: file_write(path='dr://abc123/notes.txt', content='hello')\n"
        "Example (binary): file_write(path='dr://abc123/logo.png', content='<b64>', "
        "encoding='base64')"
    ),
)
async def file_write(
    *,
    path: Annotated[str, "Target file path (dr://<catalog_id>/...). Must be under a catalog item."],
    content: Annotated[str, "File content. UTF-8 text, or base64 when encoding='base64'."],
    encoding: Annotated[
        Literal["utf-8", "base64"],
        "How 'content' is encoded: 'utf-8' (text) or 'base64' (binary). Default 'utf-8'.",
    ] = "utf-8",
    mode: Annotated[
        Literal["overwrite", "create"],
        "'overwrite' replaces existing content; 'create' fails if the file exists. Default 'overwrite'.",  # noqa: E501
    ] = "overwrite",
) -> dict[str, Any]:
    cleaned = _require_file_path(path)
    data = decode_content(content, encoding)
    if len(data) > MAX_INLINE_SIZE:
        raise ToolError(
            f"Content is {len(data)} bytes, exceeding the inline limit of "
            f"{MAX_INLINE_SIZE} bytes. Use file_import for large or remote files.",
            kind=ToolErrorKind.VALIDATION,
        )

    await _get_store().write(cleaned, data, mode=mode)
    return {"path": cleaned, "bytes_written": len(data), "mode": mode}


# ------------------------------------------------------------------ #
# file_manage  (create_dir / delete / copy / move / clone)            #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"file", "datarobot", "create", "delete", "copy", "move", "clone"},
    description=(
        "[Files—manage] Run a structural action on the filesystem. action is one of:\n"
        "  'create_dir' — create a new empty catalog item directory; returns its "
        "dr://<catalog_id>/ path. (path/target_path ignored.)\n"
        "  'delete'     — delete file(s)/director(ies) at path. Set recursive=True to "
        "remove a non-empty directory. Silent if the path does not exist.\n"
        "  'copy'       — copy path to target_path. Set recursive=True for directories.\n"
        "  'move'       — move/rename path to target_path. Set recursive=True for directories.\n"
        "  'clone'      — clone a whole catalog item directory (path) to a new catalog "
        "item; returns its id. Use files_to_omit to skip entries.\n\n"
        "Example: file_manage(action='create_dir')\n"
        "Example: file_manage(action='delete', path='dr://abc123/old/', recursive=True)\n"
        "Example: file_manage(action='copy', path='dr://abc123/a.txt', target_path='dr://abc123/b.txt')"
    ),
)
async def file_manage(
    *,
    action: Annotated[
        Literal["create_dir", "delete", "copy", "move", "clone"],
        "Structural action: 'create_dir' | 'delete' | 'copy' | 'move' | 'clone'.",
    ],
    path: Annotated[
        str | None,
        "Source/target path (dr://<catalog_id>/...). Required for all actions except create_dir.",
    ] = None,
    target_path: Annotated[
        str | None,
        "Destination path for 'copy' and 'move' (dr://<catalog_id>/...).",
    ] = None,
    recursive: Annotated[
        bool,
        "Recurse into directories for delete/copy/move. Default False.",
    ] = False,
    maxdepth: Annotated[
        int | None,
        "Maximum recursion depth for delete/copy/move. None means unlimited.",
    ] = None,
    files_to_omit: Annotated[
        list[str] | None,
        "For 'clone', catalog-relative paths to exclude from the clone.",
    ] = None,
) -> dict[str, Any]:
    if maxdepth is not None and maxdepth < 1:
        raise ToolError(
            "Argument validation error: 'maxdepth' must be >= 1 when provided.",
            kind=ToolErrorKind.VALIDATION,
        )

    store = _get_store()

    if action == "create_dir":
        catalog_id = await store.create_dir()
        return {
            "created": True,
            "catalog_id": catalog_id,
            "path": f"dr://{catalog_id}/",
        }

    if action == "delete":
        cleaned = _require_file_path(path)
        await store.delete(cleaned, recursive=recursive, maxdepth=maxdepth)
        return {"deleted": True, "path": cleaned}

    if action == "clone":
        cleaned = _require_file_path(path)
        catalog_id = await store.clone(cleaned, files_to_omit=files_to_omit)
        return {
            "cloned": True,
            "source": cleaned,
            "catalog_id": catalog_id,
            "path": f"dr://{catalog_id}/",
        }

    # copy / move both need a source and a target.
    source = _require_file_path(path)
    target = _require_file_path(target_path, "target_path")
    if action == "copy":
        await store.copy(source, target, recursive=recursive, maxdepth=maxdepth)
        return {"copied": True, "source": source, "target": target}

    await store.move(source, target, recursive=recursive, maxdepth=maxdepth)
    return {"moved": True, "source": source, "target": target}
