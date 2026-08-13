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

"""Async import tools for the DataRobot Files API filesystem.

Large or remote payloads are ingested via background jobs. ``file_import`` starts
the job and returns a ``status_id``; ``file_get_status`` performs a single
non-blocking status fetch so callers can poll until completion.
"""

from __future__ import annotations

import logging
from typing import Annotated
from typing import Any
from typing import Literal

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.files.file_system_store import is_terminal_import_failure_status
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.files_api.common_utils import OVERWRITE_STRATEGY_DOC
from datarobot_genai.drtools.files_api.common_utils import get_store as _get_store
from datarobot_genai.drtools.files_api.common_utils import require_file_path as _require_file_path
from datarobot_genai.drtools.files_api.common_utils import require_path as _require_path

logger = logging.getLogger(__name__)

_IMPORT_STATUS_NOTE = (
    "Import started. Poll file_get_status(status_id=..., target_status='completed') "
    "every few seconds until target_reached is true."
)


# ------------------------------------------------------------------ #
# file_import                                                          #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"file", "datarobot", "import", "upload", "url", "data_source"},
    description=(
        "[Files—import] Start a background import into a catalog directory. "
        "Use for large or remote files instead of file_write. Returns immediately "
        "with a status_id; poll file_get_status until completion.\n"
        "  - source='url': set url to a file or archive URL reachable by DataRobot.\n"
        "  - source='data_source': set data_source_id (and optional credential_id).\n"
        "  - unpack_archive: extract zip/tar archives when true (default).\n"
        f"  - {OVERWRITE_STRATEGY_DOC}\n\n"
        "Example: file_import(path='dr://abc123/data/', source='url', "
        "url='https://example.com/report.zip')\n"
        "Example: file_import(path='dr://abc123/in/', source='data_source', "
        "data_source_id='ds-123')"
    ),
    display_name="Files — Import",
    description_ui=(
        "Starts a background import of a large or remote file from a URL or data "
        "source into a catalog directory."
    ),
)
async def file_import(
    *,
    path: Annotated[
        str,
        "Destination directory under a catalog item (dr://<catalog_id>/folder/).",
    ],
    source: Annotated[
        Literal["url", "data_source"],
        "Import source type: 'url' or 'data_source'.",
    ],
    url: Annotated[
        str | None,
        "Remote file or archive URL. Required when source='url'.",
    ] = None,
    data_source_id: Annotated[
        str | None,
        "DataRobot data source ID. Required when source='data_source'.",
    ] = None,
    credential_id: Annotated[
        str | None,
        "Optional credential ID for data_source imports.",
    ] = None,
    unpack_archive: Annotated[
        bool,
        "Extract archive contents when true; upload as-is when false. Default true.",
    ] = True,
    overwrite: Annotated[
        Literal["rename", "replace", "skip", "error"],
        f"{OVERWRITE_STRATEGY_DOC} Default 'rename'.",
    ] = "rename",
) -> dict[str, Any]:
    cleaned = _require_file_path(path)
    store = _get_store()

    if source == "url":
        remote_url = _require_path(url, "url")
        status_id = await store.import_from_url(
            cleaned,
            remote_url,
            unpack_archive=unpack_archive,
            overwrite=overwrite,
        )
    else:
        ds_id = _require_path(data_source_id, "data_source_id")
        status_id = await store.import_from_data_source(
            cleaned,
            ds_id,
            credential_id=credential_id,
            unpack_archive=unpack_archive,
            overwrite=overwrite,
        )

    return {
        "path": cleaned,
        "source": source,
        "status_id": status_id,
        "note": _IMPORT_STATUS_NOTE,
    }


# ------------------------------------------------------------------ #
# file_get_status                                                      #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"file", "datarobot", "import", "status"},
    description=(
        "[Files—import status] Fetch the current status of a background import "
        "started by file_import. Performs ONE non-blocking fetch — it does not "
        "wait or poll. Pass target_status='completed' to get target_reached. "
        "Raises if the import enters a terminal failure state.\n\n"
        "Example: file_get_status(status_id='abc123')\n"
        "Example: file_get_status(status_id='abc123', target_status='completed')"
    ),
    display_name="Files — Import status",
    description_ui="Fetches the current status of a background import.",
)
async def file_get_status(
    *,
    status_id: Annotated[str, "Status ID returned by file_import."],
    target_status: Annotated[
        str | None,
        "Optional status to compare against, e.g. 'completed'. Does not block.",
    ] = None,
) -> dict[str, Any]:
    sid = _require_path(status_id, "status_id")
    if target_status is not None:
        target = target_status.strip()
        if not target:
            raise ToolError(
                "Argument validation error: 'target_status' cannot be empty.",
                kind=ToolErrorKind.VALIDATION,
            )
    else:
        target = None

    raw = await _get_store().get_status(sid)
    status = raw.get("status")
    if is_terminal_import_failure_status(str(status) if status is not None else None):
        raise ToolError(
            f"Import {sid!r} entered terminal status {status!r}.",
            kind=ToolErrorKind.UPSTREAM,
        )

    if target is None:
        return {"status_id": sid, "status": status, "raw": raw}

    normalized_status = str(status).lower() if status is not None else ""
    normalized_target = target.lower()
    return {
        "status_id": sid,
        "status": status,
        "target_reached": normalized_status == normalized_target,
        "raw": raw,
    }
