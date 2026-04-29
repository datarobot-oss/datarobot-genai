# Copyright 2025 DataRobot, Inc.
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

"""Google Drive MCP tools for interacting with Google Drive API."""

import logging
from typing import Annotated
from typing import Any
from typing import Literal

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.gdrive import LIMIT
from datarobot_genai.drtools.core.clients.gdrive import MAX_PAGE_SIZE
from datarobot_genai.drtools.core.clients.gdrive import SUPPORTED_FIELDS
from datarobot_genai.drtools.core.clients.gdrive import SUPPORTED_FIELDS_STR
from datarobot_genai.drtools.core.clients.gdrive import GoogleDriveClient
from datarobot_genai.drtools.core.clients.gdrive import get_gdrive_access_token
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)


@tool_metadata(
    tags={"gdrive", "google", "list", "search", "files", "find", "contents"},
    description=(
        "[GDrive—find files] Use when the user needs Google Drive file names and ids with "
        "optional folder scope, Drive query string, and pagination. Not file body text "
        "(gdrive_read_content), not SharePoint (microsoft_graph_search_content)."
    ),
)
async def gdrive_find_contents(
    *,
    page_size: Annotated[
        int, f"Maximum number of files to return per page (max {MAX_PAGE_SIZE})."
    ] = 10,
    limit: Annotated[int, f"Total maximum number of files to return (max {LIMIT})."] = 50,
    page_token: Annotated[
        str | None, "The token for the next page of results, retrieved from a previous call."
    ] = None,
    query: Annotated[
        str | None, "Optional filter to narrow results (e.g., 'trashed = false')."
    ] = None,
    folder_id: Annotated[
        str | None,
        "The ID of a specific folder to list or search within. "
        "If omitted, searches the entire Drive.",
    ] = None,
    recursive: Annotated[
        bool,
        "If True, searches all subfolders. "
        "If False and folder_id is provided, only lists immediate children.",
    ] = False,
    fields: Annotated[
        list[str] | None,
        "Optional list of metadata fields to include. Ex. id, name, mimeType. "
        f"Default = {SUPPORTED_FIELDS_STR}",
    ] = None,
) -> dict[str, Any]:
    access_token = await get_gdrive_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with GoogleDriveClient(access_token) as client:
        data = await client.list_files(
            page_size=page_size,
            page_token=page_token,
            query=query,
            limit=limit,
            folder_id=folder_id,
            recursive=recursive,
        )

    filtered_fields = set(fields).intersection(SUPPORTED_FIELDS) if fields else SUPPORTED_FIELDS
    number_of_files = len(data.files)

    return {
        "files": [file.model_dump(by_alias=True, include=filtered_fields) for file in data.files],
        "count": number_of_files,
        "nextPageToken": data.next_page_token,
    }


@tool_metadata(
    tags={"gdrive", "google", "read", "content", "file", "download"},
    description=(
        "[GDrive—read file] Use when you have a Drive file_id (from gdrive_find_contents) and need "
        "exported text or markdown for Workspace types. Not listing files, not binary media "
        "download."
    ),
)
async def gdrive_read_content(
    *,
    file_id: Annotated[str, "The ID of the file to read."],
    target_format: Annotated[
        str | None,
        "The preferred output format for Google Workspace files "
        "(e.g., 'text/markdown' for Docs, 'text/csv' for Sheets). "
        "If not specified, uses sensible defaults. Has no effect on regular files.",
    ] = None,
) -> dict[str, Any]:
    if not file_id or not file_id.strip():
        raise ToolError(
            "Argument validation error: 'file_id' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    access_token = await get_gdrive_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with GoogleDriveClient(access_token) as client:
        file_content = await client.read_file_content(file_id, target_format)

    return file_content.as_flat_dict()


@tool_metadata(
    tags={"gdrive", "google", "create", "write", "file", "folder"},
    enabled=False,
    description=(
        "[GDrive—create file] Use when creating a new Drive file or folder from name + MIME type, "
        "optional parent folder and initial text. Not search (gdrive_find_contents), not "
        "metadata-only updates (gdrive_update_metadata)."
    ),
)
async def gdrive_create_file(
    *,
    name: Annotated[str, "The name for the new file or folder."],
    mime_type: Annotated[
        str,
        "The MIME type of the file (e.g., 'text/plain', "
        "'application/vnd.google-apps.document', 'application/vnd.google-apps.folder').",
    ],
    parent_id: Annotated[
        str | None, "The ID of the parent folder where the file should be created."
    ] = None,
    initial_content: Annotated[
        str | None, "Text content to populate the new file, if applicable."
    ] = None,
) -> dict[str, Any]:
    if not name or not name.strip():
        raise ToolError(
            "Argument validation error: 'name' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    if not mime_type or not mime_type.strip():
        raise ToolError(
            "Argument validation error: 'mime_type' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    access_token = await get_gdrive_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with GoogleDriveClient(access_token) as client:
        created_file = await client.create_file(
            name=name,
            mime_type=mime_type,
            parent_id=parent_id,
            initial_content=initial_content,
        )

    return created_file.as_flat_dict()


@tool_metadata(
    tags={"gdrive", "google", "update", "metadata", "rename", "star", "trash"},
    enabled=False,
    description=(
        "[GDrive—metadata] Use when renaming, starring, or trashing an existing file by id. "
        "Not reading content (gdrive_read_content), not ACL changes (gdrive_manage_access)."
    ),
)
async def gdrive_update_metadata(
    *,
    file_id: Annotated[str, "The ID of the file or folder to update."],
    new_name: Annotated[str | None, "A new name to rename the file."] = None,
    starred: Annotated[bool | None, "Set to True to star the file or False to unstar it."] = None,
    trash: Annotated[bool | None, "Set to True to trash the file or False to restore it."] = None,
) -> dict[str, Any]:
    if not file_id or not file_id.strip():
        raise ToolError(
            "Argument validation error: 'file_id' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    if new_name is None and starred is None and trash is None:
        raise ToolError(
            "Argument validation error: at least one of 'new_name', 'starred', or 'trash' "
            "must be provided.",
            kind=ToolErrorKind.VALIDATION,
        )

    if new_name is not None and not new_name.strip():
        raise ToolError(
            "Argument validation error: 'new_name' cannot be empty or whitespace.",
            kind=ToolErrorKind.VALIDATION,
        )

    access_token = await get_gdrive_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with GoogleDriveClient(access_token) as client:
        updated_file = await client.update_file_metadata(
            file_id=file_id,
            new_name=new_name,
            starred=starred,
            trashed=trash,
        )

    return updated_file.as_flat_dict()


@tool_metadata(
    tags={"gdrive", "google", "manage", "access", "acl"},
    enabled=False,
    description=(
        "[GDrive—permissions] Use when adding, changing, or removing sharing on a file by id "
        "(email or permission id). Not rename/trash (gdrive_update_metadata), not read content."
    ),
)
async def gdrive_manage_access(
    *,
    file_id: Annotated[str, "The ID of the file or folder."],
    action: Annotated[
        Literal["add", "update", "remove"],
        "Pass exactly one of these strings: 'add', 'update', 'remove'.",
    ],
    role: Annotated[
        Literal["reader", "commenter", "writer", "fileOrganizer", "organizer", "owner"] | None,
        "Required for 'add' and 'update' (omit for 'remove'). Pass exactly one of these strings "
        "(camelCase as shown): 'reader', 'commenter', 'writer', 'fileOrganizer', 'organizer', "
        "'owner'.",
    ] = None,
    email_address: Annotated[
        str | None, "The email of the user or group (required for 'add')."
    ] = None,
    permission_id: Annotated[
        str | None, "The specific permission ID (required for 'update' or 'remove')."
    ] = None,
    transfer_ownership: Annotated[
        bool, "Whether to transfer ownership (only for 'update' to 'owner' role)."
    ] = False,
) -> dict[str, Any]:
    if not file_id or not file_id.strip():
        raise ToolError(
            "Argument validation error: 'file_id' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    if action == "add" and not email_address:
        raise ToolError(
            "'email_address' is required for action 'add'.", kind=ToolErrorKind.VALIDATION
        )

    if action in ("update", "remove") and not permission_id:
        raise ToolError(
            "'permission_id' is required for action 'update' or 'remove'.",
            kind=ToolErrorKind.VALIDATION,
        )

    if action != "remove" and not role:
        raise ToolError(
            "'role' is required for action 'add' or 'update'.", kind=ToolErrorKind.VALIDATION
        )

    access_token = await get_gdrive_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with GoogleDriveClient(access_token) as client:
        permission_id = await client.manage_access(
            file_id=file_id,
            action=action,
            role=role,
            email_address=email_address,
            permission_id=permission_id,
            transfer_ownership=transfer_ownership,
        )

    # Build response
    structured_content = {"affectedFileId": file_id}
    if action == "add":
        structured_content["newPermissionId"] = permission_id

    return structured_content
