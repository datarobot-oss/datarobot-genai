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
from typing import Literal

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.tools.clients.gdrive import GOOGLE_DRIVE_FOLDER_MIME
from datarobot_genai.drmcp.tools.clients.gdrive import LIMIT
from datarobot_genai.drmcp.tools.clients.gdrive import MAX_PAGE_SIZE
from datarobot_genai.drmcp.tools.clients.gdrive import SUPPORTED_FIELDS
from datarobot_genai.drmcp.tools.clients.gdrive import SUPPORTED_FIELDS_STR
from datarobot_genai.drmcp.tools.clients.gdrive import GoogleDriveClient
from datarobot_genai.drmcp.tools.clients.gdrive import get_gdrive_access_token

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"google", "gdrive", "list", "search", "files", "find", "contents"})
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
) -> ToolResult:
    """
    Search or list files in the user's Google Drive with pagination and filtering support.
    Use this tool to discover file names and IDs for use with other tools.

    Limit must be bigger than or equal to page size and it must be multiplication of page size.
    Ex.
        page size = 10 limit = 50
        page size = 3 limit = 3
        page size = 12 limit = 36
    """
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
    next_page_info = (
        f"Next page token needed to fetch more data: {data.next_page_token}"
        if data.next_page_token
        else "There're no more pages."
    )
    return ToolResult(
        content=f"Successfully listed {number_of_files} files. {next_page_info}",
        structured_content={
            "files": [
                file.model_dump(by_alias=True, include=filtered_fields) for file in data.files
            ],
            "count": number_of_files,
            "nextPageToken": data.next_page_token,
        },
    )


@dr_mcp_tool(tags={"google", "gdrive", "read", "content", "file", "download"})
async def gdrive_read_content(
    *,
    file_id: Annotated[str, "The ID of the file to read."],
    target_format: Annotated[
        str | None,
        "The preferred output format for Google Workspace files "
        "(e.g., 'text/markdown' for Docs, 'text/csv' for Sheets). "
        "If not specified, uses sensible defaults. Has no effect on regular files.",
    ] = None,
) -> ToolResult:
    """
    Retrieve the content of a specific file by its ID. Google Workspace files are
    automatically exported to LLM-readable formats (Push-Down).

    Usage:
        - Basic: gdrive_read_content(file_id="1ABC123def456")
        - Custom format: gdrive_read_content(file_id="1ABC...", target_format="text/plain")
        - First use gdrive_find_contents to discover file IDs

    Supported conversions (defaults):
        - Google Docs -> Markdown (text/markdown)
        - Google Sheets -> CSV (text/csv)
        - Google Slides -> Plain text (text/plain)
        - PDF files -> Extracted text (text/plain)
        - Other text files -> Downloaded as-is

    Note: Binary files (images, videos, etc.) are not supported and will return an error.
    Large Google Workspace files (>10MB) may fail to export due to API limits.

    Refer to Google Drive export formats documentation:
    https://developers.google.com/workspace/drive/api/guides/ref-export-formats
    """
    if not file_id or not file_id.strip():
        raise ToolError("Argument validation error: 'file_id' cannot be empty.")

    access_token = await get_gdrive_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with GoogleDriveClient(access_token) as client:
        file_content = await client.read_file_content(file_id, target_format)

    export_info = ""
    if file_content.was_exported:
        export_info = f" (exported from {file_content.original_mime_type})"

    return ToolResult(
        content=(
            f"Successfully retrieved content of '{file_content.name}' "
            f"({file_content.mime_type}){export_info}."
        ),
        structured_content=file_content.as_flat_dict(),
    )


@dr_mcp_tool(tags={"google", "gdrive", "create", "write", "file", "folder"}, enabled=False)
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
) -> ToolResult:
    """
    Create a new file or folder in Google Drive.

    This tool is essential for an AI agent to generate new output (like reports or
    documentation) directly into the Drive structure.

    Usage:
        - Create empty file: gdrive_create_file(name="report.txt", mime_type="text/plain")
        - Create Google Doc: gdrive_create_file(
            name="My Report",
            mime_type="application/vnd.google-apps.document",
            initial_content="# Report Title"
          )
        - Create folder: gdrive_create_file(
            name="Reports",
            mime_type="application/vnd.google-apps.folder"
          )
        - Create in subfolder: gdrive_create_file(
            name="file.txt",
            mime_type="text/plain",
            parent_id="folder_id_here",
            initial_content="File content"
          )

    Supported MIME types:
        - text/plain: Plain text file
        - application/vnd.google-apps.document: Google Doc (content auto-converted)
        - application/vnd.google-apps.spreadsheet: Google Sheet (CSV content works best)
        - application/vnd.google-apps.folder: Folder (initial_content is ignored)

    Note: For Google Workspace files, the Drive API automatically converts plain text
    content to the appropriate format.
    """
    if not name or not name.strip():
        raise ToolError("Argument validation error: 'name' cannot be empty.")

    if not mime_type or not mime_type.strip():
        raise ToolError("Argument validation error: 'mime_type' cannot be empty.")

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

    file_type = "folder" if mime_type == GOOGLE_DRIVE_FOLDER_MIME else "file"
    content_info = ""
    if initial_content and mime_type != GOOGLE_DRIVE_FOLDER_MIME:
        content_info = " with initial content"

    return ToolResult(
        content=f"Successfully created {file_type} '{created_file.name}'{content_info}.",
        structured_content=created_file.as_flat_dict(),
    )


@dr_mcp_tool(
    tags={"google", "gdrive", "update", "metadata", "rename", "star", "trash"}, enabled=False
)
async def gdrive_update_metadata(
    *,
    file_id: Annotated[str, "The ID of the file or folder to update."],
    new_name: Annotated[str | None, "A new name to rename the file."] = None,
    starred: Annotated[bool | None, "Set to True to star the file or False to unstar it."] = None,
    trash: Annotated[bool | None, "Set to True to trash the file or False to restore it."] = None,
) -> ToolResult:
    """
    Update non-content metadata fields of a Google Drive file or folder.

    This tool allows you to:
    - Rename files and folders by setting new_name
    - Star or unstar files (per-user preference) with starred
    - Move files to trash or restore them with trash

    Usage:
        - Rename: gdrive_update_metadata(file_id="1ABC...", new_name="New Name.txt")
        - Star: gdrive_update_metadata(file_id="1ABC...", starred=True)
        - Unstar: gdrive_update_metadata(file_id="1ABC...", starred=False)
        - Trash: gdrive_update_metadata(file_id="1ABC...", trash=True)
        - Restore: gdrive_update_metadata(file_id="1ABC...", trash=False)
        - Multiple: gdrive_update_metadata(file_id="1ABC...", new_name="New.txt", starred=True)

    Note:
        - At least one of new_name, starred, or trash must be provided.
        - Starring is per-user: starring a shared file only affects your view.
        - Trashing a folder trashes all contents recursively.
        - Trashing requires permissions (owner for My Drive, organizer for Shared Drives).
    """
    if not file_id or not file_id.strip():
        raise ToolError("Argument validation error: 'file_id' cannot be empty.")

    if new_name is None and starred is None and trash is None:
        raise ToolError(
            "Argument validation error: at least one of 'new_name', 'starred', or 'trash' "
            "must be provided."
        )

    if new_name is not None and not new_name.strip():
        raise ToolError("Argument validation error: 'new_name' cannot be empty or whitespace.")

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

    changes: list[str] = []
    if new_name is not None:
        changes.append(f"renamed to '{new_name}'")
    if starred is True:
        changes.append("starred")
    elif starred is False:
        changes.append("unstarred")
    if trash is True:
        changes.append("moved to trash")
    elif trash is False:
        changes.append("restored from trash")

    changes_description = ", ".join(changes)

    return ToolResult(
        content=f"Successfully updated file '{updated_file.name}': {changes_description}.",
        structured_content=updated_file.as_flat_dict(),
    )


@dr_mcp_tool(tags={"google", "gdrive", "manage", "access", "acl"}, enabled=False)
async def gdrive_manage_access(
    *,
    file_id: Annotated[str, "The ID of the file or folder."],
    action: Annotated[Literal["add", "update", "remove"], "The operation to perform."],
    role: Annotated[
        Literal["reader", "commenter", "writer", "fileOrganizer", "organizer", "owner"] | None,
        "The access level.",
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
) -> ToolResult:
    """
    Consolidated tool for sharing files and managing permissions.
    Pushes all logic to the Google Drive API permissions resource (create, update, delete).

    Usage:
        - Add role: gdrive_manage_access(
            file_id="SomeFileId",
            action="add",
            role="reader",
            email_address="dummy@user.com"
          )
        - Update role: gdrive_manage_access(
            file_id="SomeFileId",
            action="update",
            role="reader",
            permission_id="SomePermissionId"
          )
        - Remove permission: gdrive_manage_access(
            file_id="SomeFileId",
            action="remove",
            permission_id="SomePermissionId"
          )
    """
    if not file_id or not file_id.strip():
        raise ToolError("Argument validation error: 'file_id' cannot be empty.")

    if action == "add" and not email_address:
        raise ToolError("'email_address' is required for action 'add'.")

    if action in ("update", "remove") and not permission_id:
        raise ToolError("'permission_id' is required for action 'update' or 'remove'.")

    if action != "remove" and not role:
        raise ToolError("'role' is required for action 'add' or 'update'.")

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
        content = (
            f"Successfully added role '{role}' for '{email_address}' for gdrive file '{file_id}'. "
            f"New permission id '{permission_id}'."
        )
        structured_content["newPermissionId"] = permission_id
    elif action == "update":
        content = (
            f"Successfully updated role '{role}' (permission '{permission_id}') "
            f"for gdrive file '{file_id}'."
        )
    else:  # action == "remove":
        content = f"Successfully removed permission '{permission_id}' for gdrive file '{file_id}'."

    return ToolResult(content=content, structured_content=structured_content)
