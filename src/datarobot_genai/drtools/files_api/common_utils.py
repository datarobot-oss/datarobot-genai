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

import base64
import binascii

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.files.file_system_store import DR_PROTOCOL
from datarobot_genai.drmcputils.files.file_system_store import DataRobotFileSystemStore
from datarobot_genai.drmcputils.files.file_system_store import FileSystemStore

ROOT_PATH = DR_PROTOCOL


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
