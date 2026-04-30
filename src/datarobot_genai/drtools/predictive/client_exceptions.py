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

"""Helpers for DataRobot SDK HTTP errors in predictive tools."""

from typing import NoReturn

from datarobot.errors import ClientError

from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind


def raise_tool_error_for_client_error(exc: ClientError) -> NoReturn:
    """Raise :class:`ToolError` from SDK text: 404 → ``NOT_FOUND``, else ``UPSTREAM``."""
    sc = getattr(exc, "status_code", None)
    msg = f"DataRobot API error ({sc}): {exc}"
    kind = ToolErrorKind.NOT_FOUND if sc == 404 else ToolErrorKind.UPSTREAM
    raise ToolError(msg, kind=kind) from exc
