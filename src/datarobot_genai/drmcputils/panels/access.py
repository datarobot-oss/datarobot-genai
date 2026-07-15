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

"""Shared panel access helpers: ENABLE_MCP_SANDBOX entitlement gate and store factory.

Used by both the panel tools (drtools) and the panel resources (drmcpbase), so it
lives in the shared base rather than in either consumer.
"""

import logging

from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_client
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.feature_flags import FeatureFlag
from datarobot_genai.drmcputils.files.store import DataRobotFilesBlobStore
from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drmcputils.sandbox_mode import is_mcp_sandbox_disabled

logger = logging.getLogger(__name__)

MCP_SANDBOX_FEATURE_FLAG = "ENABLE_MCP_SANDBOX"


def _require_mcp_sandbox() -> None:
    """Fail-closed unless the requesting user holds the ENABLE_MCP_SANDBOX entitlement.

    The ``MCP_SANDBOX_DISABLED`` kill-switch (see
    :mod:`datarobot_genai.drmcputils.sandbox_mode`) turns this guard into a
    no-op: sandboxing is explicitly off, so there is no entitlement to check
    and no DR API call is made. Sandbox-backed execution then runs locally.
    """
    if is_mcp_sandbox_disabled():
        return
    try:
        with request_user_dr_client() as client:
            enabled = FeatureFlag.is_enabled(MCP_SANDBOX_FEATURE_FLAG, client=client)
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - any FF lookup failure denies (fail-closed)
        raise ToolError(
            "Could not verify the ENABLE_MCP_SANDBOX entitlement required for panel tools.",
            kind=ToolErrorKind.AUTHENTICATION,
        ) from exc
    if not enabled:
        raise ToolError(
            "Panel tools require the ENABLE_MCP_SANDBOX entitlement.",
            kind=ToolErrorKind.AUTHENTICATION,
        )


def _get_store() -> PanelStore:
    return PanelStore(DataRobotFilesBlobStore())
