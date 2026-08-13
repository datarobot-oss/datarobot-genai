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

from datarobot_genai.drmcputils.auth import get_request_headers
from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_client
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.feature_flags import FeatureFlag
from datarobot_genai.drmcputils.files.store import DataRobotFilesBlobStore
from datarobot_genai.drmcputils.panels.store import CONVERSATION_ID_HEADER
from datarobot_genai.drmcputils.panels.store import PanelStore

logger = logging.getLogger(__name__)

MCP_SANDBOX_FEATURE_FLAG = "ENABLE_MCP_SANDBOX"


def _require_mcp_sandbox() -> None:
    """Fail-closed unless the requesting user holds the ENABLE_MCP_SANDBOX entitlement."""
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


def _request_conversation_id() -> str | None:
    """Return the conversation id carried by the current request, if any.

    Read from the ``x-datarobot-conversation-id`` header (the same header the
    previous panel-library MCP server used), injected per request via
    :func:`datarobot_genai.drmcputils.auth.set_request_headers`. Consumers
    without conversations simply omit the header and get an unscoped store.
    """
    return get_request_headers().get(CONVERSATION_ID_HEADER)


def _get_store() -> PanelStore:
    """Build a per-request panel store, conversation-scoped when the request carries one."""
    return PanelStore(DataRobotFilesBlobStore(), conversation_id=_request_conversation_id())
