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

"""Sandbox-backed code execution.

:func:`execute_code` is a plain async function for now — deliberately *not*
registered as an MCP tool, so it doesn't collide with the ``execute`` tool that
FastMCP CodeMode exposes (see #376). A later PR will add the MCP tool layer that
calls this function, gated by the ``MCP_SANDBOX`` DR entitlement (PR
datarobot/DataRobot#154256; pattern mirrors datarobot/global-mcp#120).
"""

import dataclasses
import logging
from typing import Annotated
from typing import Any

from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.clients.datarobot import request_user_dr_client
from datarobot_genai.drtools.core.credentials import get_credentials
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind
from datarobot_genai.drtools.core.feature_flags import FeatureFlag
from datarobot_genai.drtools.sandbox.base import SandboxError
from datarobot_genai.drtools.sandbox.base import SandboxSecurityContext
from datarobot_genai.drtools.sandbox.base import SandboxTimeout
from datarobot_genai.drtools.sandbox.workload import DataRobotWorkloadSandbox

logger = logging.getLogger(__name__)

# Entitlement the future MCP tool layer will gate registration on.
MCP_SANDBOX_FEATURE_FLAG = "MCP_SANDBOX"
DEFAULT_SANDBOX_IMAGE = (
    "datarobotdev/datarobot-user-models:"
    "public_dropin_environments_dr_mcp_execute_sandbox_minimal_latest"
)


def _resolve_security_context() -> SandboxSecurityContext | None:
    # Send a tightened securityContext only when the workload-api accepts it
    # (gated by ENABLE_WORKLOAD_API_SECURITY_CONTEXT, see
    # datarobot/DataRobot#153183). On any FF lookup error, fail safe to
    # "no context" — the workload-api applies cluster defaults.
    try:
        with request_user_dr_client() as client:
            enabled = FeatureFlag.is_enabled("ENABLE_WORKLOAD_API_SECURITY_CONTEXT", client=client)
    except Exception as exc:  # noqa: BLE001
        logger.debug("WORKLOAD_API_SECURITY_CONTEXT FF check raised: %r", exc)
        return None
    return SandboxSecurityContext() if enabled else None


async def execute_code(
    code: Annotated[str, "Python source to execute. Assign to `_return` to return a value."],
    *,
    inputs: Annotated[
        dict[str, Any] | None,
        "Optional JSON-serializable mapping bound as `inputs` in the sandbox.",
    ] = None,
    timeout_s: Annotated[float, "Max wall-clock time before forced teardown."] = 30.0,
    image: Annotated[
        str, "Container image URI. Defaults to the minimal sandbox runner."
    ] = DEFAULT_SANDBOX_IMAGE,
) -> dict[str, Any]:
    """Execute Python code in an isolated, short-lived DataRobot workload container.

    The container runs with a locked-down securityContext (read-only rootfs,
    drop ALL caps, no privilege escalation) and is force-deleted in a finally
    block on success, failure, timeout, or cancellation. Returns a dict with
    stdout/stderr, an optional ``_return`` value the code may assign, duration,
    exit code, and the image used. The default image is built by DRUM at
    ``datarobotdev/datarobot-user-models:public_dropin_environments_dr_mcp_execute_sandbox_minimal_latest``
    and ships polars, pyarrow, datarobot, requests.

    This is a plain function — a later PR adds the MCP tool layer that calls it
    (gated by ``MCP_SANDBOX``), kept separate so it doesn't override FastMCP
    CodeMode's ``execute`` tool.
    """
    # Derive credentials the same way the rest of drtools / MCP does, rather
    # than reading DATAROBOT_ENDPOINT / DATAROBOT_API_TOKEN off os.environ: the
    # requesting user's token comes from the request headers (falling back to
    # the application token only in non-HTTP contexts), and the endpoint comes
    # from configured credentials. `get_datarobot_access_token` raises
    # ToolError(AUTHENTICATION) when no token is available.
    token = get_datarobot_access_token()
    endpoint = get_credentials().datarobot.datarobot_endpoint

    sandbox = DataRobotWorkloadSandbox(
        image=image,
        datarobot_endpoint=endpoint,
        datarobot_api_token=token,
        security_context=_resolve_security_context(),
    )
    try:
        result = await sandbox.run(code, inputs=inputs, timeout_s=timeout_s)
    except SandboxTimeout as exc:
        raise ToolError(
            f"Sandbox execution timed out after {timeout_s}s: {exc}",
            kind=ToolErrorKind.UPSTREAM,
        ) from exc
    except SandboxError as exc:
        raise ToolError(
            f"Sandbox execution failed: {exc}",
            kind=ToolErrorKind.UPSTREAM,
        ) from exc

    payload = dataclasses.asdict(result)
    payload["image"] = image
    return payload
