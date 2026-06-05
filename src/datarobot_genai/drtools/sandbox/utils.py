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

"""Sandbox-backed code execution."""

import dataclasses
from typing import Annotated
from typing import Any

from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind
from datarobot_genai.drtools.sandbox.base import SandboxError
from datarobot_genai.drtools.sandbox.base import SandboxTimeout
from datarobot_genai.drtools.sandbox.workload import DEFAULT_SANDBOX_IMAGE
from datarobot_genai.drtools.sandbox.workload import run_request_scoped

MCP_SANDBOX_FEATURE_FLAG = "MCP_SANDBOX"


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
    """Execute Python in a remote workload container; returns stdout/stderr and metadata."""
    try:
        result = await run_request_scoped(
            code, inputs=inputs, timeout_s=timeout_s, image=image
        )
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
