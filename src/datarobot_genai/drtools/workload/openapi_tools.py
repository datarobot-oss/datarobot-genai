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

import logging
from pathlib import Path
from typing import Annotated
from typing import Any

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.constants import OPENAPI_BUNDLED_SPEC_CANDIDATES
from datarobot_genai.drmcputils.constants import OPENAPI_DEFAULT_REMOTE_PATH
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient
from datarobot_genai.drtools.core.utils import read_spec_file

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# read_openapi_spec                                                     #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"openapi", "workload", "datarobot", "spec"},
    description=(
        "[OpenAPI spec—read] Fetch or read the DataRobot Workload API OpenAPI "
        "spec. Useful for discovering available endpoints, request/response "
        "schemas, and operation IDs without browsing external docs.\n\n"
        "Resolution order:\n"
        "1. `local_path` — read from a filesystem path (YAML or JSON).\n"
        "2. `remote_path` — fetch from the configured DataRobot endpoint "
        "   (relative path, default 'openapi.json').\n"
        "3. Bundled — search well-known container paths for a local copy.\n\n"
        "Returns a dict with 'info', 'paths', and 'components' keys.\n\n"
        "Example: read_openapi_spec()  "
        "→ fetches openapi.json from the DR endpoint\n"
        "Example: read_openapi_spec(local_path='/app/openapi.yaml')\n"
        "Example: read_openapi_spec(remote_path='openapi.json')"
    ),
)
async def read_openapi_spec(
    *,
    local_path: Annotated[
        str | None,
        "Absolute filesystem path to a local YAML or JSON OpenAPI spec file. "
        "Takes precedence over remote_path.",
    ] = None,
    remote_path: Annotated[
        str | None,
        "Path relative to the configured DataRobot endpoint to fetch the spec "
        "from (e.g. 'openapi.json'). Defaults to 'openapi.json'.",
    ] = None,
) -> dict[str, Any]:
    # 1. Explicit local file
    if local_path:
        p = Path(local_path.strip())
        if not p.exists():
            raise ToolError(
                f"Local spec file not found: {p}",
                kind=ToolErrorKind.NOT_FOUND,
            )
        spec = read_spec_file(p)
        if spec is None:
            raise ToolError(
                f"Failed to parse spec file: {p}",
                kind=ToolErrorKind.UPSTREAM,
            )
        return spec

    # 2. Remote fetch via authenticated DR REST client
    fetch_path = (remote_path or "").strip() or OPENAPI_DEFAULT_REMOTE_PATH
    try:
        return WorkloadApiClient().get_openapi_spec(fetch_path)
    except ClientError as exc:
        logger.debug("Remote spec fetch failed (%s); trying bundled paths.", exc)

    # 3. Bundled fallback
    for candidate in OPENAPI_BUNDLED_SPEC_CANDIDATES:
        spec = read_spec_file(Path(candidate))
        if spec is not None:
            logger.info("Loaded bundled OpenAPI spec from %s", candidate)
            return spec

    raise ToolError(
        "Could not load the OpenAPI spec. "
        "Provide a local_path or ensure the DataRobot endpoint is reachable.",
        kind=ToolErrorKind.NOT_FOUND,
    )
