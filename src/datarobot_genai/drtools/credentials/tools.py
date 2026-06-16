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

from typing import Annotated
from typing import Any

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.constants import CREDENTIAL_TYPE_KEYS
from datarobot_genai.drmcputils.constants import KNOWN_CREDENTIAL_TYPES
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import DataRobotApiClient
from datarobot_genai.drtools.core.utils import require_id
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata

# ------------------------------------------------------------------ #
# credential_list                                                       #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"credential", "datarobot", "list"},
    description=(
        "[Credentials—list] List all DataRobot credentials accessible to the "
        "current user. Returns credentialId, name, credentialType, and "
        "creationDate for each entry. Use credential_get for the full record "
        "of a specific credential.\n\n"
        "Example: credential_list(limit=50)"
    ),
)
async def credential_list(
    *,
    limit: Annotated[int, "Maximum credentials to return (1–100). Default 100."] = 100,
    offset: Annotated[int, "Number of credentials to skip for pagination. Default 0."] = 0,
) -> dict[str, Any]:
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    clamped_limit, note = clamp_limit(limit)
    try:
        result = DataRobotApiClient().list_credentials(limit=clamped_limit, offset=offset)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    return merge_pagination_metadata(
        {"credentials": data, "count": len(data)},
        result,
        note,
        offset=offset,
        limit=clamped_limit,
    )


# ------------------------------------------------------------------ #
# credential_get                                                        #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"credential", "datarobot", "get"},
    description=(
        "[Credentials—get] Fetch the full record of a DataRobot credential by "
        "its id. Returns credentialId, name, credentialType, description, and "
        "creationDate. Secrets (passwords, tokens) are masked by the server.\n\n"
        "Example: credential_get(credential_id='cred-abc123')"
    ),
)
async def credential_get(
    *,
    credential_id: Annotated[str, "Id of the credential to fetch."],
) -> dict[str, Any]:
    cid = require_id(credential_id, "credential_id")
    try:
        return DataRobotApiClient().get_credential(cid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# credential_keys                                                       #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"credential", "datarobot", "keys"},
    description=(
        "[Credentials—keys] Return the valid `key` field names for a given "
        "DataRobot credential type. These are the values to supply in the "
        "`key` field of a CredentialEnvironmentVariable "
        "(source: 'dr-credential') when wiring credentials into a workload's "
        "environment.\n\n"
        f"Known credential types: {', '.join(KNOWN_CREDENTIAL_TYPES)}.\n\n"
        "Example: credential_keys(credential_type='basic')  "
        "→ ['login', 'password']\n"
        "Example: credential_keys(credential_type='s3')  "
        "→ ['awsAccessKeyId', 'awsSecretAccessKey', 'awsSessionToken']"
    ),
)
async def credential_keys(
    *,
    credential_type: Annotated[
        str,
        "The credential type string (e.g. 'basic', 's3', 'gcp', 'oauth'). "
        "Matches the credentialType field returned by credential_list/get.",
    ],
) -> dict[str, Any]:
    ct = (credential_type or "").strip().lower()
    if not ct:
        raise ToolError(
            "Argument validation error: 'credential_type' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    keys = CREDENTIAL_TYPE_KEYS.get(ct)
    if keys is None:
        raise ToolError(
            f"Unknown credential type '{ct}'. Known types: {', '.join(KNOWN_CREDENTIAL_TYPES)}.",
            kind=ToolErrorKind.VALIDATION,
        )
    return {
        "credentialType": ct,
        "keys": keys,
        "usage": (
            "Use these key names in the `key` field of a "
            "CredentialEnvironmentVariable when configuring workload environment "
            "variables that reference this credential."
        ),
    }
