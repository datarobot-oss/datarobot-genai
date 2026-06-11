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

"""DataRobot use case tools."""

import logging
from typing import Annotated
from typing import Any

import datarobot as dr
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error

logger = logging.getLogger(__name__)


@tool_metadata(
    tags={"use_case", "read", "list", "daria"},
    description=(
        "[Use case—list] Use when the user needs DataRobot use cases (workspace bundles) as "
        "id+name, optionally filtered by name search. Read-only. Not assets inside a known case "
        "(usecases_list_assets), not modeling project ids (modeling_list_projects), "
        "not deployments alone "
        "(deployment_get_list)."
    ),
)
async def datarobot_usecases_list(
    *,
    search: Annotated[str | None, "Optional search filter for use case names"] = None,
    limit: Annotated[int, "Maximum number of use cases to return"] = 100,
) -> dict[str, Any]:
    if search is not None and (not search or not search.strip()):
        raise ToolError(
            "Argument validation error: 'search' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    params: dict = {"limit": limit}
    if search:
        params["search"] = search

    with ThreadSafeDataRobotClient().request_user_client():
        rest_client = dr.client.get_client()
        try:
            response = rest_client.get("useCases/", params=params)
        except ClientError as e:
            raise_tool_error_for_client_error(e)
        items = response.json().get("data", [])

    return {
        "use_cases": [{"id": item["id"], "name": item.get("name", "Untitled")} for item in items],
        "count": len(items),
    }


@tool_metadata(
    tags={"use_case", "read", "assets", "daria"},
    description=(
        "[Use case—assets] Use when you have one or more use_case_id values and need what is "
        "linked: datasets, deployments, and experiments as id+name per case. Read-only. Not "
        "discovering use case ids (datarobot_usecases_list), not Jira/Confluence content."
    ),
)
async def usecases_list_assets(
    *,
    use_case_id: Annotated[str | None, "The ID of a single DataRobot use case"] = None,
    use_case_ids: Annotated[list[str] | None, "List of use case IDs to fetch assets for"] = None,
) -> dict[str, Any]:
    ids: list[str] = []
    if use_case_ids is not None:
        if not use_case_ids:
            raise ToolError("use_case_ids must not be empty.", kind=ToolErrorKind.VALIDATION)
        # Validate each ID in the list
        for uc_id in use_case_ids:
            if not uc_id or not uc_id.strip():
                raise ToolError(
                    "Argument validation error: use_case IDs in list cannot be empty.",
                    kind=ToolErrorKind.VALIDATION,
                )
        ids = use_case_ids
    elif use_case_id:
        if not use_case_id or not use_case_id.strip():
            raise ToolError(
                "Argument validation error: 'use_case_id' cannot be empty.",
                kind=ToolErrorKind.VALIDATION,
            )
        ids = [use_case_id]
    else:
        raise ToolError(
            "Either use_case_id or use_case_ids must be provided.", kind=ToolErrorKind.VALIDATION
        )

    results: list[dict] = []
    with ThreadSafeDataRobotClient().request_user_client():
        for uc_id in ids:
            entry: dict = {"use_case_id": uc_id}

            try:
                use_case = dr.UseCase.get(uc_id)
            except Exception as exc:
                entry["error"] = str(exc)
                results.append(entry)
                continue

            entry["name"] = use_case.name

            try:
                datasets = list(use_case.list_datasets())
                entry["datasets"] = [{"id": d.id, "name": d.name} for d in datasets]
            except Exception as exc:
                entry["datasets_error"] = str(exc)

            try:
                deployments = list(use_case.list_deployments())
                entry["deployments"] = [{"id": d.id, "label": d.label} for d in deployments]
            except Exception as exc:
                entry["deployments_error"] = str(exc)

            try:
                projects = list(use_case.list_projects())
                entry["experiments"] = [{"id": p.id, "name": p.project_name} for p in projects]
            except Exception as exc:
                entry["experiments_error"] = str(exc)

            results.append(entry)

    return {"use_cases": results, "count": len(results)}
