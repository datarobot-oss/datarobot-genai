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

import base64
import binascii
import io
import itertools
import logging
from typing import Annotated
from typing import Any

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.utils import is_valid_url

logger = logging.getLogger(__name__)

# Max page / batch size for predictive `data` tools
DR_PREDICTIVE_API_PAGINATION_MAX = 1000


def _serialize_datastore_params(params: Any) -> dict[str, Any]:
    """Return JSON-serializable params; DataRobot SDK uses DataStoreParameters, not a dict."""
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    collect = getattr(params, "collect_payload", None)
    if callable(collect):
        payload = collect()
        if isinstance(payload, dict):
            return payload
    return {}


def _merge_pagination_metadata(
    final_results: dict[str, Any],
    api_response: dict[str, Any] | list,
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Add offset/limit echo and DataRobot list pagination (next, previous, total) when present."""
    if offset is not None:
        final_results["offset"] = offset
    if limit is not None:
        final_results["limit"] = limit
    if isinstance(api_response, dict):
        for key in ("next", "previous"):
            if key in api_response and api_response[key] is not None:
                final_results[key] = api_response[key]
        for total_key in ("total_count", "total"):
            if total_key in api_response and api_response[total_key] is not None:
                final_results["total_count"] = api_response[total_key]
                break
    return final_results


@tool_metadata(tags={"predictive", "data", "write", "upload", "catalog", "daria"})
async def upload_dataset_to_ai_catalog(
    *,
    file_content_base64: Annotated[
        str,
        (
            "Base64-encoded file bytes (e.g. CSV). For remote clients; "
            "mutually exclusive with file_url."
        ),
    ]
    | None = None,
    dataset_filename: Annotated[
        str,
        "Filename for base64 upload; include extension (e.g. data.csv).",
    ] = "data.csv",
    file_url: Annotated[str, "HTTPS URL of a dataset file to register in the catalog."]
    | None = None,
) -> dict[str, Any]:
    """Upload a dataset to the DataRobot AI Catalog / Data Registry from bytes or URL."""
    if not file_content_base64 and not file_url:
        raise ToolError("Either file_content_base64 or file_url must be provided.")
    if file_content_base64 and file_url:
        raise ToolError("Please provide either file_content_base64 or file_url, not both.")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    catalog_item = None
    if file_content_base64 is not None:
        raw_b64 = file_content_base64.strip()
        if not raw_b64:
            raise ToolError("Argument validation error: 'file_content_base64' cannot be empty.")
        try:
            raw = base64.b64decode(raw_b64)
        except binascii.Error as e:
            raise ToolError("Invalid base64 in file_content_base64.") from e
        if not raw:
            raise ToolError("Decoded file content is empty.")
        fname = (
            dataset_filename.strip()
            if dataset_filename and dataset_filename.strip()
            else "data.csv"
        )
        buffer = io.BytesIO(raw)
        buffer.name = fname
        catalog_item = client.Dataset.create_from_file(filelike=buffer)
    else:
        if file_url is None or not is_valid_url(file_url):
            logger.error("Invalid file URL: %s", file_url)
            raise ToolError(f"Invalid file URL: {file_url}")
        catalog_item = client.Dataset.create_from_url(file_url)

    if not catalog_item:
        raise ToolError("Failed to upload dataset.")

    return {
        "dataset_id": catalog_item.id,
        "dataset_version_id": catalog_item.version_id,
        "dataset_name": catalog_item.name,
    }


@tool_metadata(tags={"predictive", "data", "read", "list", "catalog", "daria"})
async def list_ai_catalog_items(
    offset: Annotated[
        int | None,
        "Skip this many catalog items (0-based). Example: page 2 with page size 3 uses offset=3.",
    ] = None,
    limit: Annotated[
        int,
        (
            "Max datasets to return in this call (default 1000). Use with offset to page. "
            "Values above 1000 are rejected; use offset to continue."
        ),
    ] = DR_PREDICTIVE_API_PAGINATION_MAX,
) -> dict[str, Any]:
    """List AI Catalog items (datasets) for the authenticated user.

    The ``datasets`` value is always a mapping of dataset id to name (empty mapping when none).

    Pagination: the DataRobot SDK's ``Dataset.iterate`` is called with a finite ``limit`` (default
    1000) so the gateway and MCP process avoid unbounded responses. Use ``offset`` and additional
    calls to walk large catalogs. The response echoes the effective limit and may set
    ``may_have_more`` when the page is full.
    """
    if offset is not None and offset < 0:
        raise ToolError("offset must be non-negative")

    max_per = DR_PREDICTIVE_API_PAGINATION_MAX
    if limit < 1:
        raise ToolError("limit must be at least 1")
    if limit > max_per:
        raise ToolError(f"limit cannot exceed {max_per}; use offset to page through the catalog.")
    apply_limit = limit

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    # DataRobot ``Dataset.iterate`` expects an int; do not pass ``None`` when offset is omitted.
    iterate_offset = 0 if offset is None else offset
    gen = client.Dataset.iterate(offset=iterate_offset, limit=apply_limit)
    datasets = list(itertools.islice(gen, apply_limit))

    if not datasets:
        logger.info("No AI Catalog items found")
        out: dict[str, Any] = {"datasets": {}, "count": 0}
        return _merge_pagination_metadata(out, {}, offset=offset, limit=apply_limit)

    datasets_dict = {ds.id: ds.name for ds in datasets}
    out = _merge_pagination_metadata(
        {
            "datasets": datasets_dict,
            "count": len(datasets),
        },
        {},
        offset=offset,
        limit=apply_limit,
    )
    out["may_have_more"] = len(datasets) == apply_limit
    return out


@tool_metadata(tags={"predictive", "data", "read", "dataset", "metadata", "daria"})
async def get_dataset_details(
    *,
    dataset_id: Annotated[str, "The ID of the DataRobot dataset"] | None = None,
    include_sample: Annotated[bool, "Whether to include sample rows"] = True,
    sample_offset: Annotated[
        int,
        (
            "0-based index of the first sample row to return; use with sample_rows to page "
            "through data."
        ),
    ] = 0,
    sample_rows: Annotated[
        int,
        "Max sample rows; larger values are capped at 1000 for large previews.",
    ] = 10,
) -> dict[str, Any]:
    """Get DataRobot dataset metadata and optional sample rows.

    Paginate the sample with sample_offset and sample_rows.
    """
    if not dataset_id:
        raise ToolError("Dataset ID must be provided")
    if sample_offset < 0:
        raise ToolError("sample_offset must be non-negative")
    if sample_rows < 1:
        raise ToolError("sample_rows must be at least 1")

    max_per = DR_PREDICTIVE_API_PAGINATION_MAX
    apply_sample_rows = min(sample_rows, max_per)
    if apply_sample_rows < sample_rows:
        logger.warning(
            "get_dataset_details: sample_rows %s exceeds cap %s, using %s",
            sample_rows,
            max_per,
            apply_sample_rows,
        )

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    dataset = client.Dataset.get(dataset_id)

    result: dict = {
        "id": dataset.id,
        "name": dataset.name,
        "created_at": str(dataset.created_at),
        "row_count": getattr(dataset, "row_count", None),
    }
    if include_sample:
        try:
            df = dataset.get_raw_sample_data()
            result["columns"] = list(df.columns)
            n = len(df)
            end = min(sample_offset + apply_sample_rows, n)
            if sample_offset < n:
                if hasattr(df, "iloc"):
                    sample_df = df.iloc[sample_offset:end]
                else:
                    sample_df = df[sample_offset:end]
            elif hasattr(df, "iloc"):
                sample_df = df.iloc[0:0]
            else:
                sample_df = df[0:0]
            result["sample"] = sample_df.to_dict(orient="records")
            result["sample_offset"] = sample_offset
            result["sample_rows"] = apply_sample_rows
            result["sample_count"] = len(result["sample"])
            result["sample_total_available"] = n
        except Exception as exc:
            result["sample_error"] = str(exc)

    return result


@tool_metadata(tags={"predictive", "data", "read", "datastore", "list", "daria"})
async def list_datastores(
    offset: Annotated[
        int | None,
        "Skip this many datastores (0-based). Use with limit for paged listing; omit for all.",
    ] = None,
    limit: Annotated[
        int,
        (
            "Max datastores to return (default 1000). Values above 1000 are rejected; "
            "use offset to page."
        ),
    ] = DR_PREDICTIVE_API_PAGINATION_MAX,
) -> dict[str, Any]:
    """List available DataRobot data connections (datastores)."""
    if offset is not None and offset < 0:
        raise ToolError("offset must be non-negative")

    max_per = DR_PREDICTIVE_API_PAGINATION_MAX
    if limit < 1:
        raise ToolError("limit must be at least 1")
    if limit > max_per:
        raise ToolError(f"limit cannot exceed {max_per}; use offset to page.")
    apply_limit = limit

    token = await get_datarobot_access_token()
    dr_module = DataRobotClient(token).get_client()
    rest_client = dr_module.client.get_client()

    params: dict[str, Any] = {"limit": apply_limit}
    if offset is not None:
        params["offset"] = offset
    response = rest_client.get("externalDataStores/", params=params)
    body = response.json()
    items = body.get("data", [])
    if not isinstance(items, list):
        items = []

    out: dict[str, Any] = {
        "datastores": [
            {
                "id": ds.get("id", ""),
                "canonical_name": ds.get("canonicalName", ""),
                "creator_id": ds.get("creatorId", ""),
                "params": _serialize_datastore_params(ds.get("params", {})),
            }
            for ds in items
        ],
        "count": len(items),
    }
    return _merge_pagination_metadata(out, body, offset=offset, limit=apply_limit)


@tool_metadata(tags={"predictive", "data", "read", "datastore", "browse", "daria"})
async def browse_datastore(
    *,
    datastore_id: Annotated[str, "The ID of the datastore to browse"] | None = None,
    path: Annotated[str, "The path to browse within the datastore"] | None = None,
    offset: Annotated[int, "Pagination offset"] = 0,
    limit: Annotated[
        int,
        (
            "Maximum items to return (default 1000). Values above 1000 are rejected; "
            "use offset to page."
        ),
    ] = DR_PREDICTIVE_API_PAGINATION_MAX,
    search: Annotated[str, "Search filter for items"] | None = None,
) -> dict[str, Any]:
    """Browse a DataRobot data connection to list catalogs, schemas, and tables."""
    if not datastore_id:
        raise ToolError("Datastore ID must be provided")
    if offset < 0:
        raise ToolError("offset must be non-negative")
    if limit < 1:
        raise ToolError("limit must be at least 1")

    max_per = DR_PREDICTIVE_API_PAGINATION_MAX
    if limit > max_per:
        raise ToolError(f"limit cannot exceed {max_per}; use offset to page.")
    apply_limit = limit

    token = await get_datarobot_access_token()
    dr_module = DataRobotClient(token).get_client()
    rest_client = dr_module.client.get_client()

    params: dict = {"offset": offset, "limit": apply_limit}
    if path:
        params["path"] = path
    if search:
        params["search"] = search
    response = rest_client.get(f"externalDataDrivers/{datastore_id}/tables/", params=params)
    data = response.json()
    items = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        items = [items] if items is not None else []

    out: dict[str, Any] = {
        "datastore_id": datastore_id,
        "path": path or "/",
        "items": items,
        "count": len(items),
    }
    body = data if isinstance(data, dict) else {}
    return _merge_pagination_metadata(out, body, offset=offset, limit=apply_limit)


@tool_metadata(
    tags={"predictive", "data", "read", "write", "delete", "datastore", "query", "sql", "daria"}
)
async def query_datastore(
    *,
    datastore_id: Annotated[str, "The ID of the datastore to query"] | None = None,
    sql: Annotated[str, "The SQL query to execute"] | None = None,
    offset: Annotated[int, "Number of rows to skip (0-based) for pagination"] = 0,
    limit: Annotated[
        int,
        (
            "Maximum rows to return (default 1000). Values above 1000 are rejected; "
            "use offset to page."
        ),
    ] = DR_PREDICTIVE_API_PAGINATION_MAX,
) -> dict[str, Any]:
    """Execute a SQL query against a DataRobot datastore connection.

    Only data manipulation language queries (insert, update, and delete data)
    are supported — no commits or rollbacks.
    """
    if not datastore_id:
        raise ToolError("Datastore ID must be provided")
    if not sql:
        raise ToolError("SQL query must be provided")
    if offset < 0:
        raise ToolError("offset must be non-negative")
    if limit < 1:
        raise ToolError("limit must be at least 1")

    max_per = DR_PREDICTIVE_API_PAGINATION_MAX
    if limit > max_per:
        raise ToolError(f"limit cannot exceed {max_per}; use offset to page.")
    apply_limit = limit

    token = await get_datarobot_access_token()
    dr_module = DataRobotClient(token).get_client()
    rest_client = dr_module.client.get_client()

    payload = {"query": sql, "offset": offset, "limit": apply_limit}
    response = rest_client.post(f"externalDataDrivers/{datastore_id}/execute/", json=payload)
    body = response.json()
    row_data = body.get("data", [])

    out: dict[str, Any] = {
        "rows": row_data,
        "row_count": len(row_data) if isinstance(row_data, list) else 0,
        "columns": body.get("columns", []),
    }
    return _merge_pagination_metadata(out, body, offset=offset, limit=apply_limit)


# from fastmcp import Context

# from datarobot_genai.drmcp.core.memory_management import MemoryManager, get_memory_manager


# @tool_metadata()
# async def list_ai_catalog_items(
#     ctx: Context, agent_id: str = None, storage_id: str = None
# ) -> str:
#     """
#     List all AI Catalog items (datasets) for the authenticated user.

#     Returns:
#         a resource id that can be used to retrieve the list of AI Catalog items using the
#         get_resource tool
#     """
#     token = await get_datarobot_access_token()
#     client = DataRobotClient(token).get_client()
#     datasets = client.Dataset.list()
#     if not datasets:
#         logger.info("No AI Catalog items found")
#         return "No AI Catalog items found."
#     result = "\n".join(f"{ds.id}: {ds.name}" for ds in datasets)

#     if MemoryManager.is_initialized():
#         resource_id = await get_memory_manager().store_resource(
#             data=result,
#             memory_storage_id=storage_id,
#             agent_identifier=agent_id,
#         )
#     else:
#         raise ValueError("MemoryManager is not initialized")

#     logger.info(f"Found {len(datasets)} AI Catalog items")
#     return resource_id
