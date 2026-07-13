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

"""DataRobot Vector Database (VDB) tools."""

import logging
from typing import Annotated
from typing import Any

import datarobot as dr
from datarobot.errors import ClientError
from datarobot.models.genai.vector_database import ChunkingParameters
from datarobot.models.genai.vector_database import VectorDatabase

from datarobot_genai.drmcpbase.dynamic_tools.deployment.config import build_deployment_auth_headers
from datarobot_genai.drmcpbase.dynamic_tools.deployment.config import get_deployment_base_url
from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drmcputils.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drmcputils.credentials import get_credentials
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.pagination import PAGINATION_MAX
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EMBEDDING_MODELS: frozenset[str] = frozenset(
    {
        "intfloat/e5-large-v2",
        "intfloat/e5-base-v2",
        "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-small",
        "sentence-transformers/all-MiniLM-L6-v2",
        "jinaai/jina-embedding-t-en-v1",
        "jinaai/jina-embedding-s-en-v2",
        "cl-nagoya/sup-simcse-ja-base",
    }
)
DEFAULT_CHUNKING_METHOD = "recursive"
DEFAULT_CHUNK_SIZE = 256
MAX_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP_PERCENTAGE = 10
DEFAULT_SEPARATORS = ["\n\n", "\n", " "]
VDB_BUILD_COMPLETED_STATUS = "completed"
VDB_BUILD_TERMINAL_FAILURE_STATUSES = frozenset({"error", "failed"})
VDB_DEPLOY_TERMINAL_FAILURE_STATUSES = frozenset({"failed", "errored"})

_VDB_BUILD_STATUS_NOTE = (
    "Vector database build started. Poll vdb_get(vector_database_id=..., "
    "target_status='completed') every few seconds until target_reached is true, "
    "then call vdb_deploy."
)
_VDB_DEPLOY_STATUS_NOTE = (
    "Deployment created. Poll vdb_get(deployment_id=..., target_status='active') "
    "every few seconds until target_reached is true, then use vdb_query."
)


def _normalize_status(status: str | None) -> str:
    return str(status).lower() if status is not None else ""


def _normalize_vdb_document(item: dict[str, Any]) -> dict[str, Any]:
    document: dict[str, Any] = {
        "page_content": item.get("page_content") or item.get("content", ""),
        "metadata": item.get("metadata", {}),
    }
    if "score" in item:
        document["score"] = item["score"]
    return document


def _parse_vdb_query_documents(response_data: Any) -> list[dict[str, Any]]:
    """Normalize prediction API responses into document records."""
    if isinstance(response_data, list):
        return [_normalize_vdb_document(item) for item in response_data if isinstance(item, dict)]

    rows = response_data.get("data", []) if isinstance(response_data, dict) else []
    documents: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "page_content" in row or "content" in row:
            documents.append(_normalize_vdb_document(row))
            continue

        chunks = row.get("prediction")
        if chunks is None:
            prediction_values = row.get("predictionValues")
            if isinstance(prediction_values, list) and prediction_values:
                first_value = prediction_values[0]
                if isinstance(first_value, dict):
                    chunks = first_value.get("value")
        if isinstance(chunks, list):
            for chunk in chunks:
                if isinstance(chunk, str):
                    documents.append({"page_content": chunk, "metadata": {}})
                elif isinstance(chunk, dict):
                    documents.append(_normalize_vdb_document(chunk))
    return documents


def _is_vector_database_deployment(deployment: dict[str, Any]) -> bool:
    model = deployment.get("model")
    if isinstance(model, dict) and model.get("targetType") == "VectorDatabase":
        return True
    capabilities = deployment.get("capabilities")
    return (
        isinstance(capabilities, dict)
        and capabilities.get("supportsVectorDatabaseQuerying") is True
    )


@tool_metadata(
    tags={"vdb", "read", "list", "daria"},
    description=(
        "[VDB—discover deployments] Use when the user needs deployed Vector Databases (VDBs) as "
        "ID/label/status records. Read-only. Filters deployments client-side to vector-database "
        "targets. Not predictive deployments (deployment_get_list), not AI Catalog datasets "
        "(catalog_list_datasets). Next step: vdb_query."
    ),
    display_name="Vector Database — List",
    description_ui="List deployed vector database (VDB) deployments.",
)
async def vdb_list(
    *,
    offset: Annotated[
        int | None,
        "Skip this many VDBs (0-based). Use with limit for paged listing; omit for all.",
    ] = None,
    limit: Annotated[
        int,
        ("Max VDBs to return (default 100). Values above 100 are rejected; use offset to page."),
    ] = PAGINATION_MAX,
) -> dict[str, Any]:
    if offset is not None and offset < 0:
        raise ToolError("offset must be non-negative", kind=ToolErrorKind.VALIDATION)

    limit, message = clamp_limit(limit)
    start = offset or 0

    with ThreadSafeDataRobotClient().request_user_client():
        rest_client = dr.client.get_client()
        try:
            all_deployments = list(
                dr.utils.pagination.unpaginate(
                    initial_url="deployments/",
                    initial_params={"limit": PAGINATION_MAX},
                    client=rest_client,
                )
            )
        except ClientError as e:
            raise_tool_error_for_client_error(e)

        vdb_deployments = [d for d in all_deployments if _is_vector_database_deployment(d)]
        total_count = len(vdb_deployments)
        vdbs = vdb_deployments[start : start + limit]
        api_response: dict[str, Any] = {"total_count": total_count}
        if start + limit < total_count:
            api_response["next"] = start + limit
        if start > 0:
            api_response["previous"] = max(0, start - limit)

    final_results: dict[str, Any] = {
        "vector_databases": [
            {
                "deployment_id": d["id"],
                "label": d.get("label", ""),
                "status": d.get("status", ""),
            }
            for d in vdbs
        ],
        "count": len(vdbs),
    }
    return merge_pagination_metadata(
        final_results=final_results,
        api_response=api_response,
        message=message,
        offset=offset,
        limit=limit,
    )


@tool_metadata(
    tags={"vdb", "read", "query", "search", "daria"},
    description=(
        "[VDB—semantic search] Use when the user wants to retrieve documents from a deployed "
        "Vector Database via semantic similarity (deployment_id from vdb_list). "
        "Returns matched documents with metadata. Read-only. Not deployment metadata "
        "(deployment_get_info), not predictive scoring (predict_*)."
    ),
    display_name="Vector Database — Query",
    description_ui="Run semantic search against a vector database deployment.",
)
async def vdb_query(
    *,
    deployment_id: Annotated[str, "The deployment ID of the Vector Database"] | None = None,
    query: Annotated[str, "The search query"] | None = None,
    num_results: Annotated[int, "Number of results to return"] = 5,
    retrieval_mode: Annotated[
        str, "Retrieval mode: 'similarity' or 'maximal_marginal_relevance'"
    ] = "similarity",
) -> dict[str, Any]:
    if not deployment_id:
        raise ToolError("Deployment ID must be provided", kind=ToolErrorKind.VALIDATION)
    if not query:
        raise ToolError("Query must be provided", kind=ToolErrorKind.VALIDATION)

    with ThreadSafeDataRobotClient().request_user_client() as rest_client:
        try:
            deployment = dr.Deployment.get(deployment_id)
        except ClientError as e:
            raise_tool_error_for_client_error(e)

        endpoint = get_credentials().datarobot.datarobot_endpoint
        token = get_datarobot_access_token()
        try:
            base_url = get_deployment_base_url(deployment, endpoint)
        except ValueError as e:
            raise ToolError(str(e), kind=ToolErrorKind.UPSTREAM) from e

        headers = {
            **build_deployment_auth_headers(deployment, token),
            "Content-Type": "application/json",
        }
        request_payload = [
            {
                "promptText": query,
                "num_results": num_results,
                "retrieval_mode": retrieval_mode,
            }
        ]

        try:
            response = rest_client.request(
                "POST",
                f"{base_url}predictions",
                join_endpoint=False,
                json=request_payload,
                headers=headers,
            )
        except ClientError as e:
            raise_tool_error_for_client_error(e)
        documents = _parse_vdb_query_documents(response.json())

    return {
        "deployment_id": deployment_id,
        "documents": documents,
        "count": len(documents),
    }


def _resolve_chunking_parameters(
    *,
    embedding_model: str | None,
    chunking_method: str | None,
    chunk_size: int | None,
    chunk_overlap_percentage: int | None,
    separators: list[str] | None,
) -> ChunkingParameters:
    resolved_embedding_model = (
        embedding_model.strip()
        if embedding_model and embedding_model.strip()
        else DEFAULT_EMBEDDING_MODEL
    )
    if resolved_embedding_model not in SUPPORTED_EMBEDDING_MODELS:
        supported = ", ".join(sorted(SUPPORTED_EMBEDDING_MODELS))
        raise ToolError(
            f"embedding_model must be one of: {supported}.",
            kind=ToolErrorKind.VALIDATION,
        )

    resolved_chunking_method = (
        chunking_method.strip()
        if chunking_method and chunking_method.strip()
        else DEFAULT_CHUNKING_METHOD
    )
    resolved_chunk_size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
    if resolved_chunk_size <= 0 or resolved_chunk_size > MAX_CHUNK_SIZE:
        raise ToolError(
            f"chunk_size must be between 1 and {MAX_CHUNK_SIZE}.",
            kind=ToolErrorKind.VALIDATION,
        )

    resolved_overlap = (
        chunk_overlap_percentage
        if chunk_overlap_percentage is not None
        else DEFAULT_CHUNK_OVERLAP_PERCENTAGE
    )
    if resolved_overlap < 0 or resolved_overlap > 100:
        raise ToolError(
            "chunk_overlap_percentage must be between 0 and 100.",
            kind=ToolErrorKind.VALIDATION,
        )

    resolved_separators = separators if separators is not None else list(DEFAULT_SEPARATORS)
    if resolved_chunking_method == "recursive" and not resolved_separators:
        raise ToolError(
            "separators must be provided for recursive chunking.",
            kind=ToolErrorKind.VALIDATION,
        )

    return ChunkingParameters(
        embedding_model=resolved_embedding_model,
        chunking_method=resolved_chunking_method,
        chunk_size=resolved_chunk_size,
        chunk_overlap_percentage=resolved_overlap,
        separators=resolved_separators,
    )


@tool_metadata(
    tags={"vdb", "write", "create"},
    description=(
        "[VDB—create] Use when the user wants to create a new DataRobot vector database from an "
        "AI Catalog dataset (dataset_id from catalog_list_datasets or catalog_upload_dataset) "
        "linked to a use case (use_case_id from datarobot_usecases_list). "
        "Chunking parameters are required by the platform; when omitted, defaults are applied "
        f"(embedding_model={DEFAULT_EMBEDDING_MODEL!r}, chunking_method='recursive', "
        f"chunk_size={DEFAULT_CHUNK_SIZE}, chunk_overlap_percentage="
        f"{DEFAULT_CHUNK_OVERLAP_PERCENTAGE}, separators={DEFAULT_SEPARATORS!r}). "
        "chunk_size must be <= 256. Returns vector_database_id and execution_status "
        "(building is async). REQUIRED follow-up: poll vdb_get with target_status='completed' "
        "until target_reached, then vdb_deploy. Not live deployments yet (vdb_list after deploy), "
        "not semantic search (vdb_query)."
    ),
    display_name="Vector Database — Create",
    description_ui="Create a vector database from a catalog dataset in a use case.",
)
async def vdb_create(
    *,
    dataset_id: Annotated[
        str | None,
        "AI Catalog dataset ID with document and document_file_path columns.",
    ] = None,
    use_case_id: Annotated[str | None, "Use case ID to attach the vector database to."] = None,
    name: Annotated[str | None, "Optional display name for the vector database."] = None,
    embedding_model: Annotated[
        str | None,
        (
            "Embedding model name. Defaults to sentence-transformers/all-MiniLM-L6-v2. "
            "Supported: intfloat/e5-large-v2, intfloat/e5-base-v2, "
            "intfloat/multilingual-e5-base, intfloat/multilingual-e5-small, "
            "sentence-transformers/all-MiniLM-L6-v2, jinaai/jina-embedding-t-en-v1, "
            "jinaai/jina-embedding-s-en-v2, cl-nagoya/sup-simcse-ja-base."
        ),
    ] = None,
    chunking_method: Annotated[
        str | None,
        "Chunking method, e.g. 'recursive' or 'semantic'. Defaults to 'recursive'.",
    ] = None,
    chunk_size: Annotated[
        int | None,
        f"Max chunk size in tokens (1–{MAX_CHUNK_SIZE}). Defaults to {DEFAULT_CHUNK_SIZE}.",
    ] = None,
    chunk_overlap_percentage: Annotated[
        int | None,
        f"Overlap percentage between consecutive chunks (0–100). "
        f"Defaults to {DEFAULT_CHUNK_OVERLAP_PERCENTAGE}.",
    ] = None,
    separators: Annotated[
        list[str] | None,
        "Ordered separator strings for recursive chunking. Defaults to ['\\n\\n', '\\n', ' '].",
    ] = None,
) -> dict[str, Any]:
    if not dataset_id or not dataset_id.strip():
        raise ToolError("dataset_id must be provided", kind=ToolErrorKind.VALIDATION)
    if not use_case_id or not use_case_id.strip():
        raise ToolError("use_case_id must be provided", kind=ToolErrorKind.VALIDATION)

    chunking_parameters = _resolve_chunking_parameters(
        embedding_model=embedding_model,
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap_percentage=chunk_overlap_percentage,
        separators=separators,
    )

    with ThreadSafeDataRobotClient().request_user_client():
        try:
            vector_database = VectorDatabase.create(
                dataset_id=dataset_id.strip(),
                use_case=use_case_id.strip(),
                name=name.strip() if name and name.strip() else None,
                chunking_parameters=chunking_parameters,
            )
        except ClientError as e:
            raise_tool_error_for_client_error(e)

    return {
        "vector_database_id": vector_database.id,
        "name": vector_database.name,
        "execution_status": vector_database.execution_status,
        "use_case_id": vector_database.use_case_id,
        "dataset_id": vector_database.dataset_id,
        "chunking_parameters": {
            "embedding_model": chunking_parameters.embedding_model,
            "chunking_method": chunking_parameters.chunking_method,
            "chunk_size": chunking_parameters.chunk_size,
            "chunk_overlap_percentage": chunking_parameters.chunk_overlap_percentage,
            "separators": chunking_parameters.separators,
        },
        "note": _VDB_BUILD_STATUS_NOTE,
    }


@tool_metadata(
    tags={"vdb", "read", "status", "daria"},
    description=(
        "[VDB—status] REQUIRED polling step after vdb_create or vdb_deploy. Pass exactly one of "
        "vector_database_id (build status) or deployment_id (launch status). Performs ONE "
        "non-blocking fetch — it does not wait or poll. Pass target_status to get target_reached "
        "(build: 'completed'; deployment: 'active'). Raises on terminal failure. Call again "
        "yourself every few seconds until target_reached is true.\n\n"
        "Example: vdb_get(vector_database_id='vdb-abc', target_status='completed')\n"
        "Example: vdb_get(deployment_id='dep-abc', target_status='active')"
    ),
    display_name="Vector Database — Get status",
    description_ui="Fetch vector database build or deployment launch status.",
)
async def vdb_get(
    *,
    vector_database_id: Annotated[
        str | None,
        "Vector database ID from vdb_create. Mutually exclusive with deployment_id.",
    ] = None,
    deployment_id: Annotated[
        str | None,
        "Deployment ID from vdb_deploy. Mutually exclusive with vector_database_id.",
    ] = None,
    target_status: Annotated[
        str | None,
        "Optional status to compare against, e.g. 'completed' (build) or 'active' (deployment). "
        "Does not block.",
    ] = None,
) -> dict[str, Any]:
    has_vdb_id = bool(vector_database_id and vector_database_id.strip())
    has_dep_id = bool(deployment_id and deployment_id.strip())
    if has_vdb_id == has_dep_id:
        raise ToolError(
            "Exactly one of vector_database_id or deployment_id must be provided.",
            kind=ToolErrorKind.VALIDATION,
        )

    if target_status is not None:
        target = target_status.strip()
        if not target:
            raise ToolError(
                "Argument validation error: 'target_status' cannot be empty.",
                kind=ToolErrorKind.VALIDATION,
            )
    else:
        target = None

    with ThreadSafeDataRobotClient().request_user_client():
        if has_vdb_id:
            vdb_id = vector_database_id.strip()  # type: ignore[union-attr]
            try:
                vector_database = VectorDatabase.get(vdb_id)
            except ClientError as e:
                raise_tool_error_for_client_error(e)
            status = vector_database.execution_status
            normalized = _normalize_status(status)
            if normalized in VDB_BUILD_TERMINAL_FAILURE_STATUSES:
                raise ToolError(
                    f"Vector database {vdb_id!r} entered terminal status {status!r}.",
                    kind=ToolErrorKind.UPSTREAM,
                )
            result: dict[str, Any] = {
                "vector_database_id": vdb_id,
                "execution_status": status,
            }
        else:
            dep_id = deployment_id.strip()  # type: ignore[union-attr]
            rest_client = dr.client.get_client()
            try:
                response = rest_client.get(f"deployments/{dep_id}/")
            except ClientError as e:
                raise_tool_error_for_client_error(e)
            deployment = response.json()
            status = deployment.get("status")
            normalized = _normalize_status(status)
            if normalized in VDB_DEPLOY_TERMINAL_FAILURE_STATUSES:
                raise ToolError(
                    f"Deployment {dep_id!r} entered terminal status {status!r}.",
                    kind=ToolErrorKind.UPSTREAM,
                )
            result = {
                "deployment_id": dep_id,
                "status": status,
                "label": deployment.get("label", ""),
            }

    if target is not None:
        result["target_reached"] = normalized == _normalize_status(target)
    return result


@tool_metadata(
    tags={"vdb", "write", "deploy", "daria"},
    description=(
        "[VDB—deploy] Use when the user wants to deploy a built vector database "
        "(vector_database_id from vdb_create) to a live MLOps deployment for querying. "
        "Requires build status COMPLETED (poll vdb_get first). Returns deployment_id, label, "
        "and status. REQUIRED follow-up: poll vdb_get with deployment_id and "
        "target_status='active' until target_reached, then vdb_query. Not creating the vector "
        "database (vdb_create), not predictive model deployments (deployment_create_deployment)."
    ),
    display_name="Vector Database — Deploy",
    description_ui="Deploy a vector database to create a live queryable deployment.",
)
async def vdb_deploy(
    *,
    vector_database_id: Annotated[
        str | None,
        "Vector database ID from vdb_create.",
    ] = None,
    prediction_environment_id: Annotated[
        str | None,
        "Optional prediction environment ID. When set, default_prediction_server_id is ignored.",
    ] = None,
    default_prediction_server_id: Annotated[
        str | None,
        "Optional prediction server ID. Defaults to the first available server when omitted.",
    ] = None,
    credential_id: Annotated[
        str | None,
        "Optional credential ID for connected external vector databases.",
    ] = None,
) -> dict[str, Any]:
    if not vector_database_id or not vector_database_id.strip():
        raise ToolError("vector_database_id must be provided", kind=ToolErrorKind.VALIDATION)

    with ThreadSafeDataRobotClient().request_user_client():
        try:
            vector_database = VectorDatabase.get(vector_database_id.strip())
        except ClientError as e:
            raise_tool_error_for_client_error(e)

        build_status = _normalize_status(vector_database.execution_status)
        if build_status != VDB_BUILD_COMPLETED_STATUS:
            raise ToolError(
                "Vector database build is not complete "
                f"(execution_status={vector_database.execution_status!r}). "
                "Poll vdb_get(vector_database_id=..., target_status='completed') "
                "every few seconds until target_reached is true, then call vdb_deploy.",
                kind=ToolErrorKind.UPSTREAM,
            )

        deploy_kwargs: dict[str, Any] = {}
        if prediction_environment_id and prediction_environment_id.strip():
            deploy_kwargs["prediction_environment_id"] = prediction_environment_id.strip()
        elif default_prediction_server_id and default_prediction_server_id.strip():
            deploy_kwargs["default_prediction_server_id"] = default_prediction_server_id.strip()
        else:
            try:
                prediction_servers = dr.PredictionServer.list()
            except ClientError as e:
                raise_tool_error_for_client_error(e)
            if not prediction_servers:
                raise ToolError(
                    "No prediction servers available for deployment.",
                    kind=ToolErrorKind.UPSTREAM,
                )
            deploy_kwargs["default_prediction_server_id"] = prediction_servers[0].id

        if credential_id and credential_id.strip():
            deploy_kwargs["credential_id"] = credential_id.strip()

        try:
            deployment = vector_database.deploy(**deploy_kwargs)
        except ClientError as e:
            raise_tool_error_for_client_error(e)

        # Best-effort: the deployment already exists, so never fail the tool here.
        deployment_status: str | None = None
        try:
            dep_response = dr.client.get_client().get(f"deployments/{deployment.id}/")
            deployment_status = dep_response.json().get("status")
        except Exception:
            logger.debug("Could not fetch deployment status after deploy", exc_info=True)

    return {
        "vector_database_id": vector_database_id.strip(),
        "deployment_id": deployment.id,
        "label": deployment.label,
        "status": deployment_status,
        "note": _VDB_DEPLOY_STATUS_NOTE,
    }
