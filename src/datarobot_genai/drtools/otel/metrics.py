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

"""OTel metrics tools: ``otel_metrics_catalog_list``, ``otel_metrics_values_get``.

Plan §2.5-§2.6.

``otel_metrics_values_get`` folds the doc's ``retrieve_metrics_values`` and
``retrieve_autocollected_metrics_values`` into one tool behind a ``source``
param, following the existing ``workload_activity_get(kind=...)`` precedent in
``drtools/workload/workload_observability.py``.

The trap this module exists to avoid: every sampled deployment in the original
measurement had *no* configured custom metrics, so ``source='configured'``
returning an empty ``metric_aggregations`` is the expected, common case — not a
failure to retry. ``source='autocollected'`` is the default precisely because
it is the one guaranteed to have data (built-in platform metrics, no
configuration required).
"""

from typing import Annotated
from typing import Any
from typing import Literal

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_otel_query import OtelQueryApiClient
from datarobot_genai.drtools.core.utils import require_object_id
from datarobot_genai.drtools.otel.constants import EntityType

# otel_metrics_catalog_list's client-side cap. The server itself is unpaginated
# (max_length=1000); this clamps further and reports when it bit, since 1000
# catalog entries is far more than any tool call needs to see at once.
METRICS_CATALOG_MAX = 100

# Plan §2's shared conventions: GENAI_EXPERIMENTATION is required by *every*
# /otel route, these two included — worth surfacing in the description so a
# 403 reads as configuration, not "no access to this entity's metrics".
_OTEL_403_HINT = (
    "A 403 here usually means configuration, not missing data: the "
    "GENAI_EXPERIMENTATION feature flag is required for every /otel route."
)


# ------------------------------------------------------------------ #
# otel_metrics_catalog_list                                            #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"otel", "datarobot", "metrics", "catalog", "list", "observability"},
    description=(
        "[OTel—metrics catalog] List the OTel metrics an entity actually emits "
        "— the discovery step before otel_metrics_values_get. Measured 5-224 "
        "tokens. Unpaginated server-side (up to 1000 entries); this tool clamps "
        f"to the first {METRICS_CATALOG_MAX} and adds a note when the server "
        "returned more. Use 'search' to narrow instead of paging.\n\n"
        f"{_OTEL_403_HINT}\n\n"
        "Example: otel_metrics_catalog_list(entity_type='deployment', entity_id='...')\n"
        "Example: otel_metrics_catalog_list(entity_type='deployment', "
        "entity_id='...', search='latency')"
    ),
    display_name="OTel — List metrics catalog",
    description_ui=("List the OTel metrics an entity emits: name, description, type, and units."),
)
async def otel_metrics_catalog_list(
    *,
    entity_type: Annotated[EntityType, "Type of entity the OTel data belongs to."],
    entity_id: Annotated[str, "24-character hex ID of the entity."],
    search: Annotated[str | None, "Substring match on metric name."] = None,
    metric_type: Annotated[
        str | None, "Filter by metric type, e.g. 'counter' | 'gauge' | 'histogram'."
    ] = None,
) -> dict[str, Any]:
    eid = require_object_id(entity_id, "entity_id")
    try:
        result = OtelQueryApiClient().list_metrics_summary(
            entity_type, eid, search=search, metric_type=metric_type
        )
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    truncated = len(data) > METRICS_CATALOG_MAX
    metrics = data[:METRICS_CATALOG_MAX]

    response: dict[str, Any] = {"metrics": metrics, "count": len(metrics)}
    if truncated:
        response["note"] = (
            f"Server returned more than {METRICS_CATALOG_MAX} metrics; showing "
            f"the first {METRICS_CATALOG_MAX}. Use 'search' to narrow."
        )
    return response


# ------------------------------------------------------------------ #
# otel_metrics_values_get                                              #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"otel", "datarobot", "metrics", "values", "get", "observability"},
    description=(
        "[OTel—metrics values] Get current metric values for an entity.\n\n"
        "source='autocollected' (default): built-in platform metrics (CPU, "
        "memory, request counts, ...) that always have data — no configuration "
        "required.\n"
        "source='configured': custom metric configurations defined for this "
        "entity; histogram_buckets=True returns raw buckets instead of "
        "percentiles (configured only).\n\n"
        "IMPORTANT: an empty 'metric_aggregations' for source='configured' "
        "means 'no custom metrics configured for this entity', NOT 'no data' — "
        "every deployment sampled while measuring this endpoint had none "
        "configured, so do not read an empty list as a failure and retry. "
        "Measured 5-224 tokens; no truncation in phase 1.\n\n"
        f"{_OTEL_403_HINT}\n\n"
        "Example: otel_metrics_values_get(entity_type='deployment', entity_id='...')\n"
        "Example: otel_metrics_values_get(entity_type='deployment', "
        "entity_id='...', source='configured', histogram_buckets=True)"
    ),
    display_name="OTel — Get metric values",
    description_ui=(
        "Get current OTel metric values for an entity: built-in platform "
        "metrics by default, or configured custom metrics."
    ),
)
async def otel_metrics_values_get(
    *,
    entity_type: Annotated[EntityType, "Type of entity the OTel data belongs to."],
    entity_id: Annotated[str, "24-character hex ID of the entity."],
    source: Annotated[
        Literal["configured", "autocollected"],
        "'autocollected' (default): built-in platform metrics, always has "
        "data. 'configured': custom metric configurations for this entity.",
    ] = "autocollected",
    start_time: Annotated[str | None, "RFC3339 start of the window."] = None,
    end_time: Annotated[str | None, "RFC3339 end of the window."] = None,
    histogram_buckets: Annotated[
        bool, "source='configured' only: return raw buckets instead of percentiles."
    ] = False,
) -> dict[str, Any]:
    eid = require_object_id(entity_id, "entity_id")
    client = OtelQueryApiClient()

    try:
        if source == "autocollected":
            result = client.get_autocollected_metrics_values(
                entity_type, eid, start=start_time, end=end_time
            )
        else:
            result = client.get_metrics_values(
                entity_type,
                eid,
                histogram_buckets=histogram_buckets,
                start=start_time,
                end=end_time,
            )
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    if source == "autocollected":
        data = result.get("data", []) or []
        return {"metrics": data, "count": len(data)}

    # source == "configured": a single aggregation object, not a list envelope
    # — pass the three documented fields through explicitly rather than the
    # raw response, so the contract stays exactly what §2.6 promises regardless
    # of what else the server happens to include.
    return {
        "start_time": result.get("start_time"),
        "end_time": result.get("end_time"),
        "metric_aggregations": result.get("metric_aggregations", []) or [],
    }
