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

import io
import json
import logging
from datetime import datetime
from typing import Annotated
from typing import Any

import polars as pl
from datarobot.errors import ClientError
from datarobot_predict import TimeSeriesType
from datarobot_predict.deployment import predict as dr_predict
from dateutil import parser as dateutil_parser

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind
from datarobot_genai.drtools.core.utils import predictions_result_response
from datarobot_genai.drtools.predictive.client_exceptions import raise_tool_error_for_client_error

logger = logging.getLogger(__name__)


def _parse_datetime(value: str) -> datetime:
    normalized = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return dateutil_parser.parse(str(value))


@tool_metadata(
    tags={"predictive", "prediction", "realtime", "read", "scoring"},
    description=(
        "[Predict—deployment + catalog, synchronous rows] Use when the user already has an AI "
        "Catalog dataset_id and wants realtime-style scoring through a deployment with rows "
        "returned in one response (moderate size). Not for pasted inline CSV/JSON "
        "(predict_realtime), not for async batch CSV download (predict_by_ai_catalog), not "
        "project-partition batch (predict_from_project_data)."
    ),
)
async def predict_by_ai_catalog_rt(
    *,
    deployment_id: Annotated[str, "MLOps deployment id."],
    dataset_id: Annotated[str, "AI Catalog dataset id (tabular)."],
    timeout: Annotated[int, "Client wait cap in seconds."] | None = 600,
) -> dict[str, Any]:
    if not deployment_id or not deployment_id.strip():
        raise ToolError(
            "Argument validation error: 'deployment_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    if not dataset_id or not dataset_id.strip():
        raise ToolError(
            "Argument validation error: 'dataset_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    try:
        dataset = client.Dataset.get(dataset_id)
    except ClientError as e:
        raise_tool_error_for_client_error(e)

    # 1. Preferred: built-in DataFrame helper (newer SDKs)
    if hasattr(dataset, "get_as_dataframe"):
        df = dataset.get_as_dataframe()

    # 2. Next: if there is a method returning a local file path
    elif hasattr(dataset, "download"):
        path = dataset.download("dataset.csv")
        df = pl.read_csv(path).to_pandas()

    # 3. Next: if there is a method returning a local file path
    elif hasattr(dataset, "get_file"):
        path = dataset.get_file()
        df = pl.read_csv(path).to_pandas()

    # 4. Bytes fallback
    elif hasattr(dataset, "get_bytes"):
        raw = dataset.get_bytes()
        df = pl.read_csv(io.BytesIO(raw)).to_pandas()

    # 5. Last resort: expose URL then fetch manually
    else:
        url = dataset.url
        df = pl.read_csv(url).to_pandas()

    try:
        deployment = client.Deployment.get(deployment_id=deployment_id)
    except ClientError as e:
        raise_tool_error_for_client_error(e)
    result = dr_predict(deployment, df, timeout=timeout or 600)
    predictions = result.dataframe
    prediction_results = predictions_result_response(predictions, show_explanations=True)
    content = (
        prediction_results.model_dump()
        if hasattr(prediction_results, "model_dump")
        else prediction_results
        if isinstance(prediction_results, dict)
        else prediction_results.__dict__
    )
    return content


@tool_metadata(
    tags={"predictive", "prediction", "realtime", "read", "scoring"},
    description=(
        "[Predict—inline rows now] Use when the user pastes or embeds prediction rows directly "
        "in the conversation: a CSV snippet (header + rows) or a JSON array of row objects in "
        "the dataset argument, plus deployment_id. Immediate synchronous scoring; results return "
        "in the tool response. Do not use for catalog dataset_id only (use "
        "predict_by_ai_catalog_rt or batch predict_by_ai_catalog), not for project holdout "
        "batch jobs (predict_from_project_data), and not for leaderboard model scoring jobs "
        "(score_dataset_with_model). Match feature columns to get_deployment_info. Time series: "
        "forecast_point or forecast_range_start+end, plus series_id_column if multiseries. "
        "validate_prediction_data can sanity-check CSV shape first."
    ),
)
async def predict_realtime(
    *,
    deployment_id: Annotated[str, "MLOps deployment id."],
    dataset: Annotated[
        str,
        (
            "CSV (header row + data) or JSON array of row objects; columns must match deployment "
            "inputs."
        ),
    ],
    forecast_point: Annotated[
        str,
        "Time series forecast anchor date (ISO). Omit for non-time-series or use range instead.",
    ]
    | None = None,
    forecast_range_start: Annotated[
        str,
        "Time series historical window start (ISO); pair with forecast_range_end.",
    ]
    | None = None,
    forecast_range_end: Annotated[
        str,
        "Time series historical window end (ISO); pair with forecast_range_start.",
    ]
    | None = None,
    series_id_column: Annotated[
        str,
        "Multiseries: existing column that identifies each series (e.g. store_id).",
    ]
    | None = None,
    max_explanations: Annotated[
        int,
        "Explanation count: 0 none; >0 cap; some deployments treat 0 as 'all' for SHAP—see docs.",
    ]
    | None = 0,
    max_ngram_explanations: Annotated[
        int,
        "Text model: max text-segment explanations per row; omit to skip.",
    ]
    | None = None,
    threshold_high: Annotated[
        float,
        "If set with explanations: only explain rows with prediction probability above this (0–1).",
    ]
    | None = None,
    threshold_low: Annotated[
        float,
        "If set with explanations: only explain rows with prediction probability below this (0–1).",
    ]
    | None = None,
    passthrough_columns: Annotated[
        str,
        "'all' or comma-separated input column names to copy through to the output.",
    ]
    | None = None,
    explanation_algorithm: Annotated[
        str,
        "Optional 'shap' or 'xemp'; omit to use deployment default.",
    ]
    | None = None,
    prediction_endpoint: Annotated[
        str,
        "Rare: override default prediction HTTP endpoint (e.g. dedicated inference host).",
    ]
    | None = None,
    timeout: Annotated[int, "Client wait cap in seconds."] | None = 600,
) -> dict[str, Any]:
    if not deployment_id or not deployment_id.strip():
        raise ToolError(
            "Argument validation error: 'deployment_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    ds = dataset.strip()
    if not ds:
        raise ToolError(
            "Argument validation error: 'dataset' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    # JSON array of row objects must not go through read_csv first — Polars can "parse" a
    # leading '[' as CSV and yield empty or wrong frames, which then produce API errors like
    # "no data to predict on".
    pl_df: pl.DataFrame
    if ds.startswith("["):
        try:
            data = json.loads(ds)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse dataset string as JSON: {e}") from e
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of row objects.")
        pl_df = pl.DataFrame(data)
    else:
        try:
            pl_df = pl.read_csv(io.StringIO(ds))
        except Exception:
            try:
                parsed = json.loads(ds)
                if isinstance(parsed, list):
                    pl_df = pl.DataFrame(parsed)
                elif isinstance(parsed, dict):
                    pl_df = pl.DataFrame([parsed])
                else:
                    raise ValueError("JSON dataset must be an array or object.")
            except Exception as e:
                raise ValueError(f"Could not parse dataset string as CSV or JSON: {e}") from e

    # Normalize column names: strip leading/trailing whitespace
    pl_df = pl_df.rename({c: c.strip() for c in pl_df.columns})

    if series_id_column and series_id_column not in pl_df.columns:
        raise ValueError(f"series_id_column '{series_id_column}' not found in input data.")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    try:
        deployment = client.Deployment.get(deployment_id=deployment_id)
    except ClientError as e:
        raise_tool_error_for_client_error(e)

    # Check if this is a time series prediction or regular prediction
    is_time_series = bool(forecast_point or (forecast_range_start and forecast_range_end))

    # Convert to pandas at the predict API boundary
    df = pl_df.to_pandas()
    predict_kwargs = {
        "deployment": deployment,
        "data_frame": df,
        "timeout": timeout,
    }

    # Add time series parameters if applicable
    if is_time_series:
        if forecast_point:
            forecast_point_dt = _parse_datetime(forecast_point)
            predict_kwargs["time_series_type"] = TimeSeriesType.FORECAST
            predict_kwargs["forecast_point"] = forecast_point_dt
        elif forecast_range_start and forecast_range_end:
            predictions_start_date_dt = _parse_datetime(forecast_range_start)
            predictions_end_date_dt = _parse_datetime(forecast_range_end)
            predict_kwargs["time_series_type"] = TimeSeriesType.HISTORICAL
            predict_kwargs["predictions_start_date"] = predictions_start_date_dt
            predict_kwargs["predictions_end_date"] = predictions_end_date_dt

    # Add explanation parameters
    if max_explanations != 0:
        predict_kwargs["max_explanations"] = max_explanations
    if max_ngram_explanations is not None:
        predict_kwargs["max_ngram_explanations"] = max_ngram_explanations
    if threshold_high is not None:
        predict_kwargs["threshold_high"] = threshold_high
    if threshold_low is not None:
        predict_kwargs["threshold_low"] = threshold_low
    if explanation_algorithm is not None:
        predict_kwargs["explanation_algorithm"] = explanation_algorithm

    # Add passthrough columns
    if passthrough_columns is not None:
        if passthrough_columns == "all":
            predict_kwargs["passthrough_columns"] = "all"
        else:
            # Convert comma-separated string to set
            columns_set = {col.strip() for col in passthrough_columns.split(",")}
            predict_kwargs["passthrough_columns"] = columns_set

    # Add custom prediction endpoint
    if prediction_endpoint is not None:
        predict_kwargs["prediction_endpoint"] = prediction_endpoint

    # Run prediction
    result = dr_predict(**predict_kwargs)
    predictions = result.dataframe
    prediction_results = predictions_result_response(
        predictions, show_explanations=max_explanations not in {0, "0"}
    )
    content = (
        prediction_results.model_dump()
        if hasattr(prediction_results, "model_dump")
        else prediction_results
        if isinstance(prediction_results, dict)
        else prediction_results.__dict__
    )
    return content
