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
import uuid
from datetime import datetime
from typing import Annotated

import pandas as pd
from datarobot_predict import TimeSeriesType
from datarobot_predict.deployment import predict as dr_predict
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from pydantic import BaseModel

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.core.utils import predictions_result_response
from datarobot_genai.drmcp.tools.clients.s3 import get_s3_bucket_info

logger = logging.getLogger(__name__)


class BucketInfo(BaseModel):
    bucket: str
    key: str


def make_output_settings() -> BucketInfo:
    bucket_info = get_s3_bucket_info()
    s3_key = f"{bucket_info['prefix']}{uuid.uuid4()}.csv"
    return BucketInfo(bucket=bucket_info["bucket"], key=s3_key)


@dr_mcp_tool(tags={"predictive", "prediction", "realtime", "read", "scoring"})
async def predict_by_ai_catalog_rt(
    deployment_id: Annotated[str, "The ID of the DataRobot deployment to use for prediction"]
    | None = None,
    dataset_id: Annotated[str, "ID of an AI Catalog item to use as input data"] | None = None,
    timeout: Annotated[int, "Timeout in seconds for the prediction job"] | None = 600,
) -> ToolError | ToolResult:
    """
    Make real-time predictions using a DataRobot deployment and an AI Catalog dataset using the
    datarobot-predict library.
    Use this for fast results when your data is not huge (not gigabytes). Results larger than 1MB
    will be returned as a resource id and S3 URL; smaller results will be returned inline as a CSV
    string.
    """
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")
    if not dataset_id:
        raise ToolError("Dataset ID must be provided")

    client = get_sdk_client()
    dataset = client.Dataset.get(dataset_id)

    # 1. Preferred: built-in DataFrame helper (newer SDKs)
    if hasattr(dataset, "get_as_dataframe"):
        df = dataset.get_as_dataframe()

    # 2. Next: if there is a method returning a local file path
    elif hasattr(dataset, "download"):
        path = dataset.download("dataset.csv")
        df = pd.read_csv(path)

    # 3. Next: if there is a method returning a local file path
    elif hasattr(dataset, "get_file"):
        path = dataset.get_file()
        df = pd.read_csv(path)

    # 4. Bytes fallback
    elif hasattr(dataset, "get_bytes"):
        raw = dataset.get_bytes()
        df = pd.read_csv(io.BytesIO(raw))

    # 5. Last resort: expose URL then fetch manually
    else:
        url = dataset.url
        df = pd.read_csv(url)

    deployment = client.Deployment.get(deployment_id=deployment_id)
    result = dr_predict(deployment, df, timeout=timeout or 600)
    predictions = result.dataframe
    bucket_info = make_output_settings()
    prediction_results = predictions_result_response(
        predictions,
        bucket_info.bucket,
        bucket_info.key,
        f"pred_{deployment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        True,
    )
    content = (
        prediction_results.model_dump()
        if hasattr(prediction_results, "model_dump")
        else prediction_results
        if isinstance(prediction_results, dict)
        else prediction_results.__dict__
    )
    return ToolResult(structured_content=content)


@dr_mcp_tool(tags={"predictive", "prediction", "realtime", "read", "scoring"})
async def predict_realtime(
    deployment_id: Annotated[str, "The ID of the DataRobot deployment to use for prediction"]
    | None = None,
    file_path: Annotated[
        str,
        """Path to a CSV file to use as input data. For time series with forecast_point,
        must have at least 4 historical values within the feature derivation window.""",
    ]
    | None = None,
    dataset: Annotated[
        str,
        """CSV or JSON string representing the input data.
        If provided, this takes precedence over file_path.""",
    ]
    | None = None,
    forecast_point: Annotated[
        str,
        """Date to start forecasting from (e.g., '2024-06-01').
        If provided, triggers time series FORECAST mode. Uses most recent date if None.""",
    ]
    | None = None,
    forecast_range_start: Annotated[
        str,
        """Start date for historical predictions (e.g., '2024-06-01').
        Must be used with forecast_range_end for HISTORICAL mode.""",
    ]
    | None = None,
    forecast_range_end: Annotated[
        str,
        """End date for historical predictions (e.g., '2024-06-07').
        Must be used with forecast_range_start for HISTORICAL mode.""",
    ]
    | None = None,
    series_id_column: Annotated[
        str,
        """Column name identifying different series (e.g., 'store_id', 'region').
        Must exist in the input data.""",
    ]
    | None = None,
    max_explanations: Annotated[
        int,
        """Number of prediction explanations to return per prediction.
        - 0: No explanations (default)
        - Positive integer: Specific number of explanations
        - 'all': All available explanations (SHAP only)
        Note: For SHAP, 0 means all explanations; for XEMP, 0 means none.""",
    ]
    | None = 0,
    max_ngram_explanations: Annotated[
        int,
        """Maximum number of text explanations per prediction.
        Recommended: 'all' for text models. None disables text explanations.""",
    ]
    | None = None,
    threshold_high: Annotated[
        float,
        """Only compute explanations for predictions above this threshold (0.0-1.0).
        Useful for focusing explanations on high-confidence predictions.""",
    ]
    | None = None,
    threshold_low: Annotated[
        float,
        """Only compute explanations for predictions below this threshold (0.0-1.0).
        Useful for focusing explanations on low-confidence predictions.""",
    ]
    | None = None,
    passthrough_columns: Annotated[
        str,
        """Input columns to include in output alongside predictions.
        - 'all': Include all input columns
        - 'column1,column2': Comma-separated list of specific columns
        - None: No passthrough columns (default)""",
    ]
    | None = None,
    explanation_algorithm: Annotated[
        str,
        """Algorithm for computing explanations.
        - 'shap': SHAP explanations (default for most models)
        - 'xemp': XEMP explanations (faster, less accurate)
        - None: Use deployment default""",
    ]
    | None = None,
    prediction_endpoint: Annotated[
        str,
        """Override the prediction server endpoint URL.
        Useful for custom prediction servers or Portable Prediction Server.""",
    ]
    | None = None,
    timeout: Annotated[int, "Timeout in seconds for the prediction job"] | None = 600,
) -> ToolError | ToolResult:
    """
    Make real-time predictions using a DataRobot deployment and a local CSV file or a dataset
    string.

    This is the unified prediction function that supports:
    - Regular classification/regression predictions
    - Time series forecasting with advanced parameters
    - Prediction explanations (SHAP/XEMP)
    - Text explanations for NLP models
    - Custom thresholds and passthrough columns

    For regular predictions: Just provide deployment_id and file_path or dataset
    For time series: Add forecast_point OR forecast_range_start/end
    For explanations: Set max_explanations > 0 and optionally explanation_algorithm
    For text models: Use max_ngram_explanations for text feature explanations

    When using this tool, always consider feature importance. For features with high importance,
    try to infer or ask for a reasonable value, using frequent values or domain knowledge if
    available.
    For less important features, you may leave them blank.

    Examples
    --------
    # Regular binary classification
    predict_realtime(deployment_id="abc123", file_path="data.csv")

    # With SHAP explanations
    predict_realtime(deployment_id="abc123", file_path="data.csv",
                    max_explanations=10, explanation_algorithm="shap")

    # Time series forecasting
    predict_realtime(deployment_id="abc123", file_path="ts_data.csv",
                    forecast_point="2024-06-01")

    # Multiseries time series
    predict_realtime(deployment_id="abc123", file_path="multiseries.csv",
                    forecast_point="2024-06-01", series_id_column="store_id")

    # Historical time series predictions
    predict_realtime(deployment_id="abc123", file_path="ts_data.csv",
                    forecast_range_start="2024-06-01",
                    forecast_range_end="2024-06-07")

    # Text model with explanations and passthrough
    predict_realtime(deployment_id="abc123", file_path="text_data.csv",
                    max_explanations="all", max_ngram_explanations="all",
                    passthrough_columns="document_id,customer_id")
    """
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")
    if not dataset and not file_path:
        raise ToolError("Either dataset or file_path must be provided")

    # Load input data from dataset string or file_path
    if dataset is not None:
        # Try CSV first
        try:
            df = pd.read_csv(io.StringIO(dataset))
        except Exception:
            # Try JSON
            try:
                data = json.loads(dataset)
                df = pd.DataFrame(data)
            except Exception as e:
                raise ValueError(f"Could not parse dataset string as CSV or JSON: {e}")
    elif file_path is not None:
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Either file_path or dataset must be provided.")

    # Normalize column names: strip leading/trailing whitespace
    df.columns = df.columns.str.strip()

    if series_id_column and series_id_column not in df.columns:
        raise ValueError(f"series_id_column '{series_id_column}' not found in input data.")

    client = get_sdk_client()
    deployment = client.Deployment.get(deployment_id=deployment_id)

    # Check if this is a time series prediction or regular prediction
    is_time_series = bool(forecast_point or (forecast_range_start and forecast_range_end))

    # Start with base prediction parameters
    predict_kwargs = {
        "deployment": deployment,
        "data_frame": df,
        "timeout": timeout,
    }

    # Add time series parameters if applicable
    if is_time_series:
        if forecast_point:
            forecast_point_dt = pd.to_datetime(forecast_point)
            predict_kwargs["time_series_type"] = TimeSeriesType.FORECAST
            predict_kwargs["forecast_point"] = forecast_point_dt
        elif forecast_range_start and forecast_range_end:
            predictions_start_date_dt = pd.to_datetime(forecast_range_start)
            predictions_end_date_dt = pd.to_datetime(forecast_range_end)
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
    bucket_info = make_output_settings()
    prediction_results = predictions_result_response(
        predictions,
        bucket_info.bucket,
        bucket_info.key,
        f"pred_{deployment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        max_explanations not in {0, "0"},
    )
    content = (
        prediction_results.model_dump()
        if hasattr(prediction_results, "model_dump")
        else prediction_results
        if isinstance(prediction_results, dict)
        else prediction_results.__dict__
    )
    return ToolResult(structured_content=content)
