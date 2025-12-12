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

import pandas as pd
from datarobot_predict import TimeSeriesType
from datarobot_predict.deployment import predict as dr_predict
from pydantic import BaseModel

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.core.utils import PredictionResponse
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


@dr_mcp_tool(tags={"prediction", "realtime", "scoring"})
async def predict_by_ai_catalog_rt(
    deployment_id: str,
    dataset_id: str,
    timeout: int = 600,
) -> PredictionResponse:
    """
    Make real-time predictions using a DataRobot deployment and an AI Catalog dataset using the
    datarobot-predict library.
    Use this for fast results when your data is not huge (not gigabytes). Results larger than 1MB
    will be returned as a resource id and S3 URL; smaller results will be returned inline as a CSV
    string.

    Args:
        deployment_id: The ID of the DataRobot deployment to use for prediction.
        dataset_id: ID of an AI Catalog item to use as input data.
        timeout: Timeout in seconds for the prediction job (default 600).

    Returns
    -------
        dict: {"type": "inline", "data": csv_str} for small results (<1MB), or {"type": "resource",
        "resource_id": ..., "s3_url": ...} for large results (>=1MB).
    """
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
    result = dr_predict(deployment, df, timeout=timeout)
    predictions = result.dataframe
    bucket_info = make_output_settings()
    return predictions_result_response(
        predictions,
        bucket_info.bucket,
        bucket_info.key,
        f"pred_{deployment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        True,
    )


@dr_mcp_tool(tags={"prediction", "realtime", "scoring"})
async def predict_realtime(
    deployment_id: str,
    file_path: str | None = None,
    dataset: str | None = None,
    forecast_point: str | None = None,
    forecast_range_start: str | None = None,
    forecast_range_end: str | None = None,
    series_id_column: str | None = None,
    max_explanations: int | str = 0,
    max_ngram_explanations: int | str | None = None,
    threshold_high: float | None = None,
    threshold_low: float | None = None,
    passthrough_columns: str | None = None,
    explanation_algorithm: str | None = None,
    prediction_endpoint: str | None = None,
    timeout: int = 600,
) -> PredictionResponse:
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

    Args:
        deployment_id: The ID of the DataRobot deployment to use for prediction.
        file_path: Path to a CSV file to use as input data. For time series with forecast_point,
                   must have at least 4 historical values within the feature derivation window.
        dataset: (Optional) CSV or JSON string representing the input data. If provided, this
            takes precedence over file_path.
        forecast_point: (Time Series) Date to start forecasting from (e.g., "2024-06-01").
                        If provided, triggers time series FORECAST mode. Uses most recent date if
                        None.
        forecast_range_start: (Time Series) Start date for historical predictions (e.g.,
            "2024-06-01").
                              Must be used with forecast_range_end for HISTORICAL mode.
        forecast_range_end: (Time Series) End date for historical predictions (e.g., "2024-06-07").
                            Must be used with forecast_range_start for HISTORICAL mode.
        series_id_column: (Multiseries Time Series) Column name identifying different series
                          (e.g., "store_id", "region"). Must exist in the input data.
        max_explanations: Number of prediction explanations to return per prediction.
                          - 0: No explanations (default)
                          - Positive integer: Specific number of explanations
                          - "all": All available explanations (SHAP only)
                          Note: For SHAP, 0 means all explanations; for XEMP, 0 means none.
        max_ngram_explanations: (Text Models) Maximum number of text explanations per prediction.
                                Recommended: "all" for text models. None disables text explanations.
        threshold_high: Only compute explanations for predictions above this threshold (0.0-1.0).
                        Useful for focusing explanations on high-confidence predictions.
        threshold_low: Only compute explanations for predictions below this threshold (0.0-1.0).
                       Useful for focusing explanations on low-confidence predictions.
        passthrough_columns: Input columns to include in output alongside predictions.
                             - "all": Include all input columns
                             - "column1,column2": Comma-separated list of specific columns
                             - None: No passthrough columns (default)
        explanation_algorithm: Algorithm for computing explanations.
                               - "shap": SHAP explanations (default for most models)
                               - "xemp": XEMP explanations (faster, less accurate)
                               - None: Use deployment default
        prediction_endpoint: Override the prediction server endpoint URL.
                             Useful for custom prediction servers or Portable Prediction Server.
        timeout: Request timeout in seconds (default 600).

    Returns
    -------
        dict: Prediction response with the following structure:
        - {"type": "inline", "data": "csv_string"} for results < 1MB
        - {"type": "resource", "resource_id": "...", "s3_url": "..."} for results >= 1MB

        The CSV data contains:
        - Prediction columns (e.g., class probabilities, regression values)
        - Explanation columns (if max_explanations > 0)
        - Passthrough columns (if specified)
        - Time series metadata (for forecasting: FORECAST_POINT, FORECAST_DISTANCE, etc.)

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
    return predictions_result_response(
        predictions,
        bucket_info.bucket,
        bucket_info.key,
        f"pred_{deployment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        max_explanations not in {0, "0"},
    )
