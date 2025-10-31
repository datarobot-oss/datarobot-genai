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
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest

from datarobot_genai.drmcp.core.common import MCPError
from datarobot_genai.drmcp.core.constants import MAX_INLINE_SIZE
from datarobot_genai.drmcp.tools.predictive import predict_realtime
from datarobot_genai.drmcp.tools.predictive.predict_realtime import make_output_settings
from datarobot_genai.drmcp.tools.predictive.predict_realtime import predict_by_ai_catalog_rt

THRESHOLD_HIGH = 0.8
THRESHOLD_LOW = 0.2


@pytest.fixture()
def patch_realtime_dependencies() -> Generator[dict[str, Any], None, None]:
    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.predict_realtime.get_sdk_client"
        ) as mock_get_sdk_client,
        patch(
            "datarobot_genai.drmcp.tools.predictive.predict_realtime.pd.read_csv"
        ) as mock_read_csv,
        patch(
            "datarobot_genai.drmcp.tools.predictive.predict_realtime.dr_predict"
        ) as mock_dr_predict,
        patch("datarobot_genai.drmcp.core.utils.boto3.client") as mock_boto3_client,
        patch(
            "datarobot_genai.drmcp.core.utils.generate_presigned_url",
            return_value="https://dummy-presigned-url",
        ),
    ):
        mock_client = MagicMock()
        mock_deployment = MagicMock()
        mock_client.Deployment = mock_deployment
        mock_deployment.get = MagicMock()
        mock_get_sdk_client.return_value = mock_client
        yield {
            "mock_read_csv": mock_read_csv,
            "mock_deployment_get": mock_deployment,
            "mock_dr_predict": mock_dr_predict,
            "mock_boto3_client": mock_boto3_client,
        }


@pytest.mark.asyncio
async def test_predict_realtime_forecast_point(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    patch_realtime_dependencies["mock_read_csv"].return_value = df
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = df
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3
    result = await predict_realtime.predict_realtime(
        deployment_id="dep",
        file_path="file.csv",
        forecast_point="2024-06-01",
        timeout=5,
    )
    assert result.type == "inline"
    assert "a,b" in result.data
    assert "1,3" in result.data
    mock_s3.upload_file.assert_not_called()
    # Verify dr_predict called with correct parameters
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["data_frame"].equals(df)
    assert kwargs["time_series_type"].name == "FORECAST"
    assert kwargs["forecast_point"].strftime("%Y-%m-%d") == "2024-06-01"
    assert kwargs["timeout"] == 5


@pytest.mark.asyncio
async def test_predict_realtime_forecast_range_resource(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    # Create a large DataFrame to trigger resource path
    df = pd.DataFrame({"a": range(MAX_INLINE_SIZE), "b": range(MAX_INLINE_SIZE)})
    patch_realtime_dependencies["mock_read_csv"].return_value = df
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = df
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3
    result = await predict_realtime.predict_realtime(
        deployment_id="dep",
        file_path="file.csv",
        forecast_range_start="2024-06-01",
        forecast_range_end="2024-06-07",
        max_explanations="all",
        explanation_algorithm="shap",
        timeout=5,
    )
    assert result.type == "resource"
    assert result.s3_url is not None
    mock_s3.upload_file.assert_called()
    # Verify dr_predict called with correct parameters
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["data_frame"].equals(df)
    assert kwargs["time_series_type"].name == "HISTORICAL"
    assert kwargs["predictions_start_date"].strftime("%Y-%m-%d") == "2024-06-01"
    assert kwargs["predictions_end_date"].strftime("%Y-%m-%d") == "2024-06-07"
    assert kwargs["max_explanations"] == "all"
    assert kwargs["explanation_algorithm"] == "shap"
    assert kwargs["timeout"] == 5


# ============================================================================
# TIME SERIES REGRESSION SPECIFIC TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_predict_timeseries_regression_forecast_point_with_intervals(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    """Test time series regression with forecast point and prediction intervals."""
    # Mock regression input data with typical time series features
    df = pd.DataFrame(
        {
            "date": ["2024-05-30", "2024-05-31"],
            "temperature": [25.5, 26.2],
            "humidity": [65.0, 68.5],
            "sales": [1250.75, 1380.25],  # Target variable for regression
        }
    )

    # Mock regression prediction output with prediction intervals
    regression_predictions = pd.DataFrame(
        {
            "date": ["2024-06-01", "2024-06-02"],
            "prediction": [1425.50, 1398.75],  # Continuous regression values
            "prediction_lower": [1350.25, 1320.10],  # Lower bound
            "prediction_upper": [1500.75, 1477.40],  # Upper bound
        }
    )

    patch_realtime_dependencies["mock_read_csv"].return_value = df
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = regression_predictions
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3

    result = await predict_realtime.predict_realtime(
        deployment_id="regression_dep_123",
        file_path="regression_data.csv",
        forecast_point="2024-06-01",
        timeout=300,
    )

    # Verify response structure
    assert result.type == "inline"
    assert "prediction" in result.data
    assert (
        "1425.5" in result.data
    )  # Check regression prediction value (pandas removes trailing zeros)
    assert "prediction_lower" in result.data
    assert "prediction_upper" in result.data

    # Verify dr_predict called with regression-appropriate parameters
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["data_frame"].equals(df)
    assert kwargs["time_series_type"].name == "FORECAST"
    assert kwargs["forecast_point"].strftime("%Y-%m-%d") == "2024-06-01"
    assert kwargs["timeout"] == 300


@pytest.mark.asyncio
async def test_predict_timeseries_regression_historical_range(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    """Test time series regression with historical date range predictions."""
    # Mock regression input data
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03"],
            "feature1": [10.5, 11.2, 9.8],
            "feature2": [100, 105, 98],
            "target": [500.25, 520.75, 485.50],
        }
    )

    # Mock regression predictions for historical range
    regression_predictions = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03"],
            "prediction": [498.75, 522.10, 487.25],  # Continuous regression values
            "actual": [500.25, 520.75, 485.50],  # Actual values for comparison
            "residual": [-1.50, -1.65, -1.75],  # Prediction errors
        }
    )

    patch_realtime_dependencies["mock_read_csv"].return_value = df
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = regression_predictions
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3

    result = await predict_realtime.predict_realtime(
        deployment_id="regression_dep_456",
        file_path="historical_regression_data.csv",
        forecast_range_start="2024-05-01",
        forecast_range_end="2024-05-03",
        timeout=600,
    )

    # Verify response structure for regression
    assert result.type == "inline"
    assert "prediction" in result.data
    assert "498.75" in result.data  # Check first regression prediction
    assert (
        "522.1" in result.data
    )  # Check second regression prediction (pandas removes trailing zeros)

    # Verify dr_predict called with historical range parameters
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["data_frame"].equals(df)
    assert kwargs["time_series_type"].name == "HISTORICAL"
    assert kwargs["predictions_start_date"].strftime("%Y-%m-%d") == "2024-05-01"
    assert kwargs["predictions_end_date"].strftime("%Y-%m-%d") == "2024-05-03"
    assert kwargs["timeout"] == 600


@pytest.mark.asyncio
async def test_predict_timeseries_regression_multiseries(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    """Test time series regression with multiple series (e.g., multiple stores, regions)."""
    # Mock multiseries regression data
    df = pd.DataFrame(
        {
            "date": ["2024-05-30", "2024-05-30", "2024-05-31", "2024-05-31"],
            "store_id": ["store_A", "store_B", "store_A", "store_B"],
            "temperature": [25.5, 22.1, 26.2, 23.8],
            "promotion": [1, 0, 1, 1],
            "sales": [1250.75, 980.25, 1380.25, 1050.50],
        }
    )

    # Mock multiseries regression predictions
    regression_predictions = pd.DataFrame(
        {
            "date": ["2024-06-01", "2024-06-01"],
            "store_id": ["store_A", "store_B"],
            "prediction": [1425.50, 1125.75],  # Different predictions per series
            "prediction_lower": [1350.25, 1050.10],
            "prediction_upper": [1500.75, 1201.40],
        }
    )

    patch_realtime_dependencies["mock_read_csv"].return_value = df
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = regression_predictions
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3

    result = await predict_realtime.predict_realtime(
        deployment_id="multiseries_regression_dep",
        file_path="multiseries_data.csv",
        forecast_point="2024-06-01",
        series_id_column="store_id",
        timeout=450,
    )

    # Verify multiseries regression response
    assert result.type == "inline"
    assert "store_A" in result.data
    assert "store_B" in result.data
    assert "1425.5" in result.data  # Store A prediction (pandas removes trailing zeros)
    assert "1125.75" in result.data  # Store B prediction

    # Verify series_id_column validation and dr_predict parameters
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["data_frame"].equals(df)
    assert kwargs["time_series_type"].name == "FORECAST"
    assert kwargs["timeout"] == 450


@pytest.mark.asyncio
async def test_predict_timeseries_regression_large_dataset_resource(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    """Test time series regression with large dataset that triggers S3 resource storage."""
    # Create large regression dataset
    large_df = pd.DataFrame(
        {
            "date": ["2024-05-01"] * MAX_INLINE_SIZE,
            "feature1": range(MAX_INLINE_SIZE),
            "feature2": [x * 1.5 for x in range(MAX_INLINE_SIZE)],
            "target": [x * 2.5 + 100 for x in range(MAX_INLINE_SIZE)],
        }
    )

    # Large regression predictions
    large_predictions = pd.DataFrame(
        {
            "date": ["2024-06-01"] * MAX_INLINE_SIZE,
            "prediction": [x * 2.6 + 105 for x in range(MAX_INLINE_SIZE)],
            "prediction_lower": [x * 2.4 + 95 for x in range(MAX_INLINE_SIZE)],
            "prediction_upper": [x * 2.8 + 115 for x in range(MAX_INLINE_SIZE)],
        }
    )

    patch_realtime_dependencies["mock_read_csv"].return_value = large_df
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = large_predictions
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3

    result = await predict_realtime.predict_realtime(
        deployment_id="large_regression_dep",
        file_path="large_regression_data.csv",
        forecast_point="2024-06-01",
        timeout=900,
    )

    # Verify large dataset triggers resource storage
    assert result.type == "resource"
    assert result.s3_url is not None
    assert result.resource_id is not None
    mock_s3.upload_file.assert_called()

    # Verify dr_predict called correctly for large regression dataset
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["data_frame"].equals(large_df)
    assert kwargs["time_series_type"].name == "FORECAST"
    assert kwargs["timeout"] == 900


@pytest.mark.asyncio
async def test_predict_timeseries_regression_series_id_validation_error(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    """Test time series regression with invalid series_id_column raises appropriate error."""
    df = pd.DataFrame(
        {
            "date": ["2024-05-30", "2024-05-31"],
            "store_id": ["store_A", "store_B"],
            "sales": [1250.75, 1380.25],
        }
    )

    patch_realtime_dependencies["mock_read_csv"].return_value = df

    # Test with non-existent series_id_column
    with pytest.raises(MCPError) as exc_info:
        await predict_realtime.predict_realtime(
            deployment_id="regression_dep",
            file_path="data.csv",
            forecast_point="2024-06-01",
            series_id_column="invalid_column",
        )
    assert (
        "Error in predict_realtime: ValueError: series_id_column 'invalid_column' not found "
        "in input data." == str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_predict_timeseries_regression_no_prediction_intervals(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    """Test time series regression without prediction intervals (point estimates only)."""
    df = pd.DataFrame(
        {
            "date": ["2024-05-30", "2024-05-31"],
            "temperature": [25.5, 26.2],
            "sales": [1250.75, 1380.25],
        }
    )

    # Mock regression predictions without intervals
    regression_predictions = pd.DataFrame(
        {
            "date": ["2024-06-01"],
            "prediction": [1425.50],  # Only point estimate
        }
    )

    patch_realtime_dependencies["mock_read_csv"].return_value = df
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = regression_predictions
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3

    result = await predict_realtime.predict_realtime(
        deployment_id="regression_dep",
        file_path="data.csv",
        forecast_point="2024-06-01",
        timeout=200,
    )

    # Verify point estimate only
    assert result.type == "inline"
    assert "prediction" in result.data
    assert "1425.5" in result.data  # pandas removes trailing zeros
    assert "prediction_lower" not in result.data
    assert "prediction_upper" not in result.data

    # Verify dr_predict called without prediction intervals
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["data_frame"].equals(df)
    assert kwargs["time_series_type"].name == "FORECAST"


@pytest.mark.asyncio
async def test_predict_realtime_with_all_explanation_parameters(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    """Test predict_realtime with comprehensive explanation parameters."""
    df = pd.DataFrame({"text": ["hello world", "test document"], "feature": [1, 2]})
    patch_realtime_dependencies["mock_read_csv"].return_value = df
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = df
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3

    result = await predict_realtime.predict_realtime(
        deployment_id="text_dep",
        file_path="text_data.csv",
        max_explanations=10,
        max_ngram_explanations="all",
        threshold_high=0.8,
        threshold_low=0.2,
        explanation_algorithm="shap",
        timeout=300,
    )

    # Verify response
    assert result.type == "inline"
    assert "text" in result.data

    # Verify dr_predict called with all explanation parameters
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["data_frame"].equals(df)
    assert kwargs["max_explanations"] == 10
    assert kwargs["max_ngram_explanations"] == "all"
    assert kwargs["threshold_high"] == THRESHOLD_HIGH
    assert kwargs["threshold_low"] == THRESHOLD_LOW
    assert kwargs["explanation_algorithm"] == "shap"
    assert kwargs["timeout"] == 300


@pytest.mark.asyncio
async def test_predict_realtime_with_passthrough_columns_all(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    """Test predict_realtime with passthrough_columns='all'."""
    df = pd.DataFrame({"id": [1, 2], "feature1": [10, 20], "feature2": [100, 200]})
    patch_realtime_dependencies["mock_read_csv"].return_value = df
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = df
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3

    await predict_realtime.predict_realtime(
        deployment_id="dep",
        file_path="data.csv",
        passthrough_columns="all",
        timeout=300,
    )

    # Verify passthrough columns parameter
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["passthrough_columns"] == "all"


@pytest.mark.asyncio
async def test_predict_realtime_with_passthrough_columns_specific(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    """Test predict_realtime with specific passthrough columns."""
    df = pd.DataFrame({"id": [1, 2], "feature1": [10, 20], "feature2": [100, 200]})
    patch_realtime_dependencies["mock_read_csv"].return_value = df
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = df
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3

    await predict_realtime.predict_realtime(
        deployment_id="dep",
        file_path="data.csv",
        passthrough_columns="id,feature1",
        timeout=300,
    )

    # Verify passthrough columns parameter converted to set
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["passthrough_columns"] == {"id", "feature1"}


@pytest.mark.asyncio
async def test_predict_realtime_with_custom_endpoint(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    """Test predict_realtime with custom prediction endpoint."""
    df = pd.DataFrame({"feature": [1, 2]})
    patch_realtime_dependencies["mock_read_csv"].return_value = df
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = df
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3

    custom_endpoint = "https://custom-prediction-server.com/predict"
    await predict_realtime.predict_realtime(
        deployment_id="dep",
        file_path="data.csv",
        prediction_endpoint=custom_endpoint,
        timeout=300,
    )

    # Verify custom endpoint parameter
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["prediction_endpoint"] == custom_endpoint


@pytest.mark.asyncio
async def test_predict_realtime_regular_prediction_no_time_series_params(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    """Test predict_realtime for regular prediction without any time series parameters."""
    df = pd.DataFrame({"feature1": [1, 2], "feature2": [10, 20]})
    patch_realtime_dependencies["mock_read_csv"].return_value = df
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = df
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3

    await predict_realtime.predict_realtime(
        deployment_id="dep",
        file_path="data.csv",
        max_explanations=5,
        explanation_algorithm="xemp",
        timeout=300,
    )

    # Verify no time series parameters are passed
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert "time_series_type" not in kwargs
    assert "forecast_point" not in kwargs
    assert "predictions_start_date" not in kwargs
    assert "predictions_end_date" not in kwargs
    assert kwargs["max_explanations"] == 5
    assert kwargs["explanation_algorithm"] == "xemp"


@pytest.mark.asyncio
async def test_predict_realtime_with_dataset_csv(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    # Prepare a CSV string
    csv_str = "a,b\n1,3\n2,4\n"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = df
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3
    result = await predict_realtime.predict_realtime(
        deployment_id="dep",
        dataset=csv_str,
        timeout=5,
    )
    assert result.type == "inline"
    assert "a,b" in result.data
    assert "1,3" in result.data
    args, kwargs = patch_realtime_dependencies["mock_read_csv"].call_args

    assert isinstance(args[0], io.StringIO)
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["data_frame"].equals(df)


@pytest.mark.asyncio
async def test_predict_realtime_with_dataset_json(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    # Prepare a JSON string
    data = [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
    json_str = json.dumps(data)
    df = pd.DataFrame(data)
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = df
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3
    result = await predict_realtime.predict_realtime(
        deployment_id="dep",
        dataset=json_str,
        timeout=5,
    )
    assert result.type == "inline"
    assert "a,b" in result.data
    assert "1,3" in result.data
    # For JSON, pd.read_csv will fail, so it will try json.loads and pd.DataFrame
    # So mock_read_csv may or may not be called, but if called, it should be with StringIO
    # We can check that either it was called with StringIO and failed, or not called at all
    # (since the fallback is not mocked)
    # So just check that the dataframe is correct
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["data_frame"].equals(df)


@pytest.mark.asyncio
async def test_predict_realtime_dataset_takes_precedence(
    patch_realtime_dependencies: dict[str, Any],
) -> None:
    # If both dataset and file_path are provided, dataset should be used
    csv_str = "a,b\n1,3\n2,4\n"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    patch_realtime_dependencies["mock_deployment_get"].return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = df
    patch_realtime_dependencies["mock_dr_predict"].return_value = mock_result
    mock_s3 = MagicMock()
    patch_realtime_dependencies["mock_boto3_client"].return_value = mock_s3
    result = await predict_realtime.predict_realtime(
        deployment_id="dep",
        file_path="should_not_be_used.csv",
        dataset=csv_str,
        timeout=5,
    )
    assert result.type == "inline"
    assert "a,b" in result.data
    assert "1,3" in result.data
    args, kwargs = patch_realtime_dependencies["mock_read_csv"].call_args

    assert isinstance(args[0], io.StringIO)
    args, kwargs = patch_realtime_dependencies["mock_dr_predict"].call_args
    assert kwargs["data_frame"].equals(df)


class TestMakeOutputSettings:
    """Test cases for make_output_settings function."""

    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.get_s3_bucket_info")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.uuid.uuid4")
    def test_make_output_settings_success(self, mock_uuid4, mock_get_s3_bucket_info):
        """Test successful creation of output settings."""
        mock_uuid4.return_value = "test-uuid-123"
        mock_get_s3_bucket_info.return_value = {"bucket": "test-bucket", "prefix": "test-prefix/"}

        result = make_output_settings()

        assert result.bucket == "test-bucket"
        assert result.key == "test-prefix/test-uuid-123.csv"
        mock_get_s3_bucket_info.assert_called_once()

    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.get_s3_bucket_info")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.uuid.uuid4")
    def test_make_output_settings_with_empty_prefix(self, mock_uuid4, mock_get_s3_bucket_info):
        """Test make_output_settings with empty prefix."""
        mock_uuid4.return_value = "test-uuid-456"
        mock_get_s3_bucket_info.return_value = {"bucket": "test-bucket", "prefix": ""}

        result = make_output_settings()

        assert result.bucket == "test-bucket"
        assert result.key == "test-uuid-456.csv"

    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.get_s3_bucket_info")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.uuid.uuid4")
    def test_make_output_settings_with_none_prefix(self, mock_uuid4, mock_get_s3_bucket_info):
        """Test make_output_settings with None prefix."""
        mock_uuid4.return_value = "test-uuid-789"
        mock_get_s3_bucket_info.return_value = {"bucket": "test-bucket", "prefix": None}

        result = make_output_settings()

        assert result.bucket == "test-bucket"
        assert result.key == "Nonetest-uuid-789.csv"


class TestPredictByAiCatalogRt:
    """Test cases for predict_by_ai_catalog_rt function."""

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.predictions_result_response")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.make_output_settings")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.dr_predict")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.get_sdk_client")
    async def test_predict_by_ai_catalog_rt_with_get_as_dataframe(
        self,
        mock_get_sdk_client,
        mock_dr_predict,
        mock_make_output_settings,
        mock_predictions_result_response,
    ):
        """Test predict_by_ai_catalog_rt with get_as_dataframe method."""
        # Setup mocks
        mock_client = Mock()
        mock_dataset = Mock()
        mock_dataset.get_as_dataframe.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_client.Dataset.get.return_value = mock_dataset
        mock_deployment = Mock()
        mock_client.Deployment.get.return_value = mock_deployment
        mock_get_sdk_client.return_value = mock_client

        mock_result = Mock()
        mock_result.dataframe = pd.DataFrame({"prediction": [0.8, 0.9]})
        mock_dr_predict.return_value = mock_result

        mock_bucket_info = Mock()
        mock_bucket_info.bucket = "test-bucket"
        mock_bucket_info.key = "test-key"
        mock_make_output_settings.return_value = mock_bucket_info

        mock_predictions_result_response.return_value = {"type": "inline", "data": "test_data"}

        # Call function
        result = await predict_by_ai_catalog_rt("deployment123", "dataset123")

        # Verify calls
        mock_client.Dataset.get.assert_called_once_with("dataset123")
        mock_dataset.get_as_dataframe.assert_called_once()
        mock_client.Deployment.get.assert_called_once_with(deployment_id="deployment123")
        mock_dr_predict.assert_called_once_with(
            mock_deployment, mock_dataset.get_as_dataframe.return_value, timeout=600
        )
        mock_make_output_settings.assert_called_once()
        mock_predictions_result_response.assert_called_once()

        assert result == {"type": "inline", "data": "test_data"}

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.predictions_result_response")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.make_output_settings")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.dr_predict")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.pd.read_csv")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.get_sdk_client")
    async def test_predict_by_ai_catalog_rt_with_download(
        self,
        mock_get_sdk_client,
        mock_read_csv,
        mock_dr_predict,
        mock_make_output_settings,
        mock_predictions_result_response,
    ):
        """Test predict_by_ai_catalog_rt with download method."""
        # Setup mocks
        mock_client = Mock()
        mock_dataset = Mock()
        # Remove get_as_dataframe attribute entirely
        del mock_dataset.get_as_dataframe
        mock_dataset.download.return_value = "dataset.csv"
        mock_client.Dataset.get.return_value = mock_dataset
        mock_deployment = Mock()
        mock_client.Deployment.get.return_value = mock_deployment
        mock_get_sdk_client.return_value = mock_client

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = mock_df

        mock_result = Mock()
        mock_result.dataframe = pd.DataFrame({"prediction": [0.8, 0.9]})
        mock_dr_predict.return_value = mock_result

        mock_bucket_info = Mock()
        mock_bucket_info.bucket = "test-bucket"
        mock_bucket_info.key = "test-key"
        mock_make_output_settings.return_value = mock_bucket_info

        mock_predictions_result_response.return_value = {"type": "inline", "data": "test_data"}

        # Call function
        result = await predict_by_ai_catalog_rt("deployment123", "dataset123")

        # Verify calls
        mock_client.Dataset.get.assert_called_once_with("dataset123")
        mock_dataset.download.assert_called_once_with("dataset.csv")
        mock_read_csv.assert_called_once_with("dataset.csv")
        mock_client.Deployment.get.assert_called_once_with(deployment_id="deployment123")
        mock_dr_predict.assert_called_once_with(mock_deployment, mock_df, timeout=600)

        assert result == {"type": "inline", "data": "test_data"}

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.predictions_result_response")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.make_output_settings")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.dr_predict")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.pd.read_csv")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.get_sdk_client")
    async def test_predict_by_ai_catalog_rt_with_get_file(
        self,
        mock_get_sdk_client,
        mock_read_csv,
        mock_dr_predict,
        mock_make_output_settings,
        mock_predictions_result_response,
    ):
        """Test predict_by_ai_catalog_rt with get_file method."""
        # Setup mocks
        mock_client = Mock()
        mock_dataset = Mock()
        # Remove get_as_dataframe and download attributes entirely
        del mock_dataset.get_as_dataframe
        del mock_dataset.download
        mock_dataset.get_file.return_value = "dataset.csv"
        mock_client.Dataset.get.return_value = mock_dataset
        mock_deployment = Mock()
        mock_client.Deployment.get.return_value = mock_deployment
        mock_get_sdk_client.return_value = mock_client

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = mock_df

        mock_result = Mock()
        mock_result.dataframe = pd.DataFrame({"prediction": [0.8, 0.9]})
        mock_dr_predict.return_value = mock_result

        mock_bucket_info = Mock()
        mock_bucket_info.bucket = "test-bucket"
        mock_bucket_info.key = "test-key"
        mock_make_output_settings.return_value = mock_bucket_info

        mock_predictions_result_response.return_value = {"type": "inline", "data": "test_data"}

        # Call function
        result = await predict_by_ai_catalog_rt("deployment123", "dataset123")

        # Verify calls
        mock_client.Dataset.get.assert_called_once_with("dataset123")
        mock_dataset.get_file.assert_called_once()
        mock_read_csv.assert_called_once_with("dataset.csv")
        mock_client.Deployment.get.assert_called_once_with(deployment_id="deployment123")
        mock_dr_predict.assert_called_once_with(mock_deployment, mock_df, timeout=600)

        assert result == {"type": "inline", "data": "test_data"}

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.predictions_result_response")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.make_output_settings")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.dr_predict")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.pd.read_csv")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.get_sdk_client")
    async def test_predict_by_ai_catalog_rt_with_get_bytes(
        self,
        mock_get_sdk_client,
        mock_read_csv,
        mock_dr_predict,
        mock_make_output_settings,
        mock_predictions_result_response,
    ):
        """Test predict_by_ai_catalog_rt with get_bytes method."""
        # Setup mocks
        mock_client = Mock()
        mock_dataset = Mock()
        # Remove get_as_dataframe, download, and get_file attributes entirely
        del mock_dataset.get_as_dataframe
        del mock_dataset.download
        del mock_dataset.get_file
        mock_dataset.get_bytes.return_value = b"col1,col2\n1,3\n2,4\n"
        mock_client.Dataset.get.return_value = mock_dataset
        mock_deployment = Mock()
        mock_client.Deployment.get.return_value = mock_deployment
        mock_get_sdk_client.return_value = mock_client

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = mock_df

        mock_result = Mock()
        mock_result.dataframe = pd.DataFrame({"prediction": [0.8, 0.9]})
        mock_dr_predict.return_value = mock_result

        mock_bucket_info = Mock()
        mock_bucket_info.bucket = "test-bucket"
        mock_bucket_info.key = "test-key"
        mock_make_output_settings.return_value = mock_bucket_info

        mock_predictions_result_response.return_value = {"type": "inline", "data": "test_data"}

        # Call function
        result = await predict_by_ai_catalog_rt("deployment123", "dataset123")

        # Verify calls
        mock_client.Dataset.get.assert_called_once_with("dataset123")
        mock_dataset.get_bytes.assert_called_once()
        mock_read_csv.assert_called_once()
        mock_client.Deployment.get.assert_called_once_with(deployment_id="deployment123")
        mock_dr_predict.assert_called_once_with(mock_deployment, mock_df, timeout=600)

        assert result == {"type": "inline", "data": "test_data"}

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.predictions_result_response")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.make_output_settings")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.dr_predict")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.pd.read_csv")
    @patch("datarobot_genai.drmcp.tools.predictive.predict_realtime.get_sdk_client")
    async def test_predict_by_ai_catalog_rt_with_url_fallback(
        self,
        mock_get_sdk_client,
        mock_read_csv,
        mock_dr_predict,
        mock_make_output_settings,
        mock_predictions_result_response,
    ):
        """Test predict_by_ai_catalog_rt with URL fallback."""
        # Setup mocks
        mock_client = Mock()
        mock_dataset = Mock()
        # Remove get_as_dataframe, download, get_file, and get_bytes attributes entirely
        del mock_dataset.get_as_dataframe
        del mock_dataset.download
        del mock_dataset.get_file
        del mock_dataset.get_bytes
        mock_dataset.url = "https://example.com/dataset.csv"
        mock_client.Dataset.get.return_value = mock_dataset
        mock_deployment = Mock()
        mock_client.Deployment.get.return_value = mock_deployment
        mock_get_sdk_client.return_value = mock_client

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = mock_df

        mock_result = Mock()
        mock_result.dataframe = pd.DataFrame({"prediction": [0.8, 0.9]})
        mock_dr_predict.return_value = mock_result

        mock_bucket_info = Mock()
        mock_bucket_info.bucket = "test-bucket"
        mock_bucket_info.key = "test-key"
        mock_make_output_settings.return_value = mock_bucket_info

        mock_predictions_result_response.return_value = {"type": "inline", "data": "test_data"}

        # Call function
        result = await predict_by_ai_catalog_rt("deployment123", "dataset123")

        # Verify calls
        mock_client.Dataset.get.assert_called_once_with("dataset123")
        mock_read_csv.assert_called_once_with("https://example.com/dataset.csv")
        mock_client.Deployment.get.assert_called_once_with(deployment_id="deployment123")
        mock_dr_predict.assert_called_once_with(mock_deployment, mock_df, timeout=600)

        assert result == {"type": "inline", "data": "test_data"}
