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

import json
from io import StringIO
from typing import Any

import pandas as pd
import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session

SHAP_NOT_SUPPORTED_MSG = "SHAP explanations are not supported"


def _is_shap_not_supported_error(result: Any) -> bool:
    """Return True if the result is the known 'SHAP not supported' error (test can pass)."""
    if result.isError and result.content:
        error_text = (
            result.content[0].text
            if isinstance(result.content[0], TextContent)
            else str(result.content[0])
        )
        return SHAP_NOT_SUPPORTED_MSG in error_text
    return False


@pytest.mark.asyncio
class TestMCPRealtimePredictToolsIntegration:
    """Integration tests for MCP realtime predict tools."""

    async def test_timeseries_regression_forecast_point(
        self, timeseries_regression_project: dict[str, Any], test_data_dir: Any
    ) -> None:
        """Integration test for time series regression with forecast point."""
        async with integration_test_mcp_session() as session:
            deployment_id = timeseries_regression_project["deployment_id"]
            predict_file = test_data_dir / "timeseries_regression_predict.csv"

            # Read the file as a string for the dataset argument
            with open(predict_file) as f:
                dataset_str = f.read()

            result = await session.call_tool(
                "predict_realtime",
                {
                    "deployment_id": deployment_id,
                    "dataset": dataset_str,
                    "forecast_point": "2024-02-01",
                    "timeout": 300,
                },
            )
            assert not result.isError
            result_content = result.content[0]
            assert isinstance(result_content, TextContent)
            result_data = result_content.text

            # Parse the response as JSON to get the type and data
            response_dict = json.loads(result_data)
            assert response_dict["type"] == "inline"
            assert len(response_dict["data"]) > 0

            # Parse CSV and check structure
            df = pd.read_csv(StringIO(response_dict["data"]))
            assert len(df) > 0  # Has some rows
            assert "sales (actual)_PREDICTION" in df.columns  # Has prediction column

            # Check that we got the expected number of forecast days (7 days ahead)
            assert len(df) == 7

    async def test_timeseries_regression_historical_range(
        self, timeseries_regression_project: dict[str, Any], test_data_dir: Any
    ) -> None:
        """Integration test for time series regression with historical date range."""
        async with integration_test_mcp_session() as session:
            deployment_id = timeseries_regression_project["deployment_id"]
            predict_file = test_data_dir / "timeseries_regression_predict.csv"

            result = await session.call_tool(
                "predict_realtime",
                {
                    "deployment_id": deployment_id,
                    "file_path": str(predict_file),
                    "forecast_range_start": "2024-01-31",
                    "forecast_range_end": "2024-02-02",
                    "timeout": 300,
                },
            )
            assert not result.isError
            result_content = result.content[0]
            assert isinstance(result_content, TextContent)
            result_data = result_content.text

            # Parse the response as JSON to get the type and data
            response_dict = json.loads(result_data)
            assert response_dict["type"] == "inline"
            assert len(response_dict["data"]) > 0

            # Parse CSV and check structure
            df = pd.read_csv(StringIO(response_dict["data"]))
            assert "sales (actual)_PREDICTION" in df.columns
            assert len(df) == 14

    async def test_multiseries_regression(
        self, multiseries_regression_project: dict[str, Any], test_data_dir: Any
    ) -> None:
        """Integration test for multiseries time series regression."""
        async with integration_test_mcp_session() as session:
            deployment_id = multiseries_regression_project["deployment_id"]
            predict_file = test_data_dir / "multiseries_regression_predict.csv"

            result = await session.call_tool(
                "predict_realtime",
                {
                    "deployment_id": deployment_id,
                    "file_path": str(predict_file),
                    "forecast_point": "2024-01-17",
                    "series_id_column": "store_id",
                    "timeout": 300,
                },
            )
            assert not result.isError
            result_content = result.content[0]
            assert isinstance(result_content, TextContent)
            result_data = result_content.text

            # Parse the response as JSON to get the type and data
            response_dict = json.loads(result_data)

            # Verify multiseries response
            assert response_dict["type"] in ["inline", "resource"]
            if response_dict["type"] == "inline":
                assert len(response_dict["data"]) > 0
                # Should have predictions for both stores
                assert "store_A" in response_dict["data"]
                assert "store_B" in response_dict["data"]
                # Check for prediction content
                assert "_PREDICTION" in response_dict["data"] or "sales" in response_dict["data"]
            else:
                assert response_dict["s3_url"] is not None

    async def test_regular_realtime_prediction(
        self, timeseries_regression_project: dict[str, Any], test_data_dir: Any
    ) -> None:
        """Integration test for regular real-time prediction (non-time series specific)."""
        async with integration_test_mcp_session() as session:
            deployment_id = timeseries_regression_project["deployment_id"]
            predict_file = test_data_dir / "timeseries_regression_predict.csv"

            # Use the unified predict_realtime function for regular (non-time series) predictions
            result = await session.call_tool(
                "predict_realtime",
                {
                    "deployment_id": deployment_id,
                    "file_path": str(predict_file),
                    "timeout": 300,
                },
            )
            assert not result.isError
            result_content = result.content[0]
            assert isinstance(result_content, TextContent)
            result_data = result_content.text

            # Parse the response as JSON to get the type and data
            response_dict = json.loads(result_data)
            assert response_dict["type"] in ["inline", "resource"]
            if response_dict["type"] == "inline":
                assert "_PREDICTION" in response_dict["data"] or len(response_dict["data"]) > 0
            else:
                assert response_dict["s3_url"] is not None
                assert response_dict["resource_id"] is not None

    async def test_timeseries_regression_error_handling(
        self, timeseries_regression_project: dict[str, Any], test_data_dir: Any
    ) -> None:
        """Integration test for error handling with invalid parameters."""
        async with integration_test_mcp_session() as session:
            deployment_id = timeseries_regression_project["deployment_id"]
            predict_file = test_data_dir / "timeseries_regression_predict.csv"

            # Test with invalid series_id_column
            result = await session.call_tool(
                "predict_realtime",
                {
                    "deployment_id": deployment_id,
                    "file_path": str(predict_file),
                    "forecast_point": "2024-02-01",
                    "series_id_column": "invalid_column",
                    "timeout": 300,
                },
            )
            assert result.isError
            assert (
                result.content[0].text  # type: ignore[union-attr]
                == "Error in predict_realtime: "
                "ValueError: series_id_column 'invalid_column' not found in input data."
            )

    async def test_deployment_compatibility(
        self, timeseries_regression_project: dict[str, Any]
    ) -> None:
        """Integration test to verify deployment is compatible with time series predictions."""
        deployment_id = timeseries_regression_project["deployment_id"]
        deployment = timeseries_regression_project["deployment"]
        model = timeseries_regression_project["model"]

        # Verify deployment exists and is active
        assert deployment.id == deployment_id
        assert deployment.status in [
            "active",
            "Active",
        ]  # Different DR versions may use different casing

        # Verify model is time series regression
        # Note: Model object may not have target_type attribute, but we can verify it's a
        # valid model
        assert model.id is not None

        # Verify project settings
        project = timeseries_regression_project["project"]
        # DataRobot may append "(actual)" to target names, so check for both
        assert project.target in ["sales", "sales (actual)"]

    async def test_predict_realtime_with_explanations(
        self, timeseries_regression_project: dict[str, Any], test_data_dir: Any
    ) -> None:
        """Integration test for predict_realtime with explanation parameters."""
        async with integration_test_mcp_session() as session:
            deployment_id = timeseries_regression_project["deployment_id"]
            predict_file = test_data_dir / "timeseries_regression_predict.csv"

            result = await session.call_tool(
                "predict_realtime",
                {
                    "deployment_id": deployment_id,
                    "file_path": str(predict_file),
                    "max_explanations": 3,
                    "explanation_algorithm": "shap",
                    "passthrough_columns": "all",
                    "timeout": 300,
                },
            )
            if _is_shap_not_supported_error(result):
                return  # deployment doesn't support SHAP; test passes
            assert not result.isError
            result_content = result.content[0]
            assert isinstance(result_content, TextContent)
            result_data = result_content.text

            # Parse the response as JSON to get the type and data
            response_dict = json.loads(result_data)
            assert response_dict["type"] == "inline"
            assert len(response_dict["data"]) > 0

            # Parse the CSV to check for explanation columns and passthrough columns
            df = pd.read_csv(StringIO(response_dict["data"]))

            # Should have prediction columns
            assert "sales (actual)_PREDICTION" in df.columns

            # Should have passthrough columns from input (tool strips column names)
            input_df = pd.read_csv(predict_file)
            for col in input_df.columns:
                col_stripped = col.strip()
                if col_stripped != "sales":  # sales might not be in predictions
                    assert col_stripped in df.columns, (
                        f"Passthrough column {col_stripped} missing from output"
                    )

    async def test_predict_realtime_time_series_with_explanations(
        self, timeseries_regression_project: dict[str, Any], test_data_dir: Any
    ) -> None:
        """Integration test combining time series forecasting with explanations."""
        async with integration_test_mcp_session() as session:
            deployment_id = timeseries_regression_project["deployment_id"]
            predict_file = test_data_dir / "timeseries_regression_predict.csv"

            result = await session.call_tool(
                "predict_realtime",
                {
                    "deployment_id": deployment_id,
                    "file_path": str(predict_file),
                    "forecast_point": "2024-02-01",
                    "max_explanations": 2,
                    "explanation_algorithm": "shap",
                    "passthrough_columns": "date,temperature",
                    "timeout": 300,
                },
            )
            if _is_shap_not_supported_error(result):
                return  # deployment doesn't support SHAP; test passes
            assert not result.isError
            result_content = result.content[0]
            assert isinstance(result_content, TextContent)
            result_data = result_content.text

            # Parse the response as JSON to get the type and data
            response_dict = json.loads(result_data)
            assert response_dict["type"] == "inline"
            assert len(response_dict["data"]) > 0

            # Parse and verify structure
            df = pd.read_csv(StringIO(response_dict["data"]))

            # Should have time series columns
            assert "FORECAST_POINT" in df.columns
            assert "FORECAST_DISTANCE" in df.columns
            assert "sales (actual)_PREDICTION" in df.columns

            # Should have some explanation columns (exact names depend on model)
            explanation_cols = [col for col in df.columns if "EXPLANATION" in col or "SHAP" in col]
            assert len(explanation_cols) > 0, "No explanation columns found in output"

            # Should have requested passthrough columns
            assert "date" in df.columns
            assert "temperature" in df.columns

            # Should have 7 forecast days
            assert len(df) == 7

    async def test_classification_basic_prediction(
        self, classification_project: dict[str, Any], test_data_dir: Any
    ) -> None:
        """Integration test for basic text classification prediction."""
        async with integration_test_mcp_session() as session:
            deployment_id = classification_project["deployment_id"]
            predict_file = test_data_dir / "text_classification_predict.csv"

            result = await session.call_tool(
                "predict_realtime",
                {
                    "deployment_id": deployment_id,
                    "file_path": str(predict_file),
                    "timeout": 300,
                },
            )
            assert not result.isError
            result_content = result.content[0]
            assert isinstance(result_content, TextContent)
            result_data = result_content.text

            # Parse the response as JSON to get the type and data
            response_dict = json.loads(result_data)
            assert response_dict["type"] == "inline"
            assert len(response_dict["data"]) > 0

            # Parse CSV and check structure
            df = pd.read_csv(StringIO(response_dict["data"]))
            assert len(df) > 0  # Has some rows

            # Should have prediction columns for sentiment classification
            prediction_cols = [col for col in df.columns if "_PREDICTION" in col]
            assert len(prediction_cols) > 0, "No prediction columns found"

            # For text classification, we expect positive/negative predictions
            print(f"Available columns: {list(df.columns)}")
            print(f"Prediction columns: {prediction_cols}")
            print(f"Text classification predictions successful with {len(df)} rows")

    async def test_predict_realtime_comprehensive_explanation_parameters(
        self, timeseries_regression_project: dict[str, Any], test_data_dir: Any
    ) -> None:
        """Integration test for predict_realtime with comprehensive explanation parameters."""
        async with integration_test_mcp_session(use_stub=False) as session:
            deployment_id = timeseries_regression_project["deployment_id"]
            predict_file = test_data_dir / "timeseries_regression_predict.csv"

            result = await session.call_tool(
                "predict_realtime",
                {
                    "deployment_id": deployment_id,
                    "file_path": str(predict_file),
                    "max_explanations": 5,
                    "explanation_algorithm": "shap",
                    "threshold_high": 0.8,
                    "threshold_low": 0.2,
                    "passthrough_columns": "date,temperature",
                    "timeout": 300,
                },
            )
            if _is_shap_not_supported_error(result):
                return  # deployment doesn't support SHAP; test passes
            assert not result.isError
            result_content = result.content[0]
            assert isinstance(result_content, TextContent)
            result_data = result_content.text

            # Parse the response as JSON to get the type and data
            response_dict = json.loads(result_data)
            assert response_dict["type"] == "inline"
            assert len(response_dict["data"]) > 0

            # Parse CSV and check structure
            df = pd.read_csv(StringIO(response_dict["data"]))
            assert len(df) > 0

            # Should have prediction columns
            assert "sales (actual)_PREDICTION" in df.columns

            # Should have SHAP explanation columns
            explanation_cols = [col for col in df.columns if "EXPLANATION" in col or "SHAP" in col]
            assert len(explanation_cols) > 0, "No explanation columns found"

            # Should have threshold-related information if applicable
            # (Note: threshold columns may not always be present depending on model type)

            # Should have requested passthrough columns
            assert "date" in df.columns, "Passthrough column 'date' missing"
            assert "temperature" in df.columns, "Passthrough column 'temperature' missing"

    async def test_predict_realtime_with_custom_endpoint_and_passthrough_all(
        self, timeseries_regression_project: dict[str, Any], test_data_dir: Any
    ) -> None:
        """Integration test for predict_realtime with custom endpoint and
        passthrough_columns='all'.
        """
        async with integration_test_mcp_session() as session:
            deployment_id = timeseries_regression_project["deployment_id"]
            predict_file = test_data_dir / "timeseries_regression_predict.csv"

            # Read the input file to know what columns to expect (tool strips column names)
            input_df = pd.read_csv(predict_file)
            input_columns = {col.strip() for col in input_df.columns}

            result = await session.call_tool(
                "predict_realtime",
                {
                    "deployment_id": deployment_id,
                    "file_path": str(predict_file),
                    "passthrough_columns": "all",
                    "timeout": 300,
                },
            )
            assert not result.isError
            result_content = result.content[0]
            assert isinstance(result_content, TextContent)
            result_data = result_content.text

            # Parse the response as JSON to get the type and data
            response_dict = json.loads(result_data)
            assert response_dict["type"] == "inline"
            assert len(response_dict["data"]) > 0

            # Parse CSV and check structure
            df = pd.read_csv(StringIO(response_dict["data"]))
            assert len(df) > 0

            # Should have prediction columns
            prediction_cols = [col for col in df.columns if "_PREDICTION" in col]
            assert len(prediction_cols) > 0

            # Should have all input columns as passthrough (except target which may be renamed)
            for col in input_columns:
                if col != "sales":  # Target column may not be passed through or may be renamed
                    assert col in df.columns, f"Expected passthrough column {col} not found"

            # Test successful completion with passthrough columns
            print(
                f"Successfully tested passthrough_columns='all' with {len(df.columns)} "
                f"total columns"
            )
            print(f"Input columns: {list(input_columns)}")
            print(f"Output columns: {list(df.columns)}")

    async def test_predict_realtime_with_max_ngram_explanations(
        self, timeseries_regression_project: dict[str, Any], test_data_dir: Any
    ) -> None:
        """Integration test for predict_realtime with max_ngram_explanations parameter."""
        async with integration_test_mcp_session() as session:
            deployment_id = timeseries_regression_project["deployment_id"]
            predict_file = test_data_dir / "timeseries_regression_predict.csv"

            result = await session.call_tool(
                "predict_realtime",
                {
                    "deployment_id": deployment_id,
                    "file_path": str(predict_file),
                    "max_explanations": 3,
                    "max_ngram_explanations": 5,
                    "explanation_algorithm": "shap",
                    "timeout": 300,
                },
            )
            if _is_shap_not_supported_error(result):
                return  # deployment doesn't support SHAP; test passes
            assert not result.isError
            result_content = result.content[0]
            assert isinstance(result_content, TextContent)
            result_data = result_content.text

            # Parse the response as JSON to get the type and data
            response_dict = json.loads(result_data)
            assert response_dict["type"] == "inline"
            assert len(response_dict["data"]) > 0

            # Parse CSV and check structure
            df = pd.read_csv(StringIO(response_dict["data"]))
            assert len(df) > 0

            # Should have prediction columns
            assert "sales (actual)_PREDICTION" in df.columns

            # Should have explanation columns
            explanation_cols = [col for col in df.columns if "EXPLANATION" in col or "SHAP" in col]
            assert len(explanation_cols) > 0, "No explanation columns found"

            print("Successfully tested max_ngram_explanations parameter")
            print(f"Available columns: {list(df.columns)}")
            print(f"Explanation columns: {explanation_cols}")
