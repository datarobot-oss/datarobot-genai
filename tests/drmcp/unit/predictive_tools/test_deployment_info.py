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
import tempfile
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from datarobot_genai.drmcp.core.shared import MCPError
from datarobot_genai.drmcp.tools.predictive.deployment_info import generate_prediction_data_template
from datarobot_genai.drmcp.tools.predictive.deployment_info import get_deployment_features
from datarobot_genai.drmcp.tools.predictive.deployment_info import get_deployment_info
from datarobot_genai.drmcp.tools.predictive.deployment_info import validate_prediction_data

IMPORTANCE_THRESHOLD_TEST = 0.8


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_sdk_client")
@pytest.mark.asyncio
async def test_get_deployment_info_success(mock_get_sdk_client: Any) -> None:
    """Test successful retrieval of deployment features."""
    # Setup mocks
    mock_client = MagicMock()
    mock_get_sdk_client.return_value = mock_client

    mock_deployment = MagicMock()
    mock_deployment.model = {"project_id": "proj123", "id": "model123"}

    mock_model = MagicMock()
    mock_model.model_type = "Regression"

    # Mock feature impact

    mock_project = MagicMock()
    mock_project.target = "sales"
    mock_project.target_type = "Regression"

    # Time series configuration
    class Partition:
        datetime_partition_column = "date"
        forecast_window_start = 1
        forecast_window_end = 7
        multiseries_id_columns = ["store_id"]

    mock_project.datetime_partitioning = Partition()

    # Mock features with proper attributes
    mock_features = []
    for name, ftype, importance in [
        ("temperature", "numeric", 0.8),
        ("humidity", "numeric", 0.6),
        ("sales", "numeric", 0.0),
    ]:
        feature = {
            "feature_name": name,
            "feature_type": ftype,
            "importance": importance,
            "date_format": None,
        }
        mock_features.append(feature)
    mock_deployment.get_features.return_value = mock_features

    mock_client.Deployment.get.return_value = mock_deployment
    mock_client.Model.get.return_value = mock_model
    mock_client.Project.get.return_value = mock_project

    # Test
    result = await get_deployment_info("dep123")

    # Handle potential error response
    if result.startswith("Error"):
        pytest.fail(f"get_deployment_features returned error: {result}")

    result_json = json.loads(result)

    # Assertions
    assert result_json["deployment_id"] == "dep123"
    assert result_json["model_type"] == "Regression"
    assert result_json["target"] == "sales"
    assert len(result_json["features"]) == 3

    # Check feature sorting by importance
    assert result_json["features"][0]["feature_name"] == "temperature"
    assert result_json["features"][0]["importance"] == IMPORTANCE_THRESHOLD_TEST

    # Check time series config
    assert "time_series_config" in result_json
    assert result_json["time_series_config"]["datetime_column"] == "date"
    assert result_json["time_series_config"]["series_id_columns"] == ["store_id"]


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_sdk_client")
@pytest.mark.asyncio
async def test_generate_prediction_data_template(
    mock_get_sdk_client: Any, mock_get_features: Any
) -> None:
    """Test generating prediction data template."""
    # Setup mocks
    mock_client = MagicMock()
    mock_get_sdk_client.return_value = mock_client

    mock_deployment = MagicMock()
    mock_client.Deployment.get.return_value = mock_deployment

    # Mock feature info
    features_info = {
        "deployment_id": "dep123",
        "model_type": "Regression",
        "target": "sales",
        "target_type": "Regression",
        "features": [
            {
                "name": "temperature",
                "feature_type": "numeric",
                "importance": 0.8,
                "is_target": False,
            },
            {
                "name": "category",
                "feature_type": "categorical",
                "importance": 0.6,
                "is_target": False,
            },
            {
                "name": "description",
                "feature_type": "text",
                "importance": 0.4,
                "is_target": False,
            },
            {
                "name": "sales",
                "feature_type": "numeric",
                "importance": 0,
                "is_target": True,
            },
        ],
        "time_series_config": {
            "datetime_column": "date",
            "forecast_window_start": 1,
            "forecast_window_end": 7,
            "series_id_columns": ["store_id"],
        },
        "total_features": 4,
    }
    mock_get_features.return_value = json.dumps(features_info)

    # Test
    result = await generate_prediction_data_template("dep123", n_rows=3)

    # Assertions
    assert "# Prediction Data Template" in result
    assert "Model Type: Regression" in result
    assert "temperature" in result
    assert "category" in result
    assert "description" in result
    # Check that sales column is present in the CSV data (target is included)
    lines = result.strip().split("\n")
    # Find where CSV data starts (after comment lines)
    csv_start = 0
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            csv_start = i
            break

    # Check header line contains sales
    header_line = lines[csv_start]
    assert "sales" in header_line.split(",")  # Target included in CSV columns
    assert "date" in result  # Time series datetime column
    assert "store_id" in result  # Multiseries ID column


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_sdk_client")
@pytest.mark.asyncio
async def test_validate_prediction_data_valid(
    mock_get_sdk_client: Any, mock_get_features: Any, tmp_path: Any
) -> None:
    """Test validating valid prediction data."""
    # Setup mocks
    mock_client = MagicMock()
    mock_get_sdk_client.return_value = mock_client

    # Create test CSV
    test_file = tmp_path / "test_data.csv"
    df = pd.DataFrame(
        {
            "temperature": [70, 75, 80],
            "humidity": [60, 65, 70],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    df.to_csv(test_file, index=False)

    # Mock feature info
    features_info = {
        "features": [
            {
                "feature_name": "temperature",
                "feature_type": "numeric",
                "importance": 0.8,
                "is_target": False,
            },
            {
                "feature_name": "humidity",
                "feature_type": "numeric",
                "importance": 0.6,
                "is_target": False,
            },
            {
                "feature_name": "date",
                "feature_type": "date",
                "importance": 0.4,
                "is_target": False,
            },
        ],
        "model_type": "Regression",
        "time_series_config": {
            "datetime_column": "date",
            "forecast_window_start": 1,
            "forecast_window_end": 7,
            "series_id_columns": [],
        },
    }
    mock_get_features.return_value = json.dumps(features_info)

    # Test
    result = await validate_prediction_data("dep123", str(test_file))
    result_json = json.loads(result)

    # Assertions
    assert result_json["status"] == "valid"
    assert len(result_json["errors"]) == 0
    assert len(result_json["warnings"]) == 0
    assert result_json["summary"]["rows"] == 3
    assert result_json["summary"]["columns"] == 3


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_sdk_client")
@pytest.mark.asyncio
async def test_validate_prediction_data_missing_important_feature(
    mock_get_sdk_client: Any, mock_get_features: Any, tmp_path: Any
) -> None:
    features_info = {
        "features": [
            {"name": "imp", "feature_type": "numeric", "importance": 0.9},
            {"name": "notimp", "feature_type": "text", "importance": 0.0},
        ],
        "model_type": "Test",
        "target": "",
        "target_type": "",
        "total_features": 2,
    }
    mock_get_features.return_value = json.dumps(features_info)
    mock_get_sdk_client.return_value = MagicMock()
    # Only notimp present
    df = pd.DataFrame({"notimp": ["", ""]})
    test_file = tmp_path / "test3.csv"
    df.to_csv(test_file, index=False)
    result = await validate_prediction_data("id", str(test_file))
    assert "Missing important feature: imp" in result
    assert "status" in result and "invalid" not in result
    # If present but all missing, info not warning
    df2 = pd.DataFrame({"imp": [None, None], "notimp": ["", ""]})
    test_file2 = tmp_path / "test4.csv"
    df2.to_csv(test_file2, index=False)
    result2 = await validate_prediction_data("id", str(test_file2))
    assert "is entirely missing or empty (this is allowed)" in result2
    # If present and not all missing, type check applies
    df3 = pd.DataFrame({"imp": ["bad", "bad"], "notimp": ["", ""]})
    test_file3 = tmp_path / "test5.csv"
    df3.to_csv(test_file3, index=False)
    result3 = await validate_prediction_data("id", str(test_file3))
    assert "should be numeric but is object" in result3


# Additional tests for coverage
@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_generate_prediction_data_template_error(mock_get_features: Any) -> None:
    mock_get_features.return_value = "Error: something went wrong"
    result = await generate_prediction_data_template("bad_id")
    assert result.startswith("Error: ")


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_generate_prediction_data_template_empty_features(
    mock_get_features: Any,
) -> None:
    mock_get_features.return_value = json.dumps(
        {
            "features": [],
            "model_type": "Test",
            "target": "",
            "target_type": "",
            "total_features": 0,
        }
    )
    result = await generate_prediction_data_template("empty_id")
    assert "# Total Features: 0" in result


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_generate_prediction_data_template_unknown_type(
    mock_get_features: Any,
) -> None:
    features_info = {
        "features": [{"name": "foo", "feature_type": "unknown"}],
        "model_type": "Test",
        "target": "",
        "target_type": "",
        "total_features": 1,
    }
    mock_get_features.return_value = json.dumps(features_info)
    result = await generate_prediction_data_template("id", n_rows=2)
    assert '""' in result


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_generate_prediction_data_template_none_min_max(
    mock_get_features: Any,
) -> None:
    features_info = {
        "features": [
            {"name": "num", "feature_type": "numeric", "min": None, "max": None},
            {"name": "dt", "feature_type": "date", "min": None, "max": None},
        ],
        "model_type": "Test",
        "target": "",
        "target_type": "",
        "total_features": 2,
    }
    mock_get_features.return_value = json.dumps(features_info)
    result = await generate_prediction_data_template("id")
    assert "num" in result and "dt" in result


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_generate_prediction_data_template_key_summary(
    mock_get_features: Any,
) -> None:
    features_info = {
        "features": [
            {
                "name": "cat",
                "feature_type": "summarized categorical",
                "key_summary": [{"key": "A"}, {"key": "B"}],
            }
        ],
        "model_type": "Test",
        "target": "",
        "target_type": "",
        "total_features": 1,
    }
    mock_get_features.return_value = json.dumps(features_info)
    result = await generate_prediction_data_template("id")
    assert "cat" in result


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_generate_prediction_data_template_multiseries(
    mock_get_features: Any,
) -> None:
    features_info = {
        "features": [{"name": "num", "feature_type": "numeric"}],
        "model_type": "Test",
        "target": "",
        "target_type": "",
        "total_features": 1,
        "time_series_config": {
            "datetime_column": "dt_col",
            "forecast_window_start": 1,
            "forecast_window_end": 2,
            "series_id_columns": ["series_id"],
        },
    }
    mock_get_features.return_value = json.dumps(features_info)
    result = await generate_prediction_data_template("id")
    assert "dt_col" in result and "series_id" in result


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_validate_prediction_data_error(mock_get_features: Any) -> None:
    mock_get_features.return_value = "Error: bad deployment"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b\n1,2\n")
        f.flush()
        with pytest.raises(MCPError) as exc_info:
            await validate_prediction_data("bad_id", f.name)
        assert "Error in validate_prediction_data: JSONDecodeError: Expecting value" in str(
            exc_info.value
        )


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_validate_prediction_data_missing_feature(
    mock_get_features: Any, tmp_path: Any
) -> None:
    features_info = {
        "features": [
            {
                "feature_name": "foo",
                "feature_type": "numeric",
                "importance": 1.0,
                "is_target": False,
            }
        ],
        "model_type": "Test",
    }
    mock_get_features.return_value = json.dumps(features_info)
    test_file = tmp_path / "test.csv"
    pd.DataFrame({"bar": [1, 2]}).to_csv(test_file, index=False)
    result = await validate_prediction_data("id", str(test_file))
    assert "Missing important feature" in result


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_validate_prediction_data_extra_columns(
    mock_get_features: Any, tmp_path: Any
) -> None:
    features_info = {
        "features": [
            {
                "feature_name": "foo",
                "feature_type": "numeric",
                "importance": 1.0,
                "is_target": False,
            }
        ],
        "model_type": "Test",
    }
    mock_get_features.return_value = json.dumps(features_info)
    test_file = tmp_path / "test.csv"
    pd.DataFrame({"foo": [1, 2], "extra": [3, 4]}).to_csv(test_file, index=False)
    result = await validate_prediction_data("id", str(test_file))
    assert "Extra columns found" in result


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_validate_prediction_data_type_mismatch(
    mock_get_features: Any, tmp_path: Any
) -> None:
    features_info = {
        "features": [
            {
                "feature_name": "foo",
                "feature_type": "numeric",
                "importance": 1.0,
                "is_target": False,
            }
        ],
        "model_type": "Test",
    }
    mock_get_features.return_value = json.dumps(features_info)
    test_file = tmp_path / "test.csv"
    pd.DataFrame({"foo": ["a", "b"]}).to_csv(test_file, index=False)
    result = await validate_prediction_data("id", str(test_file))
    assert "should be numeric" in result


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_validate_prediction_data_time_series_missing(
    mock_get_features: Any, tmp_path: Any
) -> None:
    features_info = {
        "features": [
            {
                "feature_name": "foo",
                "feature_type": "numeric",
                "importance": 1.0,
                "is_target": False,
            }
        ],
        "model_type": "Test",
        "time_series_config": {
            "datetime_column": "dt_col",
            "forecast_window_start": 1,
            "forecast_window_end": 2,
            "series_id_columns": ["series_id"],
        },
    }
    mock_get_features.return_value = json.dumps(features_info)
    test_file = tmp_path / "test.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(test_file, index=False)
    result = await validate_prediction_data("id", str(test_file))
    assert "Missing required datetime column" in result


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_validate_prediction_data_time_series_parse_error(
    mock_get_features: Any, tmp_path: Any
) -> None:
    features_info = {
        "features": [
            {
                "feature_name": "foo",
                "feature_type": "numeric",
                "importance": 1.0,
                "is_target": False,
            },
            {
                "feature_name": "dt_col",
                "feature_type": "date",
                "importance": 0.5,
                "is_target": False,
            },
        ],
        "model_type": "Test",
        "time_series_config": {
            "datetime_column": "dt_col",
            "forecast_window_start": 1,
            "forecast_window_end": 2,
            "series_id_columns": [],
        },
    }
    mock_get_features.return_value = json.dumps(features_info)
    test_file = tmp_path / "test.csv"
    pd.DataFrame({"foo": [1, 2], "dt_col": ["bad", "bad"]}).to_csv(test_file, index=False)
    result = await validate_prediction_data("id", str(test_file))
    assert "cannot be parsed as dates" in result


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_info")
@pytest.mark.asyncio
async def test_get_deployment_features_missing_fields(mock_get_info: Any) -> None:
    mock_info = json.dumps({})
    mock_get_info.return_value = mock_info

    result = await get_deployment_features("id")
    assert "features" in result and "total_features" in result


@pytest.mark.asyncio
async def test_get_deployment_info_custom_model() -> None:
    class DummyDeployment:
        model: dict[str, Any] = {}

        def get_features(self) -> str:
            return "[]"

        def get_capabilities(self) -> None:
            pass

    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment_info.get_sdk_client"
    ) as mock_client:
        mock_client.return_value.Deployment.get.return_value = DummyDeployment()
        result = await get_deployment_info("custom_id")
        info = json.loads(result)
        assert info["model_type"] == "custom"
        assert info["target"] == ""
        assert info["target_type"] == ""
        assert len(info["features"]) == 0


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_generate_prediction_data_template_categorical_defaults(
    mock_get_features: Any,
) -> None:
    features_info = {
        "features": [
            {"name": "cat", "feature_type": "categorical"},
            {"name": "sumcat", "feature_type": "summarized categorical"},
            {"name": "weird", "feature_type": "strange"},
        ],
        "model_type": "Test",
        "target": "",
        "target_type": "",
        "total_features": 3,
    }
    mock_get_features.return_value = json.dumps(features_info)
    result = await generate_prediction_data_template("id", n_rows=2)
    # All should be empty string columns
    assert ",," in result


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_generate_prediction_data_template_exception(
    mock_get_features: Any,
) -> None:
    mock_get_features.side_effect = Exception("fail")
    with pytest.raises(MCPError) as exc_info:
        await generate_prediction_data_template("id")
    assert "Error in generate_prediction_data_template: Exception: fail" == str(exc_info.value)


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_validate_prediction_data_file_error(mock_get_features: Any) -> None:
    mock_get_features.return_value = json.dumps({"features": [], "model_type": "Test"})
    with pytest.raises(MCPError) as exc_info:
        await validate_prediction_data("id", "/not/a/real/file.csv")
    assert (
        "Error in validate_prediction_data: FileNotFoundError: [Errno 2] No such file or "
        "directory: '/not/a/real/file.csv'" == str(exc_info.value)
    )


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_info")
@pytest.mark.asyncio
async def test_get_deployment_features_error_string(mock_get_info: Any) -> None:
    mock_get_info.return_value = "Error: not found"
    result = await get_deployment_features("id")
    info = json.loads(result)
    assert info["features"] == []
    assert info["total_features"] == 0
    assert "error" in info


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_info")
@pytest.mark.asyncio
async def test_get_deployment_features_optional_fields(mock_get_info: Any) -> None:
    base = {"features": [], "total_features": 0}
    mock_get_info.return_value = json.dumps(base)
    result = await get_deployment_features("id")
    info = json.loads(result)
    assert "features" in info and "total_features" in info
    base.update(
        {
            "time_series_config": {
                "datetime_column": "dt",
                "forecast_window_start": 1,
                "forecast_window_end": 2,
                "series_id_columns": [],
            },
            "model_type": "XGB",
            "target": "y",
            "target_type": "regression",
        }
    )
    mock_get_info.return_value = json.dumps(base)
    result = await get_deployment_features("id")
    info = json.loads(result)
    assert info["model_type"] == "XGB"
    assert info["target"] == "y"
    assert info["target_type"] == "regression"
    assert "time_series_config" in info


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_generate_prediction_data_template_frequent_values(
    mock_get_features: Any,
) -> None:
    features_info = {
        "features": [
            {
                "name": "cat",
                "feature_type": "categorical",
                "frequent_values": ["A", "B"],
            },
            {"name": "num", "feature_type": "numeric"},
            {"name": "txt", "feature_type": "text"},
        ],
        "model_type": "Test",
        "target": "",
        "target_type": "",
        "total_features": 3,
    }
    mock_get_features.return_value = json.dumps(features_info)
    result = await generate_prediction_data_template("id", n_rows=2)
    # cat should use 'A', num and txt should be empty or null
    assert "A,," in result


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_sdk_client")
@pytest.mark.asyncio
async def test_validate_prediction_data_missing_values(
    mock_get_sdk_client: Any, mock_get_features: Any, tmp_path: Any
) -> None:
    features_info = {
        "features": [
            {"name": "cat", "feature_type": "categorical"},
            {"name": "num", "feature_type": "numeric"},
            {"name": "txt", "feature_type": "text"},
        ],
        "model_type": "Test",
        "target": "",
        "target_type": "",
        "total_features": 3,
    }
    mock_get_features.return_value = json.dumps(features_info)
    mock_get_sdk_client.return_value = MagicMock()
    # All missing values (empty string)
    df = pd.DataFrame({"cat": ["", ""], "num": [None, None], "txt": ["", ""]})
    test_file = tmp_path / "test.csv"
    df.to_csv(test_file, index=False)
    result = await validate_prediction_data("id", str(test_file))
    assert "is entirely missing or empty (this is allowed)" in result
    # One column present, one missing
    df2 = pd.DataFrame({"cat": ["A", "B"]})
    test_file2 = tmp_path / "test2.csv"
    df2.to_csv(test_file2, index=False)
    result2 = await validate_prediction_data("id", str(test_file2))
    assert "Missing feature column: num" in result2
    assert "Missing feature column: txt" in result2
    # No errors or invalid status for missing values
    assert "status" in result2 and "invalid" not in result2


@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_deployment_features")
@patch("datarobot_genai.drmcp.tools.predictive.deployment_info.get_sdk_client")
@pytest.mark.asyncio
async def test_validate_prediction_data_with_csv_string(
    mock_get_sdk_client: Any, mock_get_features: Any
) -> None:
    features_info = {
        "features": [
            {"name": "cat", "feature_type": "categorical"},
            {"name": "num", "feature_type": "numeric"},
        ],
        "model_type": "Test",
        "target": "",
        "target_type": "",
        "total_features": 2,
    }
    mock_get_features.return_value = json.dumps(features_info)
    mock_get_sdk_client.return_value = MagicMock()
    # CSV string with only 'cat' column
    csv_str = "cat\nA\nB\n"
    result = await validate_prediction_data("id", csv_string=csv_str)
    assert "Missing feature column: num" in result
    assert "status" in result and "invalid" not in result
    # CSV string with both columns
    csv_str2 = "cat,num\nA,1\nB,2\n"
    result2 = await validate_prediction_data("id", csv_string=csv_str2)
    assert "status" in result2 and "invalid" not in result2
