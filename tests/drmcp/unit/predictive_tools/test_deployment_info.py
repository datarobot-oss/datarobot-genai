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

import csv
import json
import tempfile
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

from datarobot_genai.drtools.predictive.deployment_info import generate_prediction_data_template
from datarobot_genai.drtools.predictive.deployment_info import get_deployment_features
from datarobot_genai.drtools.predictive.deployment_info import get_deployment_info
from datarobot_genai.drtools.predictive.deployment_info import validate_prediction_data

IMPORTANCE_THRESHOLD_TEST = 0.8


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts to a CSV file (replaces pd.DataFrame().to_csv())."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _extract_content(result_obj: ToolResult | ToolError) -> str:
    """Extract text content from ToolResult."""
    if isinstance(result_obj, ToolError):
        raise result_obj
    if result_obj.content and isinstance(result_obj.content[0], TextContent):
        return result_obj.content[0].text
    return str(result_obj.content)


@patch("datarobot_genai.drtools.predictive.deployment_info.DataRobotClient")
@patch(
    "datarobot_genai.drtools.predictive.deployment_info.get_datarobot_access_token",
    new_callable=AsyncMock,
    return_value="token",
)
@pytest.mark.asyncio
async def test_get_deployment_info_success(
    mock_get_datarobot_access_token: Any, mock_data_robot_client: Any
) -> None:
    """Test successful retrieval of deployment features."""
    # Setup mocks
    mock_client = MagicMock()
    mock_data_robot_client.return_value.get_client.return_value = mock_client

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
    result = await get_deployment_info(deployment_id="dep123")

    # Handle potential error response
    if isinstance(result, ToolError):
        pytest.fail(f"get_deployment_info returned error: {result}")
    # Extract text content from ToolResult
    if result.content and isinstance(result.content[0], TextContent):
        result_content = result.content[0].text
    else:
        result_content = str(result.content)
    if result_content.startswith("Error"):
        pytest.fail(f"get_deployment_features returned error: {result_content}")

    result_json = json.loads(result_content)

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


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
@patch("datarobot_genai.drtools.predictive.deployment_info.DataRobotClient")
@patch(
    "datarobot_genai.drtools.predictive.deployment_info.get_datarobot_access_token",
    new_callable=AsyncMock,
    return_value="token",
)
@pytest.mark.asyncio
async def test_generate_prediction_data_template(
    mock_get_datarobot_access_token: Any,
    mock_data_robot_client: Any,
    mock_get_features: Any,
) -> None:
    """Test generating prediction data template."""
    # Setup mocks
    mock_client = MagicMock()
    mock_data_robot_client.return_value.get_client.return_value = mock_client

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
    mock_get_features.return_value = ToolResult(structured_content=features_info)

    # Test
    result_obj = await generate_prediction_data_template(deployment_id="dep123", n_rows=3)
    result = _extract_content(result_obj)
    result_json = json.loads(result)

    # Assertions on structured content
    assert result_json["deployment_id"] == "dep123"
    assert result_json["model_type"] == "Regression"
    assert result_json["target"] == "sales"
    assert "temperature" in result_json["template_data"][0]
    assert "category" in result_json["template_data"][0]
    assert "description" in result_json["template_data"][0]
    assert "sales" in result_json["template_data"][0]
    assert "date" in result_json["template_data"][0]
    assert "store_id" in result_json["template_data"][0]
    assert result_json["total_features"] == 4
    assert "time_series_config" in result_json


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
@patch("datarobot_genai.drtools.predictive.deployment_info.DataRobotClient")
@patch(
    "datarobot_genai.drtools.predictive.deployment_info.get_datarobot_access_token",
    new_callable=AsyncMock,
    return_value="token",
)
@pytest.mark.asyncio
async def test_validate_prediction_data_valid(
    mock_get_datarobot_access_token: Any,
    mock_data_robot_client: Any,
    mock_get_features: Any,
    tmp_path: Any,
) -> None:
    """Test validating valid prediction data."""
    # Setup mocks
    mock_client = MagicMock()
    mock_data_robot_client.return_value.get_client.return_value = mock_client

    # Create test CSV with stdlib
    test_file = tmp_path / "test_data.csv"
    _write_csv(
        str(test_file),
        [
            {"temperature": 70, "humidity": 60, "date": "2024-01-01"},
            {"temperature": 75, "humidity": 65, "date": "2024-01-02"},
            {"temperature": 80, "humidity": 70, "date": "2024-01-03"},
        ],
    )

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
    mock_get_features.return_value = ToolResult(structured_content=features_info)

    # Test
    result_obj = await validate_prediction_data(deployment_id="dep123", file_path=str(test_file))
    result = _extract_content(result_obj)
    result_json = json.loads(result)

    # Assertions
    assert result_json["status"] == "valid"
    assert len(result_json["errors"]) == 0
    assert len(result_json["warnings"]) == 0
    assert result_json["summary"]["rows"] == 3
    assert result_json["summary"]["columns"] == 3


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
@patch("datarobot_genai.drtools.predictive.deployment_info.DataRobotClient")
@patch(
    "datarobot_genai.drtools.predictive.deployment_info.get_datarobot_access_token",
    new_callable=AsyncMock,
    return_value="token",
)
@pytest.mark.asyncio
async def test_validate_prediction_data_missing_important_feature(
    mock_get_datarobot_access_token: Any,
    mock_data_robot_client: Any,
    mock_get_features: Any,
    tmp_path: Any,
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    mock_data_robot_client.return_value.get_client.return_value = MagicMock()
    # Only notimp present
    _write_csv(str(tmp_path / "test3.csv"), [{"notimp": ""}, {"notimp": ""}])
    result_obj = await validate_prediction_data(
        deployment_id="id", file_path=str(tmp_path / "test3.csv")
    )
    result = _extract_content(result_obj)
    assert "Missing important feature: imp" in result
    assert "status" in result and "invalid" not in result
    # If present but all missing, info not warning
    _write_csv(
        str(tmp_path / "test4.csv"), [{"imp": None, "notimp": ""}, {"imp": None, "notimp": ""}]
    )
    result_obj2 = await validate_prediction_data(
        deployment_id="id", file_path=str(tmp_path / "test4.csv")
    )
    result2 = _extract_content(result_obj2)
    assert "is entirely missing or empty (this is allowed)" in result2
    # If present and not all missing, type check applies
    _write_csv(
        str(tmp_path / "test5.csv"), [{"imp": "bad", "notimp": ""}, {"imp": "bad", "notimp": ""}]
    )
    result_obj3 = await validate_prediction_data(
        deployment_id="id", file_path=str(tmp_path / "test5.csv")
    )
    result3 = _extract_content(result_obj3)
    assert "should be numeric but is string" in result3


# Additional tests for coverage
@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_generate_prediction_data_template_error(mock_get_features: Any) -> None:
    mock_get_features.return_value = ToolResult(
        structured_content={"error": "something went wrong"}
    )
    with pytest.raises((ToolError, KeyError)) as exc_info:
        await generate_prediction_data_template(deployment_id="bad_id")
    assert "features" in str(exc_info.value)


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_generate_prediction_data_template_empty_features(
    mock_get_features: Any,
) -> None:
    mock_get_features.return_value = ToolResult(
        structured_content={
            "features": [],
            "model_type": "Test",
            "target": "",
            "target_type": "",
            "total_features": 0,
        }
    )
    result_obj = await generate_prediction_data_template(deployment_id="empty_id")
    result = _extract_content(result_obj)
    result_json = json.loads(result)
    assert result_json["total_features"] == 0
    assert result_json["template_data"] == []


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    result_obj = await generate_prediction_data_template(deployment_id="id", n_rows=2)
    result = _extract_content(result_obj)
    assert '""' in result


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    result_obj = await generate_prediction_data_template(deployment_id="id")
    result = _extract_content(result_obj)
    assert "num" in result and "dt" in result


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    result_obj = await generate_prediction_data_template(deployment_id="id")
    result = _extract_content(result_obj)
    assert "cat" in result


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    result_obj = await generate_prediction_data_template(deployment_id="id")
    result = _extract_content(result_obj)
    assert "dt_col" in result and "series_id" in result


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_validate_prediction_data_error(mock_get_features: Any) -> None:
    mock_get_features.return_value = ToolResult(structured_content={"error": "bad deployment"})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b\n1,2\n")
        f.flush()
        with pytest.raises((ToolError, KeyError)) as exc_info:
            await validate_prediction_data(deployment_id="bad_id", file_path=f.name)
        assert "features" in str(exc_info.value)


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    test_file = tmp_path / "test.csv"
    _write_csv(str(test_file), [{"bar": 1}, {"bar": 2}])
    result_obj = await validate_prediction_data(deployment_id="id", file_path=str(test_file))
    result = _extract_content(result_obj)
    assert "Missing important feature" in result


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    test_file = tmp_path / "test.csv"
    _write_csv(str(test_file), [{"foo": 1, "extra": 3}, {"foo": 2, "extra": 4}])
    result_obj = await validate_prediction_data(deployment_id="id", file_path=str(test_file))
    result = _extract_content(result_obj)
    assert "Extra columns found" in result


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    test_file = tmp_path / "test.csv"
    _write_csv(str(test_file), [{"foo": "a"}, {"foo": "b"}])
    result_obj = await validate_prediction_data(deployment_id="id", file_path=str(test_file))
    result = _extract_content(result_obj)
    assert "should be numeric" in result


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    test_file = tmp_path / "test.csv"
    _write_csv(str(test_file), [{"foo": 1}, {"foo": 2}])
    result_obj = await validate_prediction_data(deployment_id="id", file_path=str(test_file))
    result = _extract_content(result_obj)
    assert "Missing required datetime column" in result


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    test_file = tmp_path / "test.csv"
    _write_csv(str(test_file), [{"foo": 1, "dt_col": "bad"}, {"foo": 2, "dt_col": "bad"}])
    result_obj = await validate_prediction_data(deployment_id="id", file_path=str(test_file))
    result = _extract_content(result_obj)
    assert "cannot be parsed as dates" in result


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_info")
@pytest.mark.asyncio
async def test_get_deployment_features_missing_fields(mock_get_info: Any) -> None:
    mock_info = ToolResult(structured_content={})
    mock_get_info.return_value = mock_info

    result_obj = await get_deployment_features(deployment_id="id")
    result = _extract_content(result_obj)
    assert "features" in result and "total_features" in result


@pytest.mark.asyncio
async def test_get_deployment_info_custom_model() -> None:
    class DummyDeployment:
        model: dict[str, Any] = {}

        def get_features(self) -> str:
            return "[]"

        def get_capabilities(self) -> None:
            pass

    with (
        patch(
            "datarobot_genai.drtools.predictive.deployment_info.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.deployment_info.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_client.Deployment.get.return_value = DummyDeployment()
        mock_drc.return_value.get_client.return_value = mock_client
        result_obj = await get_deployment_info(deployment_id="custom_id")
        result = _extract_content(result_obj)
        info = json.loads(result)
        assert info["model_type"] == "custom"
        assert info["target"] == ""
        assert info["target_type"] == ""
        assert len(info["features"]) == 0


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    result_obj = await generate_prediction_data_template(deployment_id="id", n_rows=2)
    result = _extract_content(result_obj)
    result_json = json.loads(result)
    # All categorical/text columns should be empty string
    row = result_json["template_data"][0]
    assert row["cat"] == "" and row["sumcat"] == "" and row["weird"] == ""


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_generate_prediction_data_template_exception(
    mock_get_features: Any,
) -> None:
    mock_get_features.side_effect = Exception("fail")
    with pytest.raises(ToolError) as exc_info:
        await generate_prediction_data_template(deployment_id="id")
    assert "Error in generate_prediction_data_template: Exception: fail" == str(exc_info.value)


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
@pytest.mark.asyncio
async def test_validate_prediction_data_file_error(mock_get_features: Any) -> None:
    mock_get_features.return_value = ToolResult(
        structured_content={"features": [], "model_type": "Test"}
    )
    with pytest.raises(ToolError) as exc_info:
        await validate_prediction_data(deployment_id="id", file_path="/not/a/real/file.csv")
    assert (
        "Error in validate_prediction_data: FileNotFoundError: [Errno 2] No such file or "
        "directory: '/not/a/real/file.csv'" == str(exc_info.value)
    )


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_info")
@pytest.mark.asyncio
async def test_get_deployment_features_error_string(mock_get_info: Any) -> None:
    """When get_deployment_info returns error dict,
    get_deployment_features returns empty features.
    """
    mock_get_info.return_value = ToolResult(structured_content={"error": "not found"})
    result_obj = await get_deployment_features(deployment_id="id")
    result = _extract_content(result_obj)
    info = json.loads(result)
    assert info["features"] == []
    assert info["total_features"] == 0


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_info")
@pytest.mark.asyncio
async def test_get_deployment_features_optional_fields(mock_get_info: Any) -> None:
    base = {"features": [], "total_features": 0}
    mock_get_info.return_value = ToolResult(structured_content=base)
    result_obj = await get_deployment_features(deployment_id="id")
    result = _extract_content(result_obj)
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
    mock_get_info.return_value = ToolResult(structured_content=base)
    result_obj = await get_deployment_features(deployment_id="id")
    result = _extract_content(result_obj)
    info = json.loads(result)
    assert info["model_type"] == "XGB"
    assert info["target"] == "y"
    assert info["target_type"] == "regression"
    assert "time_series_config" in info


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    result_obj = await generate_prediction_data_template(deployment_id="id", n_rows=2)
    result = _extract_content(result_obj)
    result_json = json.loads(result)
    # cat should use 'A' (first frequent value), num and txt empty/null
    assert result_json["template_data"][0]["cat"] == "A"


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
@patch("datarobot_genai.drtools.predictive.deployment_info.DataRobotClient")
@patch(
    "datarobot_genai.drtools.predictive.deployment_info.get_datarobot_access_token",
    new_callable=AsyncMock,
    return_value="token",
)
@pytest.mark.asyncio
async def test_validate_prediction_data_missing_values(
    mock_get_datarobot_access_token: Any,
    mock_data_robot_client: Any,
    mock_get_features: Any,
    tmp_path: Any,
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    mock_data_robot_client.return_value.get_client.return_value = MagicMock()
    # All missing values (empty string)
    _write_csv(
        str(tmp_path / "test.csv"),
        [{"cat": "", "num": "", "txt": ""}, {"cat": "", "num": "", "txt": ""}],
    )
    result_obj = await validate_prediction_data(
        deployment_id="id", file_path=str(tmp_path / "test.csv")
    )
    result = _extract_content(result_obj)
    assert "is entirely missing or empty (this is allowed)" in result
    # One column present, one missing
    _write_csv(str(tmp_path / "test2.csv"), [{"cat": "A"}, {"cat": "B"}])
    result_obj2 = await validate_prediction_data(
        deployment_id="id", file_path=str(tmp_path / "test2.csv")
    )
    result2 = _extract_content(result_obj2)
    assert "Missing feature column: num" in result2
    assert "Missing feature column: txt" in result2
    # No errors or invalid status for missing values
    assert "status" in result2 and "invalid" not in result2


@patch("datarobot_genai.drtools.predictive.deployment_info.get_deployment_features")
@patch("datarobot_genai.drtools.predictive.deployment_info.DataRobotClient")
@patch(
    "datarobot_genai.drtools.predictive.deployment_info.get_datarobot_access_token",
    new_callable=AsyncMock,
    return_value="token",
)
@pytest.mark.asyncio
async def test_validate_prediction_data_with_csv_string(
    mock_get_datarobot_access_token: Any,
    mock_data_robot_client: Any,
    mock_get_features: Any,
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
    mock_get_features.return_value = ToolResult(structured_content=features_info)
    mock_data_robot_client.return_value.get_client.return_value = MagicMock()
    # CSV string with only 'cat' column
    csv_str = "cat\nA\nB\n"
    result_obj = await validate_prediction_data(deployment_id="id", csv_string=csv_str)
    result = _extract_content(result_obj)
    assert "Missing feature column: num" in result
    assert "status" in result and "invalid" not in result
    # CSV string with both columns
    csv_str2 = "cat,num\nA,1\nB,2\n"
    result_obj2 = await validate_prediction_data(deployment_id="id", csv_string=csv_str2)
    result2 = _extract_content(result_obj2)
    assert "status" in result2 and "invalid" not in result2
