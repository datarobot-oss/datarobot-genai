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

"""Tools for retrieving deployment metadata and data requirements."""

import io
import json
import logging
from datetime import datetime
from datetime import timedelta
from typing import Annotated
from typing import Any

import pandas as pd
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"predictive", "deployment", "read", "info", "metadata"})
async def get_deployment_info(
    *,
    deployment_id: Annotated[str, "The ID of the DataRobot deployment"] | None = None,
) -> ToolError | ToolResult:
    """
    Retrieve information about the deployment, including the list of
    features needed to make predictions on this deployment.
    """
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")

    client = get_sdk_client()
    deployment = client.Deployment.get(deployment_id)

    # get features from the deployment
    features_raw = deployment.get_features()
    deployment.get_capabilities()

    # Parse features if it's a JSON string
    if isinstance(features_raw, str):
        try:
            features = json.loads(features_raw)
        except json.JSONDecodeError:
            features = []
    else:
        features = features_raw

    # get model type if its not a custom model
    project = None
    if deployment.model.get("project_id") is None:
        model_type = "custom"
        target = ""
        target_type = ""
    else:
        project = client.Project.get(deployment.model["project_id"])
        model = client.Model.get(project=project, model_id=deployment.model["id"])
        model_type = model.model_type
        target = project.target
        target_type = project.target_type

    # Add model metadata
    result = {
        "deployment_id": deployment_id,
        "model_type": model_type,
        "target": target,
        "target_type": target_type,
        "features": sorted(features, key=lambda x: (x.get("importance") or 0), reverse=True),
        "total_features": len(features),
    }

    # Add time series specific information if applicable
    if project and hasattr(project, "datetime_partitioning"):
        partition = project.datetime_partitioning
        result["time_series_config"] = {
            "datetime_column": partition.datetime_partition_column,
            "forecast_window_start": partition.forecast_window_start,
            "forecast_window_end": partition.forecast_window_end,
            "series_id_columns": partition.multiseries_id_columns or [],
        }

    return ToolResult(
        content=json.dumps(result, indent=2),
        structured_content=result,
    )


@dr_mcp_tool(tags={"predictive", "deployment", "read", "template", "data"})
async def generate_prediction_data_template(
    *,
    deployment_id: Annotated[str, "The ID of the DataRobot deployment"] | None = None,
    n_rows: Annotated[int, "Number of template rows to generate"] = 1,
) -> ToolError | ToolResult:
    """Generate a template CSV with the correct structure for making predictions."""
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")
    if n_rows is None or n_rows <= 0:
        n_rows = 1

    # Get feature information
    features_result = await get_deployment_features(deployment_id=deployment_id)
    # Add error handling for empty or error responses
    # Extract text content from ToolResult
    if features_result.content and isinstance(features_result.content[0], TextContent):
        features_json = features_result.content[0].text
    else:
        features_json = str(features_result.content)
    if not features_json or features_json.strip().startswith("Error"):
        raise ToolError(f"Error with feature information: {features_json}")
    features_info = json.loads(features_json)

    # Create template data
    template_data = {}

    for feature in features_info["features"]:
        feature_name = feature["name"]
        feature_type = feature["feature_type"].lower()  # Normalize to lowercase

        # Use frequent values if available
        frequent_values = feature.get("frequent_values")
        if frequent_values and isinstance(frequent_values, list) and frequent_values:
            template_data[feature_name] = [frequent_values[0]] * n_rows
            continue

        # Use only documented feature properties
        min_val = feature.get("min", 0)
        max_val = feature.get("max", 0)

        # Handle None for min/max
        if min_val is None:
            min_val = 0 if feature_type == "numeric" else "2020-01-01"
        if max_val is None:
            max_val = 0 if feature_type == "numeric" else "2020-01-10"

        # Generate sample values based on type
        if feature_type == "numeric":
            template_data[feature_name] = [None] * n_rows
        elif feature_type == "date":
            template_data[feature_name] = [None] * n_rows
        elif feature_type == "summarized categorical":
            template_data[feature_name] = [""] * n_rows
        elif feature_type == "categorical":
            template_data[feature_name] = [""] * n_rows
        elif feature_type == "text":
            template_data[feature_name] = [""] * n_rows
        else:
            template_data[feature_name] = [""] * n_rows

    # Handle time series specific columns
    if "time_series_config" in features_info:
        ts_config = features_info["time_series_config"]

        # Ensure datetime column exists
        if ts_config["datetime_column"] not in template_data:
            base_date = datetime.now()
            dates = [base_date + timedelta(days=i) for i in range(n_rows)]
            template_data[ts_config["datetime_column"]] = dates

        # Add series ID columns if multiseries
        for series_col in ts_config["series_id_columns"]:
            if series_col not in template_data:
                template_data[series_col] = ["series_A"] * n_rows

    # Create DataFrame
    df = pd.DataFrame(template_data)

    # Add metadata comments
    result = f"# Prediction Data Template for Deployment: {deployment_id}\n"
    result += f"# Model Type: {features_info['model_type']}\n"
    result += f"# Target: {features_info['target']} (Type: {features_info['target_type']})\n"

    if "time_series_config" in features_info:
        ts = features_info["time_series_config"]
        result += f"# Time Series: datetime_column={ts['datetime_column']}, "
        result += f"forecast_window=[{ts['forecast_window_start']}, {ts['forecast_window_end']}]\n"
        if ts["series_id_columns"]:
            result += f"# Multiseries ID Columns: {', '.join(ts['series_id_columns'])}\n"

    result += f"# Total Features: {features_info['total_features']}\n"
    result += df.to_csv(index=False)

    # Build structured content with template data and metadata
    structured_content = {
        "deployment_id": deployment_id,
        "model_type": features_info["model_type"],
        "target": features_info["target"],
        "target_type": features_info["target_type"],
        "total_features": features_info["total_features"],
        "template_data": df.to_dict("records"),  # Convert DataFrame to list of dicts
    }

    if "time_series_config" in features_info:
        structured_content["time_series_config"] = features_info["time_series_config"]

    return ToolResult(
        content=str(result),
        structured_content=structured_content,
    )


@dr_mcp_tool(tags={"predictive", "deployment", "read", "validation", "data"})
async def validate_prediction_data(
    *,
    deployment_id: Annotated[str, "The ID of the DataRobot deployment"] | None = None,
    file_path: Annotated[
        str, "Path to the CSV file to validate (optional if csv_string is provided)"
    ]
    | None = None,
    csv_string: Annotated[str, "CSV data as a string (optional, used if file_path is not provided)"]
    | None = None,
) -> ToolError | ToolResult:
    """Validate if a CSV file is suitable for making predictions with a deployment."""
    # Load the data
    if csv_string is not None:
        df = pd.read_csv(io.StringIO(csv_string))
    elif file_path is not None:
        df = pd.read_csv(file_path)
    else:
        raise ToolError("Must provide either file_path or csv_string.")

    if not deployment_id:
        raise ToolError("Deployment ID must be provided")

    # Get deployment features
    features_result = await get_deployment_features(deployment_id=deployment_id)
    # Extract text content from ToolResult
    if features_result.content and isinstance(features_result.content[0], TextContent):
        features_json = features_result.content[0].text
    else:
        features_json = str(features_result.content)
    features_info = json.loads(features_json)

    validation_report: dict[str, Any] = {
        "status": "valid",
        "errors": [],
        "warnings": [],
        "info": [],
    }

    # Check each required feature
    required_features = [f for f in features_info["features"]]
    data_columns = set(df.columns)

    # Threshold for considering a feature as important
    importance_threshold = 0.1

    for feature in required_features:
        feature_name = feature["name"] if "name" in feature else feature["feature_name"]

        # Check if feature exists
        if feature_name not in data_columns:
            if feature.get("importance", 0) > importance_threshold:
                validation_report["warnings"].append(
                    f"Missing important feature: {feature_name} (importance: "
                    f"{feature.get('importance', 0):.2f})"
                )
            else:
                validation_report["warnings"].append(
                    f"Missing feature column: {feature_name} (column will be treated as missing "
                    f"values)"
                )
            continue

        # Check for missing values (allowed)
        if df[feature_name].isnull().all() or (df[feature_name] == "").all():
            validation_report["info"].append(
                f"Feature {feature_name} is entirely missing or empty (this is allowed)"
            )
            continue

        # Check data type compatibility (only if not all missing)
        col_dtype = str(df[feature_name].dtype)
        if feature["feature_type"] == "numeric" and not pd.api.types.is_numeric_dtype(
            df[feature_name].dropna()
        ):
            validation_report["warnings"].append(
                f"Feature {feature_name} should be numeric but is {col_dtype}"
            )

    # Check for extra columns
    expected_features = {
        f["name"] if "name" in f else f["feature_name"] for f in features_info["features"]
    }
    extra_columns = data_columns - expected_features
    if extra_columns:
        validation_report["info"].append(
            f"Extra columns found (will be ignored): {', '.join(extra_columns)}"
        )

    # Time series specific validation
    if "time_series_config" in features_info:
        ts_config = features_info["time_series_config"]

        # Check datetime column
        if ts_config["datetime_column"] not in data_columns:
            validation_report["errors"].append(
                f"Missing required datetime column: {ts_config['datetime_column']}"
            )
            validation_report["status"] = "invalid"
        elif (
            not df[ts_config["datetime_column"]].isnull().all()
            and not (df[ts_config["datetime_column"]] == "").all()
        ):
            try:
                pd.to_datetime(df[ts_config["datetime_column"]])
            except ValueError:
                validation_report["errors"].append(
                    f"Datetime column {ts_config['datetime_column']} cannot be parsed as dates"
                )
                validation_report["status"] = "invalid"

        # Check series ID columns for multiseries
        for series_col in ts_config["series_id_columns"]:
            if series_col not in data_columns:
                validation_report["errors"].append(
                    f"Missing required series ID column: {series_col}"
                )
                validation_report["status"] = "invalid"

    # Add summary
    validation_report["summary"] = {
        "file_path": file_path,
        "rows": len(df),
        "columns": len(df.columns),
        "deployment_id": deployment_id,
        "model_type": features_info["model_type"],
    }

    return ToolResult(
        content=json.dumps(validation_report, indent=2),
        structured_content=validation_report,
    )


@dr_mcp_tool(tags={"predictive", "deployment", "read", "features", "info"})
async def get_deployment_features(
    *,
    deployment_id: Annotated[str, "The ID of the DataRobot deployment"] | None = None,
) -> ToolError | ToolResult:
    """Retrieve only the features list for a deployment, as JSON string."""
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")

    info_result = await get_deployment_info(deployment_id=deployment_id)
    # Extract text content from ToolResult
    if info_result.content and isinstance(info_result.content[0], TextContent):
        info_json = info_result.content[0].text
    else:
        info_json = str(info_result.content)
    if not info_json.strip().startswith("{"):
        raise ToolError(f"Error with deployment info: {info_json}")
    info = json.loads(info_json)
    # Only keep features, time_series_config, and total_features
    result = {
        "features": info.get("features", []),
        "total_features": info.get("total_features", 0),
    }
    if "time_series_config" in info:
        result["time_series_config"] = info["time_series_config"]
    if "model_type" in info:
        result["model_type"] = info["model_type"]
    if "target" in info:
        result["target"] = info["target"]
    if "target_type" in info:
        result["target_type"] = info["target_type"]

    return ToolResult(
        content=json.dumps(result, indent=2),
        structured_content=result,
    )
