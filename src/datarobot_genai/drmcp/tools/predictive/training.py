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

"""Tools for analyzing datasets and suggesting ML use cases."""

import json
import logging
from dataclasses import asdict
from dataclasses import dataclass
from typing import Annotated
from typing import Any

import pandas as pd
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool

logger = logging.getLogger(__name__)


@dataclass
class UseCaseSuggestion:
    """Represents a suggested use case based on dataset analysis."""

    name: str
    description: str
    suggested_target: str
    problem_type: str
    confidence: float
    reasoning: str


@dataclass
class DatasetInsight:
    """Contains insights about a dataset for use case discovery."""

    total_columns: int
    total_rows: int
    numerical_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    text_columns: list[str]
    potential_targets: list[str]
    missing_data_summary: dict[str, float]


def _get_dataset_or_raise(client: Any, dataset_id: str) -> tuple[Any, pd.DataFrame]:
    """Fetch dataset and return it with its dataframe, with proper error handling.

    Args:
        client: DataRobot SDK client instance
        dataset_id: The ID of the dataset to fetch

    Returns
    -------
        Tuple of (dataset object, dataframe)

    Raises
    ------
        ToolError: If dataset is not found (404) or other error occurs
    """
    try:
        dataset = client.Dataset.get(dataset_id)
        return dataset, dataset.get_as_dataframe()
    except Exception as e:
        error_str = str(e)
        # Check if it's a 404 error (dataset not found)
        if "404" in error_str or "Not Found" in error_str:
            raise ToolError(
                f"Dataset '{dataset_id}' not found. Please verify the dataset ID exists "
                "and you have access to it."
            )
        # For other errors, provide context
        raise ToolError(f"Failed to retrieve dataset '{dataset_id}': {error_str}")


@dr_mcp_tool(tags={"predictive", "training", "read", "analysis", "dataset"})
async def analyze_dataset(
    *,
    dataset_id: Annotated[str, "The ID of the DataRobot dataset to analyze"] | None = None,
) -> ToolError | ToolResult:
    """Analyze a dataset to understand its structure and potential use cases."""
    if not dataset_id:
        raise ToolError("Dataset ID must be provided")

    client = get_sdk_client()
    dataset, df = _get_dataset_or_raise(client, dataset_id)

    # Analyze dataset structure
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Identify potential text columns (categorical with high cardinality)
    text_cols = []
    for col in categorical_cols:
        if df[col].str.len().mean() > 20:  # Text detection
            text_cols.append(col)
            categorical_cols.remove(col)  # Remove from categorical columns

    # Calculate missing data
    missing_data = {}
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 0:
            missing_data[col] = missing_pct

    # Identify potential target columns
    potential_targets = _identify_potential_targets(df, numerical_cols, categorical_cols)

    insights = DatasetInsight(
        total_columns=len(df.columns),
        total_rows=len(df),
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        text_columns=text_cols,
        potential_targets=potential_targets,
        missing_data_summary=missing_data,
    )
    insights_dict = asdict(insights)

    return ToolResult(
        content=json.dumps(insights_dict, indent=2),
        structured_content=insights_dict,
    )


@dr_mcp_tool(tags={"predictive", "training", "read", "analysis", "usecase"})
async def suggest_use_cases(
    *,
    dataset_id: Annotated[str, "The ID of the DataRobot dataset to analyze"] | None = None,
) -> ToolError | ToolResult:
    """Analyze a dataset and suggest potential machine learning use cases."""
    if not dataset_id:
        raise ToolError("Dataset ID must be provided")

    client = get_sdk_client()
    dataset, df = _get_dataset_or_raise(client, dataset_id)

    # Get dataset insights first
    insights_result = await analyze_dataset(dataset_id=dataset_id)
    insights = insights_result.structured_content

    suggestions = []
    for target_col in insights["potential_targets"]:
        target_suggestions = _analyze_target_for_use_cases(df, target_col)
        suggestions.extend([asdict(s) for s in target_suggestions])

    # Sort by confidence score
    suggestions.sort(key=lambda x: x["confidence"], reverse=True)

    return ToolResult(
        content=json.dumps(suggestions, indent=2),
        structured_content={"use_case_suggestions": suggestions},
    )


@dr_mcp_tool(tags={"predictive", "training", "read", "analysis", "eda"})
async def get_exploratory_insights(
    *,
    dataset_id: Annotated[str, "The ID of the DataRobot dataset to analyze"] | None = None,
    target_col: Annotated[str, "Optional target column to focus EDA insights on"] | None = None,
) -> ToolError | ToolResult:
    """Generate exploratory data insights for a dataset."""
    if not dataset_id:
        raise ToolError("Dataset ID must be provided")

    client = get_sdk_client()
    dataset, df = _get_dataset_or_raise(client, dataset_id)

    # Get dataset insights first
    insights_result = await analyze_dataset(dataset_id=dataset_id)
    insights = insights_result.structured_content

    eda_insights = {
        "dataset_summary": {
            "total_rows": int(insights["total_rows"]),  # Convert to native Python int
            "total_columns": int(insights["total_columns"]),  # Convert to native Python int
            "memory_usage": int(df.memory_usage().sum()),  # Convert to native Python int
        },
        "target_analysis": {},
        "feature_correlations": {},
        "missing_data": insights["missing_data_summary"],
        "data_types": {
            "numerical": insights["numerical_columns"],
            "categorical": insights["categorical_columns"],
            "datetime": insights["datetime_columns"],
            "text": insights["text_columns"],
        },
    }

    # Target-specific analysis
    if target_col and target_col in df.columns:
        target_data = df[target_col]
        target_analysis = {
            "column_name": target_col,
            "data_type": str(target_data.dtype),
            "unique_values": int(target_data.nunique()),  # Convert to native Python int
            "missing_values": int(target_data.isnull().sum()),  # Convert to native Python int
            "missing_percentage": float(
                target_data.isnull().sum() / len(df) * 100
            ),  # Already float
        }

        if pd.api.types.is_numeric_dtype(target_data):
            target_analysis.update(
                {
                    "min_value": float(target_data.min()),  # Convert to native Python float
                    "max_value": float(target_data.max()),  # Convert to native Python float
                    "mean": float(target_data.mean()),  # Convert to native Python float
                    "median": float(target_data.median()),  # Convert to native Python float
                    "std_dev": float(target_data.std()),  # Convert to native Python float
                }
            )
        else:
            value_counts = target_data.value_counts()
            target_analysis.update(
                {
                    "value_counts": {
                        str(k): int(v) for k, v in value_counts.items()
                    },  # Convert both key and value
                    "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                }
            )

        eda_insights["target_analysis"] = target_analysis

        # Feature correlations with target (for numerical features)
        if pd.api.types.is_numeric_dtype(target_data):
            numerical_features = [col for col in insights["numerical_columns"] if col != target_col]
            if numerical_features:
                correlations = {}
                for feature in numerical_features:
                    corr = df[feature].corr(target_data)
                    if not pd.isna(corr):
                        correlations[feature] = float(corr)  # Convert to native Python float

                # Sort by absolute correlation
                eda_insights["feature_correlations"] = dict(
                    sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                )

    return ToolResult(
        content=json.dumps(eda_insights, indent=2),
        structured_content=eda_insights,
    )


def _identify_potential_targets(
    df: pd.DataFrame, numerical_cols: list[str], categorical_cols: list[str]
) -> list[str]:
    """Identify columns that could potentially be targets."""
    potential_targets = []

    # Look for common target column names
    target_keywords = [
        "target",
        "label",
        "class",
        "outcome",
        "result",
        "prediction",
        "predict",
        "sales",
        "revenue",
        "price",
        "amount",
        "value",
        "score",
        "rating",
        "churn",
        "conversion",
        "fraud",
        "default",
        "failure",
        "success",
        "risk",
        "probability",
        "likelihood",
        "status",
        "category",
        "type",
    ]

    for col in df.columns:
        col_lower = col.lower()

        # Check if column name contains target keywords
        if any(keyword in col_lower for keyword in target_keywords):
            potential_targets.append(col)
            continue

        # For numerical columns, check if they might be targets
        if col in numerical_cols:
            # Skip ID-like columns
            if "id" in col_lower or col_lower.endswith("_id"):
                continue

            # Check for bounded values (might be scores/ratings)
            if df[col].min() >= 0 and df[col].max() <= 100:
                potential_targets.append(col)
                continue

            # Check for binary-like numerical values
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                potential_targets.append(col)
                continue

        # For categorical columns, check cardinality
        if col in categorical_cols:
            unique_vals = df[col].nunique()
            # Good targets have reasonable cardinality (2-20 classes typically)
            if 2 <= unique_vals <= 20:
                potential_targets.append(col)

    return potential_targets


def _analyze_target_for_use_cases(df: pd.DataFrame, target_col: str) -> list[UseCaseSuggestion]:
    """Analyze a specific target column and suggest use cases."""
    suggestions: list[UseCaseSuggestion] = []

    target_data = df[target_col].dropna()
    if len(target_data) == 0:
        return suggestions

    # Determine if it's numerical or categorical
    if pd.api.types.is_numeric_dtype(target_data):
        unique_count = target_data.nunique()

        if unique_count == 2:
            # Binary classification
            values = sorted(target_data.unique())
            suggestions.append(
                UseCaseSuggestion(
                    name="Binary Classification",
                    description=f"Predict whether {target_col} will be {values[0]} or {values[1]}",
                    suggested_target=target_col,
                    problem_type="Binary Classification",
                    confidence=0.8,
                    reasoning=f"Column {target_col} has exactly 2 unique values, suggesting binary "
                    f"classification",
                )
            )
        elif unique_count <= 10:
            # Multiclass classification
            suggestions.append(
                UseCaseSuggestion(
                    name="Multiclass Classification",
                    description=f"Classify {target_col} into {unique_count} categories",
                    suggested_target=target_col,
                    problem_type="Multiclass Classification",
                    confidence=0.7,
                    reasoning=f"Column {target_col} has {unique_count} discrete values, suggesting "
                    f"classification",
                )
            )

        # Always suggest regression for numeric columns with more than 2 unique values
        if unique_count > 2:
            suggestions.append(
                UseCaseSuggestion(
                    name="Regression Modeling",
                    description=f"Predict the value of {target_col}",
                    suggested_target=target_col,
                    problem_type="Regression",
                    confidence=0.6
                    + (
                        0.1 if unique_count > 10 else 0
                    ),  # higher confidence for columns with more unique values for regression
                    reasoning=(
                        f"Column {target_col} is numerical with {unique_count} unique values, "
                        f"suggesting regression"
                    ),
                )
            )

    else:
        # Categorical target
        unique_count = target_data.nunique()

        if unique_count == 2:
            suggestions.append(
                UseCaseSuggestion(
                    name="Binary Classification",
                    description=f"Predict the category of {target_col}",
                    suggested_target=target_col,
                    problem_type="Binary Classification",
                    confidence=0.9,
                    reasoning=f"Column {target_col} is categorical with 2 classes",
                )
            )
        elif unique_count <= 20:
            suggestions.append(
                UseCaseSuggestion(
                    name="Multiclass Classification",
                    description=f"Classify {target_col} into {unique_count} categories",
                    suggested_target=target_col,
                    problem_type="Multiclass Classification",
                    confidence=0.8,
                    reasoning=f"Column {target_col} is categorical with {unique_count} manageable "
                    f"classes",
                )
            )

    # Add specific use case suggestions based on column names
    col_lower = target_col.lower()
    if "sales" in col_lower or "revenue" in col_lower:
        suggestions.append(
            UseCaseSuggestion(
                name="Sales Forecasting",
                description=f"Forecast future {target_col} values",
                suggested_target=target_col,
                problem_type="Regression",
                confidence=0.9,
                reasoning="Sales/revenue data is ideal for forecasting models",
            )
        )
    elif "churn" in col_lower:
        suggestions.append(
            UseCaseSuggestion(
                name="Customer Churn Prediction",
                description="Predict which customers are likely to churn",
                suggested_target=target_col,
                problem_type="Binary Classification",
                confidence=0.95,
                reasoning="Churn prediction is a classic binary classification problem",
            )
        )
    elif "fraud" in col_lower:
        suggestions.append(
            UseCaseSuggestion(
                name="Fraud Detection",
                description="Detect fraudulent transactions or activities",
                suggested_target=target_col,
                problem_type="Binary Classification",
                confidence=0.95,
                reasoning="Fraud detection is typically a binary classification problem",
            )
        )
    elif "price" in col_lower or "cost" in col_lower:
        suggestions.append(
            UseCaseSuggestion(
                name="Price Prediction",
                description=f"Predict optimal {target_col}",
                suggested_target=target_col,
                problem_type="Regression",
                confidence=0.85,
                reasoning="Price prediction is a common regression use case",
            )
        )

    return suggestions


@dr_mcp_tool(tags={"predictive", "training", "write", "autopilot", "model"})
async def start_autopilot(
    *,
    target: Annotated[str, "Name of the target column for modeling"] | None = None,
    project_id: Annotated[
        str, "Optional, the ID of the DataRobot project or a new project if no id is provided"
    ]
    | None = None,
    mode: Annotated[str, "Optional, Autopilot mode ('quick', 'comprehensive', or 'manual')"]
    | None = "quick",
    dataset_url: Annotated[
        str,
        """
        Optional, The URL to the dataset to upload
        (optional if dataset_id is provided) for a new project.
        """,
    ]
    | None = None,
    dataset_id: Annotated[
        str,
        """
        Optional, The ID of an existing dataset in AI Catalog
        (optional if dataset_url is provided) for a new project.
        """,
    ]
    | None = None,
    project_name: Annotated[
        str, "Optional, name for the project if no id is provided, creates a new project"
    ]
    | None = "MCP Project",
    use_case_id: Annotated[
        str,
        "Optional, ID of the use case to associate this project (required for next-gen platform)",
    ]
    | None = None,
) -> ToolError | ToolResult:
    """Start automated model training (Autopilot) for a project."""
    client = get_sdk_client()

    if not project_id:
        if not dataset_url and not dataset_id:
            raise ToolError("Either dataset_url or dataset_id must be provided")
        if dataset_url and dataset_id:
            raise ToolError("Please provide either dataset_url or dataset_id, not both")

        if dataset_url:
            dataset = client.Dataset.create_from_url(dataset_url)
        else:
            dataset = client.Dataset.get(dataset_id)

        project = client.Project.create_from_dataset(
            dataset.id, project_name=project_name, use_case=use_case_id
        )
    else:
        project = client.Project.get(project_id)

    if not target:
        raise ToolError("Target variable must be specified")

    try:
        # Start modeling
        project.analyze_and_model(target=target, mode=mode)

        result = {
            "project_id": project.id,
            "target": target,
            "mode": mode,
            "status": project.get_status(),
            "use_case_id": project.use_case_id,
        }

        return ToolResult(
            content=json.dumps(result, indent=2),
            structured_content=result,
        )

    except Exception as e:
        raise ToolError(
            content=json.dumps(
                {
                    "error": f"Failed to start Autopilot: {str(e)}",
                    "project_id": project.id if project else None,
                    "target": target,
                    "mode": mode,
                },
                indent=2,
            )
        )


@dr_mcp_tool(tags={"prediction", "training", "read", "model", "evaluation"})
async def get_model_roc_curve(
    *,
    project_id: Annotated[str, "The ID of the DataRobot project"] | None = None,
    model_id: Annotated[str, "The ID of the model to analyze"] | None = None,
    source: Annotated[
        str,
        """
        The source of the data to use for the ROC curve
        ('validation' or 'holdout' or 'crossValidation')
        """,
    ]
    | str = "validation",
) -> ToolError | ToolResult:
    """Get detailed ROC curve for a specific model."""
    if not project_id:
        raise ToolError("Project ID must be provided")
    if not model_id:
        raise ToolError("Model ID must be provided")

    client = get_sdk_client()
    project = client.Project.get(project_id)
    model = client.Model.get(project=project, model_id=model_id)

    try:
        roc_curve = model.get_roc_curve(source=source)
        roc_data = {
            "roc_points": [
                {
                    "accuracy": point.get("accuracy", 0),
                    "f1_score": point.get("f1_score", 0),
                    "false_negative_score": point.get("false_negative_score", 0),
                    "true_negative_score": point.get("true_negative_score", 0),
                    "true_negative_rate": point.get("true_negative_rate", 0),
                    "matthews_correlation_coefficient": point.get(
                        "matthews_correlation_coefficient", 0
                    ),
                    "true_positive_score": point.get("true_positive_score", 0),
                    "positive_predictive_value": point.get("positive_predictive_value", 0),
                    "false_positive_score": point.get("false_positive_score", 0),
                    "false_positive_rate": point.get("false_positive_rate", 0),
                    "negative_predictive_value": point.get("negative_predictive_value", 0),
                    "true_positive_rate": point.get("true_positive_rate", 0),
                    "threshold": point.get("threshold", 0),
                }
                for point in roc_curve.roc_points
            ],
            "negative_class_predictions": roc_curve.negative_class_predictions,
            "positive_class_predictions": roc_curve.positive_class_predictions,
            "source": source,
        }

        return ToolResult(
            content=json.dumps({"data": roc_data}, indent=2),
            structured_content={"data": roc_data},
        )
    except Exception as e:
        raise ToolError(f"Failed to get ROC curve: {str(e)}")


@dr_mcp_tool(tags={"predictive", "training", "read", "model", "evaluation"})
async def get_model_feature_impact(
    *,
    project_id: Annotated[str, "The ID of the DataRobot project"] | None = None,
    model_id: Annotated[str, "The ID of the model to analyze"] | None = None,
) -> ToolError | ToolResult:
    """Get detailed feature impact for a specific model."""
    if not project_id:
        raise ToolError("Project ID must be provided")
    if not model_id:
        raise ToolError("Model ID must be provided")

    client = get_sdk_client()
    project = client.Project.get(project_id)
    model = client.Model.get(project=project, model_id=model_id)
    # Get feature impact
    model.request_feature_impact()
    feature_impact = model.get_or_request_feature_impact()

    return ToolResult(
        content=json.dumps({"data": feature_impact}, indent=2),
        structured_content={"data": feature_impact},
    )


@dr_mcp_tool(tags={"predictive", "training", "read", "model", "evaluation"})
async def get_model_lift_chart(
    *,
    project_id: Annotated[str, "The ID of the DataRobot project"] | None = None,
    model_id: Annotated[str, "The ID of the model to analyze"] | None = None,
    source: Annotated[
        str,
        """
        The source of the data to use for the lift chart
        ('validation' or 'holdout' or 'crossValidation')
        """,
    ]
    | str = "validation",
) -> ToolError | ToolResult:
    """Get detailed lift chart for a specific model."""
    if not project_id:
        raise ToolError("Project ID must be provided")
    if not model_id:
        raise ToolError("Model ID must be provided")

    client = get_sdk_client()
    project = client.Project.get(project_id)
    model = client.Model.get(project=project, model_id=model_id)

    # Get lift chart
    lift_chart = model.get_lift_chart(source=source)

    lift_chart_data = {
        "bins": [
            {
                "actual": bin["actual"],
                "predicted": bin["predicted"],
                "bin_weight": bin["bin_weight"],
            }
            for bin in lift_chart.bins
        ],
        "source_model_id": lift_chart.source_model_id,
        "target_class": lift_chart.target_class,
    }

    return ToolResult(
        content=json.dumps({"data": lift_chart_data}, indent=2),
        structured_content={"data": lift_chart_data},
    )
