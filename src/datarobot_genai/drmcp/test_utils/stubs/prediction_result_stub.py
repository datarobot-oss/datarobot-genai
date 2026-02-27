# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stub for datarobot_predict.deployment.predict used in integration tests with DR client stubs."""

from typing import Any
from typing import NamedTuple

import pandas as pd


class StubPredictionResult(NamedTuple):
    """Stub for datarobot_predict PredictionResult (dataframe + response_headers)."""

    dataframe: pd.DataFrame = pd.DataFrame()
    response_headers: dict[str, Any] = {}


def test_create_prediction_result(
    deployment: Any,
    data_frame: pd.DataFrame,
    max_explanations: int | str = 0,
    max_ngram_explanations: int | str | None = None,
    threshold_high: float | None = None,
    threshold_low: float | None = None,
    time_series_type: Any = None,
    forecast_point: Any = None,
    predictions_start_date: Any = None,
    predictions_end_date: Any = None,
    passthrough_columns: str | set[str] | None = None,
    explanation_algorithm: str | None = None,
    prediction_endpoint: str | None = None,
    relax_known_in_advance_features_check: bool | None = None,
    timeout: int = 600,
    **kwargs: Any,
) -> StubPredictionResult:
    """Stub for datarobot_predict.deployment.predict.

    Returns a result with row count and columns matching integration test expectations
    (sales (actual)_PREDICTION, FORECAST_*, explanation columns; 7 rows for forecast,
    14 for historical range).
    """
    n_in = len(data_frame)
    # Time series: integration tests expect 7 rows for forecast_point, 14 for historical range
    if forecast_point is not None:
        n_rows = 7
    elif predictions_start_date is not None and predictions_end_date is not None:
        n_rows = 14
    else:
        n_rows = n_in

    # Build output with target row count (repeat input rows if needed)
    if n_rows <= n_in:
        out_df = data_frame.iloc[:n_rows].copy()
    else:
        out_df = pd.concat([data_frame] * (n_rows // n_in + 1), ignore_index=True).iloc[:n_rows]

    # Prediction column: use "sales (actual)_PREDICTION" when input has "sales" (integration tests)
    if "sales" in data_frame.columns:
        prediction_col = "sales (actual)_PREDICTION"
    else:
        prediction_col = "stub_PREDICTION"
    if prediction_col not in out_df.columns:
        out_df[prediction_col] = [0.0] * n_rows

    # Time series: add FORECAST_POINT and FORECAST_DISTANCE when forecast params present
    if forecast_point is not None or (
        predictions_start_date is not None and predictions_end_date is not None
    ):
        if "FORECAST_POINT" not in out_df.columns:
            out_df["FORECAST_POINT"] = (
                [forecast_point] * n_rows if forecast_point is not None else [pd.NaT] * n_rows
            )
        if "FORECAST_DISTANCE" not in out_df.columns:
            out_df["FORECAST_DISTANCE"] = list(range(1, n_rows + 1))

    # Explanations: add at least one column so tests asserting "EXPLANATION" or "SHAP" in cols pass
    if max_explanations not in (0, "0"):
        if not any("EXPLANATION" in c or "SHAP" in c for c in out_df.columns):
            out_df["EXPLANATION_1"] = [0.0] * n_rows

    return StubPredictionResult(dataframe=out_df, response_headers={})
