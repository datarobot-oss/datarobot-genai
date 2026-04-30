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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import polars as pl
import pytest
from datarobot.errors import ClientError

from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind
from datarobot_genai.drtools.predictive import model
from datarobot_genai.drtools.predictive.model import ModelEncoder
from datarobot_genai.drtools.predictive.model import model_to_dict


def _patch_model_client(mock_client: MagicMock):
    """Return (patch get_datarobot_access_token, patch DataRobotClient).
    Use: with p1, p2 as mock_drc.
    """
    return (
        patch(
            "datarobot_genai.drtools.predictive.model.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.model.DataRobotClient"),
    )


@pytest.mark.asyncio
async def test_get_best_model_success() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_model1 = MagicMock(id="m1", model_type="XGBoost", metrics={"AUC": {"validation": 0.9}})
    mock_model2 = MagicMock(
        id="m2", model_type="Random Forest", metrics={"AUC": {"validation": 0.8}}
    )
    mock_project.get_models.return_value = [mock_model1, mock_model2]
    mock_client.Project.get.return_value = mock_project
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        result = await model.get_best_model(project_id="pid", metric="AUC")
    assert isinstance(result, dict)
    assert result["project_id"] == "pid"
    assert result["best_model"]["model_type"] == "XGBoost"
    expected_auc = 0.9
    assert result["best_model"]["metrics"]["AUC"]["validation"] == expected_auc


@pytest.mark.asyncio
async def test_get_best_model_no_models() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_project.get_models.return_value = []
    mock_client.Project.get.return_value = mock_project
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(ToolError, match="No models found for this project."):
            await model.get_best_model(project_id="pid", metric="AUC")


@pytest.mark.asyncio
async def test_get_best_model_project_not_found() -> None:
    mock_client = MagicMock()
    mock_client.Project.get.return_value = None
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(ToolError, match="Project with ID pid not found."):
            await model.get_best_model(project_id="pid", metric="AUC")


@pytest.mark.asyncio
async def test_get_best_model_project_client_error_404() -> None:
    mock_client = MagicMock()
    mock_client.Project.get.side_effect = ClientError(
        "404 client error: {'message': 'Not Found'}",
        status_code=404,
        json={"message": "Not Found"},
    )
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(ToolError) as exc_info:
            await model.get_best_model(project_id="missing-proj", metric="AUC")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND
    assert "404" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_best_model_error() -> None:
    with patch(
        "datarobot_genai.drtools.predictive.model.get_datarobot_access_token",
        new_callable=AsyncMock,
        side_effect=Exception("fail"),
    ):
        with pytest.raises(Exception) as exc_info:
            await model.get_best_model(project_id="pid", metric="AUC")
        assert "fail" == str(exc_info.value)


@pytest.mark.asyncio
async def test_score_dataset_with_model_success() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_dr_model = MagicMock()
    mock_job = MagicMock(id="jobid")
    mock_catalog_ds = MagicMock()
    mock_catalog_ds.id = "catalog_ds"
    mock_prediction_ds = MagicMock()
    mock_prediction_ds.id = "pred_ds_uploaded"
    mock_client.Dataset.get.return_value = mock_catalog_ds
    mock_project.upload_dataset_from_catalog.return_value = mock_prediction_ds
    mock_dr_model.request_predictions.return_value = mock_job
    mock_dr_model.model_type = "type1"
    mock_dr_model.metrics = {"AUC": 0.9}
    mock_client.Model.get.return_value = mock_dr_model
    mock_client.Project.get.return_value = mock_project
    p1, p2 = _patch_model_client(mock_client)
    ds_id = "catalog_ds"
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        result = await model.score_dataset_with_model(
            project_id="pid", model_id="mid", dataset_id=ds_id
        )
    mock_client.Project.get.assert_called_once_with("pid")
    mock_client.Model.get.assert_called_once_with(mock_project, "mid")
    mock_client.Dataset.get.assert_called_once_with(ds_id)
    mock_project.upload_dataset_from_catalog.assert_called_once_with(dataset_id="catalog_ds")
    mock_dr_model.request_predictions.assert_called_once_with(dataset_id="pred_ds_uploaded")
    assert isinstance(result, dict)
    assert result["scoring_job_id"] == "jobid"
    assert result["catalog_dataset_id"] == "catalog_ds"
    assert result["prediction_dataset_id"] == "pred_ds_uploaded"


@pytest.mark.asyncio
async def test_score_dataset_with_model_empty_dataset_id() -> None:
    with pytest.raises(ToolError, match="Dataset ID must be provided"):
        await model.score_dataset_with_model(
            project_id="pid",
            model_id="mid",
            dataset_id="   ",
        )


@pytest.mark.asyncio
async def test_score_dataset_with_model_project_not_found() -> None:
    project_id = "pid"
    mock_client = MagicMock()
    mock_client.Project.get.side_effect = ClientError(
        "404 client error: {'message': 'Not Found'}",
        status_code=404,
        json={"message": "Not Found"},
    )
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(ToolError) as exc_info:
            await model.score_dataset_with_model(
                project_id=project_id,
                model_id="mid",
                dataset_id="dsid",
            )
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND
    assert "not found" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_score_dataset_with_model_model_not_found() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_project.get_models.return_value = []
    mock_client.Project.get.return_value = mock_project
    mock_client.Model.get.side_effect = ClientError(
        "404 client error: {'message': 'Leaderboard Item Not Found'}",
        status_code=404,
        json={"message": "Leaderboard Item Not Found"},
    )
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(ToolError) as exc_info:
            await model.score_dataset_with_model(
                project_id="pid",
                model_id="mid",
                dataset_id="dsid",
            )
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


@pytest.mark.asyncio
async def test_score_dataset_with_model_client_error_non_404_is_upstream() -> None:
    mock_client = MagicMock()
    mock_client.Project.get.side_effect = ClientError(
        "503: service unavailable",
        status_code=503,
        json={},
    )
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(ToolError) as exc_info:
            await model.score_dataset_with_model(
                project_id="pid",
                model_id="mid",
                dataset_id="dsid",
            )
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_score_dataset_with_model_error() -> None:
    with patch(
        "datarobot_genai.drtools.predictive.model.get_datarobot_access_token",
        new_callable=AsyncMock,
        side_effect=Exception("fail"),
    ):
        with pytest.raises(Exception) as exc_info:
            await model.score_dataset_with_model(
                project_id="pid",
                model_id="mid",
                dataset_id="dsid",
            )
        assert "fail" == str(exc_info.value)


@pytest.mark.asyncio
async def test_get_model_details_success() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_project.target = "target_col"
    mock_project.metric = "AUC"
    mock_model = MagicMock(
        id="mid",
        model_type="XGBoost",
        featurelist_name="Informative Features",
        metrics={"AUC": {"validation": 0.9}},
        sample_pct=64,
    )
    mock_model.get_or_request_feature_impact.return_value = [{"feature": "f1", "impact": 0.8}]
    mock_client.Project.get.return_value = mock_project
    mock_client.Model.get.return_value = mock_model
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        result = await model.get_model_details(project_id="pid", model_id="mid")
    assert isinstance(result, dict)
    assert result["model_id"] == "mid"
    assert result["model_type"] == "XGBoost"
    assert result["target"] == "target_col"


@pytest.mark.asyncio
async def test_get_model_details_missing_project_id() -> None:
    """Required param project_id is enforced by signature (wrapped as ToolError)."""
    with pytest.raises(TypeError, match="project_id"):
        await model.get_model_details(model_id="mid")


@pytest.mark.asyncio
async def test_get_model_details_missing_model_id() -> None:
    """Required param model_id is enforced by signature (wrapped as ToolError)."""
    with pytest.raises(TypeError, match="model_id"):
        await model.get_model_details(project_id="pid")


@pytest.mark.asyncio
async def test_get_model_details_feature_impact_error() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_project.target = "target_col"
    mock_project.metric = "AUC"
    mock_model = MagicMock(id="mid", model_type="XGBoost", metrics={}, sample_pct=64)
    mock_model.get_or_request_feature_impact.side_effect = Exception("not available")
    mock_client.Project.get.return_value = mock_project
    mock_client.Model.get.return_value = mock_model
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        result = await model.get_model_details(
            project_id="pid", model_id="mid", include_feature_impact=True
        )
    assert "feature_impact_error" in result


@pytest.mark.asyncio
async def test_is_eligible_for_timeseries_training_success() -> None:
    mock_client = MagicMock()
    mock_dataset = MagicMock()
    # Build with polars, convert to pandas at boundary (SDK returns pandas)
    dates = pl.date_range(
        pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=199), eager=True
    )
    pandas_df = pl.DataFrame(
        {
            "date": dates,
            "target": range(200),
            "feature": range(200),
        }
    ).to_pandas()
    mock_dataset.get_as_dataframe.return_value = pandas_df
    mock_client.Dataset.get.return_value = mock_dataset
    with (
        patch(
            "datarobot_genai.drtools.predictive.model.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.model.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await model.is_eligible_for_timeseries_training(
            dataset_id="ds1", datetime_column="date", target_column="target"
        )
    assert result["status"] == "ELIGIBLE"
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_is_eligible_for_timeseries_training_too_few_rows() -> None:
    mock_client = MagicMock()
    mock_dataset = MagicMock()
    # Build with polars, convert to pandas at boundary (SDK returns pandas)
    dates = pl.date_range(
        pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=49), eager=True
    )
    pandas_df = pl.DataFrame(
        {
            "date": dates,
            "target": range(50),
        }
    ).to_pandas()
    mock_dataset.get_as_dataframe.return_value = pandas_df
    mock_client.Dataset.get.return_value = mock_dataset
    with (
        patch(
            "datarobot_genai.drtools.predictive.model.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.model.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await model.is_eligible_for_timeseries_training(
            dataset_id="ds1", datetime_column="date", target_column="target"
        )
    assert result["status"] == "NOT_ELIGIBLE"
    assert any("Too few rows" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_is_eligible_for_timeseries_training_missing_params() -> None:
    with pytest.raises(TypeError, match="dataset_id"):
        await model.is_eligible_for_timeseries_training(
            datetime_column="date", target_column="target"
        )


def _build_dataset_mock(pandas_df) -> MagicMock:
    mock_client = MagicMock()
    mock_dataset = MagicMock()
    mock_dataset.get_as_dataframe.return_value = pandas_df
    mock_client.Dataset.get.return_value = mock_dataset
    return mock_client


async def _run_eligibility(
    pandas_df, *, datetime_column="date", target_column="target", series_id_column=None
):
    mock_client = _build_dataset_mock(pandas_df)
    with (
        patch(
            "datarobot_genai.drtools.predictive.model.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.model.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        return await model.is_eligible_for_timeseries_training(
            dataset_id="ds1",
            datetime_column=datetime_column,
            target_column=target_column,
            series_id_column=series_id_column,
        )


@pytest.mark.asyncio
async def test_eligibility_reports_daily_cadence_and_no_gaps() -> None:
    dates = pl.date_range(
        pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=199), eager=True
    )
    pandas_df = pl.DataFrame({"date": dates, "target": range(200)}).to_pandas()

    result = await _run_eligibility(pandas_df)

    assert result["status"] == "ELIGIBLE"
    assert result["cadence"]["median_timestep_human"] == "1 day"
    assert result["cadence"]["pct_series_with_gaps"] == 0.0
    assert any("Median timestep: 1 day" in line for line in result["info"])


@pytest.mark.asyncio
async def test_eligibility_flags_gap_percentage_for_irregular_data() -> None:
    # 200 daily timestamps with one missing date partway through
    dates = list(
        pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=199), eager=True)
    )
    del dates[100]
    pandas_df = pl.DataFrame({"date": dates, "target": range(199)}).to_pandas()

    result = await _run_eligibility(pandas_df)

    # DR's backend treats non-regular cadence as still eligible — verdict
    # must remain ELIGIBLE; only the diagnostics change.
    assert result["status"] == "ELIGIBLE"
    assert result["cadence"]["pct_series_with_gaps"] == 1.0
    assert any("gap larger than the median timestep" in line for line in result["info"])


@pytest.mark.asyncio
async def test_eligibility_multiseries_cadence_per_series() -> None:
    daily = list(
        pl.date_range(
            pl.date(2020, 1, 1),
            pl.date(2020, 1, 1) + pl.duration(days=149),
            interval="1d",
            eager=True,
        )
    )
    biweekly = list(
        pl.date_range(
            pl.date(2020, 1, 1),
            pl.date(2020, 1, 1) + pl.duration(weeks=2 * 149),
            interval="2w",
            eager=True,
        )
    )[:150]
    pandas_df = pl.DataFrame(
        {
            "date": daily + biweekly,
            "series": ["a"] * 150 + ["b"] * 150,
            "target": list(range(150)) + list(range(150)),
        }
    ).to_pandas()

    result = await _run_eligibility(pandas_df, series_id_column="series")

    assert result["status"] == "ELIGIBLE"
    assert result["cadence"]["n_series"] == 2
    # Some fraction of series will register gaps relative to the global median.
    assert 0.0 < result["cadence"]["pct_series_with_gaps"] <= 1.0


@pytest.mark.asyncio
async def test_eligibility_handles_weekly_cadence_label() -> None:
    dates = pl.date_range(
        pl.date(2020, 1, 1),
        pl.date(2020, 1, 1) + pl.duration(weeks=199),
        interval="1w",
        eager=True,
    )
    pandas_df = pl.DataFrame({"date": dates, "target": range(200)}).to_pandas()

    result = await _run_eligibility(pandas_df)

    assert result["status"] == "ELIGIBLE"
    assert result["cadence"]["median_timestep_human"] == "1 week"
    assert result["cadence"]["pct_series_with_gaps"] == 0.0


def test_humanize_timestep_seconds_known_cadences() -> None:
    assert model._humanize_timestep_seconds(0) == "unknown"
    assert model._humanize_timestep_seconds(60) == "1 minute"
    assert model._humanize_timestep_seconds(3600) == "1 hour"
    assert model._humanize_timestep_seconds(86400) == "1 day"
    assert model._humanize_timestep_seconds(86400 * 2) == "2 days"
    assert model._humanize_timestep_seconds(604800) == "1 week"
    assert model._humanize_timestep_seconds(90) == "90.0 seconds"


def test_compute_cadence_returns_none_when_insufficient_data() -> None:
    import datetime as _dt

    df = pl.DataFrame({"t": [_dt.datetime(2024, 1, 1)]})
    assert model._compute_cadence(df, "t", None) is None


@pytest.mark.asyncio
async def test_eligibility_error_for_missing_datetime_column_lists_columns() -> None:
    dates = pl.date_range(
        pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=199), eager=True
    )
    pandas_df = pl.DataFrame({"date": dates, "target": range(200)}).to_pandas()

    result = await _run_eligibility(pandas_df, datetime_column="not_a_column")

    assert result["status"] == "NOT_ELIGIBLE"
    err = next(e for e in result["errors"] if "not_a_column" in e)
    # Must list the available columns and call out case-sensitivity.
    assert "Available columns" in err
    assert "date" in err
    assert "target" in err
    assert "case-sensitive" in err


@pytest.mark.asyncio
async def test_eligibility_error_for_missing_target_lists_columns() -> None:
    dates = pl.date_range(
        pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=199), eager=True
    )
    pandas_df = pl.DataFrame({"date": dates, "target": range(200)}).to_pandas()

    result = await _run_eligibility(pandas_df, target_column="missing_target")

    assert result["status"] == "NOT_ELIGIBLE"
    err = next(e for e in result["errors"] if "missing_target" in e)
    assert "Available columns" in err


@pytest.mark.asyncio
async def test_eligibility_error_for_high_null_target_explains_remediation() -> None:
    dates = pl.date_range(
        pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=199), eager=True
    )
    # 50% nulls in target — over the 30% threshold.
    target = [None if i % 2 == 0 else i for i in range(200)]
    pandas_df = pl.DataFrame({"date": dates, "target": target}).to_pandas()

    result = await _run_eligibility(pandas_df)

    assert result["status"] == "NOT_ELIGIBLE"
    err = next(e for e in result["errors"] if "null" in e)
    assert "50.0%" in err
    # Must explain WHY (DR needs populated target) and HOW (filter / aggregate).
    assert "filter" in err.lower() or "aggregate" in err.lower()


@pytest.mark.asyncio
async def test_eligibility_error_for_too_few_rows_explains_remediation() -> None:
    dates = pl.date_range(
        pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=49), eager=True
    )
    pandas_df = pl.DataFrame({"date": dates, "target": range(50)}).to_pandas()

    result = await _run_eligibility(pandas_df)

    assert result["status"] == "NOT_ELIGIBLE"
    err = next(e for e in result["errors"] if "Too few rows" in e)
    assert "100" in err
    assert "aggregation" in err.lower() or "history" in err.lower()


@pytest.mark.asyncio
async def test_eligibility_allows_all_null_target_for_scoring_dataset() -> None:
    """Scoring datasets have an all-null target; treat as INFO, not an error."""
    dates = pl.date_range(
        pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=199), eager=True
    )
    pandas_df = pl.DataFrame(
        {"date": dates, "target": [None] * 200}, schema={"date": pl.Date, "target": pl.Int64}
    ).to_pandas()

    result = await _run_eligibility(pandas_df)

    assert result["status"] == "ELIGIBLE"
    assert any("scoring dataset" in line for line in result["info"])


@pytest.mark.asyncio
async def test_eligibility_flags_row_level_duplicates() -> None:
    """Two rows with identical (date, series_id) must be flagged as a blocking error."""
    dates = list(
        pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=199), eager=True)
    )
    # 200 unique dates + one repeat of the first date (same series).
    dates.append(dates[0])
    pandas_df = pl.DataFrame({"date": dates, "target": list(range(201))}).to_pandas()

    result = await _run_eligibility(pandas_df)

    assert result["status"] == "NOT_ELIGIBLE"
    err = next(e for e in result["errors"] if "duplicated key" in e)
    assert "Aggregate" in err or "disambiguate" in err


@pytest.mark.asyncio
async def test_eligibility_flags_duplicates_per_series_in_multiseries() -> None:
    dates_a = list(
        pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=99), eager=True)
    )
    dates_b = list(
        pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=99), eager=True)
    )
    # Inject a duplicate (same date) for series 'a' but not 'b' — must
    # detect the dupe inside series 'a' even though the date appears in 'b' too.
    dates_a.append(dates_a[0])
    pandas_df = pl.DataFrame(
        {
            "series": ["a"] * 101 + ["b"] * 100,
            "date": dates_a + dates_b,
            "target": list(range(201)),
        }
    ).to_pandas()

    result = await _run_eligibility(pandas_df, series_id_column="series")

    assert result["status"] == "NOT_ELIGIBLE"
    assert any("duplicated key" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_eligibility_error_for_unparseable_datetime_shows_samples() -> None:
    # Strings that can't be parsed as dates/timestamps.
    pandas_df = pl.DataFrame(
        {
            "date": ["banana", "kumquat", "fig"] + ["pear"] * 197,
            "target": list(range(200)),
        }
    ).to_pandas()

    result = await _run_eligibility(pandas_df)

    assert result["status"] == "NOT_ELIGIBLE"
    err = next(e for e in result["errors"] if "could not be parsed" in e)
    # Must show example bad values and suggest ISO-8601.
    assert "Sample values" in err
    assert "ISO-8601" in err


class TestModelToDict:
    """Test cases for model_to_dict function."""

    def test_model_to_dict_success(self):
        """Test successful conversion of model to dict."""
        model = Mock()
        model.id = "model123"
        model.model_type = "regression"
        model.metrics = {"AUC": 0.85}

        result = model_to_dict(model)

        assert result == {"id": "model123", "model_type": "regression", "metrics": {"AUC": 0.85}}

    def test_model_to_dict_attribute_error(self):
        """Test model_to_dict handles AttributeError gracefully."""

        # Create a class that raises AttributeError when accessing model_type
        class ModelWithError:
            def __init__(self):
                self.id = "model123"
                self.metrics = {"AUC": 0.85}

            @property
            def model_type(self):
                raise AttributeError("No model_type")

        model = ModelWithError()
        result = model_to_dict(model)

        assert result == {"id": "model123", "model_type": "unknown"}

    def test_model_to_dict_all_attributes_missing(self):
        """Test model_to_dict when all attributes are missing."""

        # Create a class that raises AttributeError when accessing any attribute
        class ModelWithAllErrors:
            @property
            def id(self):
                raise AttributeError("No id")

            @property
            def model_type(self):
                raise AttributeError("No model_type")

            @property
            def metrics(self):
                raise AttributeError("No metrics")

        model = ModelWithAllErrors()
        result = model_to_dict(model)

        assert result == {"id": "unknown", "model_type": "unknown"}

    def test_model_to_dict_partial_attributes_missing(self):
        """Test model_to_dict when some attributes are missing."""

        # Create a class that raises AttributeError when accessing model_type
        class ModelWithPartialError:
            def __init__(self):
                self.id = "model123"
                self.metrics = {"AUC": 0.85}

            @property
            def model_type(self):
                raise AttributeError("No model_type")

        model = ModelWithPartialError()
        result = model_to_dict(model)

        assert result == {"id": "model123", "model_type": "unknown"}


class TestModelEncoder:
    """Test cases for ModelEncoder class."""

    def test_model_encoder_with_model_object(self):
        """Test ModelEncoder with DataRobot Model object."""
        # Create a mock that behaves like a Model object
        model = Mock()
        model.id = "model123"
        model.model_type = "regression"
        model.metrics = {"AUC": 0.85}

        # Mock the isinstance check to return True for Model
        with patch("datarobot_genai.drtools.predictive.model.isinstance", return_value=True):
            encoder = ModelEncoder()
            result = encoder.default(model)

            assert result == {
                "id": "model123",
                "model_type": "regression",
                "metrics": {"AUC": 0.85},
            }

    def test_model_encoder_with_non_model_object(self):
        """Test ModelEncoder with non-Model object falls back to default."""
        encoder = ModelEncoder()

        # Test with a string - should raise TypeError
        with pytest.raises(TypeError, match="Object of type str is not JSON serializable"):
            encoder.default("test_string")

    def test_model_encoder_json_serialization(self):
        """Test ModelEncoder with JSON serialization."""
        # Create a mock that behaves like a Model object
        model = Mock()
        model.id = "model123"
        model.model_type = "regression"
        model.metrics = {"AUC": 0.85}

        # Mock the isinstance check to return True for Model
        with patch("datarobot_genai.drtools.predictive.model.isinstance", return_value=True):
            data = {"model": model, "other": "value"}
            result = json.dumps(data, cls=ModelEncoder)

            expected = (
                '{"model": {"id": "model123", "model_type": "regression", '
                '"metrics": {"AUC": 0.85}}, "other": "value"}'
            )
            assert result == expected
