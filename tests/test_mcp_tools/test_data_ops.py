"""Tests for generic data operation tools (filter, aggregate, sort, transform)."""
from __future__ import annotations

import importlib
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from datarobot_genai.mcp_tools._registry import (
    clear_registry,
    get_all_tools,
    get_tools_by_category,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


def _make_mock_dataset(df: pd.DataFrame) -> MagicMock:
    mock_ds = MagicMock()
    mock_ds.id = "ds-test"
    mock_ds.name = "Test Dataset"
    mock_ds.get_as_dataframe.return_value = df
    return mock_ds


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Dave"],
            "age": [25, 35, 28, 42],
            "revenue": [100.0, 250.0, 180.0, 320.0],
            "region": ["East", "West", "East", "West"],
        }
    )


def _reload_data_ops():
    import datarobot_genai.mcp_tools.data_ops.filter as m1
    import datarobot_genai.mcp_tools.data_ops.aggregate as m2
    import datarobot_genai.mcp_tools.data_ops.sort as m3
    import datarobot_genai.mcp_tools.data_ops.transform as m4

    for mod in (m1, m2, m3, m4):
        importlib.reload(mod)


class TestDataOpsRegistration:
    def test_all_tools_registered(self):
        _reload_data_ops()
        tools = get_all_tools()
        for name in ("filter_data", "aggregate_data", "sort_data", "transform_data"):
            assert name in tools, f"Tool '{name}' not registered"

    def test_category_is_data_ops(self):
        _reload_data_ops()
        data_ops = get_tools_by_category("data_ops")
        assert len(data_ops) == 4


class TestFilterData:
    @pytest.mark.asyncio
    async def test_filter_greater_than(self):
        from datarobot_genai.mcp_tools.data_ops.filter import filter_data

        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            result = await filter_data("ds-test", [{"column": "age", "op": ">", "value": 30}])

        assert result["filtered_row_count"] == 2
        assert all(r["age"] > 30 for r in result["rows"])

    @pytest.mark.asyncio
    async def test_filter_equals(self):
        from datarobot_genai.mcp_tools.data_ops.filter import filter_data

        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            result = await filter_data("ds-test", [{"column": "region", "op": "==", "value": "East"}])

        assert result["filtered_row_count"] == 2
        assert all(r["region"] == "East" for r in result["rows"])

    @pytest.mark.asyncio
    async def test_filter_contains(self):
        from datarobot_genai.mcp_tools.data_ops.filter import filter_data

        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            result = await filter_data("ds-test", [{"column": "name", "op": "contains", "value": "a"}])

        names = [r["name"] for r in result["rows"]]
        assert all("a" in n for n in names)

    @pytest.mark.asyncio
    async def test_filter_invalid_column_raises(self):
        from datarobot_genai.mcp_tools.data_ops.filter import filter_data

        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            with pytest.raises(ValueError, match="not found"):
                await filter_data("ds-test", [{"column": "nonexistent", "op": "==", "value": 1}])

    @pytest.mark.asyncio
    async def test_filter_column_projection(self):
        from datarobot_genai.mcp_tools.data_ops.filter import filter_data

        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            result = await filter_data(
                "ds-test",
                [{"column": "age", "op": ">", "value": 20}],
                columns=["name", "age"],
            )

        assert result["columns"] == ["name", "age"]


class TestAggregateData:
    @pytest.mark.asyncio
    async def test_aggregate_sum(self):
        from datarobot_genai.mcp_tools.data_ops.aggregate import aggregate_data

        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            result = await aggregate_data(
                "ds-test",
                group_by=["region"],
                aggregations=[{"column": "revenue", "func": "sum"}],
            )

        assert result["row_count"] == 2
        rows_by_region = {r["region"]: r for r in result["rows"]}
        assert rows_by_region["East"]["revenue"] == pytest.approx(280.0)
        assert rows_by_region["West"]["revenue"] == pytest.approx(570.0)

    @pytest.mark.asyncio
    async def test_aggregate_count(self):
        from datarobot_genai.mcp_tools.data_ops.aggregate import aggregate_data

        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            result = await aggregate_data(
                "ds-test",
                group_by=["region"],
                aggregations=[{"column": "age", "func": "count"}],
            )

        assert result["row_count"] == 2

    @pytest.mark.asyncio
    async def test_aggregate_invalid_func_raises(self):
        from datarobot_genai.mcp_tools.data_ops.aggregate import aggregate_data

        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            with pytest.raises(ValueError, match="Unsupported aggregation function"):
                await aggregate_data(
                    "ds-test",
                    group_by=["region"],
                    aggregations=[{"column": "revenue", "func": "mode"}],
                )


class TestSortData:
    @pytest.mark.asyncio
    async def test_sort_ascending(self):
        from datarobot_genai.mcp_tools.data_ops.sort import sort_data

        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            result = await sort_data("ds-test", [{"column": "age", "direction": "asc"}])

        ages = [r["age"] for r in result["rows"]]
        assert ages == sorted(ages)

    @pytest.mark.asyncio
    async def test_sort_descending(self):
        from datarobot_genai.mcp_tools.data_ops.sort import sort_data

        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            result = await sort_data("ds-test", [{"column": "revenue", "direction": "desc"}])

        revenues = [r["revenue"] for r in result["rows"]]
        assert revenues == sorted(revenues, reverse=True)

    @pytest.mark.asyncio
    async def test_sort_invalid_column_raises(self):
        from datarobot_genai.mcp_tools.data_ops.sort import sort_data

        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            with pytest.raises(ValueError, match="not found"):
                await sort_data("ds-test", [{"column": "nonexistent", "direction": "asc"}])


class TestTransformData:
    @pytest.mark.asyncio
    async def test_transform_basic(self):
        from datarobot_genai.mcp_tools.data_ops.transform import transform_data

        code = "df = df[df['age'] > 30]"
        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            result = await transform_data("ds-test", code)

        assert result["result_row_count"] == 2
        assert all(r["age"] > 30 for r in result["sample"])

    @pytest.mark.asyncio
    async def test_transform_add_column(self):
        from datarobot_genai.mcp_tools.data_ops.transform import transform_data

        code = "df['revenue_per_age'] = df['revenue'] / df['age']"
        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            result = await transform_data("ds-test", code)

        assert "revenue_per_age" in result["columns"]

    @pytest.mark.asyncio
    async def test_transform_syntax_error_returns_error(self):
        from datarobot_genai.mcp_tools.data_ops.transform import transform_data

        code = "this is not valid python !!!"
        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            result = await transform_data("ds-test", code)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_transform_missing_df_assignment(self):
        from datarobot_genai.mcp_tools.data_ops.transform import transform_data

        code = "x = 1 + 2"  # doesn't reassign df
        with patch("datarobot.Dataset.get", return_value=_make_mock_dataset(_sample_df())):
            result = await transform_data("ds-test", code)

        # df is still in namespace (unchanged), so it should succeed
        assert result["result_row_count"] == 4  # original df unchanged
