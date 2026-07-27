# Copyright 2026 DataRobot, Inc.
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

"""Unit tests for cuOpt solution -> panels persistence over PanelStore."""

import io
import json

import pandas as pd
import polars as pl

from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drtools.cuopt import persistence as cpers
from datarobot_genai.drtools.cuopt import tables as ct
from tests.drtools.unit.cuopt.conftest import lp_solve_result


def test_pandas_to_polars_json_encodes_nested_objects() -> None:
    # GIVEN a pandas frame with nested objects polars cannot infer
    df = pd.DataFrame({"a": [1, 2], "nested": [{"k": 1}, [1, 2]]})

    # WHEN converting to polars
    frame = cpers._pandas_to_polars(df)

    # THEN nested values are JSON-encoded so Parquet serialization succeeds
    assert frame["a"].to_list() == [1, 2]
    assert [json.loads(v) for v in frame["nested"].to_list()] == [{"k": 1}, [1, 2]]


async def test_persist_cuopt_panels_orders_and_links_panels(store: PanelStore) -> None:
    # GIVEN extracted LP solution tables and a parent panel id for lineage
    result = lp_solve_result()
    tables, summary = ct.extract_solution_tables("lp", result)

    # WHEN persisting the panels
    response = await cpers.persist_cuopt_panels(
        tables=tables,
        summary_text=summary,
        result_dict=result,
        problem_type="lp",
        parent_id="parent-1",
    )

    # THEN panels come back ordered summary Text -> raw Json -> Dataset tables
    assert response["status"] == "success"
    assert response["problem_type"] == "lp"
    assert response["comments"] == summary
    panels = response["panels"]
    types = [p["type"] for p in panels]
    assert types[0] == "text"
    assert types[1] == "json"
    assert set(types[2:]) == {"dataset"}
    assert [p["title"] for p in panels[2:]] == [
        "Primal Solution",
        "Dual Solution",
        "Solution Metrics",
    ]
    # AND every panel links back to the parent for lineage
    assert all(p["parents"] == ["parent-1"] for p in panels)

    # AND the Json panel carries the raw solution while Datasets round-trip as Parquet
    solution_panel = await store.get(panels[1]["id"])
    assert solution_panel.data == result["solution"]
    assert solution_panel.title == "cuOpt Solution - LP"
    frame = pl.read_parquet(io.BytesIO(await store.get_payload(panels[2]["id"])))
    assert frame["variable"].to_list() == ["x1", "x2"]
    assert frame["value"].to_list() == [0.5, 0.5]


async def test_persist_cuopt_panels_without_parent_or_summary(store: PanelStore) -> None:
    # GIVEN a result with no summary text and no parent panel
    result = {"status": "success", "solution": {"assignments": [{"a": 1}]}}
    tables = [
        {
            "title": "Assignments",
            "description": "Assignments returned by the solver.",
            "dataframe": pd.DataFrame([{"a": 1}]),
        }
    ]

    # WHEN persisting
    response = await cpers.persist_cuopt_panels(
        tables=tables,
        summary_text="",
        result_dict=result,
        problem_type="vrp",
        parent_id=None,
    )

    # THEN no Text panel is written, parents stay empty, and comments are empty
    types = [p["type"] for p in response["panels"]]
    assert types == ["json", "dataset"]
    assert all(p["parents"] == [] for p in response["panels"])
    assert response["comments"] == ""


async def test_persist_cuopt_panels_writes_to_staging(store: PanelStore) -> None:
    # GIVEN a minimal successful result
    result = lp_solve_result()
    tables, summary = ct.extract_solution_tables("lp", result)

    # WHEN persisting
    await cpers.persist_cuopt_panels(
        tables=tables,
        summary_text=summary,
        result_dict=result,
        problem_type="lp",
        parent_id=None,
    )

    # THEN all panels land in the staging source (review-before-promote flow)
    staged = await store.list(source="staging")
    assert len(staged) == 5  # summary + raw solution + 3 dataset tables
    assert await store.list(source="main") == []
