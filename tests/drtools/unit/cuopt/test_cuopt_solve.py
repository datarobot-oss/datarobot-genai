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

"""Unit tests for the high-level ``solve_with_cuopt_deployment`` entry point.

The deployment call seam is mocked (canned ``validate``/``solve`` results) and
panel writes go to the in-memory FakeBlobStore-backed PanelStore.
"""

import io

import polars as pl
import pytest

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.models import Json
from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drtools.cuopt import solve as solve_mod
from datarobot_genai.drtools.cuopt.solve import solve_with_cuopt_deployment
from tests.drtools.unit.cuopt.conftest import lp_solve_result
from tests.drtools.unit.cuopt.conftest import patch_cuopt_client
from tests.drtools.unit.cuopt.conftest import sample_lp_data
from tests.drtools.unit.cuopt.conftest import sample_milp_data
from tests.drtools.unit.cuopt.conftest import vrp_solve_result


async def test_solve_requires_data(store: PanelStore) -> None:
    # GIVEN no payload at all
    # WHEN solving
    result = await solve_with_cuopt_deployment(data=None)  # type: ignore[arg-type]

    # THEN an actionable error is returned instead of raising
    assert result["status"] == "error"
    assert "data parameter is required" in result["error"]


async def test_solve_preview_valid_adds_next_step(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a payload the deployment validates successfully
    client = patch_cuopt_client(monkeypatch, validate_result={"status": "valid", "message": "ok"})

    # WHEN previewing (default)
    result = await solve_with_cuopt_deployment(data=sample_milp_data())

    # THEN the validation result gains a next-step hint and nothing is solved
    assert result["status"] == "valid"
    assert "preview_only=False" in result["next_step"]
    client.validate.assert_awaited_once()
    client.solve.assert_not_awaited()


async def test_solve_preview_invalid_passthrough(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a payload the deployment rejects
    patch_cuopt_client(monkeypatch, validate_result={"status": "invalid", "error": "bad payload"})

    # WHEN previewing THEN the invalid result passes through unmodified
    result = await solve_with_cuopt_deployment(data=sample_milp_data())
    assert result == {"status": "invalid", "error": "bad payload"}


async def test_solve_lp_persists_panels_with_lineage(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN an LP problem stored as a Json panel and a successful solve
    patch_cuopt_client(monkeypatch, solve_result=lp_solve_result())
    input_panel = await store.create(
        Json(title="LP problem", data=sample_lp_data()), source="staging"
    )

    # WHEN solving with the panel id
    result = await solve_with_cuopt_deployment(data=input_panel.id, preview_only=False)

    # THEN the solve succeeds and reports the LP problem type
    assert result["status"] == "success"
    assert result["problem_type"] == "lp"
    assert "Detected problem type: LP" in result["comments"]

    # AND panels come back ordered summary Text -> raw Json -> Dataset tables
    panels = result["panels"]
    types = [p["type"] for p in panels]
    assert types[0] == "text"
    assert types[1] == "json"
    assert set(types[2:]) == {"dataset"}
    assert [p["title"] for p in panels[2:]] == [
        "Primal Solution",
        "Dual Solution",
        "Solution Metrics",
    ]
    # AND every output panel links back to the input Json panel
    assert all(p["parents"] == [input_panel.id] for p in panels)

    # AND the summary Text carries the summary; the Json carries the raw solution
    summary_panel = await store.get(panels[0]["id"])
    assert "Request ID: req-lp" in summary_panel.text
    solution_panel = await store.get(panels[1]["id"])
    assert solution_panel.data == lp_solve_result()["solution"]
    assert solution_panel.title == "cuOpt Solution - LP"

    # AND Dataset payloads are Parquet and round-trip
    frame = pl.read_parquet(io.BytesIO(await store.get_payload(panels[2]["id"])))
    assert frame["variable"].to_list() == ["x1", "x2"]
    assert frame["value"].to_list() == [0.5, 0.5]


async def test_solve_vrp_inline_dict_no_lineage(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN an inline native VRP payload and a successful solve
    client = patch_cuopt_client(monkeypatch, solve_result=vrp_solve_result())

    # WHEN solving with the inline dict
    result = await solve_with_cuopt_deployment(
        data={"fleet_data": {}, "task_data": {}}, preview_only=False
    )

    # THEN panels have no lineage parents and the native payload passes through
    assert result["problem_type"] == "vrp"
    assert all(p["parents"] == [] for p in result["panels"])
    client.solve.assert_awaited_once_with({"fleet_data": {}, "task_data": {}})


async def test_solve_inline_dict_gates_before_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inline-dict path must not call ``client.solve`` for an unentitled user.

    Wren regression: the sandbox gate used to run *after* ``client.solve``, so
    an unentitled caller would still trigger a solve (and only fail once
    persisting the resulting panels). It must fire before any solve call.
    """
    # GIVEN an unentitled user (sandbox gate denies)
    client = patch_cuopt_client(monkeypatch, solve_result=vrp_solve_result())

    def _deny() -> None:
        raise ToolError("Panel tools require the ENABLE_MCP_SANDBOX entitlement.")

    monkeypatch.setattr(solve_mod, "_require_mcp_sandbox", _deny)

    # WHEN solving an inline dict THEN the gate fires before any solve call
    with pytest.raises(ToolError, match="ENABLE_MCP_SANDBOX"):
        await solve_with_cuopt_deployment(
            data={"fleet_data": {}, "task_data": {}}, preview_only=False
        )
    client.solve.assert_not_awaited()


async def test_solve_converts_high_level_vrp(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a high-level VRPData payload (depot + customers)
    client = patch_cuopt_client(monkeypatch, solve_result=vrp_solve_result())

    # WHEN solving
    await solve_with_cuopt_deployment(
        data={
            "depot": {"x": 0, "y": 0},
            "customers": [{"id": "A", "x": 3, "y": 4}],
        },
        preview_only=False,
    )

    # THEN the payload sent to the deployment is the native conversion
    sent = client.solve.await_args.args[0]
    assert "fleet_data" in sent and "task_data" in sent
    assert sent["task_data"]["task_ids"] == ["A"]


async def test_solve_invalid_vrp_conversion_error(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a high-level VRP payload whose customers have no coordinates
    patch_cuopt_client(monkeypatch)

    # WHEN solving THEN the conversion failure is an actionable error dict
    result = await solve_with_cuopt_deployment(
        data={"depot": {}, "customers": [{"id": "A"}]}, preview_only=False
    )
    assert result["status"] == "error"
    assert "Failed to convert VRPData" in result["error"]


async def test_solve_solver_failure_builds_error_response(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN the deployment reports an HTTP-level failure with cuOpt details
    patch_cuopt_client(
        monkeypatch,
        solve_result={
            "status": "error",
            "error": "HTTP 400",
            "details": {"error": "Invalid constraint bounds"},
        },
    )

    # WHEN solving
    result = await solve_with_cuopt_deployment(data=sample_milp_data(), preview_only=False)

    # THEN a structured error response is built and nothing is persisted
    assert result["status"] == "error"
    assert "Invalid constraint bounds" in result["error"]
    assert "hint" in result
    assert await store.list(source="staging") == []


async def test_solve_non_json_panel_rejected(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a panel id pointing at a Dataset (not Json) panel
    patch_cuopt_client(monkeypatch)
    dataset_panel = await store.create(
        Dataset(title="not json", row_count=0, columns=[]), source="staging"
    )

    # WHEN solving THEN the type mismatch is an actionable error
    result = await solve_with_cuopt_deployment(data=dataset_panel.id)
    assert result["status"] == "error"
    assert "needs a Json panel" in result["error"]


async def test_solve_missing_panel_errors(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a panel id that does not exist
    patch_cuopt_client(monkeypatch)

    # WHEN solving THEN the load failure is an actionable error
    result = await solve_with_cuopt_deployment(data="nope-1")
    assert result["status"] == "error"
    assert "Failed to load Json panel" in result["error"]


async def test_solve_unsupported_data_type(store: PanelStore) -> None:
    # GIVEN a payload that is neither a dict nor a panel id
    # WHEN solving THEN the type is rejected with a hint
    result = await solve_with_cuopt_deployment(data=[1, 2, 3])  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "Unsupported data type" in result["error"]
