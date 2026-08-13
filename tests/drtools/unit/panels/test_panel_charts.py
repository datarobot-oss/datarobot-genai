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

"""Unit tests for the sandbox-backed chart panel tool (execute_code mocked).

The sandbox ``execute_code`` call is mocked to return a canned figure dict, so
these tests exercise the stored-payload convention (the frozen BPA frontend
contract ``{"format": "plotly", "spec": <figure-json>}``), lineage, and the
validation error paths without a container. Ported from wren-mcp's
``tests/test_facades_heavy.py`` chart cases.
"""

import datetime
import io
import json
from datetime import datetime as dtime
from typing import Any

import polars as pl
import pytest

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.models import Json
from datarobot_genai.drmcputils.panels.models import Text
from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drtools.panels import charts as charts_mod

from .conftest import FakeBlobStore

_PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> PanelStore:
    """Inject an in-memory store and skip the entitlement guard."""
    panel_store = PanelStore(FakeBlobStore())
    monkeypatch.setattr(charts_mod, "_require_mcp_sandbox", lambda: None)
    monkeypatch.setattr(charts_mod, "_get_store", lambda: panel_store)
    return panel_store


async def _make_dataset_panel(
    store: PanelStore,
    frame: pl.DataFrame,
    *,
    title: str = "src",
    source: str = "staging",
) -> str:
    buf = io.BytesIO()
    frame.write_parquet(buf)
    panel = Dataset(title=title, row_count=frame.height, columns=frame.columns)
    created = await store.create(
        panel,
        source=source,
        payload=buf.getvalue(),
        payload_name="src.parquet",
        content_type=_PARQUET_CONTENT_TYPE,
    )
    assert created.id is not None
    return created.id


def _canned_figure() -> dict[str, Any]:
    return {
        "data": [{"type": "bar", "x": ["a", "b"], "y": [1, 2]}],
        "layout": {"title": {"text": "Demo"}},
    }


def _patch_execute(
    monkeypatch: pytest.MonkeyPatch, return_value: Any, **extra: Any
) -> dict[str, Any]:
    """Patch charts_mod._execute_code to return a canned sandbox payload; record the call."""
    captured: dict[str, Any] = {}

    async def _fake(code: str, *, inputs: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        captured["code"] = code
        captured["inputs"] = inputs
        return {"stdout": "", "stderr": "", "return_value": return_value, **extra}

    monkeypatch.setattr(charts_mod, "_execute_code", _fake)
    return captured


async def test_create_chart_panel_stores_frozen_contract_payload(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a Dataset panel and a sandbox that returns a canned Plotly figure
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"x": [1, 2], "y": [3, 4]}))
    captured = _patch_execute(monkeypatch, _canned_figure())

    # WHEN creating a chart panel from it
    result = await charts_mod.create_chart_panel(
        panel_id=panel_id, code="_return = fig.to_plotly_json()", title="Bars"
    )

    # THEN the Chart panel carries lineage + execution context
    assert result["type"] == "chart"
    assert result["chart_library"] == "plotly"
    assert result["parents"] == [panel_id]
    assert result["execution_context"] == {
        "kind": "chart",
        "code": "_return = fig.to_plotly_json()",
    }
    # THEN the stored blob follows the frozen BPA frontend contract.
    payload = json.loads(await store.get_payload(result["id"]))
    assert payload == {"format": "plotly", "spec": _canned_figure()}
    # THEN content_type is application/json (blob metadata is the BlobRef; name .json).
    assert result["payload_name"] == "Bars.json"
    # THEN the source rows were bound for the sandbox under the chart preamble.
    assert captured["inputs"] == {"rows": [{"x": 1, "y": 3}, {"x": 2, "y": 4}]}
    assert captured["code"].startswith(charts_mod._CHART_PREAMBLE)


async def test_create_chart_panel_temporal_columns_are_json_safe(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a Dataset panel with native date/datetime columns; execute_code
    # serializes inputs with a bare json.dumps, so rows must arrive JSON-coerced
    # (to_dicts() would keep native temporal objects and crash it).
    panel_id = await _make_dataset_panel(
        store,
        pl.DataFrame(
            {
                "d": [datetime.date(2024, 6, 1)],
                "ts": [dtime(2024, 6, 1, 10, 30)],
                "v": [1.5],
            }
        ),
    )
    captured = _patch_execute(monkeypatch, _canned_figure())

    # WHEN creating a chart panel from it
    await charts_mod.create_chart_panel(panel_id=panel_id, code="_return = f", title="T")

    # THEN the rows bound for the sandbox are plain JSON values
    json.dumps(captured["inputs"])  # must not raise
    assert captured["inputs"]["rows"] == [
        {"d": "2024-06-01", "ts": "2024-06-01 10:30:00", "v": 1.5}
    ]


def _canned_vega_spec() -> dict[str, Any]:
    return {
        "mark": "bar",
        "data": {"values": [{"a": "x", "b": 1}]},
        "encoding": {
            "x": {"field": "a", "type": "nominal"},
            "y": {"field": "b", "type": "quantitative"},
        },
    }


async def test_create_chart_panel_altair_stores_vega_spec(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a Dataset panel and a sandbox returning a Vega-Lite spec
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"x": [1]}))
    _patch_execute(monkeypatch, _canned_vega_spec())

    # WHEN creating an altair chart panel
    result = await charts_mod.create_chart_panel(
        panel_id=panel_id, code="_return = spec", title="C", chart_library="altair"
    )

    # THEN the blob carries the altair discriminator and the spec verbatim
    assert result["chart_library"] == "altair"
    payload = json.loads(await store.get_payload(result["id"]))
    assert payload == {"format": "altair", "spec": _canned_vega_spec()}
    # Unlike plotly, an altair spec must NOT be given a synthetic `layout` key.
    assert "layout" not in payload["spec"]


@pytest.mark.parametrize(
    "spec",
    [
        {"mark": "bar"},
        {"layer": []},
        {"hconcat": []},
        {"$schema": "https://vega.github.io/schema/vega-lite/v5.json"},
        {"facet": {}, "spec": {}},
    ],
)
async def test_create_chart_panel_altair_accepts_composite_specs(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch, spec: dict[str, Any]
) -> None:
    # GIVEN composite Vega-Lite shapes altair's .to_dict() can emit
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"x": [1]}))
    _patch_execute(monkeypatch, spec)

    # WHEN/THEN each is accepted and stored unchanged
    result = await charts_mod.create_chart_panel(
        panel_id=panel_id, code="_return = spec", title="C", chart_library="altair"
    )
    payload = json.loads(await store.get_payload(result["id"]))
    assert payload == {"format": "altair", "spec": spec}


async def test_create_chart_panel_altair_rejects_non_vega_dict(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a plotly figure returned while claiming chart_library='altair' — a dict,
    # but not a Vega-Lite spec. Mislabelling must fail loudly rather than store an
    # unrenderable altair panel.
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"x": [1]}))
    _patch_execute(monkeypatch, {"data": [{"type": "bar"}], "layout": {}})

    # WHEN/THEN creating the chart panel raises naming Vega-Lite
    with pytest.raises(ToolError) as exc:
        await charts_mod.create_chart_panel(
            panel_id=panel_id, code="_return = f", title="C", chart_library="altair"
        )
    assert "Vega-Lite" in str(exc.value)


async def test_create_chart_panel_folium_stores_html_payload(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a sandbox returning rendered leaflet HTML
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"lat": [40.7]}))
    html = "<html><head></head><body><div id='map'></div></body></html>"
    _patch_execute(monkeypatch, html)

    # WHEN creating a folium chart panel
    result = await charts_mod.create_chart_panel(
        panel_id=panel_id,
        code="_return = m.get_root().render()",
        title="Map",
        chart_library="folium",
    )

    # THEN folium's discriminator is "html", NOT "folium" — the frontend switches on it
    assert result["chart_library"] == "folium"
    payload = json.loads(await store.get_payload(result["id"]))
    assert payload == {"format": "html", "html": html}


@pytest.mark.parametrize("value", [{"data": []}, "", "   ", 42])
async def test_create_chart_panel_folium_requires_html_string(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    # GIVEN a folium run returning anything but a non-empty HTML string
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"x": [1]}))
    _patch_execute(monkeypatch, value)

    # WHEN/THEN it raises, pointing at the correct render call
    with pytest.raises(ToolError) as exc:
        await charts_mod.create_chart_panel(
            panel_id=panel_id, code="_return = m", title="M", chart_library="folium"
        )
    assert "get_root().render()" in str(exc.value)


async def test_create_chart_panel_folium_missing_in_image_names_the_override(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a sandbox image without folium — the one failure mode with a deployment
    # fix rather than a code fix, so the error must point at DR_MCP_SANDBOX_IMAGE
    # instead of telling the agent to fix `code`.
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"x": [1]}))
    _patch_execute(
        monkeypatch,
        None,
        stderr="Traceback ...\nModuleNotFoundError: No module named 'folium'",
    )

    # WHEN/THEN the error names the image override
    with pytest.raises(ToolError) as exc:
        await charts_mod.create_chart_panel(
            panel_id=panel_id, code="import folium", title="M", chart_library="folium"
        )
    assert "DR_MCP_SANDBOX_IMAGE" in str(exc.value)


async def test_create_chart_panel_rejects_unknown_library(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a chart_library outside the supported set
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"x": [1]}))
    _patch_execute(monkeypatch, _canned_figure())

    # WHEN/THEN it is rejected by name rather than silently stored
    with pytest.raises(ToolError) as exc:
        await charts_mod.create_chart_panel(
            panel_id=panel_id, code="_return = f", title="C", chart_library="bokeh"
        )
    assert "bokeh" in str(exc.value)


async def test_create_chart_panel_defaults_missing_layout(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a sandbox figure without a 'layout' key
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"x": [1]}))
    _patch_execute(monkeypatch, {"data": [{"type": "scatter"}]})

    # WHEN creating the chart panel
    result = await charts_mod.create_chart_panel(
        panel_id=panel_id, code="_return = f", title="NoLayout"
    )

    # THEN the stored spec is defaulted to an empty layout dict
    payload = json.loads(await store.get_payload(result["id"]))
    assert payload["spec"]["layout"] == {}


async def test_create_chart_panel_none_return_raises(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a sandbox run where `_return` was never assigned (or the figure was
    # not JSON-serializable, in which case the runner drops it) -> None
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"x": [1]}))
    _patch_execute(monkeypatch, None, stderr="boom")

    # WHEN/THEN creating the chart panel raises an actionable error
    with pytest.raises(ToolError) as exc:
        await charts_mod.create_chart_panel(panel_id=panel_id, code="pass", title="X")
    assert "_return" in str(exc.value)


async def test_create_chart_panel_non_figure_raises(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a JSON-serializable but non-figure return (dict without 'data')
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"x": [1]}))
    _patch_execute(monkeypatch, {"not": "a figure"})

    # WHEN/THEN creating the chart panel raises mentioning the 'data' key
    with pytest.raises(ToolError) as exc:
        await charts_mod.create_chart_panel(panel_id=panel_id, code="_return = {}", title="X")
    assert "data" in str(exc.value)


async def test_create_chart_panel_non_dict_raises(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a non-dict sandbox return value
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"x": [1]}))
    _patch_execute(monkeypatch, [1, 2, 3])

    # WHEN/THEN creating the chart panel raises
    with pytest.raises(ToolError):
        await charts_mod.create_chart_panel(panel_id=panel_id, code="_return = []", title="X")


async def test_create_chart_panel_json_source_binds_data(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a Json source panel — folium's GeoJSON input path (the wren original
    # accepted a JsonRef here for the same reason)
    geojson = {"type": "FeatureCollection", "features": []}
    created = await store.create(Json(title="j", data=geojson), source="staging")
    assert created.id is not None
    captured = _patch_execute(monkeypatch, "<html></html>")

    # WHEN charting from it
    result = await charts_mod.create_chart_panel(
        panel_id=created.id,
        code="_return = m.get_root().render()",
        title="Map",
        chart_library="folium",
    )

    # THEN its inline data is bound as `data`, with no `df`/rows binding
    assert captured["inputs"] == {"data": geojson}
    assert captured["code"].startswith(charts_mod._JSON_CHART_PREAMBLE)
    assert result["parents"] == [created.id]


async def test_create_chart_panel_text_source_raises(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a source panel that is neither Dataset nor Json — nothing chartable to bind
    _patch_execute(monkeypatch, _canned_figure())
    created = await store.create(Text(title="t", text="hi"), source="staging")
    assert created.id is not None

    # WHEN/THEN creating the chart panel raises, naming both supported types
    with pytest.raises(ToolError) as exc:
        await charts_mod.create_chart_panel(panel_id=created.id, code="_return = f", title="X")
    assert "Dataset" in str(exc.value) and "Json" in str(exc.value)
