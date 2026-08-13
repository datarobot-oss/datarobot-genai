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

"""Sandbox-backed chart panels.

``create_chart_panel`` runs user Python in the isolated DataRobot workload
sandbox (``execute_code``) over a source panel — a Dataset bound as a polars
DataFrame ``df``, or a Json panel bound as ``data`` — and stores the result as a
Chart panel. Three charting libraries are supported, each with its own
stored-blob shape (``content_type="application/json"``, passed through verbatim
by the consuming app's ``/blob`` endpoint to the BPA frontend's ``ChartPayload``
union in ``frontend_web/src/types/chartData.ts``):

==============  ===========================================  =================
chart_library   stored blob                                  ``_return`` is
==============  ===========================================  =================
``plotly``      ``{"format": "plotly", "spec": <fig-json>}``  figure-JSON dict
``altair``      ``{"format": "altair", "spec": <vl-spec>}``   Vega-Lite dict
``folium``      ``{"format": "html", "html": "<html>…"}``     HTML string
==============  ===========================================  =================

plotly and altair need no charting package in the sandbox — both formats are
plain JSON the code assigns directly. folium does: leaflet HTML is not
hand-authorable, so the sandbox image must ship ``folium`` (point
``DR_MCP_SANDBOX_IMAGE`` at a chart-capable image). Rendering deliberately stays
*inside* the sandbox rather than in this process: folium's expressive surface for
real choropleths is ``style_function``/``highlight_function`` Python callables,
which no JSON spec can carry, and ``Map.render()`` is a Jinja2 expansion that
materializes the whole feature set in memory — both belong in a capped,
disposable container, not in a shared MCP server.

Ported from wren-mcp's ``app/tools/facades_heavy.py`` (MODEL-24091), including
the altair/folium restoration from wren-mcp #101.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Annotated
from typing import Any

import polars as pl

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.panels.access import _get_store
from datarobot_genai.drmcputils.panels.access import _require_mcp_sandbox
from datarobot_genai.drmcputils.panels.models import BasePanel
from datarobot_genai.drmcputils.panels.models import Chart
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.models import Json
from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.sandbox import execute_code as _execute_code

logger = logging.getLogger(__name__)

_JSON_CONTENT_TYPE = "application/json"

# Unlike the other panel tools (which default to DEFAULT_SOURCE, 'main'), charts
# default to the session-scoped staging area — preserved from the wren-mcp
# original so the BPA facade delegation keeps its behavior.
_STAGING_SOURCE = "staging"

# A Dataset source is bound as a polars DataFrame `df`; user code must assign
# the chart result (see `_CHART_FORMATS`) to `_return`. Neither plotly nor altair
# is installed in the execution environment — both formats are plain JSON.
_CHART_PREAMBLE = "import polars as pl\ndf = pl.DataFrame(inputs['rows'])"

# A Json source (e.g. the GeoJSON a folium map layers) is bound as `data`. The
# panel's payload lives inline on the model, so nothing is read from the blob.
_JSON_CHART_PREAMBLE = "data = inputs['data']"

# `chart_library` -> the `format` discriminator of the stored blob. Frozen BPA
# frontend contract: frontend_web/src/types/chartData.ts dispatches ChartPanel
# on this value to PlotlyChart / AltairChart / FoliumChart.
_CHART_FORMATS = {"plotly": "plotly", "altair": "altair", "folium": "html"}

# Top-level keys that identify a Vega-Lite spec. A single-view spec carries
# `mark`; composite specs use one of the operators; `$schema` covers anything
# altair's `.to_dict()` emits that the others miss.
_VEGA_LITE_KEYS = frozenset(
    {
        "$schema",
        "mark",
        "encoding",
        "layer",
        "hconcat",
        "vconcat",
        "concat",
        "facet",
        "repeat",
    }
)


@tool_metadata(
    tags={"panels", "write", "chart", "sandbox", "daria"},
    description=(
        "[Panels—chart] Execute Python in the sandbox to build a chart from a source panel "
        "and save it as a Chart panel. A Dataset source is bound as a polars DataFrame `df`; "
        "a Json source is bound as `data` (use a Json panel for GeoJSON that a folium map "
        "layers). Assign the result to `_return`; what it must be depends on `chart_library`:\n"
        "- 'plotly' (default): figure JSON as a plain dict, e.g. "
        "`_return = {'data': [...], 'layout': {...}}`. The plotly package is NOT installed — "
        "do not `import plotly`; build the trace and layout dicts directly.\n"
        "- 'altair': a Vega-Lite spec as a plain dict, e.g. "
        "`_return = {'mark': 'bar', 'encoding': {...}, 'data': {'values': [...]}}`. The altair "
        "package is NOT installed — do not `import altair`; write the spec directly.\n"
        "- 'folium': the map's HTML as a string. folium IS installed for this library: "
        "`import folium; m = folium.Map(...); _return = m.get_root().render()`. Use "
        "`get_root().render()`, not `_repr_html_()` — the app already frames the HTML in an "
        "iframe, so `_repr_html_()` would nest a second one.\n"
        "Build values from plain Python lists (e.g. df['x'].to_list()), not numpy arrays, so "
        "the result is JSON-serializable, and make the chart dark-mode compatible."
    ),
    display_name="Panels — Create chart",
    description_ui=(
        "Runs user-supplied Python charting code over a dataset or JSON panel in the sandbox "
        "and saves the resulting plotly, altair, or folium chart as a chart panel."
    ),
)
async def create_chart_panel(
    *,
    panel_id: Annotated[str, "Source panel to plot — a Dataset (bound as `df`) or Json (`data`)."],
    code: Annotated[str, "Python operating on `df`/`data`; assign the chart to `_return`."],
    title: Annotated[str, "Title for the resulting Chart panel."],
    description: Annotated[str | None, "Optional description for the panel."] = None,
    source: Annotated[str, "Target source ('main' or 'staging')."] = _STAGING_SOURCE,
    chart_library: Annotated[str, "One of 'plotly' (default), 'altair', 'folium'."] = "plotly",
) -> dict[str, Any]:
    """Run charting code over a source panel in the sandbox and save a Chart panel.

    A Dataset source's rows are bound as a polars DataFrame ``df``; a Json
    source's inline payload is bound as ``data``. Your ``code`` must assign the
    chart to the magic ``_return`` variable, in the shape ``chart_library``
    dictates — a plotly figure-JSON dict, a Vega-Lite spec dict, or (folium) the
    rendered HTML string::

        # chart_library='plotly'
        _return = {
            'data': [{'type': 'bar', 'x': df['region'].to_list(), 'y': df['rev'].to_list()}],
            'layout': {'title': {'text': 'Revenue by region'}},
        }

        # chart_library='altair'
        _return = {
            'mark': 'bar',
            'data': {'values': [{'region': r, 'rev': v} for r, v in zip(...)]},
            'encoding': {'x': {'field': 'region', 'type': 'nominal'},
                         'y': {'field': 'rev', 'type': 'quantitative'}},
        }

        # chart_library='folium'  (a Json source's GeoJSON bound as `data`)
        import folium
        m = folium.Map(location=[40.7, -74.0], zoom_start=10)
        folium.GeoJson(data).add_to(m)
        _return = m.get_root().render()

    Neither ``plotly`` nor ``altair`` is installed in the execution environment
    (workload image or local) — ``import`` fails with ``ModuleNotFoundError``.
    Both formats are plain JSON, so construct the dict directly. ``folium`` *is*
    required for the folium library and must be present in the sandbox image; if
    it is missing the error names ``DR_MCP_SANDBOX_IMAGE``. Build values from
    plain lists (``df[col].to_list()``) rather than numpy arrays so the result is
    JSON-serializable; a result that is not comes back empty and raises an
    actionable error.
    """
    _require_mcp_sandbox()
    if not panel_id:
        raise ToolError("panel_id must be provided", kind=ToolErrorKind.VALIDATION)
    if not code:
        raise ToolError("code must be provided", kind=ToolErrorKind.VALIDATION)
    if not title:
        raise ToolError("title must be provided", kind=ToolErrorKind.VALIDATION)
    if chart_library not in _CHART_FORMATS:
        raise ToolError(
            f"Unsupported chart_library {chart_library!r}; expected one of "
            f"{', '.join(sorted(_CHART_FORMATS))}.",
            kind=ToolErrorKind.VALIDATION,
        )

    store = _get_store()
    source_panel = await store.get(panel_id)
    preamble, inputs = await _bind_chart_source(store, source_panel, panel_id)

    result = await _execute_code(f"{preamble}\n{code}", inputs=inputs)
    payload = json.dumps(_build_chart_payload(chart_library, result)).encode("utf-8")

    panel = Chart(
        title=title,
        description=description,
        parents=[panel_id],
        chart_library=chart_library,
        execution_context={"kind": "chart", "code": code},
    )
    created = await store.create(
        panel,
        source=source,
        payload=payload,
        payload_name=f"{title}.json",
        content_type=_JSON_CONTENT_TYPE,
    )
    return created.model_dump(mode="json")


async def _bind_chart_source(
    store: PanelStore, source_panel: BasePanel, panel_id: str
) -> tuple[str, dict[str, Any]]:
    """Resolve a chart source panel to its sandbox preamble and bound inputs.

    Dataset -> Parquet payload decoded to JSON-safe rows, bound as ``df``.
    Json -> the panel's inline ``data`` mapping, bound as ``data`` (folium's
    GeoJSON input; the wren original accepted a JsonRef here for the same
    reason). Any other panel type has nothing chartable.
    """
    if isinstance(source_panel, Dataset):
        raw = await store.get_payload(source_panel)
        if raw is None:
            raise ToolError(
                f"Dataset panel {panel_id} has no stored payload to chart.",
                kind=ToolErrorKind.VALIDATION,
            )
        # Round-trip through polars' JSON writer instead of to_dicts(): to_dicts()
        # keeps native date/datetime/Decimal objects, which execute_code's plain
        # json.dumps(inputs) cannot encode; write_json coerces them to strings.
        rows = json.loads(pl.read_parquet(io.BytesIO(raw)).write_json())
        return _CHART_PREAMBLE, {"rows": rows}

    if isinstance(source_panel, Json):
        return _JSON_CHART_PREAMBLE, {"data": source_panel.data}

    raise ToolError(
        f"Panel {panel_id} is a {source_panel.type.value} panel; create_chart_panel "
        "supports Dataset panels (bound as `df`) and Json panels (bound as `data`).",
        kind=ToolErrorKind.VALIDATION,
    )


def _build_chart_payload(chart_library: str, result: dict[str, Any]) -> dict[str, Any]:
    """Validate the sandbox return value and wrap it in the stored-blob contract.

    Dispatches on ``chart_library`` to the shape the BPA frontend's
    ``ChartPayload`` union expects (see the module docstring). ``format`` is the
    discriminator the frontend switches on — note folium maps to ``"html"``,
    not ``"folium"``.
    """
    value = result.get("return_value")
    if value is None:
        raise _no_return_error(chart_library, result)

    fmt = _CHART_FORMATS[chart_library]
    if chart_library == "folium":
        if not isinstance(value, str) or not value.strip():
            raise ToolError(
                "folium chart code must return the map's HTML as a non-empty string; "
                "assign `_return = m.get_root().render()`. Got "
                f"{type(value).__name__}.",
                kind=ToolErrorKind.VALIDATION,
            )
        return {"format": fmt, "html": value}

    if not isinstance(value, dict):
        raise ToolError(
            f"{chart_library} chart code must return a spec dict; got {type(value).__name__}.",
            kind=ToolErrorKind.VALIDATION,
        )

    if chart_library == "plotly":
        if "data" not in value:
            raise ToolError(
                "plotly chart code must return a figure JSON dict (with a 'data' key); "
                "assign `_return = {'data': [...], 'layout': {...}}`.",
                kind=ToolErrorKind.VALIDATION,
            )
        # The frontend's Plotly component reads layout unconditionally.
        value.setdefault("layout", {})
    elif not _VEGA_LITE_KEYS & value.keys():
        raise ToolError(
            "altair chart code must return a Vega-Lite spec dict; expected at least one of "
            f"{', '.join(sorted(_VEGA_LITE_KEYS))}. Assign e.g. `_return = {{'mark': 'bar', "
            "'data': {'values': [...]}, 'encoding': {...}}}` — do not `import altair`, it is "
            "not installed.",
            kind=ToolErrorKind.VALIDATION,
        )

    return {"format": fmt, "spec": value}


def _no_return_error(chart_library: str, result: dict[str, Any]) -> ToolError:
    """Build the error for a sandbox run that produced no ``_return`` value.

    ``None`` means the code never assigned ``_return``, or assigned something the
    runner could not JSON-encode (it drops the value in that case). A missing
    ``folium`` is the one failure mode with a deployment fix rather than a code
    fix, so it gets called out by name.
    """
    stderr = (result.get("stderr") or "").strip()
    if chart_library == "folium" and "folium" in stderr and "ModuleNotFoundError" in stderr:
        return ToolError(
            "The configured sandbox image does not ship `folium`, which the folium chart "
            "library requires. Point DR_MCP_SANDBOX_IMAGE at a chart-capable sandbox image, "
            "or use chart_library='plotly' or 'altair' (neither needs a package in the "
            f"sandbox). Sandbox stderr: {stderr[-500:]}",
            kind=ToolErrorKind.UPSTREAM,
        )

    detail = f" Sandbox stderr: {stderr[-500:]}" if stderr else ""
    if chart_library == "folium":
        expected = "the map HTML string (`_return = m.get_root().render()`)"
    elif chart_library == "altair":
        expected = "a Vega-Lite spec dict (`_return = {'mark': ..., 'encoding': ...}`)"
    else:
        expected = "the figure JSON dict (`_return = {'data': [...], 'layout': {...}}`)"
    return ToolError(
        f"Chart code did not return a {chart_library} chart. Assign {expected} to `_return`. "
        "If the value is not JSON-serializable (e.g. built from numpy arrays), rebuild it from "
        f"plain Python lists (df[col].to_list()) so it serializes cleanly.{detail}",
        kind=ToolErrorKind.VALIDATION,
    )
