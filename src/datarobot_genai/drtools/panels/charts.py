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
sandbox (``execute_code``) over a source Dataset panel bound as a polars
DataFrame ``df``, and stores the resulting Plotly figure JSON as a Chart panel.
The stored blob is a **frozen BPA frontend contract** — ``{"format": "plotly",
"spec": <figure-json>}`` with ``content_type="application/json"`` — which the
consuming app's ``/blob`` endpoint passes through verbatim to the frontend.
Do not alter that payload shape.

Ported from wren-mcp's ``app/tools/facades_heavy.py`` (MODEL-24091).
"""

from __future__ import annotations

import io
import json
import logging
from typing import Annotated
from typing import Any
from typing import cast

import polars as pl

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.panels.access import _get_store
from datarobot_genai.drmcputils.panels.access import _require_mcp_sandbox
from datarobot_genai.drmcputils.panels.models import Chart
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.sandbox import execute_code as _execute_code

logger = logging.getLogger(__name__)

_JSON_CONTENT_TYPE = "application/json"

# Unlike the other panel tools (which default to DEFAULT_SOURCE, 'main'), charts
# default to the session-scoped staging area — preserved from the wren-mcp
# original so the BPA facade delegation keeps its behavior.
_STAGING_SOURCE = "staging"

# The source dataset is bound as a polars DataFrame `df`; user code must assign
# the Plotly figure JSON (fig.to_plotly_json()) to `_return`.
_CHART_PREAMBLE = "import polars as pl\ndf = pl.DataFrame(inputs['rows'])"


@tool_metadata(
    tags={"panels", "write", "chart", "sandbox", "daria"},
    description=(
        "[Panels—chart] Execute Python in the sandbox to build a chart from a Dataset panel "
        "and save it as a Chart panel. The source dataset is bound as a polars DataFrame `df`; "
        "build a Plotly figure and assign its JSON to `_return`, e.g. "
        "`_return = fig.to_plotly_json()`. Build the figure from plain Python lists "
        "(e.g. df['x'].to_list()) so the result is JSON-serializable, and make it "
        "dark-mode compatible."
    ),
    display_name="Panels — Create chart",
    description_ui=(
        "Runs user-supplied Python charting code over a dataset panel in the sandbox and "
        "saves the resulting Plotly figure as a chart panel."
    ),
)
async def create_chart_panel(
    *,
    panel_id: Annotated[str, "Source Dataset panel to plot."],
    code: Annotated[str, "Python operating on `df`; assign the Plotly figure JSON to `_return`."],
    title: Annotated[str, "Title for the resulting Chart panel."],
    description: Annotated[str | None, "Optional description for the panel."] = None,
    source: Annotated[str, "Target source ('main' or 'staging')."] = _STAGING_SOURCE,
    chart_library: Annotated[str, "Charting library label stored on the panel."] = "plotly",
) -> dict[str, Any]:
    """Run charting code over a Dataset panel in the sandbox and save a Chart panel.

    The source Dataset panel's rows are bound as a polars DataFrame ``df`` inside
    the sandbox. Your ``code`` must assign the Plotly figure's JSON to the magic
    ``_return`` variable, e.g.::

        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(x=df['region'].to_list(), y=df['rev'].to_list()))
        _return = fig.to_plotly_json()

    Build traces from plain lists (``df[col].to_list()``) rather than numpy
    arrays so the figure JSON is serializable; a figure that is not
    JSON-serializable comes back empty and raises an actionable error.
    """
    _require_mcp_sandbox()
    if not panel_id:
        raise ToolError("panel_id must be provided", kind=ToolErrorKind.VALIDATION)
    if not code:
        raise ToolError("code must be provided", kind=ToolErrorKind.VALIDATION)
    if not title:
        raise ToolError("title must be provided", kind=ToolErrorKind.VALIDATION)

    store = _get_store()
    source_panel = await store.get(panel_id)
    if not isinstance(source_panel, Dataset):
        raise ToolError(
            f"Panel {panel_id} is a {source_panel.type.value} panel; create_chart_panel "
            "only supports Dataset panels.",
            kind=ToolErrorKind.VALIDATION,
        )
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

    result = await _execute_code(f"{_CHART_PREAMBLE}\n{code}", inputs={"rows": rows})
    figure = result.get("return_value")
    _validate_figure(figure, result)

    spec = cast(dict[str, Any], figure)
    spec.setdefault("layout", {})
    # FROZEN BPA frontend contract: {"format": "plotly", "spec": <figure-json>}
    # stored as application/json. Do not alter this shape.
    payload = json.dumps({"format": "plotly", "spec": spec}).encode("utf-8")

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


def _validate_figure(figure: Any, result: dict[str, Any]) -> None:
    """Validate the sandbox return value is a Plotly figure JSON dict.

    A serialized Plotly figure is a dict carrying a ``data`` list (and usually a
    ``layout`` dict). When the code never assigns ``_return`` — or the figure is
    not JSON-serializable, in which case the sandbox runner drops it — the
    return value comes back ``None``; surface an actionable error telling the
    agent to build the figure from plain lists.
    """
    if figure is None:
        detail = ""
        stderr = (result.get("stderr") or "").strip()
        if stderr:
            detail = f" Sandbox stderr: {stderr[-500:]}"
        raise ToolError(
            "Chart code did not return a Plotly figure. Assign the figure JSON to `_return`, "
            "e.g. `_return = fig.to_plotly_json()`. If the figure is not JSON-serializable "
            "(e.g. built from numpy arrays), rebuild the traces from plain Python lists "
            f"(df[col].to_list()) so it serializes cleanly.{detail}",
            kind=ToolErrorKind.VALIDATION,
        )
    if not isinstance(figure, dict) or "data" not in figure:
        raise ToolError(
            "Chart code must return a Plotly figure JSON dict (with a 'data' key); "
            "assign `_return = fig.to_plotly_json()`. Got "
            f"{type(figure).__name__}.",
            kind=ToolErrorKind.VALIDATION,
        )
