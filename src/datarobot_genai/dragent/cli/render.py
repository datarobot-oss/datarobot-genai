# Copyright 2026 DataRobot, Inc. and its affiliates.
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

"""Shared terminal rendering for AG-UI events.

Both ``remote.py`` (SSE JSON dicts) and ``console.py`` (typed AG-UI objects)
need identical rendering logic.  Each caller normalises its event into a
canonical ``(event_type, fields)`` pair and calls :func:`render_event`.
"""

from __future__ import annotations

import logging

import click
from ag_ui.core import Event
from ag_ui.core import EventType

from datarobot_genai.core.agents.render import render_event

logger = logging.getLogger(__name__)


def render_sse_event(ev: dict[str, object]) -> str | None:
    """Render an SSE JSON dict event. Returns the run-error message if RUN_ERROR, else None."""
    raw_type = str(ev.get("type", ""))
    if not raw_type:
        logger.debug("Unhandled SSE event type: %s", raw_type)
        return None
    try:
        event_type = EventType(raw_type)
    except ValueError:
        logger.debug("Unknown SSE event type: %s", raw_type)
        return None

    if event_type == EventType.RUN_ERROR:
        return str(ev.get("message", "Unknown error"))

    rendered = render_event(
        event_type,
        delta=str(ev.get("delta", "")),
        name=str(ev.get("tool_call_name", "") or ev.get("step_name", "") or ev.get("name", "")),
        content=str(ev.get("content", "")),
        message=str(ev.get("message", "Unknown error")),
    )
    if rendered is not None:
        click.echo(rendered, nl=False)
    return None


def render_object_event(event: Event) -> bool:
    """Render a typed AG-UI event object.

    Returns True only when a streaming assistant text or reasoning content/chunk
    delta was printed, so the console can tell whether to skip the final result
    dump (see :meth:`DRAgentConsoleFrontEndPlugin.run_workflow`).
    """
    event_type = event.type

    # The console frontend prints its own "Run finished" message after the
    # workflow completes, so skip it here to avoid duplicate output.
    if event_type in (EventType.RUN_FINISHED, EventType.RUN_ERROR):
        return False

    delta = str(getattr(event, "delta", "") or "")
    name = str(
        getattr(event, "tool_call_name", "")
        or getattr(event, "step_name", "")
        or getattr(event, "name", "")
        or ""
    )
    content = str(getattr(event, "content", "") or "")
    message = str(getattr(event, "message", "Unknown error") or "")

    rendered = render_event(
        event_type,
        delta=delta,
        name=name,
        content=content,
        message=message,
    )
    if rendered is not None:
        click.echo(rendered, nl=False)

    return event_type in (
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_CHUNK,
        EventType.REASONING_MESSAGE_CONTENT,
        EventType.REASONING_MESSAGE_CHUNK,
    ) and bool(delta)
