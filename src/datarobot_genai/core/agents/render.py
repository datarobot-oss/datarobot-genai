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

"""Pure string rendering for AG-UI event types (no I/O)."""

from __future__ import annotations

import logging

from ag_ui.core import Event
from ag_ui.core.events import EventType
from colorama import Fore
from colorama import Style

logger = logging.getLogger(__name__)

TOOL_RESULT_MAX_LEN = 1000


def render_ag_ui_event(event: Event) -> str | None:
    """Render an AG-UI event.

    Parameters
    ----------
    event:
        The AG-UI event to render.
    """
    event_type = event.type

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
    return rendered


def render_event(
    event_type: EventType,
    *,
    delta: str = "",
    name: str = "",
    content: str = "",
    message: str = "",
) -> str | None:
    """Build a terminal-friendly string for a single event, or None to skip.

    Parameters
    ----------
    event_type:
        The AG-UI :class:`~ag_ui.core.events.EventType` for this event.
    delta:
        Text delta for content/args events.
    name:
        Name for tool-call-start, step-started, or custom events.
    content:
        Content for tool-call-result events.
    message:
        Error message for run-error events.

    Returns
    -------
        A terminal friendly string if the event was rendered, otherwise None.
        Print with print(rendered, end='') or click.echo(rendered, nl=False)
    """
    rendered: str | None = None
    if event_type in (EventType.TEXT_MESSAGE_CONTENT, EventType.TEXT_MESSAGE_CHUNK):
        rendered = f"{Fore.CYAN}{delta}{Style.RESET_ALL}"
    elif event_type == EventType.TEXT_MESSAGE_END:
        rendered = "\n"
    elif event_type == EventType.TEXT_MESSAGE_START:
        rendered = None
    elif event_type in (EventType.REASONING_MESSAGE_CONTENT, EventType.REASONING_MESSAGE_CHUNK):
        rendered = f"{Fore.YELLOW}{delta}{Style.RESET_ALL}"
    elif event_type in (EventType.REASONING_END, EventType.REASONING_MESSAGE_END):
        rendered = "\n"
    elif event_type in (EventType.REASONING_START, EventType.REASONING_MESSAGE_START):
        rendered = None
    elif event_type == EventType.TOOL_CALL_START:
        rendered = (
            f"\n{Fore.MAGENTA}\u25b6 Tool Call: {Fore.MAGENTA}{Style.DIM}{name}{Style.RESET_ALL}"
            f"\n{Fore.MAGENTA}  Arguments: {Style.RESET_ALL}"
        )
    elif event_type == EventType.TOOL_CALL_ARGS:
        rendered = f"{Fore.MAGENTA}{Style.DIM}{delta}{Style.RESET_ALL}"
    elif event_type == EventType.TOOL_CALL_END:
        rendered = "\n"
    elif event_type == EventType.TOOL_CALL_RESULT:
        if len(content) > TOOL_RESULT_MAX_LEN:
            content = content[:TOOL_RESULT_MAX_LEN] + "\u2026"
        rendered = f"{Fore.MAGENTA}  Result: {Fore.MAGENTA}{Style.DIM}{content}{Style.RESET_ALL}\n"
    elif event_type == EventType.STEP_STARTED:
        rendered = f"{Style.DIM}Step: {name}{Style.RESET_ALL}\n"
    elif event_type == EventType.STEP_FINISHED:
        rendered = None
    elif event_type == EventType.RUN_STARTED:
        rendered = f"{Style.DIM}Run started{Style.RESET_ALL}\n"
    elif event_type == EventType.RUN_FINISHED:
        rendered = f"\n{Fore.GREEN}\u2705 Run finished.{Style.RESET_ALL}\n"
    elif event_type == EventType.RUN_ERROR:
        rendered = f"{Fore.RED}\u274c Run failed: {message}{Style.RESET_ALL}\n"
    elif event_type == EventType.CUSTOM:
        if name != "Heartbeat":
            rendered = f"{Style.DIM}[{name}]{Style.RESET_ALL}\n"
    else:
        logger.debug(f"Unhandled event type: {event_type.value}")

    return rendered
