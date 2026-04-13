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
from colorama import Fore
from colorama import Style

logger = logging.getLogger(__name__)

TOOL_RESULT_MAX_LEN = 1000

# Canonical event type strings (used as the key for rendering).
TEXT_CONTENT = "TEXT_CONTENT"
TEXT_END = "TEXT_END"
TEXT_START = "TEXT_START"
REASONING_CONTENT = "REASONING_CONTENT"
REASONING_END = "REASONING_END"
REASONING_START = "REASONING_START"
TOOL_CALL_START = "TOOL_CALL_START"
TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
TOOL_CALL_END = "TOOL_CALL_END"
TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
STEP_STARTED = "STEP_STARTED"
STEP_FINISHED = "STEP_FINISHED"
RUN_STARTED = "RUN_STARTED"
RUN_FINISHED = "RUN_FINISHED"
RUN_ERROR = "RUN_ERROR"
CUSTOM = "CUSTOM"


def render_event(
    event_type: str,
    *,
    delta: str = "",
    name: str = "",
    content: str = "",
    message: str = "",
    err: bool = True,
) -> None:
    """Render a single AG-UI event to the terminal.

    Parameters
    ----------
    event_type:
        One of the canonical constants defined in this module.
    delta:
        Text delta for content/args events.
    name:
        Name for tool-call-start, step-started, or custom events.
    content:
        Content for tool-call-result events.
    message:
        Error message for run-error events.
    err:
        If True, write to stderr (default). Set to False to write to stdout.
    """
    if event_type == TEXT_CONTENT:
        click.echo(f"{Fore.CYAN}{delta}{Style.RESET_ALL}", nl=False, err=err)
    elif event_type == TEXT_END:
        click.echo("", err=err)
    elif event_type == TEXT_START:
        pass
    elif event_type == REASONING_CONTENT:
        click.echo(f"{Fore.YELLOW}{delta}{Style.RESET_ALL}", nl=False, err=True)
    elif event_type == REASONING_END:
        click.echo("", err=True)
    elif event_type == REASONING_START:
        pass
    elif event_type == TOOL_CALL_START:
        click.echo(
            f"\n{Fore.MAGENTA}\u25b6 Tool Call: {Fore.MAGENTA}{Style.DIM}{name}{Style.RESET_ALL}",
            err=True,
        )
        click.echo(
            f"{Fore.MAGENTA}  Arguments: {Style.RESET_ALL}",
            nl=False,
            err=True,
        )
    elif event_type == TOOL_CALL_ARGS:
        click.echo(
            f"{Fore.MAGENTA}{Style.DIM}{delta}{Style.RESET_ALL}",
            nl=False,
            err=True,
        )
    elif event_type == TOOL_CALL_END:
        click.echo("", err=True)
    elif event_type == TOOL_CALL_RESULT:
        if len(content) > TOOL_RESULT_MAX_LEN:
            content = content[:TOOL_RESULT_MAX_LEN] + "\u2026"
        click.echo(
            f"{Fore.MAGENTA}  Result: {Fore.MAGENTA}{Style.DIM}{content}{Style.RESET_ALL}\n",
            err=True,
        )
    elif event_type == STEP_STARTED:
        click.echo(f"{Style.DIM}Step: {name}{Style.RESET_ALL}", err=True)
    elif event_type == STEP_FINISHED:
        pass
    elif event_type == RUN_STARTED:
        click.echo(f"{Style.DIM}Run started{Style.RESET_ALL}", err=True)
    elif event_type == RUN_FINISHED:
        click.echo(f"\n{Fore.GREEN}\u2705 Run finished.{Style.RESET_ALL}")
    elif event_type == RUN_ERROR:
        pass  # Caller handles this (raise ClickException / break loop).
    elif event_type == CUSTOM:
        if name != "Heartbeat":
            click.echo(f"{Style.DIM}[{name}]{Style.RESET_ALL}", err=True)
    else:
        logger.debug("Unhandled event type: %s", event_type)


# ---------------------------------------------------------------------------
# Mapping helpers for the two calling conventions
# ---------------------------------------------------------------------------

# SSE JSON dict "type" string -> canonical event type
_SSE_TYPE_MAP: dict[str, str] = {
    "TEXT_MESSAGE_CONTENT": TEXT_CONTENT,
    "TEXT_MESSAGE_CHUNK": TEXT_CONTENT,
    "TEXT_MESSAGE_END": TEXT_END,
    "TEXT_MESSAGE_START": TEXT_START,
    "REASONING_MESSAGE_CONTENT": REASONING_CONTENT,
    "REASONING_MESSAGE_END": REASONING_END,
    "REASONING_START": REASONING_START,
    "REASONING_MESSAGE_START": REASONING_START,
    "REASONING_END": REASONING_END,
    "TOOL_CALL_START": TOOL_CALL_START,
    "TOOL_CALL_ARGS": TOOL_CALL_ARGS,
    "TOOL_CALL_END": TOOL_CALL_END,
    "TOOL_CALL_RESULT": TOOL_CALL_RESULT,
    "STEP_STARTED": STEP_STARTED,
    "STEP_FINISHED": STEP_FINISHED,
    "RUN_STARTED": RUN_STARTED,
    "RUN_FINISHED": RUN_FINISHED,
    "RUN_ERROR": RUN_ERROR,
    "CUSTOM": CUSTOM,
}

# Typed AG-UI object class name -> canonical event type
_OBJECT_TYPE_MAP: dict[str, str] = {
    "TextMessageContentEvent": TEXT_CONTENT,
    "TextMessageEndEvent": TEXT_END,
    "TextMessageStartEvent": TEXT_START,
    "ReasoningMessageContentEvent": REASONING_CONTENT,
    "ReasoningMessageEndEvent": REASONING_END,
    "ReasoningStartEvent": REASONING_START,
    "ReasoningMessageStartEvent": REASONING_START,
    "ReasoningEndEvent": REASONING_END,
    "ToolCallStartEvent": TOOL_CALL_START,
    "ToolCallArgsEvent": TOOL_CALL_ARGS,
    "ToolCallEndEvent": TOOL_CALL_END,
    "ToolCallResultEvent": TOOL_CALL_RESULT,
    "StepStartedEvent": STEP_STARTED,
    "StepFinishedEvent": STEP_FINISHED,
    "RunStartedEvent": RUN_STARTED,
    "RunFinishedEvent": RUN_FINISHED,
    "RunErrorEvent": RUN_ERROR,
    "CustomEvent": CUSTOM,
}


def render_sse_event(ev: dict[str, object]) -> str | None:
    """Render an SSE JSON dict event. Returns the run-error message if RUN_ERROR, else None.

    Text and run-lifecycle events go to stdout; everything else to stderr.
    This matches the original ``remote.py`` behaviour where the agent's text
    reply is the primary (pipe-able) output.
    """
    raw_type = str(ev.get("type", ""))
    canonical = _SSE_TYPE_MAP.get(raw_type)
    if canonical is None:
        logger.debug("Unhandled SSE event type: %s", raw_type)
        return None

    if canonical == RUN_ERROR:
        return str(ev.get("message", "Unknown error"))

    # Text content and run finished go to stdout (pipe-friendly); rest to stderr.
    err = canonical not in (TEXT_CONTENT, TEXT_END, TEXT_START, RUN_FINISHED)

    render_event(
        canonical,
        delta=str(ev.get("delta", "")),
        name=str(ev.get("tool_call_name", "") or ev.get("step_name", "") or ev.get("name", "")),
        content=str(ev.get("content", "")),
        err=err,
    )
    return None


def render_object_event(event: object) -> bool:
    """Render a typed AG-UI event object. Returns True if text was emitted."""
    class_name = type(event).__name__
    canonical = _OBJECT_TYPE_MAP.get(class_name)
    if canonical is None:
        logger.debug("Unhandled object event type: %s", class_name)
        return False

    delta = getattr(event, "delta", "")
    name = (
        getattr(event, "tool_call_name", "")
        or getattr(event, "step_name", "")
        or getattr(event, "name", "")
    )
    content = getattr(event, "content", "")

    render_event(
        canonical,
        delta=str(delta) if delta else "",
        name=str(name) if name else "",
        content=str(content) if content else "",
        err=True,
    )
    return canonical in (TEXT_CONTENT, REASONING_CONTENT) and bool(delta)
