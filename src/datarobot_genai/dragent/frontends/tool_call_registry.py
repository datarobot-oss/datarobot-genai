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

"""Per-run handoff of LLM-issued ``tool_call_id`` from converter to adaptor.

Two-phase correlation tolerates parallel same-name calls completing out of
order: ``register`` (converter) appends to a per-name FIFO; ``bind`` (adaptor,
``FUNCTION_START``) pops the head into a UUID-keyed map; ``pop`` (adaptor,
``FUNCTION_END``) drains by UUID.

Deferred end events prevent ``ToolCallEndEvent`` from racing ahead of
``ToolCallArgsEvent`` chunks still being streamed by the converter.
"""

from __future__ import annotations

from collections import deque
from contextvars import ContextVar
from typing import Any

_state: ContextVar[tuple[dict[str, deque[str]], dict[str, str], set[str], dict[str, list[Any]]]] = (
    ContextVar("dragent_tc")
)


def _get() -> tuple[dict[str, deque[str]], dict[str, str], set[str], dict[str, list[Any]]]:
    try:
        return _state.get()
    except LookupError:
        s: tuple[dict[str, deque[str]], dict[str, str], set[str], dict[str, list[Any]]] = (
            {},
            {},
            set(),
            {},
        )
        _state.set(s)
        return s


def register_tool_call(name: str, tool_call_id: str) -> None:
    pending, _, _, _ = _get()
    pending.setdefault(name, deque()).append(tool_call_id)


def bind_tool_call(name: str, nat_uuid: str) -> str | None:
    pending, bound, _, _ = _get()
    queue = pending.get(name)
    if not queue:
        return None
    bound[nat_uuid] = tc_id = queue.popleft()
    return tc_id


def pop_tool_call(nat_uuid: str) -> str | None:
    return _get()[1].pop(nat_uuid, None)


def mark_args_done(tool_call_id: str) -> list[Any]:
    """Mark argument streaming complete for *tool_call_id*.

    Returns (and removes) any end/result events that the step adaptor
    deferred while arguments were still in flight.
    """
    _, _, args_done, deferred = _get()
    args_done.add(tool_call_id)
    return deferred.pop(tool_call_id, [])


def is_args_done(tool_call_id: str) -> bool:
    """Check whether the stream converter has finished emitting args for this call."""
    return tool_call_id in _get()[2]


def defer_tool_end(tool_call_id: str, events: list[Any]) -> None:
    """Stash end/result events until argument streaming finishes."""
    _get()[3][tool_call_id] = events


def reset() -> None:
    _state.set(({}, {}, set(), {}))
