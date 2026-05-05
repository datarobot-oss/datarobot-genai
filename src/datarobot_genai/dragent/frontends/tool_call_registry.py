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
"""

from __future__ import annotations

from collections import deque
from contextvars import ContextVar

_state: ContextVar[tuple[dict[str, deque[str]], dict[str, str]]] = ContextVar("dragent_tc")


def _get() -> tuple[dict[str, deque[str]], dict[str, str]]:
    try:
        return _state.get()
    except LookupError:
        s: tuple[dict[str, deque[str]], dict[str, str]] = ({}, {})
        _state.set(s)
        return s


def register_tool_call(name: str, tool_call_id: str) -> None:
    pending, _ = _get()
    pending.setdefault(name, deque()).append(tool_call_id)


def bind_tool_call(name: str, nat_uuid: str) -> str | None:
    pending, bound = _get()
    queue = pending.get(name)
    if not queue:
        return None
    bound[nat_uuid] = tc_id = queue.popleft()
    return tc_id


def pop_tool_call(nat_uuid: str) -> str | None:
    return _get()[1].pop(nat_uuid, None)


def reset() -> None:
    _state.set(({}, {}))
