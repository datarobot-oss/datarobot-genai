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

The LLM stream and the intermediate-step stream see different ids for the
same call (provider id vs. NAT ``payload.UUID``); ``ToolCallResult`` must
bind to the LLM's id. FIFO by name; parallel same-name calls finishing
out of dispatch order would correlate incorrectly.
"""

from __future__ import annotations

from collections import deque
from contextvars import ContextVar

_pending: ContextVar[dict[str, deque[str]] | None] = ContextVar(
    "dragent_tool_call_pending", default=None
)


def register_tool_call(name: str, tool_call_id: str) -> None:
    """Record ``tool_call_id`` for the next ``FUNCTION_END`` of ``name``."""
    pending = _pending.get()
    if pending is None:
        pending = {}
        _pending.set(pending)
    pending.setdefault(name, deque()).append(tool_call_id)


def pop_tool_call(name: str) -> str | None:
    """Pop the next pending ``tool_call_id`` for ``name``, or ``None``."""
    pending = _pending.get()
    if pending is None:
        return None
    queue = pending.get(name)
    if not queue:
        return None
    tool_call_id = queue.popleft()
    if not queue:
        pending.pop(name, None)
    return tool_call_id


def reset() -> None:
    """Clear pending state for the current context. For tests."""
    _pending.set({})
