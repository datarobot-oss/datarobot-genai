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

"""Request-scoped caller token for the agent card registry and its L2 cache.

Inside :func:`registry_caller_token_scope`, the agent card registry and the L2
MemorySpace cache authenticate with the caller's token only -- never with
``DATAROBOT_API_TOKEN`` and with no fallback to it. Outside a scope (background
refresh, startup provisioning, providers that do not opt in) they keep using the
process's DataRobot API key.

A context variable rather than a parameter: the token has to reach the L2 cache
through the registry's cache backends, including the write-behind tasks they
schedule, which copy the current context when they are created.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_CALLER_TOKEN: ContextVar[str | None] = ContextVar("dragent_registry_caller_token", default=None)


def current_registry_caller_token() -> str | None:
    """Return the caller token in scope, or ``None`` to use the DataRobot API key."""
    return _CALLER_TOKEN.get()


@contextmanager
def registry_caller_token_scope(token: str | None) -> Iterator[None]:
    """Use *token* for registry HTTP and L2 cache access within the block.

    ``None`` is a no-op scope, so callers need not branch on whether a token exists.
    """
    if not token:
        yield
        return
    reset = _CALLER_TOKEN.set(token)
    try:
        yield
    finally:
        _CALLER_TOKEN.reset(reset)
