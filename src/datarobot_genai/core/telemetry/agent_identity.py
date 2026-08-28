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

"""Propagates ``gen_ai.agent.name`` across a tool-call boundary via OTel Baggage.

A tool-call span (e.g. an MCP tool served by ``drmcp``, possibly over a
network hop) has no attribute of its own for which agent invoked it - trace
parentage doesn't carry span attributes across that boundary. Baggage does,
via auto-instrumented HTTP clients that inject whatever is in the active
context on every outgoing request, so attaching it here needs no explicit
``inject``/``extract`` call pairs; the receiving side just reads it back out.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import baggage
from opentelemetry.context import Context
from opentelemetry.context import Token
from opentelemetry.context import attach
from opentelemetry.context import detach

# Matches the GEN_AI_AGENT_NAME span attribute name (opentelemetry.semconv /
# datarobot_opentelemetry.semconv) - same key, different carrier.
GEN_AI_AGENT_NAME_BAGGAGE_KEY = "gen_ai.agent.name"


def attach_agent_name_baggage(agent_name: str | None) -> Token[Context] | None:
    """Attach ``agent_name`` into OTel Baggage on the current context.

    Returns the detach token, or ``None`` when ``agent_name`` is falsy (no
    context change made). Callers must pass whatever this returns to
    :func:`detach_agent_name_baggage` when the wrapped scope ends - always,
    even when it's ``None``, so a no-op attach is a no-op detach too.
    """
    if not agent_name:
        return None
    return attach(baggage.set_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY, agent_name))


def detach_agent_name_baggage(token: Token[Context] | None) -> None:
    if token is not None:
        detach(token)


@contextmanager
def agent_name_baggage(agent_name: str | None) -> Iterator[None]:
    """Context-manager form of :func:`attach_agent_name_baggage` for callers
    with a natural ``with`` scope around the agent's execution.
    """
    token = attach_agent_name_baggage(agent_name)
    try:
        yield
    finally:
        detach_agent_name_baggage(token)
