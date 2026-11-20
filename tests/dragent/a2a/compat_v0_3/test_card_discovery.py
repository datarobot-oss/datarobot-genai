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
#
# dragent's A2A client discovers an agent that publishes a v0.3 card and talks to it over v0.3.
#
# The agent here is dragent's own server, so this is the round trip every dragent-to-dragent
# call makes. Requests go through httpx to an in-process Starlette app. A request hook records
# each call's host and JSON-RPC method, which show where the client sent the message and in
# which protocol dialect.

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from typing import NamedTuple

import httpx
import pytest
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from starlette.applications import Starlette
from starlette.routing import Mount

from datarobot_genai.dragent.a2a.client.client_base import A2ABaseClient
from datarobot_genai.dragent.a2a.server.front_end_config import A2AFrontEndConfig
from datarobot_genai.dragent.frontends.a2a import DRAgentA2AFrontEndPluginWorker
from datarobot_genai.dragent.frontends.a2a import create_agent_card

from .sdk_adapter import card_rpc_url
from .sdk_adapter import new_text_message

REPLY_TEXT = "pong"


class _Call(NamedTuple):
    host: str
    method: str


class _EchoExecutor(AgentExecutor):
    """Replies to every message with one agent text message."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            new_text_message(REPLY_TEXT, context_id=context.context_id, task_id=context.task_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


async def _dragent_agent_app() -> Starlette:
    """Build an agent served by dragent itself, mounted under ``/a2a`` as in production."""
    frontend_config = A2AFrontEndConfig(
        name="dragent Agent", description="dragent", host="localhost", port=8000
    )
    card = await create_agent_card(frontend_config, cross_app_access=None, skills=[])
    worker = DRAgentA2AFrontEndPluginWorker(
        SimpleNamespace(general=SimpleNamespace(front_end=frontend_config))  # type: ignore[arg-type]
    )
    server = worker.create_a2a_server(
        card,
        _EchoExecutor(),  # type: ignore[arg-type]
        enable_unauthenticated_well_known_route=True,
    )
    return Starlette(routes=[Mount("/a2a", app=server.build())])


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[[Starlette], list[_Call]]:
    """Route every httpx client created from now on to *app*; return the calls it records."""
    real_async_client = httpx.AsyncClient

    def install(app: Starlette) -> list[_Call]:
        calls: list[_Call] = []

        async def record(request: httpx.Request) -> None:
            if request.method == "POST":
                calls.append(_Call(request.url.host, json.loads(request.content)["method"]))

        def client_for_app(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = httpx.ASGITransport(app=app)
            kwargs["event_hooks"] = {"request": [record]}
            return real_async_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client_for_app)
        return calls

    return install


async def test_reads_a_v03_card_from_dragent_and_calls_the_agent_over_v03(serve: Any) -> None:
    # GIVEN an agent served by dragent, which publishes a v0.3 card
    calls = serve(await _dragent_agent_app())

    # WHEN dragent's client discovers it and sends a message
    async with A2ABaseClient(base_url="http://localhost:8000/a2a/", streaming=False) as client:
        card = client.agent_card
        events = [event async for event in client.send_message("ping")]
        reply = client.extract_text_from_events(events)

    # THEN the card is read, the message goes to the card's RPC URL as v0.3 message/send,
    # and the reply text comes back
    assert card is not None
    assert card.name == "dragent Agent"
    assert card_rpc_url(card) == "http://localhost:8000/a2a/"
    assert calls == [_Call("localhost", "message/send")]
    assert reply == REPLY_TEXT
