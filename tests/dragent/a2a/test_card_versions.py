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
# dragent's A2A client against real in-process A2A servers of each protocol version.
#
# Every request goes through httpx to a Starlette app, so these tests cover discovery,
# card parsing, transport selection and the JSON-RPC exchange end to end. A request hook
# records each call's host, JSON-RPC method and A2A-Version header, which show which
# interface the client chose and which protocol version it spoke.

import json
from collections.abc import Callable
from typing import Any
from typing import NamedTuple

import httpx
import pytest
from a2a.helpers import new_text_message
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes
from a2a.server.routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentInterface
from a2a.types import AgentSkill
from starlette.applications import Starlette

from datarobot_genai.dragent.a2a.cards import card_rpc_url
from datarobot_genai.dragent.a2a.client.client_base import A2ABaseClient

V10_URL = "http://agent-v10.test/"
V03_URL = "http://agent-v03.test/"
REPLY_TEXT = "pong"


class _Call(NamedTuple):
    host: str
    method: str
    a2a_version: str | None


class _EchoExecutor(AgentExecutor):
    """Replies to every message with one agent text message."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            new_text_message(REPLY_TEXT, context_id=context.context_id, task_id=context.task_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def _sdk_agent_app(interfaces: dict[str, str], *, streaming: bool = False) -> Starlette:
    """Build an a2a-sdk 1.x agent with one JSON-RPC interface per ``{version: url}``.

    The v0.3 compatibility layer is on only when the card advertises v0.3, so an agent with
    only a v1.0 interface rejects v0.3 calls.
    """
    card = AgentCard(
        name="SDK Agent",
        description="Plain a2a-sdk 1.x agent",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(url=url, protocol_binding="JSONRPC", protocol_version=version)
            for version, url in interfaces.items()
        ],
        capabilities=AgentCapabilities(streaming=streaming),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[AgentSkill(id="echo", name="Echo", description="Echoes", tags=["echo"])],
    )
    handler = DefaultRequestHandler(
        agent_executor=_EchoExecutor(), task_store=InMemoryTaskStore(), agent_card=card
    )
    speaks_v03 = any(version.startswith("0.3") for version in interfaces)
    return Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(handler, rpc_url="/", enable_v0_3_compat=speaks_v03),
        ]
    )


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[[Starlette], list[_Call]]:
    """Route every httpx client created from now on to *app*; return the calls it records."""
    real_async_client = httpx.AsyncClient

    def install(app: Starlette) -> list[_Call]:
        calls: list[_Call] = []

        async def record(request: httpx.Request) -> None:
            if request.method == "POST":
                method = json.loads(request.content)["method"]
                calls.append(_Call(request.url.host, method, request.headers.get("a2a-version")))

        def client_for_app(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = httpx.ASGITransport(app=app)
            kwargs["event_hooks"] = {"request": [record]}
            return real_async_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client_for_app)
        return calls

    return install


async def _ask(base_url: str, *, streaming: bool = False) -> tuple[AgentCard, str]:
    async with A2ABaseClient(base_url=base_url, streaming=streaming) as client:
        assert client.agent_card is not None
        events = [event async for event in client.send_message("ping")]
        return client.agent_card, client.extract_text_from_events(events)


async def test_reads_a_v10_card_and_calls_the_agent_over_v10(serve: Any) -> None:
    # GIVEN an agent that publishes a v1.0 card and speaks only v1.0
    calls = serve(_sdk_agent_app({"1.0": V10_URL}))

    # WHEN dragent's client discovers it and sends a message
    card, reply = await _ask(V10_URL)

    # THEN the card is read into the SDK's AgentCard, and the call uses v1.0
    assert isinstance(card, AgentCard)
    assert card.name == "SDK Agent"
    assert card_rpc_url(card) == V10_URL
    assert reply == REPLY_TEXT
    assert calls == [_Call("agent-v10.test", "SendMessage", "1.0")]


async def test_streams_from_a_v10_agent(serve: Any) -> None:
    # GIVEN a v1.0 agent that supports streaming
    calls = serve(_sdk_agent_app({"1.0": V10_URL}, streaming=True))

    # WHEN dragent's client streams a message to it
    _, reply = await _ask(V10_URL, streaming=True)

    # THEN the reply arrives over the v1.0 streaming method
    assert reply == REPLY_TEXT
    assert calls == [_Call("agent-v10.test", "SendStreamingMessage", "1.0")]


async def test_prefers_v10_when_a_card_offers_both_versions(serve: Any) -> None:
    # GIVEN an agent whose card lists a v0.3 and a v1.0 JSON-RPC interface, v0.3 first
    calls = serve(_sdk_agent_app({"0.3": V03_URL, "1.0": V10_URL}))

    # WHEN dragent's client sends it a message
    card, reply = await _ask(V03_URL)

    # THEN it calls the v1.0 interface, which is also the URL dragent reports for the card
    assert reply == REPLY_TEXT
    assert calls == [_Call("agent-v10.test", "SendMessage", "1.0")]
    assert card_rpc_url(card) == V10_URL
