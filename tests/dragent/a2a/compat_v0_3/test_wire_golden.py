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
# Pins what an A2A v0.3 caller sees on the wire: the agent card (full, redacted and
# extended) and the JSON-RPC responses to message/send and message/stream. The expected
# documents live in ``golden/``. They were recorded from dragent on a2a-sdk 0.3, so a
# change to the SDK underneath must leave every one of them as it is.
#
# Part of the v0.3 backward-compatibility suite (see this package's __init__). Re-recording is
# only for a deliberate v0.3 change: A2A_GOLDEN_UPDATE=1 uv run pytest <this file>

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
from starlette.testclient import TestClient

from datarobot_genai.dragent.a2a.server.agent_executor_adapter import NATWorkflowAgentExecutor
from datarobot_genai.dragent.a2a.server.front_end_config import A2ACapabilitiesConfig
from datarobot_genai.dragent.a2a.server.front_end_config import A2AFrontEndConfig
from datarobot_genai.dragent.cross_app_access_config import CrossApplicationAccessConfig
from datarobot_genai.dragent.cross_app_access_config import CrossAppTokenExchange
from datarobot_genai.dragent.cross_app_access_config import CrossAppTokenRequest
from datarobot_genai.dragent.frontends.a2a import DRAgentA2AFrontEndPluginWorker
from datarobot_genai.dragent.frontends.a2a import create_agent_card
from datarobot_genai.dragent.frontends.fastapi import _PerUserCompatibleAgentExecutor
from datarobot_genai.dragent.frontends.register import DRAgentA2AExternalConfig
from datarobot_genai.dragent.frontends.register import DRAgentA2ARedactedAgentCardConfig

from .sdk_adapter import new_text_message
from .sdk_adapter import v03

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("A2A_GOLDEN_UPDATE") == "1"

AUTH_HEADERS = {"x-datarobot-user-id": "64baa56996fb36e3eeeefc44"}
CARD_PATH = "/.well-known/agent-card.json"
REPLY_TEXT = "pong"

# Generated per request, so they are replaced before comparing.
_RANDOM_ID_KEYS = frozenset({"messageId", "taskId", "contextId"})


class _EchoExecutor(AgentExecutor):
    """Replies to every message with one agent text message."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            new_text_message(REPLY_TEXT, context_id=context.context_id, task_id=context.task_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def _normalise(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<id>" if key in _RANDOM_ID_KEYS and isinstance(item, str) else _normalise(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    return value


def _assert_matches_golden(name: str, actual: Any) -> None:
    path = GOLDEN_DIR / f"{name}.json"
    actual = _normalise(actual)
    if UPDATE:
        GOLDEN_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    expected = json.loads(path.read_text())
    assert actual == expected, f"{name} differs from {path.name}"


def _minimal_frontend_config() -> A2AFrontEndConfig:
    return A2AFrontEndConfig(name="Echo Agent", description="Echoes", host="localhost", port=8000)


def _full_frontend_config() -> A2AFrontEndConfig:
    return A2AFrontEndConfig(
        name="Full Agent",
        description="Every card feature turned on",
        host="localhost",
        port=8000,
        version="2.1.0",
        capabilities=A2ACapabilitiesConfig(streaming=True, push_notifications=False),
        server_auth=OAuth2ResourceServerConfig(
            issuer_url="https://issuer.example.com/oauth2/default",
            scopes=["agent:read", "agent:write"],
        ),
    )


_CROSS_APP_ACCESS = CrossApplicationAccessConfig(
    token_endpoint_auth_method="private_key_jwt",
    token_exchange=CrossAppTokenExchange(
        trusted_issuer="https://your-org.oktapreview.com",
        audience="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7",
    ),
    token_request=CrossAppTokenRequest(
        token_url="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7/v1/token",
        audience="https://app.datarobot.com/dr_org_id/my_agent_id",
        scopes=["agent:write", "agent:invoke"],
    ),
)

_SKILLS = [
    v03.AgentSkill(
        id="summarize",
        name="Summarize",
        description="Summarizes text",
        tags=["text", "nlp"],
        examples=["Summarize this article"],
    ),
    v03.AgentSkill(id="translate", name="Translate", description="Translates text", tags=[]),
]

_DEPLOYMENT_ENV = {
    "MLOPS_DEPLOYMENT_ID": "dep-abc123",
    "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
}


class _FailingSession:
    async def __aenter__(self) -> None:
        raise RuntimeError("workflow exploded")

    async def __aexit__(self, *args: object) -> None:
        return None


def _failing_workflow_executor() -> NATWorkflowAgentExecutor:
    """Build the real NAT executor over a session manager whose workflow always fails."""
    session_manager = MagicMock()
    session_manager.workflow.config.workflow.type = "failing_workflow"
    session_manager.session = _FailingSession
    return NATWorkflowAgentExecutor(session_manager)


async def _build_client(
    frontend_config: A2AFrontEndConfig,
    *,
    full: bool,
    enable_skills: bool,
    executor: AgentExecutor | None = None,
) -> TestClient:
    if full:
        with patch.dict(os.environ, _DEPLOYMENT_ENV):
            card = await create_agent_card(
                frontend_config,
                cross_app_access=_CROSS_APP_ACCESS,
                skills=_SKILLS,
                external=DRAgentA2AExternalConfig(id="catalog-id-xyz"),
            )
    else:
        card = await create_agent_card(frontend_config, cross_app_access=None, skills=[])

    worker = DRAgentA2AFrontEndPluginWorker(
        SimpleNamespace(general=SimpleNamespace(front_end=frontend_config))  # type: ignore[arg-type]
    )
    server = worker.create_a2a_server(
        card,
        executor or _EchoExecutor(),  # type: ignore[arg-type]
        enable_unauthenticated_well_known_route=True,
        redacted_agent_card_config=DRAgentA2ARedactedAgentCardConfig(enable_skills=enable_skills),
    )
    return TestClient(server.build())


@pytest.fixture(params=["minimal", "full"])
def scenario(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
async def client(scenario: str) -> AsyncIterator[TestClient]:
    if scenario == "minimal":
        built = await _build_client(_minimal_frontend_config(), full=False, enable_skills=False)
    else:
        built = await _build_client(_full_frontend_config(), full=True, enable_skills=True)
    with built as test_client:
        yield test_client


def _rpc(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": "req-1", "method": method}
    if params is not None:
        body["params"] = params
    return body


def _user_message(text: str) -> dict[str, Any]:
    return {
        "message": {
            "kind": "message",
            "role": "user",
            "messageId": "msg-1",
            "parts": [{"kind": "text", "text": text}],
        }
    }


async def test_authenticated_card(client: TestClient, scenario: str) -> None:
    # GIVEN an authenticated caller
    # WHEN it fetches the well-known agent card
    response = client.get(CARD_PATH, headers=AUTH_HEADERS)

    # THEN it gets the full v0.3 card, unchanged
    assert response.status_code == 200
    _assert_matches_golden(f"{scenario}_card_authenticated", response.json())


async def test_anonymous_card(client: TestClient, scenario: str) -> None:
    # GIVEN an anonymous caller and an agent that opted in to anonymous discovery
    # WHEN it fetches the well-known agent card
    response = client.get(CARD_PATH)

    # THEN it gets the redacted v0.3 card, unchanged
    assert response.status_code == 200
    _assert_matches_golden(f"{scenario}_card_anonymous", response.json())


async def test_extended_card_rpc(client: TestClient, scenario: str) -> None:
    # GIVEN an authenticated v0.3 caller
    # WHEN it asks for the authenticated extended card over JSON-RPC
    response = client.post(
        "/", json=_rpc("agent/getAuthenticatedExtendedCard"), headers=AUTH_HEADERS
    )

    # THEN the v0.3 response is unchanged
    assert response.status_code == 200
    _assert_matches_golden(f"{scenario}_rpc_extended_card", response.json())


async def test_message_send_rpc(client: TestClient, scenario: str) -> None:
    # GIVEN an authenticated v0.3 caller
    # WHEN it sends a message with message/send
    response = client.post(
        "/", json=_rpc("message/send", _user_message("ping")), headers=AUTH_HEADERS
    )

    # THEN the v0.3 response is unchanged
    assert response.status_code == 200
    _assert_matches_golden(f"{scenario}_rpc_message_send", response.json())


async def test_message_stream_rpc(client: TestClient, scenario: str) -> None:
    # GIVEN an authenticated v0.3 caller
    # WHEN it sends a message with message/stream
    events = []
    with client.stream(
        "POST", "/", json=_rpc("message/stream", _user_message("ping")), headers=AUTH_HEADERS
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line.removeprefix("data:").strip()))

    # THEN the v0.3 server-sent events are unchanged
    _assert_matches_golden(f"{scenario}_rpc_message_stream", events)


async def test_extended_card_rpc_without_identity(client: TestClient, scenario: str) -> None:
    # GIVEN a v0.3 caller without an authenticated identity
    # WHEN it asks for the authenticated extended card over JSON-RPC
    response = client.post("/", json=_rpc("agent/getAuthenticatedExtendedCard"))

    # THEN it is refused with the same v0.3 error as before
    assert response.status_code == 200
    _assert_matches_golden(f"{scenario}_rpc_extended_card_without_identity", response.json())


@pytest.fixture
async def failing_client() -> AsyncIterator[TestClient]:
    built = await _build_client(
        _minimal_frontend_config(),
        full=False,
        enable_skills=False,
        executor=_failing_workflow_executor(),
    )
    with built as test_client:
        yield test_client


async def test_message_send_when_workflow_fails(failing_client: TestClient) -> None:
    # GIVEN an agent whose workflow raises
    # WHEN a v0.3 caller sends a message with message/send
    response = failing_client.post(
        "/", json=_rpc("message/send", _user_message("ping")), headers=AUTH_HEADERS
    )

    # THEN the caller sees the same v0.3 response as before
    assert response.status_code == 200
    _assert_matches_golden("failing_rpc_message_send", response.json())


async def test_message_stream_when_workflow_fails(failing_client: TestClient) -> None:
    # GIVEN an agent whose workflow raises
    # WHEN a v0.3 caller sends a message with message/stream
    events = []
    with failing_client.stream(
        "POST", "/", json=_rpc("message/stream", _user_message("ping")), headers=AUTH_HEADERS
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line.removeprefix("data:").strip()))

    # THEN the caller sees the same v0.3 events as before
    _assert_matches_golden("failing_rpc_message_stream", events)


async def test_message_send_with_invalid_auth_context() -> None:
    # GIVEN dragent's per-user executor and a caller whose signed authorization-context
    # header does not validate
    built = await _build_client(
        _minimal_frontend_config(),
        full=False,
        enable_skills=False,
        executor=_PerUserCompatibleAgentExecutor(MagicMock()),
    )
    headers = {"x-datarobot-authorization-context": "not-a-valid-token"}

    # WHEN it sends a message
    with built as test_client:
        response = test_client.post(
            "/", json=_rpc("message/send", _user_message("ping")), headers=headers
        )

    # THEN it is refused with JSON-RPC invalid params, before the workflow runs
    assert response.status_code == 200
    error = response.json()["error"]
    assert error["code"] == -32602
    assert "X-DataRobot-Authorization-Context" in error["message"]
