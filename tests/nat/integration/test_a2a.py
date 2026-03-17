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
# Integration tests for the ``authenticated_a2a_client`` function group.

from pathlib import Path

import httpx
import pytest
import respx
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from nat.builder.context import ContextState
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.a2a.client.client_impl import A2AClientFunctionGroup
from nat.runtime.loader import load_config

import datarobot_genai.dragent.auth_a2a_client  # noqa: F401 — registers authenticated_a2a_client
import datarobot_genai.nat.datarobot_auth_provider  # noqa: F401 — registers datarobot_api_key
from datarobot_genai.dragent.auth_a2a_client import AuthenticatedA2AClientConfig
from datarobot_genai.dragent.auth_a2a_client import AuthenticatedA2AClientFunctionGroup
from datarobot_genai.nat.datarobot_auth_provider import DataRobotAPIKeyAuthProviderConfig

# ---------------------------------------------------------------------------
# Mock payloads
# ---------------------------------------------------------------------------

# Must match the default URL in workflow_with_a2a.yaml
_BASE_URL = "http://agent.example.com:8080"

_AGENT_CARD_JSON = AgentCard(
    name="Test Agent",
    description="A mock A2A agent for integration tests",
    url=f"{_BASE_URL}/",
    version="1.2.3",
    skills=[],
    capabilities=AgentCapabilities(streaming=False),
    default_input_modes=["text"],
    default_output_modes=["text"],
).model_dump(by_alias=True)

# JSONRPC-wrapped send-message response.
#
# The A2A SDK's JSONRPC transport sends all messages to the agent-card URL
# (``agent_card.url``, which is ``http://agent.example.com:8080/`` here) via a
# standard JSON-RPC 2.0 envelope.  The ``result`` field holds a ``Task``
# object whose ``kind`` discriminator must be ``"task"`` and whose
# ``status.state`` must be the *string* enum value ``"completed"``.
_SEND_MESSAGE_RESPONSE = {
    "jsonrpc": "2.0",
    "id": "test-request-id",
    "result": {
        "id": "task-1",
        "contextId": "ctx-1",
        "kind": "task",
        "status": {"state": "completed"},
        "artifacts": [
            {
                "artifactId": "art-1",
                "parts": [{"kind": "text", "text": "I can help with many things!"}],
            }
        ],
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow_path() -> Path:
    return Path(__file__).parent / "workflow_with_a2a.yaml"


@pytest.fixture(scope="module")
def nat_config(workflow_path: Path):
    """Parse the YAML once per module; all tests share the same Config object."""
    return load_config(workflow_path)


@pytest.fixture(autouse=True)
def set_context_user_id() -> None:
    """Inject a user_id into the NAT ContextVar so per-user function groups initialise."""
    ContextState.get().user_id.set("integration-test-user")


@pytest.fixture
def mock_a2a_endpoints():
    """Intercept httpx calls with respx so no real A2A agent is needed.

    Mocked routes:
    * GET  /.well-known/agent-card.json — returns a minimal AgentCard
    * POST /                            — JSONRPC endpoint (the A2A SDK's JSONRPC
                                          transport posts all messages to the agent-card
                                          URL, i.e. the server root)

    ``assert_all_called=False`` lets individual tests omit the send-message call
    (e.g. tests that only exercise function-group setup or agent-info retrieval).
    """
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{_BASE_URL}/.well-known/agent-card.json").mock(
            return_value=httpx.Response(200, json=_AGENT_CARD_JSON)
        )
        router.post(f"{_BASE_URL}/").mock(
            return_value=httpx.Response(200, json=_SEND_MESSAGE_RESPONSE)
        )
        yield router


@pytest.fixture
async def function_group(nat_config, mock_a2a_endpoints):
    """Build an AuthenticatedA2AClientFunctionGroup via WorkflowBuilder against mocked endpoints."""
    fg_config = nat_config.function_groups["a2a_agent"]
    auth_config = nat_config.authentication["datarobot_auth"]

    async with WorkflowBuilder() as builder:
        await builder.add_auth_provider("datarobot_auth", auth_config)
        await builder.add_function_group("a2a_agent", fg_config)
        fg = await builder.get_function_group("a2a_agent")
        yield fg


# ---------------------------------------------------------------------------
# Tests: YAML config parsing (no network required)
# ---------------------------------------------------------------------------


class TestConfigParsedFromYaml:
    def test_function_group_is_authenticated_a2a_client(self, nat_config):
        assert isinstance(nat_config.function_groups["a2a_agent"], AuthenticatedA2AClientConfig)

    def test_auth_provider_is_datarobot_api_key(self, nat_config):
        assert isinstance(
            nat_config.authentication["datarobot_auth"], DataRobotAPIKeyAuthProviderConfig
        )

    def test_function_group_references_auth_provider(self, nat_config):
        assert nat_config.function_groups["a2a_agent"].auth_provider == "datarobot_auth"


# ---------------------------------------------------------------------------
# Tests: WorkflowBuilder + mocked httpx (respx)
# ---------------------------------------------------------------------------


class TestAuthenticatedA2AClientGroup:
    """Builds the function group via WorkflowBuilder and exercises it against
    a respx-mocked A2A agent — no real network connection required.
    """

    async def test_function_group_type(self, function_group):
        """WorkflowBuilder produces an AuthenticatedA2AClientFunctionGroup."""
        assert isinstance(function_group, AuthenticatedA2AClientFunctionGroup)
        assert isinstance(function_group, A2AClientFunctionGroup)

    async def test_all_standard_functions_registered(self, function_group):
        """All seven standard A2A functions are exposed under the group name prefix."""
        all_fns = await function_group.get_all_functions()

        # Function names are prefixed with the group name (e.g. a2a_agent__call)
        assert "a2a_agent__call" in all_fns
        assert "a2a_agent__get_skills" in all_fns
        assert "a2a_agent__get_info" in all_fns
        assert "a2a_agent__get_task" in all_fns
        assert "a2a_agent__cancel_task" in all_fns
        assert "a2a_agent__send_message" in all_fns
        assert "a2a_agent__send_message_streaming" in all_fns

    async def test_call_agent(self, function_group):
        """The high-level ``call`` function returns a non-empty text response."""
        all_fns = await function_group.get_all_functions()
        call_fn = all_fns["a2a_agent__call"]
        response = await call_fn.ainvoke({"query": "What can you help me with?"})

        assert isinstance(response, str)
        assert len(response) > 0

    async def test_get_agent_info(self, function_group):
        """The ``get_info`` helper returns the agent name and version."""
        all_fns = await function_group.get_all_functions()
        get_info_fn = all_fns["a2a_agent__get_info"]
        info = await get_info_fn.ainvoke(None)

        assert "name" in info
        assert "version" in info
