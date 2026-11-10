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

import os
import time
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.core.mcp import MCPConfig
from datarobot_genai.dragent import mcp_xaa
from datarobot_genai.dragent.inbound_token import OAUTH_ACCESS_TOKEN_HEADER
from datarobot_genai.dragent.inbound_token import InboundTokenExpiredError
from datarobot_genai.dragent.plugins.okta_a2a_auth import ApiTokenExchange
from datarobot_genai.dragent.plugins.okta_a2a_auth import _CrossAppFlowParams
from tests.dragent.helpers import make_jwt

MCP_URL = "https://app.datarobot.com/api/v2/deployments/abc/directAccess/mcp/"
METADATA_URL = (
    "https://app.datarobot.com/api/v2/deployments/abc/directAccess"
    "/.well-known/oauth-protected-resource"
)
AGENT_IDENTITY = {"IDP_AGENT_ID": "wlp-agent", "IDP_AGENT_PRIVATE_KEY_JWK": "{}"}
CALLER_TOKEN = make_jwt(sub="alice")
CALLER_HEADERS = {"X-DataRobot-External-Access-Token": CALLER_TOKEN}
REQUIRED_XAA_SETTINGS = {
    "MCP_XAA_TRUSTED_ISSUER": "https://org.okta.com",
    "MCP_XAA_EXCHANGE_AUDIENCE": "https://org.okta.com/oauth2/aus1",
    "MCP_XAA_TOKEN_URL": "https://org.okta.com/oauth2/aus1/v1/token",
    "MCP_XAA_SCOPES": "dr.impersonation, ,read_data",
}
XAA_SETTINGS = {**REQUIRED_XAA_SETTINGS, "MCP_XAA_TOKEN_AUDIENCE": "https://mcp.example.com"}
SETTINGS_PARAMS = _CrossAppFlowParams(
    trusted_issuer="https://org.okta.com",
    exchange_audience="https://org.okta.com/oauth2/aus1",
    token_url="https://org.okta.com/oauth2/aus1/v1/token",
    target_audience="https://mcp.example.com",
    id_jag_scopes=["dr.impersonation", "read_data"],
    token_endpoint_auth_method="private_key_jwt",
)
DISCOVERED_PARAMS = _CrossAppFlowParams(
    trusted_issuer="https://discovered.okta.com",
    exchange_audience="https://discovered.okta.com/oauth2/aus2",
    token_url="https://discovered.okta.com/oauth2/aus2/v1/token",
    target_audience=None,
    id_jag_scopes=[],
    token_endpoint_auth_method="private_key_jwt",
)


def _without(settings: dict[str, str], name: str) -> dict[str, str]:
    return {key: value for key, value in settings.items() if key != name}


@pytest.fixture(autouse=True)
def clear_caches() -> Iterator[None]:
    mcp_xaa.clear_caches()
    yield
    mcp_xaa.clear_caches()


@pytest.fixture
def exchange_token() -> Iterator[AsyncMock]:
    exchange = Mock(exchange_token=AsyncMock(return_value="xaa-token"))
    with patch.object(mcp_xaa, "get_token_exchange", return_value=exchange):
        yield exchange.exchange_token


@pytest.fixture
def fetch_xaa_params() -> Iterator[AsyncMock]:
    with patch.object(
        mcp_xaa, "fetch_xaa_params", new_callable=AsyncMock, return_value=DISCOVERED_PARAMS
    ) as fetch:
        yield fetch


async def _exchange(env: dict[str, str], headers: dict[str, str]) -> str:
    with patch.dict(os.environ, env, clear=True):
        return await mcp_xaa.exchange_caller_token(MCPConfig(forwarded_headers=headers), MCP_URL)


async def test_settings_are_used_before_discovery(
    exchange_token: AsyncMock, fetch_xaa_params: AsyncMock
) -> None:
    # GIVEN the MCP_XAA_* settings and a caller token
    # WHEN the caller's token is exchanged
    token = await _exchange({**AGENT_IDENTITY, **XAA_SETTINGS}, CALLER_HEADERS)

    # THEN the exchange gets the parameters from the settings and the server is not asked
    assert token == "xaa-token"
    exchange_token.assert_awaited_once_with(SETTINGS_PARAMS, CALLER_TOKEN)
    fetch_xaa_params.assert_not_awaited()


async def test_parameters_are_discovered_once_when_no_setting_is_set(
    exchange_token: AsyncMock, fetch_xaa_params: AsyncMock
) -> None:
    # GIVEN no MCP_XAA_* setting
    # WHEN two requests exchange a token
    for _ in range(2):
        await _exchange(AGENT_IDENTITY, CALLER_HEADERS)

    # THEN the MCP server's metadata is fetched once and reused
    fetch_xaa_params.assert_awaited_once()
    assert fetch_xaa_params.await_args.args[0] == METADATA_URL
    exchange_token.assert_awaited_with(DISCOVERED_PARAMS, CALLER_TOKEN)


@pytest.mark.parametrize(
    ("env", "headers", "error"),
    [
        pytest.param(
            {**AGENT_IDENTITY, **XAA_SETTINGS},
            {"x-datarobot-api-key": "caller-key"},
            "no caller token in 'x-datarobot-external-access-token'",
            id="no-caller-token",
        ),
        pytest.param(
            {**AGENT_IDENTITY, **XAA_SETTINGS},
            {OAUTH_ACCESS_TOKEN_HEADER: "opaque-token"},
            "it is not a JWT",
            id="opaque-caller-token",
        ),
        pytest.param(
            {**AGENT_IDENTITY, "MCP_XAA_TOKEN_AUDIENCE": "https://mcp.example.com"},
            CALLER_HEADERS,
            "Missing: MCP_XAA_TRUSTED_ISSUER, MCP_XAA_EXCHANGE_AUDIENCE, MCP_XAA_TOKEN_URL, "
            "MCP_XAA_SCOPES",
            id="only-an-optional-setting",
        ),
        pytest.param(
            {**AGENT_IDENTITY, **_without(XAA_SETTINGS, "MCP_XAA_SCOPES")},
            CALLER_HEADERS,
            r"Missing: MCP_XAA_SCOPES\.",
            id="no-scopes",
        ),
        pytest.param(
            XAA_SETTINGS,
            CALLER_HEADERS,
            "Set IDP_AGENT_ID and IDP_AGENT_PRIVATE_KEY_JWK",
            id="no-agent-identity",
        ),
    ],
)
async def test_exchange_that_cannot_happen_raises(
    env: dict[str, str],
    headers: dict[str, str],
    error: str,
    exchange_token: AsyncMock,
    fetch_xaa_params: AsyncMock,
) -> None:
    # GIVEN a request or a configuration the exchange cannot work with
    # WHEN / THEN it raises and says what is missing, without calling Okta
    with pytest.raises(RuntimeError, match=error):
        await _exchange(env, headers)
    exchange_token.assert_not_awaited()
    fetch_xaa_params.assert_not_awaited()


@pytest.mark.parametrize(
    "error",
    [RuntimeError("HTTP 401"), KeyError("trusted_issuer"), ValueError("not JSON")],
    ids=["unreadable", "incomplete-metadata", "not-json"],
)
@pytest.mark.usefixtures("exchange_token")
async def test_failed_discovery_points_at_the_settings(
    error: Exception, fetch_xaa_params: AsyncMock
) -> None:
    # GIVEN an MCP server whose metadata cannot be read
    fetch_xaa_params.side_effect = error

    # WHEN two requests need the parameters
    # THEN each fails, names the settings that replace discovery, and asks the server again
    for _ in range(2):
        with pytest.raises(RuntimeError, match=r"Set the MCP_XAA_\* settings on the agent"):
            await _exchange(AGENT_IDENTITY, CALLER_HEADERS)
    assert fetch_xaa_params.await_count == 2


async def test_exchanged_tokens_are_cached_per_caller() -> None:
    # GIVEN the real exchange with only the calls to Okta replaced, and a token that lives 200s
    env = {**AGENT_IDENTITY, **XAA_SETTINGS, "XAA_TOKEN_EXCHANGE_IMPL": "http"}
    now = time.time()
    okta = AsyncMock(side_effect=[make_jwt(exp=now + 200), make_jwt(exp=now + 3600)])
    alice = {OAUTH_ACCESS_TOKEN_HEADER: make_jwt(sub="alice", exp=now + 3600)}
    bob = {OAUTH_ACCESS_TOKEN_HEADER: make_jwt(sub="bob", exp=now + 3600)}

    # WHEN alice calls twice and bob once
    with patch.object(ApiTokenExchange, "_exchange_token_impl", okta):
        tokens = [await _exchange(env, headers) for headers in (alice, alice, bob)]

    # THEN alice's short-lived token is reused and bob gets a token of his own
    assert okta.await_count == 2
    assert tokens[0] == tokens[1] != tokens[2]


async def test_expired_caller_token_is_refused_before_okta() -> None:
    # GIVEN the real exchange and a caller token that expired ten minutes ago
    env = {**AGENT_IDENTITY, **XAA_SETTINGS, "XAA_TOKEN_EXCHANGE_IMPL": "http"}
    headers = {OAUTH_ACCESS_TOKEN_HEADER: make_jwt(sub="alice", exp=time.time() - 600)}

    # WHEN / THEN the exchange raises without calling Okta
    with (
        patch.object(ApiTokenExchange, "_exchange_token_impl", new_callable=AsyncMock) as okta,
        pytest.raises(InboundTokenExpiredError),
    ):
        await _exchange(env, headers)
    okta.assert_not_awaited()
