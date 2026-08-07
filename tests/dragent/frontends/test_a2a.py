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
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from a2a.types import AgentSkill
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig

from datarobot_genai.dragent.cross_app_access_config import CrossApplicationAccessConfig
from datarobot_genai.dragent.cross_app_access_config import CrossAppTokenExchange
from datarobot_genai.dragent.cross_app_access_config import CrossAppTokenRequest
from datarobot_genai.dragent.frontends.a2a import BEARER_SECURITY_DESCRIPTION
from datarobot_genai.dragent.frontends.a2a import BEARER_SECURITY_SCHEME_NAME
from datarobot_genai.dragent.frontends.a2a import CROSS_APP_EXTENSION_DESCRIPTION
from datarobot_genai.dragent.frontends.a2a import CROSS_APP_SECURITY_SCHEME_FLOW_REF
from datarobot_genai.dragent.frontends.a2a import CROSS_APP_SECURITY_SCHEME_REF
from datarobot_genai.dragent.frontends.a2a import EXTERNAL_IDENTITY_URI
from datarobot_genai.dragent.frontends.a2a import INTERNAL_IDENTITY_URI
from datarobot_genai.dragent.frontends.a2a import JWT_BEARER_GRANT_TYPE_URI
from datarobot_genai.dragent.frontends.a2a import OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE
from datarobot_genai.dragent.frontends.a2a import TOKEN_EXCHANGE_GRANT_TYPE_URI
from datarobot_genai.dragent.frontends.a2a import TOKEN_EXCHANGE_REQUESTED_TOKEN_TYPE
from datarobot_genai.dragent.frontends.a2a import _public_card_modifier
from datarobot_genai.dragent.frontends.a2a import create_agent_card
from datarobot_genai.dragent.frontends.a2a import get_a2a_endpoint_url
from datarobot_genai.dragent.frontends.a2a import redact_agent_card
from datarobot_genai.dragent.frontends.register import DRAgentA2AExternalConfig
from datarobot_genai.dragent.frontends.session import _a2a_headers


@pytest.fixture
def a2a_frontend_config():
    return A2AFrontEndConfig(
        name="My Agent", description="Does things", host="localhost", port=8000
    )


class TestRedactAgentCard:
    async def test_strips_skills_and_identity_extensions(self, a2a_frontend_config):
        skill = AgentSkill(id="summarize", name="Summarize", description="Summarizes text", tags=[])
        external = DRAgentA2AExternalConfig(id="catalog-id-xyz")
        env = {
            "MLOPS_DEPLOYMENT_ID": "dep-abc123",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        }
        with patch.dict(os.environ, env):
            card = await create_agent_card(
                a2a_frontend_config,
                cross_app_access=None,
                skills=[skill],
                external=external,
            )

        redacted = redact_agent_card(card)

        assert redacted.skills == []
        assert redacted.supports_authenticated_extended_card is True
        assert redacted.security_schemes is not None
        assert BEARER_SECURITY_SCHEME_NAME in redacted.security_schemes
        assert card.capabilities.extensions is not None
        uris = [ext.uri for ext in card.capabilities.extensions]
        assert INTERNAL_IDENTITY_URI in uris
        assert EXTERNAL_IDENTITY_URI in uris
        redacted_uris = [ext.uri for ext in (redacted.capabilities.extensions or [])]
        assert INTERNAL_IDENTITY_URI not in redacted_uris
        assert EXTERNAL_IDENTITY_URI not in redacted_uris

    async def test_preserves_non_identity_extensions(self, a2a_frontend_config):
        cross_app_access = CrossApplicationAccessConfig(
            token_endpoint_auth_method="private_key_jwt",
            token_exchange=CrossAppTokenExchange(
                trusted_issuer="https://your-org.oktapreview.com",
                audience="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7",
            ),
            token_request=CrossAppTokenRequest(
                token_url="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7/v1/token",
                audience="https://app.datarobot.com/dr_org_id/my_agent_id",
            ),
        )
        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=cross_app_access, skills=[]
        )

        redacted = redact_agent_card(card)

        assert redacted.capabilities.extensions is not None
        assert any(ext.uri == JWT_BEARER_GRANT_TYPE_URI for ext in redacted.capabilities.extensions)
        assert redacted.security_schemes is not None
        assert "oauth2" in redacted.security_schemes


class TestAgentCardIdentitySelection:
    async def test_public_card_modifier_returns_full_card_when_authenticated(
        self, a2a_frontend_config
    ):
        skill = AgentSkill(id="summarize", name="Summarize", description="Summarizes text", tags=[])
        card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[skill])
        token = _a2a_headers.set({"x-datarobot-user-id": "64baa56996fb36e3eeeefc44"})
        try:
            result = _public_card_modifier(card)
        finally:
            _a2a_headers.reset(token)

        assert result.skills == card.skills

    async def test_public_card_modifier_redacts_when_unauthenticated(self, a2a_frontend_config):
        skill = AgentSkill(id="summarize", name="Summarize", description="Summarizes text", tags=[])
        card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[skill])
        token = _a2a_headers.set({})
        try:
            result = _public_card_modifier(card)
        finally:
            _a2a_headers.reset(token)

        assert result.skills == []


class TestCreateAgentCard:
    async def test_default_skill_when_skills_empty(self, a2a_frontend_config):
        card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[])
        assert len(card.skills) == 1
        assert card.skills[0].id == "call"
        assert card.skills[0].name == "My Agent"
        assert card.skills[0].description == "Does things"
        assert card.supports_authenticated_extended_card is True

    async def test_configured_skills_used_when_present(self, a2a_frontend_config):
        skill = AgentSkill(id="summarize", name="Summarize", description="Summarizes text", tags=[])
        card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[skill])
        assert len(card.skills) == 1
        assert card.skills[0].id == "summarize"

    async def test_agent_card_fields_from_frontend_config(self):
        cfg = A2AFrontEndConfig(
            name="My Agent",
            description="Does things",
            version="2.0.0",
            host="localhost",
            port=9000,
        )
        card = await create_agent_card(cfg, cross_app_access=None, skills=[])
        assert card.name == "My Agent"
        assert card.description == "Does things"
        assert card.version == "2.0.0"
        assert card.url == "http://localhost:9000/a2a/"

    async def test_security_schemes_set_when_cross_application_access_present(
        self, a2a_frontend_config
    ):
        cross_app_access = CrossApplicationAccessConfig(
            token_endpoint_auth_method="private_key_jwt",
            token_exchange=CrossAppTokenExchange(
                trusted_issuer="https://your-org.oktapreview.com",
                audience="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7",
            ),
            token_request=CrossAppTokenRequest(
                token_url="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7/v1/token",
                audience="https://app.datarobot.com/dr_org_id/my_agent_id",
                scopes=["blog:write"],
            ),
        )
        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=cross_app_access, skills=[]
        )

        assert "oauth2" in card.security_schemes
        oauth_scheme = card.security_schemes["oauth2"].root
        assert oauth_scheme.type == "oauth2"
        assert oauth_scheme.description == OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE

        # Only client_credentials flow, no authorization_code
        assert oauth_scheme.flows.authorization_code is None
        flow = oauth_scheme.flows.client_credentials
        assert flow.token_url == "https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7/v1/token"
        assert flow.scopes == {"blog:write": "Permission: blog:write"}

        assert card.security == [{"oauth2": ["blog:write"]}]

        # JWT Bearer extension: nested params — token_url/scopes must NOT appear here
        assert card.capabilities.extensions is not None
        assert len(card.capabilities.extensions) == 1
        ext = card.capabilities.extensions[0]
        assert ext.uri == JWT_BEARER_GRANT_TYPE_URI
        assert ext.description == CROSS_APP_EXTENSION_DESCRIPTION
        assert ext.params == {
            "ref": {
                "scheme": CROSS_APP_SECURITY_SCHEME_REF,
                "flow": CROSS_APP_SECURITY_SCHEME_FLOW_REF,
            },
            "tokenEndpointAuthMethod": "private_key_jwt",
            "tokenExchange": {
                "grantType": TOKEN_EXCHANGE_GRANT_TYPE_URI,
                "requestedTokenType": TOKEN_EXCHANGE_REQUESTED_TOKEN_TYPE,
                "trustedIssuer": "https://your-org.oktapreview.com",
                "audience": "https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7",
            },
            "tokenRequest": {
                "grantType": JWT_BEARER_GRANT_TYPE_URI,
                "audience": "https://app.datarobot.com/dr_org_id/my_agent_id",
            },
        }
        # Verify OpenAPI/extension strict separation: token_url and scopes are NOT in params
        assert "token_url" not in ext.params
        assert "scopes" not in ext.params

    async def test_security_schemes_from_server_auth(self, a2a_frontend_config):
        a2a_frontend_config.server_auth = MagicMock(
            issuer_url="https://issuer.example.com",
            discovery_url=None,
            scopes=["read"],
        )
        card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[])

        oauth_scheme = card.security_schemes["oauth2"].root
        assert oauth_scheme.description == OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE
        # Only authorization_code flow, no client_credentials
        assert oauth_scheme.flows.authorization_code is not None
        assert (
            oauth_scheme.flows.authorization_code.authorization_url
            == "https://issuer.example.com/oauth/authorize"
        )
        assert (
            oauth_scheme.flows.authorization_code.token_url
            == "https://issuer.example.com/oauth/token"
        )
        assert oauth_scheme.flows.client_credentials is None
        assert card.security == [{"oauth2": ["read"]}]

    async def test_both_server_auth_and_cross_application_access(self, a2a_frontend_config):
        # server_auth → authorization_code flow
        a2a_frontend_config.server_auth = MagicMock(
            issuer_url="https://issuer.example.com",
            discovery_url=None,
            scopes=["read"],
        )

        # cross_application_access → client_credentials flow + JWT Bearer extension
        cross_app_access = CrossApplicationAccessConfig(
            token_endpoint_auth_method="private_key_jwt",
            token_exchange=CrossAppTokenExchange(
                trusted_issuer="https://your-org.oktapreview.com",
                audience="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7",
            ),
            token_request=CrossAppTokenRequest(
                token_url="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7/v1/token",
                audience="https://app.datarobot.com/dr_org_id/my_agent_id",
                scopes=["blog:write"],
            ),
        )

        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=cross_app_access, skills=[]
        )

        # Single oauth2 scheme with both flows
        assert len(card.security_schemes) == 1
        oauth_scheme = card.security_schemes["oauth2"].root
        assert oauth_scheme.description == OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE

        assert oauth_scheme.flows.authorization_code is not None
        assert (
            oauth_scheme.flows.authorization_code.authorization_url
            == "https://issuer.example.com/oauth/authorize"
        )

        assert oauth_scheme.flows.client_credentials is not None
        assert (
            oauth_scheme.flows.client_credentials.token_url
            == "https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7/v1/token"
        )

        # Merged scopes (deduplicated)
        assert card.security == [{"oauth2": ["read", "blog:write"]}]

        # Cross-app extension: nested params; token_url/scopes only under OpenAPI flows
        assert card.capabilities.extensions is not None
        ext = card.capabilities.extensions[0]
        assert ext.uri == JWT_BEARER_GRANT_TYPE_URI
        assert ext.description == CROSS_APP_EXTENSION_DESCRIPTION
        assert ext.params == {
            "ref": {
                "scheme": CROSS_APP_SECURITY_SCHEME_REF,
                "flow": CROSS_APP_SECURITY_SCHEME_FLOW_REF,
            },
            "tokenEndpointAuthMethod": "private_key_jwt",
            "tokenExchange": {
                "grantType": TOKEN_EXCHANGE_GRANT_TYPE_URI,
                "requestedTokenType": TOKEN_EXCHANGE_REQUESTED_TOKEN_TYPE,
                "trustedIssuer": "https://your-org.oktapreview.com",
                "audience": "https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7",
            },
            "tokenRequest": {
                "grantType": JWT_BEARER_GRANT_TYPE_URI,
                "audience": "https://app.datarobot.com/dr_org_id/my_agent_id",
            },
        }
        assert "token_url" not in ext.params
        assert "scopes" not in ext.params

    async def test_default_bearer_security_schemes_when_no_auth_configured(
        self, a2a_frontend_config
    ):
        card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[])

        assert BEARER_SECURITY_SCHEME_NAME in card.security_schemes
        bearer_scheme = card.security_schemes[BEARER_SECURITY_SCHEME_NAME].root
        assert bearer_scheme.type == "http"
        assert bearer_scheme.scheme == "bearer"
        assert bearer_scheme.description == BEARER_SECURITY_DESCRIPTION
        assert card.security == [{BEARER_SECURITY_SCHEME_NAME: []}]

    async def test_internal_identity_extension_when_deployment_id_set(self, a2a_frontend_config):
        """GIVEN MLOPS_DEPLOYMENT_ID is set WHEN create_agent_card is called THEN the internal
        identity extension is present with the deployment_id.
        """
        env = {
            "MLOPS_DEPLOYMENT_ID": "dep-abc123",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        }
        with patch.dict(os.environ, env):
            card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[])

        assert card.capabilities.extensions is not None
        uris = [ext.uri for ext in card.capabilities.extensions]
        assert INTERNAL_IDENTITY_URI in uris
        internal = next(e for e in card.capabilities.extensions if e.uri == INTERNAL_IDENTITY_URI)
        assert internal.required is True
        assert internal.params == {"deployment_id": "dep-abc123"}

    async def test_internal_identity_extension_when_workload_id_set(self, a2a_frontend_config):
        """GIVEN WORKLOAD_ID is set WHEN create_agent_card is called THEN the internal
        identity extension is present with the workload_id.
        """
        env = {
            "WORKLOAD_ID": "wl-abc123",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        }
        with patch.dict(os.environ, env, clear=True):
            card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[])

        assert card.capabilities.extensions is not None
        uris = [ext.uri for ext in card.capabilities.extensions]
        assert INTERNAL_IDENTITY_URI in uris
        internal = next(e for e in card.capabilities.extensions if e.uri == INTERNAL_IDENTITY_URI)
        assert internal.required is True
        assert internal.params == {"workload_id": "wl-abc123"}

    async def test_no_internal_identity_extension_in_local_dev(self, a2a_frontend_config):
        """GIVEN MLOPS_DEPLOYMENT_ID is not set WHEN create_agent_card is called THEN the
        internal identity extension is absent.
        """
        with patch.dict(os.environ, {}, clear=True):
            card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[])

        extensions = card.capabilities.extensions or []
        assert not any(e.uri == INTERNAL_IDENTITY_URI for e in extensions)

    async def test_external_identity_extension_when_external_id_set(self, a2a_frontend_config):
        """GIVEN external.id is provided WHEN create_agent_card is called THEN the external
        identity extension is present with the correct id.
        """
        external = DRAgentA2AExternalConfig(id="catalog-id-xyz")
        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=None, skills=[], external=external
        )

        assert card.capabilities.extensions is not None
        uris = [ext.uri for ext in card.capabilities.extensions]
        assert EXTERNAL_IDENTITY_URI in uris
        ext = next(e for e in card.capabilities.extensions if e.uri == EXTERNAL_IDENTITY_URI)
        assert ext.required is False
        assert ext.params == {"id": "catalog-id-xyz"}

    async def test_no_external_identity_extension_when_external_absent(self, a2a_frontend_config):
        """GIVEN external is None WHEN create_agent_card is called THEN no external identity
        extension is present.
        """
        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=None, skills=[], external=None
        )

        extensions = card.capabilities.extensions or []
        assert not any(e.uri == EXTERNAL_IDENTITY_URI for e in extensions)

    async def test_external_url_overrides_agent_card_url(self, a2a_frontend_config):
        """GIVEN external.url is set WHEN create_agent_card is called THEN the agent card url
        uses the external URL exactly as provided.
        """
        external = DRAgentA2AExternalConfig(url="https://custom.example.com/agent/")
        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=None, skills=[], external=external
        )

        assert card.url == "https://custom.example.com/agent/"

    async def test_external_url_used_as_provided(self, a2a_frontend_config):
        """GIVEN external.url is set without a trailing slash WHEN create_agent_card is called
        THEN the url is used exactly as provided, without modification.
        """
        external = DRAgentA2AExternalConfig(url="https://custom.example.com/agent")
        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=None, skills=[], external=external
        )

        assert card.url == "https://custom.example.com/agent"

    async def test_all_extensions_combined(self, a2a_frontend_config):
        """GIVEN cross_app_access, MLOPS_DEPLOYMENT_ID, and external.id are all set WHEN
        create_agent_card is called THEN all three extensions are present.
        """
        cross_app_access = CrossApplicationAccessConfig(
            token_endpoint_auth_method="private_key_jwt",
            token_exchange=CrossAppTokenExchange(
                trusted_issuer="https://your-org.oktapreview.com",
                audience="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7",
            ),
            token_request=CrossAppTokenRequest(
                token_url="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7/v1/token",
                audience="https://app.datarobot.com/dr_org_id/my_agent_id",
            ),
        )
        external = DRAgentA2AExternalConfig(id="catalog-id-combined")
        env = {
            "MLOPS_DEPLOYMENT_ID": "dep-combined",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        }
        with patch.dict(os.environ, env):
            card = await create_agent_card(
                a2a_frontend_config,
                cross_app_access=cross_app_access,
                skills=[],
                external=external,
            )

        assert card.capabilities.extensions is not None
        uris = [ext.uri for ext in card.capabilities.extensions]
        assert JWT_BEARER_GRANT_TYPE_URI in uris
        assert INTERNAL_IDENTITY_URI in uris
        assert EXTERNAL_IDENTITY_URI in uris


class TestGetA2aEndpointUrl:
    def test_default(self):
        assert get_a2a_endpoint_url("localhost", 8000) == "http://localhost:8000/a2a/"

    @pytest.mark.parametrize(
        "env,expected",
        [
            (
                {
                    "MLOPS_DEPLOYMENT_ID": "abc123",
                    "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
                },
                "https://app.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/",
            ),
            (
                {
                    "MLOPS_DEPLOYMENT_ID": "abc123",
                    "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2/",
                },
                "https://app.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/",
            ),
            (
                {
                    "MLOPS_DEPLOYMENT_ID": "abc123",
                    "DATAROBOT_PUBLIC_API_ENDPOINT": "https://public.datarobot.com/api/v2",
                    "DATAROBOT_ENDPOINT": "https://internal.k8s.local/api/v2",
                },
                "https://public.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/",
            ),
        ],
    )
    def test_deployment(self, env, expected):
        with patch.dict(os.environ, env, clear=True):
            assert get_a2a_endpoint_url("localhost", 8000) == expected

    def test_deployment_missing_endpoint_raises(self):
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": "abc123"}, clear=True):
            with pytest.raises(
                ValueError, match="DATAROBOT_PUBLIC_API_ENDPOINT or DATAROBOT_ENDPOINT must be set"
            ):
                get_a2a_endpoint_url("localhost", 8000)
