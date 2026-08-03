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
from unittest.mock import patch

import pytest
from a2a.types import AgentSkill
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig

from datarobot_genai.dragent.cross_app_access_config import CrossApplicationAccessConfig
from datarobot_genai.dragent.cross_app_access_config import CrossAppTokenExchange
from datarobot_genai.dragent.cross_app_access_config import CrossAppTokenRequest
from datarobot_genai.dragent.frontends.a2a import EXTERNAL_IDENTITY_URI
from datarobot_genai.dragent.frontends.a2a import INTERNAL_IDENTITY_URI
from datarobot_genai.dragent.frontends.a2a import JWT_BEARER_GRANT_TYPE_URI
from datarobot_genai.dragent.frontends.a2a import _public_card_modifier
from datarobot_genai.dragent.frontends.a2a import create_agent_card
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
