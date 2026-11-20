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
# v0.3 agent cards are read the way earlier releases read them: the JSON-RPC URL, the OAuth
# flows (including the nulls older cache entries carry) and the Cross-Application Access
# parameters. Cards in this form are what agents published and what dragent cached before.

from typing import Any

from datarobot_genai.dragent.plugins.okta_a2a_auth import _parse_cross_app_params

from .sdk_adapter import authorization_code_token_url
from .sdk_adapter import card_rpc_url
from .sdk_adapter import parse_card

URL = "https://agent.example.com/a2a/"

_TRUSTED_ISSUER = "https://okta.example.com"
_AGENT_TOKEN_URL = "https://okta.example.com/oauth2/ausYYY/v1/token"
_TARGET_AUDIENCE = "https://api.agent.example.com"
_EXCHANGE_AUDIENCE = "https://okta.example.com/oauth2/ausYYY"

_CROSS_APP_EXTENSION = {
    "uri": "urn:ietf:params:oauth:grant-type:jwt-bearer",
    "params": {
        "tokenEndpointAuthMethod": "private_key_jwt",
        "tokenExchange": {"trustedIssuer": _TRUSTED_ISSUER, "audience": _EXCHANGE_AUDIENCE},
        "tokenRequest": {
            "grantType": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "audience": _TARGET_AUDIENCE,
        },
    },
}


def _v03_card(**overrides: Any) -> dict[str, Any]:
    return {
        "name": "Test Agent",
        "description": "A test agent",
        "url": URL,
        "version": "1.0.0",
        "capabilities": {},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
        **overrides,
    }


def test_v03_card_url_is_the_rpc_url() -> None:
    # GIVEN the JSON of a v0.3 card, whose ``url`` is where JSON-RPC is served
    card_json = _v03_card()

    # WHEN it is parsed
    card = parse_card(card_json)

    # THEN dragent calls that URL, and the card's own fields are intact
    assert card_rpc_url(card) == URL
    assert card.name == "Test Agent"
    assert card.version == "1.0.0"


def test_null_oauth_flows_from_older_cache_entries_are_ignored() -> None:
    # GIVEN a v0.3 card as dragent before 0.33.0 cached it: every OAuth flow key present,
    # the unused ones null
    card_json = _v03_card(
        securitySchemes={
            "oauth2": {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": "https://issuer.example.com/authorize",
                        "tokenUrl": "https://issuer.example.com/token",
                        "scopes": {"read": "Permission: read"},
                    },
                    "clientCredentials": None,
                    "implicit": None,
                    "password": None,
                },
            }
        }
    )

    # WHEN it is parsed
    card = parse_card(card_json)

    # THEN the one real flow is kept
    assert authorization_code_token_url(card) == "https://issuer.example.com/token"


def test_legacy_card_with_both_flows_under_oauth2_uses_client_credentials() -> None:
    # GIVEN a v0.3 card listing both an authorization-code and a client-credentials flow
    # under oauth2, with the Cross-Application Access extension
    card_json = _v03_card(
        url="https://agent.example.com/",
        capabilities={"streaming": False, "extensions": [_CROSS_APP_EXTENSION]},
        securitySchemes={
            "oauth2": {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": "https://okta.example.com/authorize",
                        "tokenUrl": "https://okta.example.com/token",
                        "scopes": {"openid": "OpenID scope"},
                    },
                    "clientCredentials": {
                        "tokenUrl": _AGENT_TOKEN_URL,
                        "scopes": {"read_data": "Read access"},
                    },
                },
            }
        },
    )

    # WHEN it is parsed and read for Cross-Application Access
    params = _parse_cross_app_params(parse_card(card_json))

    # THEN the client-credentials flow is the one used
    assert params.token_url == _AGENT_TOKEN_URL
    assert params.id_jag_scopes == ["read_data"]
    assert params.trusted_issuer == _TRUSTED_ISSUER
