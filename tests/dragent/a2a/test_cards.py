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

import copy
from typing import Any

import pytest
from a2a.client import ClientFactory
from a2a.types import AgentCard
from a2a.types import AgentInterface
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel

from datarobot_genai.dragent.a2a.cards import AgentCardModel
from datarobot_genai.dragent.a2a.cards import best_interface
from datarobot_genai.dragent.a2a.cards import card_rpc_url
from datarobot_genai.dragent.a2a.cards import parse_agent_card

URL = "https://agent.example.com/a2a/"

V03_CARD: dict[str, Any] = {
    "name": "Test Agent",
    "description": "A test agent",
    "url": URL,
    "version": "1.0.0",
    "capabilities": {},
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "skills": [],
}

V10_CARD: dict[str, Any] = {
    "name": "Test Agent",
    "description": "A test agent",
    "version": "1.0.0",
    "supportedInterfaces": [
        {"url": URL, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
    ],
    "capabilities": {},
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "skills": [],
}


def _v03_card_with_two_oauth_flows() -> dict[str, Any]:
    card = copy.deepcopy(V03_CARD)
    card["securitySchemes"] = {
        "oauth": {
            "type": "oauth2",
            "flows": {
                "authorizationCode": {
                    "authorizationUrl": "https://idp.example.com/authorize",
                    "tokenUrl": "https://idp.example.com/token",
                    "scopes": {"read": "Read"},
                },
                "clientCredentials": {
                    "tokenUrl": "https://idp.example.com/token",
                    "scopes": {"read": "Read"},
                },
            },
        }
    }
    return card


class TestParseAgentCard:
    def test_v03_card_url_becomes_rpc_url(self) -> None:
        # GIVEN a v0.3 card with a top-level url
        # WHEN parsing it
        card = parse_agent_card(V03_CARD)

        # THEN the url is the card's RPC url
        assert card.name == "Test Agent"
        assert card_rpc_url(card) == URL

    def test_v10_card_reads_supported_interfaces(self) -> None:
        # GIVEN a v1.0 card with supportedInterfaces
        # WHEN parsing it
        card = parse_agent_card(V10_CARD)

        # THEN the interface is kept and gives the RPC url
        assert len(card.supported_interfaces) == 1
        assert card.supported_interfaces[0].protocol_binding == "JSONRPC"
        assert card_rpc_url(card) == URL

    def test_oauth2_scheme_with_several_flows_keeps_client_credentials(self) -> None:
        # GIVEN a v0.3 card whose oauth2 scheme lists two flows
        data = _v03_card_with_two_oauth_flows()

        # WHEN parsing it
        card = parse_agent_card(data)

        # THEN only the clientCredentials flow remains
        scheme = card.security_schemes["oauth"].oauth2_security_scheme
        assert scheme.flows.WhichOneof("flow") == "client_credentials"
        assert scheme.flows.client_credentials.token_url == "https://idp.example.com/token"

    def test_parsing_does_not_mutate_input(self) -> None:
        # GIVEN a v0.3 card with two oauth flows
        data = _v03_card_with_two_oauth_flows()
        original = copy.deepcopy(data)

        # WHEN parsing it
        parse_agent_card(data)

        # THEN the input still lists both flows
        assert data == original

    @pytest.mark.parametrize(
        "bad",
        [
            {"name": 5},
            {"name": "x", "skills": "no"},
            {"bad": True},
            {"name": "x", "description": "no url"},
            {"url": "http://agent.example.com/a2a/"},
        ],
        ids=["wrong-type", "wrong-container", "unknown-keys", "no-url", "no-name"],
    )
    def test_invalid_card_raises_value_error(self, bad: dict[str, Any]) -> None:
        # GIVEN card data that is wrongly typed, or that a client could not call
        # WHEN parsing it
        # THEN a ValueError is raised
        with pytest.raises(ValueError):
            parse_agent_card(bad)


class TestCardRpcUrl:
    def test_prefers_jsonrpc_interface(self) -> None:
        # GIVEN a card whose first interface is not JSON-RPC
        card = AgentCard(
            supported_interfaces=[
                AgentInterface(url="https://rest.example.com", protocol_binding="HTTP+JSON"),
                AgentInterface(url="https://rpc.example.com", protocol_binding="JSONRPC"),
            ]
        )

        # WHEN asking for the RPC url
        # THEN the JSON-RPC interface wins
        assert card_rpc_url(card) == "https://rpc.example.com"


class TestBestInterface:
    @pytest.mark.parametrize(
        "interfaces",
        [
            pytest.param([("JSONRPC", "0.3"), ("JSONRPC", "1.0")], id="v1.0-over-v0.3"),
            pytest.param([("JSONRPC", "0.3.0"), ("JSONRPC", "1.0.0")], id="patch-ignored"),
            pytest.param([("JSONRPC", "1.1"), ("JSONRPC", "2.0"), ("JSONRPC", "0.3")], id="newer"),
            pytest.param([("JSONRPC", "0.2"), ("JSONRPC", "")], id="no-version-last"),
            pytest.param([("JSONRPC", "0.2")], id="too-old"),
            pytest.param([("JSONRPC", "1")], id="major-only"),
            pytest.param([("JSONRPC", "v1.0"), ("JSONRPC", "0.3")], id="v-prefix"),
            pytest.param([("JSONRPC", "1.0rc1"), ("JSONRPC", "0.3")], id="pre-release"),
            pytest.param([("GRPC", "1.0"), ("JSONRPC", "0.3")], id="other-binding"),
            pytest.param([("HTTP+JSON", "1.0")], id="no-jsonrpc"),
            pytest.param([], id="empty"),
        ],
    )
    def test_matches_the_sdk_client(self, interfaces: list[tuple[str, str]]) -> None:
        # GIVEN a card's interfaces
        candidates = [
            AgentInterface(
                url=f"https://agent-{i}.example.com",
                protocol_binding=binding,
                protocol_version=version,
            )
            for i, (binding, version) in enumerate(interfaces)
        ]

        # WHEN dragent and the SDK's client each pick the JSON-RPC interface
        ours = best_interface(candidates, "JSONRPC")
        sdk = ClientFactory._find_best_interface(candidates, protocol_bindings=["JSONRPC"])

        # THEN they pick the same one, so ``card_rpc_url`` is the URL the client calls
        assert (ours.url if ours else None) == (sdk.url if sdk else None)


class _Holder(BaseModel):
    card: AgentCardModel


class TestAgentCardModel:
    def test_serialize_then_validate_round_trips(self) -> None:
        # GIVEN a model holding a card
        holder = _Holder(card=parse_agent_card(V03_CARD))

        # WHEN dumping to JSON and validating it back
        restored = _Holder.model_validate_json(holder.model_dump_json())

        # THEN the card is equal and the JSON holds lossless v1.0 data
        assert restored.card == holder.card
        assert holder.model_dump(mode="json")["card"] == MessageToDict(holder.card)

    def test_validates_from_v03_json(self) -> None:
        # GIVEN v0.3 card JSON, as written by earlier releases
        # WHEN validating it into the model
        holder = _Holder.model_validate({"card": V03_CARD})

        # THEN it parses to the same card as parse_agent_card
        assert holder.card == parse_agent_card(V03_CARD)
        assert card_rpc_url(holder.card) == URL


def test_null_oauth_flows_from_older_cache_entries_are_ignored() -> None:
    # GIVEN a v0.3 card as dragent before 0.33.0 cached it: every OAuth flow key present,
    # the unused ones null
    card_data = {
        "name": "Agent",
        "description": "d",
        "url": "http://agent.example.com/a2a/",
        "version": "1.0.0",
        "capabilities": {},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
        "securitySchemes": {
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
        },
    }

    # WHEN parsing it
    card = parse_agent_card(card_data)

    # THEN the one real flow is kept
    flows = card.security_schemes["oauth2"].oauth2_security_scheme.flows
    assert flows.WhichOneof("flow") == "authorization_code"
    assert flows.authorization_code.token_url == "https://issuer.example.com/token"
