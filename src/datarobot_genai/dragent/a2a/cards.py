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

"""Reading A2A agent cards of either protocol version into ``a2a-sdk`` 1.x's ``AgentCard``.

Used wherever dragent consumes other agents' cards: the A2A clients, the agent card
registry and its cache, and the Okta cross-app provider.
"""

import copy
import logging
import re
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Annotated
from typing import Any

import httpx
from a2a.client.card_resolver import parse_agent_card as _sdk_parse_agent_card
from a2a.types import AgentCard
from a2a.types import AgentInterface
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from a2a.utils.constants import PROTOCOL_VERSION_0_3
from a2a.utils.constants import PROTOCOL_VERSION_1_0
from a2a.utils.constants import TransportProtocol
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import ParseError
from pydantic import PlainSerializer
from pydantic import PlainValidator
from pydantic import WithJsonSchema

logger = logging.getLogger(__name__)

JSONRPC = TransportProtocol.JSONRPC.value


def card_rpc_url(card: AgentCard) -> str:
    """Return the URL a client sends JSON-RPC calls to.

    v0.3 cards carry it as the top-level ``url``, which parses into the card's first
    interface; v1.0 cards list it in ``supportedInterfaces``. When a card offers JSON-RPC at
    several protocol versions, this picks the interface the SDK's ``ClientFactory`` picks
    (v1.0 first, then the newest supported), so it is the endpoint the client actually
    calls. Falls back to the first interface of any binding, else ``""``.
    """
    interface = best_interface(card.supported_interfaces, JSONRPC)
    if interface is not None:
        return interface.url
    return card.supported_interfaces[0].url if card.supported_interfaces else ""


# ``major.minor`` of a protocol version; the patch level and any suffix do not matter.
_MAJOR_MINOR = re.compile(r"\s*[vV]?(\d+)(?:\.(\d+))?")


def _major_minor(version: str) -> tuple[int, int] | None:
    match = _MAJOR_MINOR.match(version)
    if match is None:
        return None
    return int(match[1]), int(match[2] or 0)


_V1_0 = _major_minor(PROTOCOL_VERSION_1_0) or (1, 0)
_V0_3 = _major_minor(PROTOCOL_VERSION_0_3) or (0, 3)


def best_interface(
    interfaces: Sequence[AgentInterface], protocol_binding: str
) -> AgentInterface | None:
    """Return the interface the SDK's client calls for ``protocol_binding``, if any.

    The first v1.0 interface wins, then the first newer than v1.0, then the first from v0.3 on,
    then the first without a version. This follows ``ClientFactory._find_best_interface`` of
    a2a-sdk 1.x, which is private and so not called here; ``test_cards`` checks they agree.
    """
    newer_than_v1_0: AgentInterface | None = None
    from_v0_3: AgentInterface | None = None
    without_version: AgentInterface | None = None
    for interface in interfaces:
        if interface.protocol_binding != protocol_binding:
            continue
        if not interface.protocol_version:
            if without_version is None:
                without_version = interface
            continue
        version = _major_minor(interface.protocol_version)
        if version is None:
            continue
        if version == _V1_0:
            return interface
        if newer_than_v1_0 is None and version > _V1_0:
            newer_than_v1_0 = interface
        if from_v0_3 is None and version >= _V0_3:
            from_v0_3 = interface
    return newer_than_v1_0 or from_v0_3 or without_version


def _keep_one_oauth_flow(card_data: dict[str, Any]) -> None:
    """Reduce every OAuth2 scheme of a v0.3 card to a single flow, in place.

    A v0.3 scheme may list several flows, as dragent's own cards did before 0.33.0. A v1.0
    scheme has exactly one and the SDK's parser rejects more, which would make such an agent
    uncallable. Cross-application access reads ``clientCredentials``, so that flow is kept.
    """
    for name, scheme in (card_data.get("securitySchemes") or {}).items():
        flows = scheme.get("flows") if isinstance(scheme, dict) else None
        if not isinstance(flows, dict):
            continue
        # Cache entries written before 0.33.0 list every flow, the unused ones as null.
        flows = {key: flow for key, flow in flows.items() if flow is not None}
        if len(flows) > 1:
            kept = "clientCredentials" if "clientCredentials" in flows else next(iter(flows))
            logger.info(
                "Scheme %r of an agent card lists several OAuth flows; keeping %s.", name, kept
            )
            flows = {kept: flows[kept]}
        scheme["flows"] = flows


def parse_agent_card(card_data: Mapping[str, Any]) -> AgentCard:
    """Parse agent card JSON of either protocol version into a protobuf ``AgentCard``.

    The input is not modified.

    Raises
    ------
    ValueError
        The data is not a valid agent card, or lacks a name or an interface URL.
    """
    if not isinstance(card_data, Mapping):
        raise ValueError("Invalid agent card: not a JSON object")
    data = copy.deepcopy(dict(card_data))
    _keep_one_oauth_flow(data)
    try:
        card = _sdk_parse_agent_card(data)
    except ParseError as exc:
        raise ValueError(f"Invalid agent card: {exc}") from exc
    # Protobuf parsing ignores unknown keys, so any JSON object would otherwise parse into
    # an empty card. Require what a client needs to call the agent at all.
    if not card.name:
        raise ValueError("Invalid agent card: missing 'name'")
    if not card_rpc_url(card):
        raise ValueError("Invalid agent card: missing 'url' / 'supportedInterfaces'")
    return card


async def fetch_agent_card(
    httpx_client: httpx.AsyncClient,
    base_url: str,
    agent_card_path: str = AGENT_CARD_WELL_KNOWN_PATH,
) -> AgentCard:
    """GET the agent card at *agent_card_path* under *base_url* and parse it.

    Replaces ``a2a.client.A2ACardResolver``, which rejects v0.3 cards whose OAuth2 scheme
    lists several flows (see :func:`parse_agent_card`).

    Raises
    ------
    httpx.HTTPError
        The request failed or returned an error status.
    ValueError
        The response is not a valid agent card.
    """
    response = await httpx_client.get(f"{base_url.rstrip('/')}/{agent_card_path.lstrip('/')}")
    response.raise_for_status()
    return parse_agent_card(response.json())


def _validate_agent_card(value: Any) -> AgentCard:
    return value if isinstance(value, AgentCard) else parse_agent_card(value)


# A protobuf ``AgentCard`` as a pydantic field: serialized as v1.0 JSON, validated from card
# JSON of either version, so cache entries written by earlier releases (v0.3) still load.
AgentCardModel = Annotated[
    AgentCard,
    PlainValidator(_validate_agent_card),
    PlainSerializer(MessageToDict, return_type=dict),
    WithJsonSchema({"type": "object"}),
]
