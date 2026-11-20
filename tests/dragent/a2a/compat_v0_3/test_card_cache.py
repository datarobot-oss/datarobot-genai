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
# Agent cards in the registry cache. A cache written by the previous release holds v0.3 card
# JSON, and a record must survive a round trip through JSON, so a new release can read what the
# old one stored (Redis, files) without refetching or failing.

from datetime import UTC
from datetime import datetime

from datarobot_genai.dragent.agent_card_registry_backends import AgentCardCacheRecord

from .sdk_adapter import parse_card

_V03_CARD = {
    "name": "Test Agent",
    "description": "A test agent",
    "url": "https://agent.example.com/a2a/",
    "version": "1.0.0",
    "skills": [],
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False},
}


def test_json_round_trip_preserves_card() -> None:
    # GIVEN a cache record
    record = AgentCardCacheRecord(card=parse_card(_V03_CARD))

    # WHEN it is dumped to JSON and loaded back
    restored = AgentCardCacheRecord.model_validate_json(record.model_dump_json())

    # THEN the card is unchanged
    assert restored.card == record.card


def test_version_1_payload_with_v03_card_still_validates() -> None:
    # GIVEN a version-1 payload written by the previous release (v0.3 card JSON)
    payload = {
        "version": 1,
        "fetched_at": datetime.now(UTC).isoformat(),
        "card": _V03_CARD,
        "source": "registry",
        "deployment_id": "dep-1",
    }

    # WHEN it is validated
    record = AgentCardCacheRecord.model_validate(payload)

    # THEN the card parses
    assert record.card == parse_card(_V03_CARD)
