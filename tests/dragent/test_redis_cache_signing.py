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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.dragent.agent_card_registry_backends import AgentCardCacheRecord
from datarobot_genai.dragent.redis_cache_signing import open_redis_model
from datarobot_genai.dragent.redis_cache_signing import resolve_redis_signing_key
from datarobot_genai.dragent.redis_cache_signing import seal_redis_model


class TestRedisCacheSigning:
    def test_seal_and_open_round_trip(self):
        key = resolve_redis_signing_key("unit-test-secret")
        record = AgentCardCacheRecord(card=MagicMock())
        sealed = seal_redis_model(record, signing_key=key)
        opened = open_redis_model(AgentCardCacheRecord, sealed, signing_key=key)
        assert opened is not None
        assert opened.card is record.card

    def test_rejects_tampered_payload(self):
        key = resolve_redis_signing_key("unit-test-secret")
        record = AgentCardCacheRecord(card=MagicMock())
        sealed = seal_redis_model(record, signing_key=key)
        tampered = sealed.replace("registry", "attacker")
        assert open_redis_model(AgentCardCacheRecord, tampered, signing_key=key) is None

    def test_rejects_unsigned_payload(self):
        key = resolve_redis_signing_key("unit-test-secret")
        unsigned = AgentCardCacheRecord(card=MagicMock()).model_dump_json()
        assert open_redis_model(AgentCardCacheRecord, unsigned, signing_key=key) is None

    def test_requires_signing_secret(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="AGENT_CARD_REGISTRY_REDIS_SIGNING_KEY"):
                resolve_redis_signing_key()
