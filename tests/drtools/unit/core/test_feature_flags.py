# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Iterator
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drtools.core import feature_flags as feature_flags_module
from datarobot_genai.drtools.core.feature_flags import FeatureFlag
from datarobot_genai.drtools.core.feature_flags import is_tool_feature_enabled


def _mock_client(token: str | None = "tok", *, value: bool = True, name: str = "FLAG") -> Mock:
    client = Mock()
    client.token = token
    client.post.return_value.json.return_value = {"entitlements": [{"name": name, "value": value}]}
    return client


def _mock_principal_for(client: Mock) -> str:
    import hashlib

    token = client.token or ""
    if not token:
        return "__no_token__"
    return hashlib.sha256(token.encode()).hexdigest()[:16]


class TestFeatureFlag:
    @pytest.fixture
    def module_under_test(self) -> str:
        return "datarobot_genai.drtools.core.feature_flags"

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> Iterator[None]:
        feature_flags_module._eval_cache.clear()
        yield
        feature_flags_module._eval_cache.clear()

    @pytest.fixture
    def mock_feature_flag_create(self) -> Iterator[Mock]:
        with patch.object(FeatureFlag, "create") as mock_func:
            yield mock_func

    def test_create_posts_with_provided_client(self) -> None:
        client = _mock_client(name="MY_FLAG", value=True)

        output = FeatureFlag.create("MY_FLAG", client=client)

        client.post.assert_called_once_with(
            "entitlements/evaluate/",
            json={"entitlements": [{"name": "MY_FLAG"}]},
        )
        assert output == FeatureFlag(name="MY_FLAG", enabled=True)

    def test_is_enabled_caches_within_ttl(self, mock_feature_flag_create: Mock) -> None:
        mock_feature_flag_create.return_value = FeatureFlag(name="MY_FLAG", enabled=True)
        client = _mock_client()

        first = FeatureFlag.is_enabled("MY_FLAG", client=client)
        second = FeatureFlag.is_enabled("MY_FLAG", client=client)

        assert first is True and second is True
        mock_feature_flag_create.assert_called_once_with("MY_FLAG", client=client)

    def test_is_enabled_bypasses_cache_when_ttl_zero(self, mock_feature_flag_create: Mock) -> None:
        mock_feature_flag_create.return_value = FeatureFlag(name="MY_FLAG", enabled=True)
        client = _mock_client()

        FeatureFlag.is_enabled("MY_FLAG", client=client, ttl_seconds=0)
        FeatureFlag.is_enabled("MY_FLAG", client=client, ttl_seconds=0)

        assert mock_feature_flag_create.call_count == 2

    def test_is_enabled_separates_cache_by_principal(self, mock_feature_flag_create: Mock) -> None:
        client_a = Mock(token="user-a")
        client_b = Mock(token="user-b")
        mock_feature_flag_create.side_effect = [
            FeatureFlag(name="MY_FLAG", enabled=True),
            FeatureFlag(name="MY_FLAG", enabled=False),
        ]

        a = FeatureFlag.is_enabled("MY_FLAG", client=client_a)
        b = FeatureFlag.is_enabled("MY_FLAG", client=client_b)
        a_again = FeatureFlag.is_enabled("MY_FLAG", client=client_a)
        b_again = FeatureFlag.is_enabled("MY_FLAG", client=client_b)

        assert (a, a_again) == (True, True)
        assert (b, b_again) == (False, False)
        assert mock_feature_flag_create.call_count == 2

    def test_is_enabled_expires_after_ttl(
        self, mock_feature_flag_create: Mock, module_under_test: str
    ) -> None:
        mock_feature_flag_create.return_value = FeatureFlag(name="MY_FLAG", enabled=True)
        client = _mock_client()

        with patch(f"{module_under_test}.time.monotonic") as mock_time:
            mock_time.side_effect = [0.0, 1000.0]
            FeatureFlag.is_enabled("MY_FLAG", client=client, ttl_seconds=300)
            FeatureFlag.is_enabled("MY_FLAG", client=client, ttl_seconds=300)

        assert mock_feature_flag_create.call_count == 2

    def test_is_mcp_tools_gallery_support_enabled_forwards_client(
        self, mock_feature_flag_create: Mock
    ) -> None:
        mock_feature_flag_create.return_value = FeatureFlag(
            name="ENABLE_MCP_TOOLS_GALLERY_SUPPORT", enabled=True
        )
        client = _mock_client()

        output = FeatureFlag.is_mcp_tools_gallery_support_enabled(client=client)

        mock_feature_flag_create.assert_called_once_with(
            "ENABLE_MCP_TOOLS_GALLERY_SUPPORT", client=client
        )
        assert output is True

    def test_principal_key_returns_sentinel_when_token_missing(
        self, mock_feature_flag_create: Mock
    ) -> None:
        # Clients without a `.token` attribute (or with empty token) should share
        # the same cache slot so unauthenticated lookups don't blow up.
        mock_feature_flag_create.return_value = FeatureFlag(name="FLAG", enabled=True)
        client_no_token = Mock(spec=[])  # no .token attribute
        client_empty = Mock(token="")

        FeatureFlag.is_enabled("FLAG", client=client_no_token)
        FeatureFlag.is_enabled("FLAG", client=client_empty)

        # Both lookups hit the same cache entry → only one upstream call
        assert mock_feature_flag_create.call_count == 1
        assert ("FLAG", "__no_token__") in feature_flags_module._eval_cache

    def test_prune_expired_drops_only_expired_entries_when_over_cap(
        self, mock_feature_flag_create: Mock, module_under_test: str
    ) -> None:
        # Pre-fill the cache to capacity with entries of mixed ages so that
        # _prune_expired must run and selectively drop only the expired ones.
        max_entries = feature_flags_module._MAX_CACHE_ENTRIES
        ttl = 10.0
        for i in range(max_entries):
            # Half are "old" (will be expired), half are "fresh".
            ts = 0.0 if i % 2 == 0 else 1000.0
            feature_flags_module._eval_cache[(f"FLAG_{i}", "p")] = (ts, True)
        assert len(feature_flags_module._eval_cache) == max_entries

        mock_feature_flag_create.return_value = FeatureFlag(name="NEW", enabled=True)
        client = _mock_client()
        with patch(f"{module_under_test}.time.monotonic", return_value=1001.0):
            FeatureFlag.is_enabled("NEW", client=client, ttl_seconds=ttl)

        # All "old" half should be gone (1001.0 - 0.0 >= 10.0 = expired);
        # all "fresh" half should remain (1001.0 - 1000.0 < 10.0); new entry added.
        for i in range(max_entries):
            present = (f"FLAG_{i}", "p") in feature_flags_module._eval_cache
            if i % 2 == 0:
                assert not present, f"FLAG_{i} should have been pruned (expired)"
            else:
                assert present, f"FLAG_{i} should still be cached (fresh)"
        assert ("NEW", _mock_principal_for(client)) in feature_flags_module._eval_cache

    def test_prune_expired_evicts_oldest_when_no_expired_entries(
        self, mock_feature_flag_create: Mock, module_under_test: str
    ) -> None:
        # Pre-fill the cache to capacity with entries that are all still within
        # TTL — _prune_expired must fall through to LRU-style oldest eviction
        # to make room for the new entry.
        max_entries = feature_flags_module._MAX_CACHE_ENTRIES
        ttl = 10_000.0
        # Ascending timestamps so the smallest-ts entry is the unambiguous oldest.
        for i in range(max_entries):
            feature_flags_module._eval_cache[(f"FLAG_{i}", "p")] = (float(i), True)

        mock_feature_flag_create.return_value = FeatureFlag(name="NEW", enabled=True)
        client = _mock_client()
        with patch(f"{module_under_test}.time.monotonic", return_value=10.0):
            FeatureFlag.is_enabled("NEW", client=client, ttl_seconds=ttl)

        # Oldest (FLAG_0 @ ts=0.0) should be evicted; rest still present; new added.
        assert ("FLAG_0", "p") not in feature_flags_module._eval_cache
        assert ("FLAG_1", "p") in feature_flags_module._eval_cache
        assert ("NEW", _mock_principal_for(client)) in feature_flags_module._eval_cache
        assert len(feature_flags_module._eval_cache) == max_entries


class TestIsToolFeatureEnabled:
    """The shared gating policy used by every MCP tool registry."""

    def test_no_flag_is_always_enabled_without_evaluating(self) -> None:
        evaluator = Mock()
        assert is_tool_feature_enabled(None, evaluator=evaluator) is True
        evaluator.assert_not_called()

    def test_delegates_to_evaluator_when_flagged(self) -> None:
        evaluator = Mock(return_value=True)
        assert is_tool_feature_enabled("MCP_SANDBOX", evaluator=evaluator) is True
        evaluator.assert_called_once_with("MCP_SANDBOX")

    def test_disabled_flag_gates_off(self) -> None:
        assert is_tool_feature_enabled("MCP_SANDBOX", evaluator=lambda _: False) is False

    def test_fails_closed_when_evaluator_raises(self) -> None:
        def boom(_: str) -> bool:
            raise RuntimeError("DR client unavailable")

        assert is_tool_feature_enabled("MCP_SANDBOX", evaluator=boom) is False
