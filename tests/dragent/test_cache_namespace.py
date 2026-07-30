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

from unittest.mock import patch

import pytest

from datarobot_genai.dragent.cache_namespace import build_namespaced_redis_prefix
from datarobot_genai.dragent.cache_namespace import require_cache_namespace
from datarobot_genai.dragent.cache_namespace import resolve_cache_namespace


class TestResolveCacheNamespace:
    def test_explicit_namespace(self):
        assert resolve_cache_namespace("my-agent-1") == "my-agent-1"

    def test_deployment_id_from_env(self):
        with patch.dict("os.environ", {"MLOPS_DEPLOYMENT_ID": "dep-abc"}, clear=False):
            assert resolve_cache_namespace() == "dep-abc"

    def test_workload_id_from_env(self):
        with patch.dict(
            "os.environ",
            {"WORKLOAD_ID": "wl-xyz"},
            clear=True,
        ):
            assert resolve_cache_namespace() == "wl-xyz"

    def test_explicit_overrides_deployment_id(self):
        with patch.dict("os.environ", {"MLOPS_DEPLOYMENT_ID": "dep-abc"}, clear=False):
            assert resolve_cache_namespace("custom-ns") == "custom-ns"

    def test_returns_none_without_sources(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_cache_namespace() is None

    def test_rejects_unsafe_namespace(self):
        with pytest.raises(ValueError, match="Cache namespace"):
            resolve_cache_namespace("bad namespace!")


class TestBuildNamespacedRedisPrefix:
    def test_appends_namespace_segment(self):
        assert build_namespaced_redis_prefix("dragent:", "dep-1") == "dragent:dep-1:"

    def test_normalizes_base_without_trailing_colon(self):
        assert build_namespaced_redis_prefix("dragent", "dep-1") == "dragent:dep-1:"


class TestRequireCacheNamespace:
    def test_raises_when_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="AGENT_CARD_REGISTRY_CACHE_NAMESPACE"):
                require_cache_namespace()
