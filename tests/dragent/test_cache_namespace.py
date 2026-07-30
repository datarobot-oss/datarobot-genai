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

from datarobot_genai.dragent.cache_namespace import ResolvedCacheNamespace
from datarobot_genai.dragent.cache_namespace import build_namespaced_redis_prefix
from datarobot_genai.dragent.cache_namespace import require_cache_namespace
from datarobot_genai.dragent.cache_namespace import resolve_cache_namespace


class TestResolveCacheNamespace:
    def test_explicit_namespace_for_local_dev(self):
        with patch.dict("os.environ", {}, clear=True):
            resolved = resolve_cache_namespace("my-agent-1")
        assert resolved == ResolvedCacheNamespace("my-agent-1", "explicit")

    def test_deployment_id_from_env(self):
        with patch.dict("os.environ", {"MLOPS_DEPLOYMENT_ID": "dep-abc"}, clear=True):
            resolved = resolve_cache_namespace()
        assert resolved == ResolvedCacheNamespace("dep-abc", "deployment")

    def test_workload_id_from_env(self):
        with patch.dict("os.environ", {"WORKLOAD_ID": "wl-xyz"}, clear=True):
            resolved = resolve_cache_namespace()
        assert resolved == ResolvedCacheNamespace("wl-xyz", "workload")

    def test_deployment_id_wins_over_explicit_on_hosted_runtime(self):
        with patch.dict("os.environ", {"MLOPS_DEPLOYMENT_ID": "dep-abc"}, clear=True):
            resolved = resolve_cache_namespace("custom-ns")
        assert resolved == ResolvedCacheNamespace("dep-abc", "deployment")

    def test_rejects_conflicting_explicit_on_hosted_deployment(self):
        with patch.dict("os.environ", {"MLOPS_DEPLOYMENT_ID": "dep-abc"}, clear=True):
            with pytest.raises(ValueError, match="cannot override MLOPS_DEPLOYMENT_ID"):
                resolve_cache_namespace("other-ns")

    def test_rejects_conflicting_explicit_on_hosted_workload(self):
        with patch.dict("os.environ", {"WORKLOAD_ID": "wl-xyz"}, clear=True):
            with pytest.raises(ValueError, match="cannot override WORKLOAD_ID"):
                resolve_cache_namespace("other-ns")

    def test_returns_none_without_sources(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_cache_namespace() is None

    def test_rejects_unsafe_namespace(self):
        with pytest.raises(ValueError, match="Cache namespace"):
            resolve_cache_namespace("bad namespace!")


class TestBuildNamespacedRedisPrefix:
    def test_deployment_prefix(self):
        resolved = ResolvedCacheNamespace("dep-1", "deployment")
        assert build_namespaced_redis_prefix("dragent:", resolved) == "dragent:dep:dep-1:"

    def test_workload_prefix(self):
        resolved = ResolvedCacheNamespace("wl-1", "workload")
        assert build_namespaced_redis_prefix("dragent:", resolved) == "dragent:wl:wl-1:"

    def test_explicit_dev_prefix(self):
        resolved = ResolvedCacheNamespace("local-1", "explicit")
        assert build_namespaced_redis_prefix("dragent", resolved) == "dragent:dev:local-1:"


class TestRequireCacheNamespace:
    def test_raises_when_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="MLOPS_DEPLOYMENT_ID"):
                require_cache_namespace()
