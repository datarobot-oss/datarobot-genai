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

"""Per-deployment Redis cache namespace for dragent shared caches.

When many user-modifiable agent deployments share one Redis instance, keys must
be isolated per local deployment (or an explicit namespace) so co-tenants cannot
read, overwrite, or poison each other's agent card or XAA token entries.
"""

from __future__ import annotations

import re

from datarobot_genai.core.runtime import get_deployment_id
from datarobot_genai.core.runtime import get_workload_id

_NAMESPACE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_REDIS_NAMESPACE_REQUIRED_MSG = (
    "Redis cache backends require a per-deployment namespace. Set "
    "AGENT_CARD_REGISTRY_CACHE_NAMESPACE, or run on a DataRobot-hosted "
    "deployment (MLOPS_DEPLOYMENT_ID) or workload (WORKLOAD_ID) so the "
    "namespace can be derived automatically."
)


def _validate_namespace(namespace: str) -> str:
    """Return *namespace* when it is a safe Redis key segment."""
    candidate = namespace.strip()
    if not candidate or not _NAMESPACE_PATTERN.fullmatch(candidate):
        raise ValueError(
            "Cache namespace must be 1-128 characters of [A-Za-z0-9._-] "
            f"and start with an alphanumeric character; got {namespace!r}."
        )
    return candidate


def resolve_cache_namespace(explicit: str | None = None) -> str | None:
    """Resolve the cache namespace for the current process.

    Priority:

    1. Explicit ``AGENT_CARD_REGISTRY_CACHE_NAMESPACE`` (or constructor arg).
    2. Platform-injected ``MLOPS_DEPLOYMENT_ID``.
    3. Platform-injected ``WORKLOAD_ID``.
    """
    if explicit is not None and explicit.strip():
        return _validate_namespace(explicit)
    if deployment_id := get_deployment_id():
        return _validate_namespace(deployment_id)
    if workload_id := get_workload_id():
        return _validate_namespace(workload_id)
    return None


def require_cache_namespace(explicit: str | None = None) -> str:
    """Return the cache namespace or raise when Redis backends cannot be used."""
    namespace = resolve_cache_namespace(explicit)
    if namespace is None:
        raise ValueError(_REDIS_NAMESPACE_REQUIRED_MSG)
    return namespace


def build_namespaced_redis_prefix(base_prefix: str, namespace: str) -> str:
    """Compose ``{base_prefix}{namespace}:`` for Redis key isolation."""
    normalized_base = base_prefix if base_prefix.endswith(":") else f"{base_prefix}:"
    return f"{normalized_base}{_validate_namespace(namespace)}:"
