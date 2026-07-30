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

"""Per-deployment/workload Redis cache namespace for dragent shared caches.

Platform-injected deployment or workload IDs always define the namespace on
hosted runtimes. A manual ``AGENT_CARD_REGISTRY_CACHE_NAMESPACE`` is accepted
only for local development (no platform IDs) and cannot override hosted IDs.
"""

from __future__ import annotations

import re
from typing import Literal
from typing import NamedTuple

from datarobot_genai.core.runtime import get_deployment_id
from datarobot_genai.core.runtime import get_workload_id

NamespaceSource = Literal["deployment", "workload", "explicit"]

_NAMESPACE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_SOURCE_PREFIX_SEGMENT: dict[NamespaceSource, str] = {
    "deployment": "dep",
    "workload": "wl",
    "explicit": "dev",
}

_REDIS_NAMESPACE_REQUIRED_MSG = (
    "Redis cache backends require a per-deployment or per-workload namespace. "
    "Run on a DataRobot-hosted deployment (MLOPS_DEPLOYMENT_ID) or workload "
    "(WORKLOAD_ID), or set AGENT_CARD_REGISTRY_CACHE_NAMESPACE for local "
    "development only."
)


class ResolvedCacheNamespace(NamedTuple):
    """A cache namespace bound to a single deployment or workload."""

    namespace: str
    source: NamespaceSource


def _validate_namespace(namespace: str) -> str:
    """Return *namespace* when it is a safe Redis key segment."""
    candidate = namespace.strip()
    if not candidate or not _NAMESPACE_PATTERN.fullmatch(candidate):
        raise ValueError(
            "Cache namespace must be 1-128 characters of [A-Za-z0-9._-] "
            f"and start with an alphanumeric character; got {namespace!r}."
        )
    return candidate


def _explicit_namespace(explicit: str | None) -> str | None:
    if explicit is None or not explicit.strip():
        return None
    return _validate_namespace(explicit)


def resolve_cache_namespace(explicit: str | None = None) -> ResolvedCacheNamespace | None:
    """Resolve the cache namespace for the current process.

    On hosted runtimes, ``MLOPS_DEPLOYMENT_ID`` or ``WORKLOAD_ID`` always wins.
    ``AGENT_CARD_REGISTRY_CACHE_NAMESPACE`` is used only when neither platform ID
    is set (local development). Setting a conflicting explicit namespace on a
    hosted runtime raises :class:`ValueError`.
    """
    explicit_ns = _explicit_namespace(explicit)

    if deployment_id := get_deployment_id():
        ns = _validate_namespace(deployment_id)
        if explicit_ns and explicit_ns != ns:
            raise ValueError(
                "AGENT_CARD_REGISTRY_CACHE_NAMESPACE cannot override "
                f"MLOPS_DEPLOYMENT_ID on a hosted deployment "
                f"(explicit={explicit_ns!r}, deployment={ns!r})."
            )
        return ResolvedCacheNamespace(ns, "deployment")

    if workload_id := get_workload_id():
        ns = _validate_namespace(workload_id)
        if explicit_ns and explicit_ns != ns:
            raise ValueError(
                "AGENT_CARD_REGISTRY_CACHE_NAMESPACE cannot override "
                f"WORKLOAD_ID on a hosted workload "
                f"(explicit={explicit_ns!r}, workload={ns!r})."
            )
        return ResolvedCacheNamespace(ns, "workload")

    if explicit_ns:
        return ResolvedCacheNamespace(explicit_ns, "explicit")

    return None


def require_cache_namespace(explicit: str | None = None) -> ResolvedCacheNamespace:
    """Return the cache namespace or raise when Redis backends cannot be used."""
    resolved = resolve_cache_namespace(explicit)
    if resolved is None:
        raise ValueError(_REDIS_NAMESPACE_REQUIRED_MSG)
    return resolved


def build_namespaced_redis_prefix(
    base_prefix: str,
    resolved: ResolvedCacheNamespace,
) -> str:
    """Compose ``{base}{kind}:{namespace}:`` for Redis key isolation."""
    normalized_base = base_prefix if base_prefix.endswith(":") else f"{base_prefix}:"
    kind = _SOURCE_PREFIX_SEGMENT[resolved.source]
    return f"{normalized_base}{kind}:{resolved.namespace}:"
