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

"""Ensure agent card registry L2 provisioning runs before the registry singleton locks in L1.

On enclave workloads, ``memory_space_cache.try_resolve_memory_space_id`` provisions the
registry L2 MemorySpace. ``configure_datarobot_memory_client`` skips ``dr.Client()``'s
``GET /version/`` compatibility check on enclave endpoints because gateways expose the
memory Session API but not the full control-hub ``/api/v2`` surface.

This module retries provisioning at lifespan startup, resets the registry singleton when a
space is adopted, and logs raw HTTP probes when SDK error handling masks the real failure.
"""

from __future__ import annotations

import logging
import os

import requests

from datarobot_genai.core.config import resolve_config
from datarobot_genai.core.runtime import get_workload_id
from datarobot_genai.dragent.agent_card_registry import reset_default_registry
from datarobot_genai.dragent.deployment_urls import WORKLOAD_EXTERNAL_HOST_ENV
from datarobot_genai.dragent.deployment_urls import WORKLOAD_EXTERNAL_PREFIX_ENV
from datarobot_genai.dragent.deployment_urls import resolve_external_workload_api_endpoint
from datarobot_genai.dragent.memory_space_cache import is_enclave_l2_workload
from datarobot_genai.dragent.memory_space_cache import registry_cache_deduplication_key
from datarobot_genai.dragent.memory_space_cache import try_resolve_memory_space_id

logger = logging.getLogger(__name__)


class _BootstrapState:
    """Mutable container for one-time bootstrap state."""

    bootstrapped: bool = False


def _resolve_api_token() -> str | None:
    """Return the DataRobot API token from the registered config or the environment."""
    try:
        token = resolve_config().resolve_datarobot_api_token()
    except Exception:
        token = None
    if token:
        return token
    return os.getenv("DATAROBOT_API_TOKEN", "").strip() or None


def _memory_api_auth_headers(api_token: str) -> dict[str, str]:
    return {"Authorization": f"Token {api_token}"}


def _log_memory_api_probe_response(
    *,
    method: str,
    url: str,
    response: requests.Response,
) -> None:
    body_preview = (response.text or "")[:1000]
    logger.info(
        "Agent card registry L2 memory API probe: %s %s -> status=%s content-type=%s body=%r",
        method,
        url,
        response.status_code,
        response.headers.get("content-type"),
        body_preview,
    )


def _log_memory_health_ready_probe(*, endpoint: str, api_token: str) -> None:
    """Log the raw HTTP response from ``GET /memory/health/ready/`` to verify auth."""
    url = f"{endpoint.rstrip('/')}/memory/health/ready/"
    try:
        response = requests.get(
            url,
            headers=_memory_api_auth_headers(api_token),
            timeout=30,
        )
        _log_memory_api_probe_response(method="GET", url=url, response=response)
    except requests.RequestException as exc:
        logger.error(
            "Agent card registry L2 memory API probe: GET %s failed: %s",
            url,
            exc,
        )


def _log_memory_space_create_probe(
    *,
    endpoint: str,
    api_token: str,
    deduplication_key: str,
) -> None:
    """Log the raw HTTP response from ``POST /memory/new/`` when SDK error handling fails.

    The DataRobot SDK's ``_http_message`` raises ``KeyError: 'content-type'`` when the
    enclave gateway returns an error without that header, which masks the real failure.
    """
    url = f"{endpoint.rstrip('/')}/memory/new/"
    payload = {
        "description": "Agent card registry L2 cache",
        "deduplication_key": deduplication_key,
    }
    try:
        response = requests.post(
            url,
            data=payload,
            headers=_memory_api_auth_headers(api_token),
            timeout=30,
        )
        _log_memory_api_probe_response(method="POST", url=url, response=response)
    except requests.RequestException as exc:
        logger.error(
            "Agent card registry L2 memory API probe: POST %s failed: %s",
            url,
            exc,
        )


def _log_memory_api_probes_on_failure(
    *,
    endpoint: str,
    api_token: str,
    deduplication_key: str,
) -> None:
    """Run health and create probes so token, routing, and write failures are visible."""
    _log_memory_health_ready_probe(endpoint=endpoint, api_token=api_token)
    _log_memory_space_create_probe(
        endpoint=endpoint,
        api_token=api_token,
        deduplication_key=deduplication_key,
    )


def _registry_l2_gate_status() -> dict[str, object]:
    """Return the provisioning gate values for structured startup logging."""
    host = os.getenv(WORKLOAD_EXTERNAL_HOST_ENV, "").strip()
    prefix = os.getenv(WORKLOAD_EXTERNAL_PREFIX_ENV, "").strip()
    return {
        "enclave_host_set": bool(host),
        "enclave_prefix_set": bool(prefix),
        "enclave_api_endpoint": resolve_external_workload_api_endpoint(),
        "workload_id": get_workload_id(),
        "enclave_l2_workload": is_enclave_l2_workload(),
        "api_token_set": _resolve_api_token() is not None,
    }


def ensure_registry_l2_cache_provisioned(*, phase: str) -> str | None:
    """Provision the registry L2 MemorySpace on enclave workloads and reset the singleton."""
    gates = _registry_l2_gate_status()
    logger.info(
        "Agent card registry L2 cache check (%s): enclave_host=%s enclave_prefix=%s "
        "enclave_api_endpoint=%s workload_id=%s enclave_l2_workload=%s api_token_set=%s",
        phase,
        gates["enclave_host_set"],
        gates["enclave_prefix_set"],
        gates["enclave_api_endpoint"],
        gates["workload_id"],
        gates["enclave_l2_workload"],
        gates["api_token_set"],
    )

    if not gates["enclave_l2_workload"]:
        if gates["enclave_api_endpoint"] is None:
            logger.info(
                "Agent card registry L2 cache skipped (%s): not behind enclave API gateway "
                "(%s and %s must both be set).",
                phase,
                WORKLOAD_EXTERNAL_HOST_ENV,
                WORKLOAD_EXTERNAL_PREFIX_ENV,
            )
        else:
            logger.info(
                "Agent card registry L2 cache skipped (%s): %s is unset "
                "(required for enclave workload deduplication).",
                phase,
                "WORKLOAD_ID",
            )
        return None

    if not gates["api_token_set"]:
        logger.info(
            "Agent card registry L2 cache skipped (%s): DATAROBOT_API_TOKEN is not set.",
            phase,
        )
        return None

    workload_id = gates["workload_id"]
    assert isinstance(workload_id, str)
    deduplication_key = registry_cache_deduplication_key(workload_id)
    logger.info(
        "Agent card registry L2 cache provisioning (%s): creating or adopting MemorySpace "
        "(dedup_key=%s, endpoint=%s).",
        phase,
        deduplication_key,
        gates["enclave_api_endpoint"],
    )
    space_id = try_resolve_memory_space_id()
    if space_id:
        reset_default_registry()
        logger.info(
            "Agent card registry L2 MemorySpace ready (%s, space_id=%s)",
            phase,
            space_id,
        )
        return space_id

    enclave_endpoint = gates["enclave_api_endpoint"]
    api_token = _resolve_api_token()
    if isinstance(enclave_endpoint, str) and api_token:
        _log_memory_api_probes_on_failure(
            endpoint=enclave_endpoint,
            api_token=api_token,
            deduplication_key=deduplication_key,
        )

    logger.warning(
        "Agent card registry L2 cache provisioning failed (%s): MemorySpace.create "
        "returned no space id — see the memory API probe logs above (GET "
        "memory/health/ready, then POST memory/new) for HTTP status and response "
        "bodies (the DataRobot SDK may have raised KeyError: 'content-type' when "
        "parsing the error response).",
        phase,
    )
    return None


def bootstrap_registry_l2_cache() -> None:
    """Provision registry L2 before workflow config locks the registry singleton to L1."""
    if _BootstrapState.bootstrapped:
        return
    _BootstrapState.bootstrapped = True
    logger.info("Agent card registry L2 bootstrap: ensuring enclave MemorySpace provisioning.")
    ensure_registry_l2_cache_provisioned(phase="import")
