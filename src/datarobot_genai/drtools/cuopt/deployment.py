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

"""cuOpt deployment resolution and the deployment call seam.

The cuOpt NIM is a regular DataRobot deployment (a tool-tagged serverless GPU
deployment). This module resolves *which* deployment to talk to -- the explicit
``CUOPT_DEPLOYMENT_ID`` env var / runtime parameter wins, falling back to the
same ``tool=tool`` tag lookup dynamic tool discovery uses -- and builds the
submit/poll client on the shared deployment seam
(:func:`get_deployment_base_url` / :func:`build_deployment_auth_headers` from
``drmcputils.deployment``), the same helpers the dynamic deployment tools use.
Only the cuOpt-specific async submit+poll protocol lives here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import datarobot as dr
import httpx
from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic_settings import SettingsConfigDict

from datarobot_genai.drmcputils.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_client
from datarobot_genai.drmcputils.deployment import build_deployment_auth_headers
from datarobot_genai.drmcputils.deployment import get_deployment_base_url
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)

CUOPT_TIMEOUT_SECONDS = 300
_POLL_INTERVAL_SECONDS = 1.0
_CUOPT_LABEL_MARKER = "cuopt"
_TOOL_TAG = "tool"


class CuOptDeploymentSettings(DataRobotAppFrameworkBaseSettings):
    """Resolves ``CUOPT_DEPLOYMENT_ID`` from env vars, ``.env``, file secrets,
    and the ``MLOPS_RUNTIME_PARAM_CUOPT_DEPLOYMENT_ID`` runtime parameter.
    """

    cuopt_deployment_id: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_ignore_empty=True,
    )


def _list_tool_tagged_deployments() -> list[dict[str, Any]]:
    """Fetch raw deployment records tagged ``tool`` from DataRobot.

    Mirrors dynamic tool discovery's deployment listing (the API filters tags
    with OR semantics, so callers must re-check name *and* value with AND).
    """
    with request_user_dr_client(headers_auth_only=False) as client:
        return list(
            dr.utils.pagination.unpaginate(
                initial_url="deployments/",
                initial_params={"tag_values": _TOOL_TAG, "tag_keys": _TOOL_TAG},
                client=client,
            )
        )


def find_cuopt_tool_deployment_id() -> str | None:
    """Discover the cuOpt NIM among tool-tagged deployments by label.

    Returns
    -------
        The deployment id of the single tool-tagged deployment whose label
        contains ``cuopt`` (case-insensitive), or ``None`` when there is none.

    Raises
    ------
        ToolError: If multiple candidates match (set ``CUOPT_DEPLOYMENT_ID``
            to disambiguate).
    """
    candidates: list[str] = []
    for deployment in _list_tool_tagged_deployments():
        is_tool = any(
            tag.get("name") == _TOOL_TAG and tag.get("value") == _TOOL_TAG
            for tag in deployment.get("tags", [])
        )
        if not is_tool:
            continue
        label = str(deployment.get("label") or "")
        if _CUOPT_LABEL_MARKER in label.lower():
            candidates.append(str(deployment["id"]))

    if not candidates:
        return None
    if len(candidates) > 1:
        raise ToolError(
            "Multiple tool-tagged cuOpt deployments found: "
            f"{', '.join(sorted(candidates))}. Set CUOPT_DEPLOYMENT_ID to pick one.",
            kind=ToolErrorKind.VALIDATION,
        )
    return candidates[0]


def resolve_cuopt_deployment_id() -> str:
    """Resolve the cuOpt deployment id, or ``""`` when unconfigured.

    The explicit ``CUOPT_DEPLOYMENT_ID`` env var / runtime parameter wins;
    otherwise the tool-tagged deployment lookup is consulted. Settings are
    instantiated fresh so runtime-param / env changes are always honored.
    """
    configured = CuOptDeploymentSettings().cuopt_deployment_id
    if configured:
        return configured
    try:
        return find_cuopt_tool_deployment_id() or ""
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - discovery is best-effort; unconfigured is the signal
        logger.warning("Tool-tagged cuOpt deployment lookup failed: %s", exc)
        return ""


class CuOptValidationError(Exception):
    """Raised when the cuOpt server returns a validation error (400/422)."""

    def __init__(
        self,
        status_code: int,
        error_message: str,
        raw_response: dict[str, Any] | None = None,
    ):
        self.status_code = status_code
        self.error_message = error_message
        self.raw_response = raw_response or {}
        super().__init__(f"cuOpt Validation Error (HTTP {status_code}): {error_message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "error": self.error_message,
            "details": self.raw_response,
        }


class CuOptDeploymentClient:
    """Submit/poll client for a DataRobot cuOpt NIM deployment.

    ``base_url`` and ``headers`` are pre-computed by :func:`get_cuopt_client`
    through the shared deployment seam; this class only implements the cuOpt
    request protocol (submit to ``/request``, poll ``/solution/{id}``).
    """

    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str],
        timeout: int = CUOPT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self.timeout = timeout

    @staticmethod
    def _response_json_or_text(
        response: httpx.Response, text_limit: int | None = None
    ) -> dict[str, Any]:
        try:
            body = response.json()
            return body if isinstance(body, dict) else {"response": body}
        except Exception:  # noqa: BLE001 - non-JSON bodies degrade to raw text
            raw = response.text
            if text_limit is not None:
                raw = raw[:text_limit]
            return {"raw": raw}

    @classmethod
    def _http_status_error_payload(cls, error: httpx.HTTPStatusError) -> dict[str, Any]:
        return {
            "status": "error",
            "error": f"HTTP {error.response.status_code}",
            "details": cls._response_json_or_text(error.response, text_limit=500),
        }

    async def validate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate a cuOpt payload without solving (``?validation_only=true``)."""
        logger.info("Validating cuOpt payload")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/request",
                    params={"validation_only": "true"},
                    json=data,
                    headers=self.headers,
                )

                if response.status_code == httpx.codes.OK:
                    result = response.json()
                    logger.info("Payload validated successfully")
                    return {
                        "status": "valid",
                        "message": "Payload validated successfully. Ready to solve.",
                        "response": result,
                    }
                else:
                    logger.warning("Validation failed: HTTP %s", response.status_code)
                    return {
                        "status": "invalid",
                        "error": f"Validation failed (HTTP {response.status_code})",
                        "details": self._response_json_or_text(response),
                    }

        except Exception as e:  # noqa: BLE001 - surface as a structured result, not a raise
            logger.error("Validation error: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def solve(self, data: dict[str, Any]) -> dict[str, Any]:
        """Submit a cuOpt payload and poll for the solution."""
        logger.info("Submitting optimization request")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/request",
                    json=data,
                    headers=self.headers,
                )
                response.raise_for_status()

                request_data = response.json()
                request_id = request_data.get("reqId") or request_data.get("requestId")

                if not request_id:
                    return {
                        "status": "error",
                        "error": "No request ID returned",
                        "response": request_data,
                    }

                logger.info("Request submitted: %s, polling for result...", request_id)

            result = await self._poll_for_result(request_id)

            response_data = result.get("response", {})
            solver_response = response_data.get("solver_response", {})
            infeasible_response = response_data.get("solver_infeasible_response", {})

            if solver_response:
                solution = solver_response
                status = "success"
            elif infeasible_response:
                solution = infeasible_response
                status = "infeasible"
            else:
                solution = {}
                status = "failed"

            logger.info("Optimization completed with status: %s", status)

            return {
                "request_id": request_id,
                "status": status,
                "solution": solution,
                "cost": solution.get("solution_cost", 0.0),
            }

        except httpx.HTTPStatusError as e:
            return self._http_status_error_payload(e)
        except CuOptValidationError as e:
            return {
                "status": "error",
                "error": e.error_message,
                "details": e.raw_response,
            }
        except Exception as e:  # noqa: BLE001 - surface as a structured result, not a raise
            logger.error("Solve error: %s", e, exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def _poll_for_result(self, request_id: str) -> dict[str, Any]:
        """Poll for the optimization result until ready or the deadline passes."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        attempt = 0
        last_status_code: int | None = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while loop.time() < deadline:
                attempt += 1
                response = await client.get(
                    f"{self.base_url}/solution/{request_id}",
                    headers=self.headers,
                )
                last_status_code = response.status_code

                if response.status_code == httpx.codes.OK:
                    result = response.json()

                    if "response" in result:
                        logger.info("Solution ready for request %s", request_id)
                        return result  # type: ignore[no-any-return]

                    if result.get("status") == "failed":
                        raise Exception(
                            f"Optimization failed: {result.get('error', 'Unknown error')}"
                        )

                elif response.status_code in (
                    httpx.codes.BAD_REQUEST,
                    httpx.codes.UNPROCESSABLE_ENTITY,
                ):
                    error_data = self._response_json_or_text(response)
                    error_msg = str(error_data.get("error") or response.text)
                    raise CuOptValidationError(
                        status_code=response.status_code,
                        error_message=error_msg,
                        raw_response=error_data,
                    )

                elif response.status_code == httpx.codes.NOT_FOUND:
                    pass  # not ready yet -- keep polling
                else:
                    response.raise_for_status()

                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, remaining))

        raise Exception(
            "Optimization request timed out waiting for result "
            f"(request_id={request_id}, timeout={self.timeout}s, "
            f"attempts={attempt}, last_status={last_status_code})"
        )


def get_cuopt_client(timeout: int = CUOPT_TIMEOUT_SECONDS) -> CuOptDeploymentClient:
    """Build a :class:`CuOptDeploymentClient` for the resolved cuOpt deployment.

    The deployment's base URL and auth headers are computed through the same
    seam the dynamic deployment tools use (``get_deployment_base_url`` /
    ``build_deployment_auth_headers`` with the request-scoped DataRobot
    credentials), then extended with the cuOpt NIM's ``directAccess`` route.

    Raises
    ------
        ToolError: If no cuOpt deployment is configured or discoverable.
    """
    deployment_id = resolve_cuopt_deployment_id()
    if not deployment_id:
        raise ToolError(
            "cuOpt deployment is not configured; set CUOPT_DEPLOYMENT_ID or deploy "
            "a tool-tagged cuOpt NIM deployment.",
            kind=ToolErrorKind.VALIDATION,
        )

    with ThreadSafeDataRobotClient().request_user_client(headers_auth_only=False) as api_client:
        deployment = dr.Deployment.get(deployment_id)
        try:
            base_url = get_deployment_base_url(deployment, api_client.endpoint)
        except ValueError as e:
            raise ToolError(str(e), kind=ToolErrorKind.UPSTREAM) from e
        auth_headers = build_deployment_auth_headers(deployment, api_client.token)

    return CuOptDeploymentClient(
        base_url=f"{base_url}directAccess/nim/cuopt",
        headers={**auth_headers, "Content-Type": "application/json"},
        timeout=timeout,
    )
