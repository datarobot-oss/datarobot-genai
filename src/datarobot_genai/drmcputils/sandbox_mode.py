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

"""``MCP_SANDBOX_DISABLED`` — env-var kill-switch for MCP sandboxing.

When set to a truthy value, sandboxing is turned off altogether:

- :func:`datarobot_genai.drmcputils.panels.access._require_mcp_sandbox`
  skips the ``ENABLE_MCP_SANDBOX`` entitlement lookup (no DR API call), and
- :func:`datarobot_genai.drtools.core.sandbox.utils.execute_code` runs tool
  code in a plain local subprocess on the MCP server instead of submitting a
  DataRobot workload-api container.

The default — unset, empty, falsy, or unparseable — keeps sandboxing ON
(fail-closed), so staging/production behavior is unchanged unless an operator
explicitly opts out. Intended uses: local development, and deployments where
the workload-api sandbox does not exist yet.

SECURITY: the kill-switch removes every isolation guarantee — LLM-generated
code executes directly in the MCP server's environment with its privileges.
Never set it on shared or multi-tenant deployments.

Lives in ``drmcputils`` so both consumers (the panels entitlement gate in
``drmcputils``/``drmcpbase`` and the sandbox executor in ``drtools``) share
one definition.
"""

import logging

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict

logger = logging.getLogger(__name__)

MCP_SANDBOX_DISABLED_ENV_VAR = "MCP_SANDBOX_DISABLED"

# Warn-once latch so every request doesn't repeat the (deliberately loud)
# warning. Module-level so tests can reset it.
_warning_emitted = False


class _MCPSandboxModeSettings(DataRobotAppFrameworkBaseSettings):
    """Resolves ``MCP_SANDBOX_DISABLED`` from env vars, ``.env``, file secrets,
    and ``MLOPS_RUNTIME_PARAM_`` runtime parameters (same sources as the other
    settings classes in this repo; fields map by name).
    """

    mcp_sandbox_disabled: bool = Field(
        default=False,
        description=(
            "Kill-switch: disable MCP sandboxing entirely. Skips the "
            "ENABLE_MCP_SANDBOX entitlement check and executes tool code "
            "locally on the MCP server (NO isolation)."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_ignore_empty=True,
    )


def is_mcp_sandbox_disabled() -> bool:
    """Whether the ``MCP_SANDBOX_DISABLED`` kill-switch is on.

    Fail-closed: any settings/parse error keeps sandboxing enabled. Emits a
    loud warning (once per process) when the kill-switch is active.
    """
    try:
        disabled = _MCPSandboxModeSettings().mcp_sandbox_disabled
    except Exception:  # noqa: BLE001 - unparseable config must not turn the sandbox off
        logger.debug(
            "Could not parse %s; keeping MCP sandboxing enabled (fail-closed).",
            MCP_SANDBOX_DISABLED_ENV_VAR,
            exc_info=True,
        )
        return False
    if disabled and not _warning_emitted:
        logger.warning(
            "MCP sandboxing is DISABLED (%s=true): the ENABLE_MCP_SANDBOX "
            "entitlement check is skipped and tool code executes directly on "
            "the MCP server process with NO isolation. Only use this for "
            "local development or single-tenant deployments without the "
            "DataRobot workload-api sandbox; never on shared deployments.",
            MCP_SANDBOX_DISABLED_ENV_VAR,
        )
        # Assign via globals() to avoid the `global` statement (same pattern as
        # the credentials singleton in this package).
        globals()["_warning_emitted"] = True
    return disabled
