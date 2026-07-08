# Copyright 2025 DataRobot, Inc. and its affiliates.
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

import typing

from a2a.types import AgentSkill
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from datarobot_genai.dragent.deployment_urls import DEFAULT_A2A_MOUNT_PATH
from datarobot_genai.dragent.deployment_urls import normalize_a2a_mount_path

from .server_auth import CrossApplicationAccessConfig


class DRAgentA2AExternalConfig(BaseModel):
    """Customer-provided external identity and URL override for the agent card."""

    id: str | None = Field(
        default=None, description="External agent identifier for catalog discovery."
    )
    url: str | None = Field(
        default=None, description="Custom external URL override for the agent card endpoint."
    )


class DRAgentA2AConfig(BaseModel):
    """DR-owned wrapper around NAT's A2AFrontEndConfig with optional skill definitions."""

    server: A2AFrontEndConfig = Field(description="NAT A2A server configuration.")
    cross_application_access: CrossApplicationAccessConfig | None = Field(
        default=None,
        description=(
            "Configuration for Cross-Application Access utilizing a hybrid RFC 8693 / "
            "RFC 7523 flow."
        ),
    )
    skills: list[AgentSkill] = Field(
        default=[],
        description="Skills to advertise in the A2A agent card. "
        "If empty, a single default skill is generated from the agent name and description.",
    )
    external: DRAgentA2AExternalConfig | None = Field(
        default=None,
        description="External identity and URL override for the agent card.",
    )
    mount_path: str = Field(
        default=DEFAULT_A2A_MOUNT_PATH,
        validate_default=True,
        description=(
            "HTTP path under which A2A endpoints are mounted. "
            "Accepts bare segments ('a2a'), rooted paths ('/api/a2a'), or '/' for a "
            "root mount. The value is normalized at parse time: a leading slash is "
            "added if absent, trailing slashes are stripped. "
            "Defaults to 'a2a' (i.e. mounted at '/a2a/')."
        ),
    )

    @field_validator("mount_path")
    @classmethod
    def _normalize_mount_path(cls, v: str) -> str:
        return normalize_a2a_mount_path(v)


class DRAgentFastApiFrontEndConfig(FastApiFrontEndConfig, name="dragent_fastapi"):  # type: ignore
    a2a: DRAgentA2AConfig | None = Field(
        default=None,
        description="Expose this agent via the Agent2Agent protocol. "
        "A2A server endpoints are mounted under the path configured by "
        "``a2a.mount_path`` (default: '/a2a/').",
    )
    workflow: typing.Annotated[
        FastApiFrontEndConfig.EndpointBase,
        Field(description="Endpoint for the default workflow."),
    ] = FastApiFrontEndConfig.EndpointBase(
        method="POST",
        path="/v1/workflow",
        openai_api_v1_path="/chat/completions",
        legacy_path="/generate",
        legacy_openai_api_path="/chat",
        description="Executes the default NAT workflow from the loaded configuration ",
    )
