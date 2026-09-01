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

"""Agent Manifest: a static reflection of a running agent's declared NAT
workflow structure, served at ``/.well-known/agent-manifest.json`` alongside
the existing A2A ``/.well-known/agent-card.json``.

Unlike the A2A agent card (hand-authored ``skills``/``description`` metadata,
see :mod:`~datarobot_genai.dragent.frontends.a2a`), this manifest is derived
entirely from ``workflow.yaml``'s own declared structure - the
``functions``/``function_groups``/``workflow`` sections of the NAT
:class:`~nat.data_models.config.Config` - so it needs no separate
configuration and can never drift from what the agent actually declares.

The manifest's own wire format is provisional: workload-api's manifest-parsing
consumer hasn't landed yet, so there is no authoritative schema to match
against. This shape may need to change once that lands.
"""

from __future__ import annotations

from typing import Literal

from nat.data_models.config import Config
from pydantic import BaseModel
from pydantic import Field


class AgentManifestRootAgent(BaseModel):
    """The workflow's entry-point function - ``workflow:`` in workflow.yaml."""

    name: str = Field(description="Display name, falling back to the registered type")
    type: str = Field(description="Registered NAT component type, e.g. 'streaming_memory_agent'")
    description: str | None = Field(
        default=None, description="Only present when the concrete config type declares one"
    )


class AgentManifestComponent(BaseModel):
    """One declared ``functions:`` or ``function_groups:`` entry."""

    name: str = Field(description="The workflow.yaml key this component is declared under")
    type: str = Field(description="Registered NAT component type")
    kind: Literal["function", "function_group"]
    description: str | None = Field(
        default=None, description="Only present when the concrete config type declares one"
    )


class AgentManifest(BaseModel):
    root_agent: AgentManifestRootAgent
    components: list[AgentManifestComponent]


def _root_agent_from_config(config: Config) -> AgentManifestRootAgent:
    workflow_config = config.workflow
    return AgentManifestRootAgent(
        name=workflow_config.name or workflow_config.type,
        type=workflow_config.type,
        description=getattr(workflow_config, "description", None),
    )


def _components_from_config(config: Config) -> list[AgentManifestComponent]:
    components = [
        AgentManifestComponent(
            name=name,
            type=function_config.type,
            kind="function",
            description=getattr(function_config, "description", None),
        )
        for name, function_config in config.functions.items()
    ]
    components += [
        AgentManifestComponent(
            name=name,
            type=group_config.type,
            kind="function_group",
            description=getattr(group_config, "description", None),
        )
        for name, group_config in config.function_groups.items()
    ]
    return components


def build_agent_manifest(config: Config) -> AgentManifest:
    """Build the manifest from a NAT ``Config`` alone - no build/execution needed.

    ``config.workflow``/``config.functions``/``config.function_groups`` are
    already fully-parsed ``workflow.yaml`` declarations by the time a
    ``Config`` exists, so this never runs any agent code.
    """
    return AgentManifest(
        root_agent=_root_agent_from_config(config),
        components=_components_from_config(config),
    )
