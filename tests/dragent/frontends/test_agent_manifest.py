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

from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig

from datarobot_genai.dragent.frontends.agent_manifest import build_agent_manifest


class _FakeToolConfig(FunctionBaseConfig, name="fake_tool"):  # type: ignore[call-arg, misc]
    """A function config whose concrete type declares ``description`` -
    mirrors how real function configs (e.g. StreamingMemoryAgentConfig) do,
    unlike the base FunctionBaseConfig, which has no such field.

    Not a type NAT's plugin registry knows about, so ``Config(...)``'s normal
    validation would reject it (it looks up every declared ``_type`` against
    installed plugins) - tests build the ``Config`` via ``model_construct``
    instead, which skips that lookup and just stores the objects directly.
    ``build_agent_manifest`` only reads attributes off an already-built
    ``Config``, so this is a faithful enough double for it.
    """

    description: str | None = None


class _FakeGroupConfig(FunctionGroupBaseConfig, name="fake_group"):  # type: ignore[call-arg, misc]
    description: str | None = None


def test_build_agent_manifest_on_empty_config() -> None:
    """A config with no declared functions/function_groups and the default
    (unset) workflow still produces a valid manifest, not an error - this is
    the shape `worker` fixtures across this test suite construct by default.
    """
    config = Config(general=GeneralConfig())

    manifest = build_agent_manifest(config)

    assert manifest.components == []
    assert manifest.root_agent.type == "EmptyFunctionConfig"
    # No explicit name was set, so it falls back to the type.
    assert manifest.root_agent.name == "EmptyFunctionConfig"
    assert manifest.root_agent.description is None


def test_build_agent_manifest_lists_functions_and_function_groups() -> None:
    config = Config.model_construct(
        general=GeneralConfig(),
        functions={
            "planner": _FakeToolConfig(description="plans things"),
            "writer": _FakeToolConfig(),
        },
        function_groups={"mcp_tools": _FakeGroupConfig(description="exposes MCP tools")},
        workflow=_FakeToolConfig(name="My Root", description="the entry point"),
    )

    manifest = build_agent_manifest(config)

    assert manifest.root_agent.name == "My Root"
    assert manifest.root_agent.type == "fake_tool"
    assert manifest.root_agent.description == "the entry point"

    by_name = {c.name: c for c in manifest.components}
    assert by_name["planner"].kind == "function"
    assert by_name["planner"].type == "fake_tool"
    assert by_name["planner"].description == "plans things"
    # A function without a description reports None, not an empty string or KeyError.
    assert by_name["writer"].description is None
    assert by_name["mcp_tools"].kind == "function_group"
    assert by_name["mcp_tools"].description == "exposes MCP tools"


def test_build_agent_manifest_root_agent_falls_back_to_type_without_a_name() -> None:
    config = Config.model_construct(general=GeneralConfig(), workflow=_FakeToolConfig())

    manifest = build_agent_manifest(config)

    assert manifest.root_agent.name == "fake_tool"
    assert manifest.root_agent.type == "fake_tool"
