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

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.runtime.session import SessionManager
from nat.utils.data_models.schema_validator import validate_schema
from nat.utils.io.yaml_tools import yaml_load
from nat.utils.type_utils import StrPath


def load_config(config_file: StrPath, headers: dict[str, str] | None = None) -> Config:
    """
    Load a NAT configuration file with injected headers. It ensures that all plugins are
    loaded and then validates the configuration file against the Config schema.

    Parameters
    ----------
    config_file : StrPath
        The path to the configuration file

    Returns
    -------
    Config
        The validated Config object
    """
    # Ensure all of the plugins are loaded
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    config_yaml = yaml_load(config_file)

    add_headers_to_datarobot_mcp_auth(config_yaml, headers)

    # Validate configuration adheres to NAT schemas
    validated_nat_config = validate_schema(config_yaml, Config)

    return validated_nat_config


def add_headers_to_datarobot_mcp_auth(config_yaml: dict, headers: dict[str, str] | None) -> None:
    if headers:
        if authentication := config_yaml.get("authentication"):
            for auth_name in authentication:
                auth_config = authentication[auth_name]
                if auth_config.get("_type") == "datarobot_mcp_auth":
                    auth_config["headers"] = headers


@asynccontextmanager
async def load_workflow(
    config_file: StrPath, max_concurrency: int = -1, headers: dict[str, str] | None = None
) -> AsyncGenerator[Workflow, None]:
    """
    Load the NAT configuration file and create a Runner object. This is the primary entry point for
    running NAT workflows with injected headers.

    Parameters
    ----------
    config_file : StrPath
        The path to the configuration file
    max_concurrency : int, optional
        The maximum number of parallel workflow invocations to support. Specifying 0 or -1 will
        allow an unlimited count, by default -1
    """
    # Load the config object
    config = load_config(config_file, headers=headers)

    # Must yield the workflow function otherwise it cleans up
    async with WorkflowBuilder.from_config(config=config) as workflow:
        yield SessionManager(await workflow.build(), max_concurrency=max_concurrency)
