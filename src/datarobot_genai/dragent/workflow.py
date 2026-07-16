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

"""Load and run NAT workflows for DRAgent entry points."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.runtime.session import SessionManager
from nat.utils.data_models.schema_validator import validate_schema
from nat.utils.io.yaml_tools import yaml_load
from nat.utils.type_utils import StrPath

from datarobot_genai.dragent.workflow_paths import publish_dragent_config_file_env


def load_config(config_file: StrPath) -> Config:
    """Load a NAT configuration file and validate it against the Config schema.

    Ensures all plugins are registered before validation.
    """
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    config_yaml = yaml_load(config_file)
    return validate_schema(config_yaml, Config)


@asynccontextmanager
async def load_workflow(
    config_file: StrPath,
    max_concurrency: int = -1,
) -> AsyncGenerator[SessionManager]:
    """Load a NAT workflow and yield a ``SessionManager`` for running it.

    This is the primary entry point for in-process DRAgent workflow execution
    (inline runner, tests, and similar callers). HTTP-fronted deployments use
    NAT's own server lifecycle instead.

    Parameters
    ----------
    config_file
        Path to the workflow YAML.
    max_concurrency
        Maximum parallel workflow invocations. ``0`` or ``-1`` means unlimited.
    """
    publish_dragent_config_file_env(config_file)

    config = load_config(config_file)

    async with WorkflowBuilder.from_config(config=config) as builder:
        session_manager = await SessionManager.create(
            config=config, shared_builder=builder, max_concurrency=max_concurrency
        )

        try:
            yield session_manager
        finally:
            await session_manager.shutdown()
