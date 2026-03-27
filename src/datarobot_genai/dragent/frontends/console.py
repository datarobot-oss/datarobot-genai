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

from __future__ import annotations

import logging
from pathlib import Path

import click
from nat.data_models.front_end import FrontEndBaseConfig
from nat.front_ends.simple_base.simple_front_end_plugin_base import SimpleFrontEndPluginBase
from nat.runtime.session import SessionManager
from pydantic import Field

logger = logging.getLogger(__name__)


class DRAgentConsoleFrontEndConfig(FrontEndBaseConfig, name="dragent_console"):  # type: ignore
    """Frontend config for running a dragent workflow from the console."""

    input_query: list[str] | None = Field(
        default=None,
        alias="input",
        description="Input prompt(s) to submit to the workflow.",
    )
    input_file: Path | None = Field(
        default=None,
        description="Path to a text file containing the input prompt.",
    )


class DRAgentConsoleFrontEndPlugin(SimpleFrontEndPluginBase[DRAgentConsoleFrontEndConfig]):
    """Console frontend for dragent workflows with AG-UI support."""

    async def pre_run(self) -> None:
        if (
            self.front_end_config.input_query is not None
            and self.front_end_config.input_file is not None
        ):
            raise click.UsageError("Must specify either --input or --input-file, not both")
        if self.front_end_config.input_query is None and self.front_end_config.input_file is None:
            raise click.UsageError("Must specify either --input or --input-file")

    async def run_workflow(self, session_manager: SessionManager) -> None:
        if self.front_end_config.input_query:
            input_text = self.front_end_config.input_query[0]
        elif self.front_end_config.input_file:
            with open(self.front_end_config.input_file, encoding="utf-8") as f:
                input_text = f.read()
        else:
            raise click.UsageError("No input provided.")

        user_id = "cli-user"

        async with session_manager.session(user_id=user_id) as session:
            async with session.run(input_text) as runner:
                result = await runner.result(to_type=str)

        click.echo(result)
