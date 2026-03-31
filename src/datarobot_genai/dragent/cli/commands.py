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

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = "agent/workflow.yaml"


# Config loading and frontend invocation adapted from NAT's StartCommandGroup.invoke_subcommand().
# Source: https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/src/nat/cli/commands/start.py
# NAT doesn't expose a programmatic API for this — replace if one becomes available.
def _run_nat_frontend(
    config_file: Path,
    frontend_name: str,
    override: tuple[tuple[str, str], ...],
    **frontend_kwargs: Any,
) -> None:
    """Load a NAT config and run a registered frontend plugin."""
    from nat.cli.cli_utils.config_override import load_and_override_config
    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.data_models.config import Config
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins
    from nat.utils.data_models.schema_validator import validate_schema

    # Discover all plugins so frontends and config objects are registered
    discover_and_register_plugins(PluginTypes.FRONT_END)
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    config_dict = load_and_override_config(config_file, override)
    config = validate_schema(config_dict, Config)

    # Look up the registered frontend
    registry = GlobalTypeRegistry.get()
    all_front_ends = registry.get_registered_front_ends()

    front_end_info = None
    for fe in all_front_ends:
        if fe.local_name == frontend_name:
            front_end_info = registry.get_front_end(config_type=fe.config_type)
            break

    if front_end_info is None:
        raise click.ClickException(
            f"Frontend '{frontend_name}' not found. "
            f"Available: {[fe.local_name for fe in all_front_ends]}"
        )

    # Set the frontend config if it doesn't match
    if not isinstance(config.general.front_end, front_end_info.config_type):
        config.general.front_end = front_end_info.config_type()

    # Apply any frontend-specific kwargs
    front_end_config = config.general.front_end
    for param, value in frontend_kwargs.items():
        if value is not None:
            setattr(front_end_config, param, value)

    async def run_plugin() -> None:
        async with front_end_info.build_fn(front_end_config, config) as front_end_plugin:
            await front_end_plugin.run()

    try:
        asyncio.run(run_plugin())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


@click.group(
    name="dragent",
    invoke_without_command=True,
    no_args_is_help=True,
    help="DRAgent CLI - run and serve dragent workflows.",
)
def dragent_command() -> None:
    pass


@dragent_command.command(name="serve", help="Start the dragent HTTP server.")
@click.option(
    "--config-file",
    "config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CONFIG_FILE,
    show_default=True,
    help="Path to workflow YAML config.",
)
@click.option("--port", type=int, default=None, help="Server port (default: from config or 8080).")
@click.option("--reload", type=bool, default=None, help="Enable auto-reload.")
@click.option(
    "--override", type=(str, str), multiple=True, help="Override config values using dot notation."
)
def serve_command(
    config_file: Path, port: int | None, reload: bool | None, override: tuple[tuple[str, str], ...]
) -> None:
    _run_nat_frontend(
        config_file=config_file,
        frontend_name="dragent_fastapi",
        override=override,
        port=port,
        reload=reload,
    )


@dragent_command.command(name="run", help="Execute a dragent workflow locally (in-process).")
@click.option(
    "--config-file",
    "config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CONFIG_FILE,
    show_default=True,
    help="Path to workflow YAML config.",
)
@click.option(
    "--input", "input_query", type=str, default=None, help="Prompt string to send to the workflow."
)
@click.option(
    "--input-file",
    "input_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to file containing the prompt.",
)
@click.option(
    "--override", type=(str, str), multiple=True, help="Override config values using dot notation."
)
def run_command(
    config_file: Path,
    input_query: str | None,
    input_file: Path | None,
    override: tuple[tuple[str, str], ...],
) -> None:
    if input_query is None and input_file is None:
        raise click.UsageError("Must specify either --input or --input-file.")
    if input_query is not None and input_file is not None:
        raise click.UsageError("Must specify either --input or --input-file, not both.")

    _run_nat_frontend(
        config_file=config_file,
        frontend_name="dragent_console",
        override=override,
        input_query=[input_query] if input_query is not None else None,
        input_file=input_file,
    )
