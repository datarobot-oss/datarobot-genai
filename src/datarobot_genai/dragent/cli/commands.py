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

import json

import click
from nat.cli.commands.start import StartCommandGroup
from nat.cli.type_registry import RegisteredFrontEndInfo

from .remote import build_agui_payload
from .remote import get_auth_context_headers
from .remote import get_local_auth_context_headers
from .remote import normalize_base_url
from .remote import require_auth
from .remote import stream_agui_events

_FRONTEND_COMMANDS: dict[str, dict[str, str]] = {
    "dragent_fastapi": {"alias": "serve", "help": "Start the dragent HTTP server."},
    "dragent_console": {"alias": "run", "help": "Execute a dragent workflow locally (in-process)."},
}


class DRAgentCommandGroup(StartCommandGroup):
    """NAT StartCommandGroup filtered to dragent frontends with friendly aliases."""

    def _build_params(self, front_end: RegisteredFrontEndInfo) -> list[click.Parameter]:
        params = super()._build_params(front_end)
        for param in params:
            if not isinstance(param, click.Option):
                continue
            if "--config_file" in param.opts:
                param.required = False
                param.envvar = "DRAGENT_CONFIG_FILE"
            elif "--port" in param.opts:
                param.envvar = "AGENT_PORT"
            elif "--user_prompt" in param.opts:
                param.opts.insert(0, "--input")
            elif "--input_file" in param.opts:
                param.opts.insert(0, "--file")
        return params

    def invoke_subcommand(  # type: ignore[override]
        self,
        ctx: click.Context,
        cmd_name: str,
        config_file: object,
        override: tuple[tuple[str, str], ...],
        **kwargs: object,
    ) -> int | None:
        if config_file is None:
            raise click.ClickException(
                "No config file provided. "
                "Pass --config_file <path> or set the DRAGENT_CONFIG_FILE env var."
            )
        return super().invoke_subcommand(ctx, cmd_name, config_file, override, **kwargs)

    def _load_commands(self) -> dict[str, click.Command]:
        # Depends on StartCommandGroup caching into self._commands (nvidia-nat 1.4.1).
        # If NAT renames/removes this attribute, _load_commands will need updating.
        commands: dict[str, click.Command] | None = self._commands  # type: ignore[has-type,assignment]
        if commands is not None:
            return commands

        # Reset cache so the parent actually populates it, then filter + alias.
        self._commands = None  # type: ignore[assignment]
        loaded = super()._load_commands()

        filtered: dict[str, click.Command] = {}
        for original_name, meta in _FRONTEND_COMMANDS.items():
            if original_name not in loaded:
                raise RuntimeError(
                    f"Frontend '{original_name}' not registered. "
                    f"Ensure nat.front_ends entry points are installed."
                )
            cmd = loaded[original_name]
            cmd.name = meta["alias"]
            cmd.help = meta["help"]
            filtered[meta["alias"]] = cmd

        filtered["query"] = query_command
        self._commands = filtered  # type: ignore[assignment]
        return filtered


@click.command(
    name="dragent",
    cls=DRAgentCommandGroup,
    invoke_without_command=True,
    no_args_is_help=True,
    help="DRAgent CLI - run and serve dragent workflows.",
)
@click.option(
    "--api-token",
    "api_token",
    envvar="DATAROBOT_API_TOKEN",
    default=None,
    help="DataRobot API token.",
)
@click.option(
    "--base-url",
    "base_url",
    envvar="DATAROBOT_ENDPOINT",
    default=None,
    help="DataRobot API endpoint.",
)
@click.pass_context
def dragent_command(ctx: click.Context, api_token: str | None, base_url: str | None) -> None:
    ctx.ensure_object(dict)
    ctx.obj["api_token"] = api_token
    ctx.obj["base_url"] = base_url


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@click.command(
    name="query",
    help="Query a dragent server. Use --local for localhost (reads AGENT_PORT env var), "
    "or --deployment-id for a DataRobot deployment.",
)
@click.option(
    "--local", "local", is_flag=True, help="Query localhost using --port or AGENT_PORT env var."
)
@click.option(
    "--port",
    "port",
    default=None,
    type=int,
    envvar="AGENT_PORT",
    help="Port for --local. Falls back to AGENT_PORT env var.",
)
@click.option(
    "--deployment-id",
    "--deployment_id",
    "deployment_id",
    default=None,
    help="DataRobot deployment ID.",
)
@click.option("--input", "--user_prompt", "input_query", default=None, help="Prompt string.")
@click.option(
    "--file",
    "--input-file",
    "input_file",
    default=None,
    help="Path to a text file whose contents are used as the prompt.",
)
@click.option(
    "--show-payload", "show_payload", is_flag=True, help="Show the request payload sent to the API."
)
@click.pass_context
def query_command(
    ctx: click.Context,
    local: bool,
    port: int | None,
    deployment_id: str | None,
    input_query: str | None,
    input_file: str | None,
    show_payload: bool,
) -> None:
    if local and deployment_id:
        raise click.UsageError("Specify either --local or --deployment-id, not both.")
    if not local and not deployment_id:
        raise click.UsageError("Specify --local or --deployment-id.")
    if input_query is None and input_file is None:
        raise click.UsageError("Specify --input or --file.")
    if input_query is not None and input_file is not None:
        raise click.UsageError("Specify --input or --file, not both.")

    if input_file is not None:
        try:
            with open(input_file, encoding="utf-8") as f:
                prompt_text = f.read()
        except FileNotFoundError:
            raise click.ClickException(f"File not found: {input_file}")
        except OSError as exc:
            raise click.ClickException(f"Cannot read {input_file}: {exc}")
        if not prompt_text.strip():
            raise click.UsageError("Input file is empty.")
        payload = build_agui_payload(prompt_text)
    else:
        assert input_query is not None
        payload = build_agui_payload(input_query)

    if deployment_id:
        api_token, base_url = require_auth(ctx)
        base_url = normalize_base_url(base_url)
        target_url = f"{base_url}/api/v2/deployments/{deployment_id}/directAccess/generate/stream"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}",
            **get_auth_context_headers(api_token, base_url),
        }
    else:
        if not port:
            raise click.UsageError(
                "Port is required for --local. Pass --port or set AGENT_PORT env var."
            )
        target_url = f"http://localhost:{port}/generate/stream"
        headers = {
            "Content-Type": "application/json",
            **get_local_auth_context_headers(),
        }

    if show_payload:
        click.echo(json.dumps(payload, indent=2))
    stream_agui_events(target_url, payload, headers)
