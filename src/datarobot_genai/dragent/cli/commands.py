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
            if isinstance(param, click.Option) and "--config_file" in param.opts:
                param.required = False
                param.default = "agent/agent/workflow.yaml"
                param.show_default = True
                param.type = click.Path(exists=True)
                break
        return params

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

        filtered["run-deployment"] = run_deployment_command
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
# run-deployment
# ---------------------------------------------------------------------------


@click.command(name="run-deployment", help="Query a deployed model via the DataRobot API.")
@click.option("--deployment-id", "deployment_id", required=True, help="Deployment ID.")
@click.option("--input", "input_query", required=True, help="Prompt string.")
@click.option(
    "--show-payload", "show_payload", is_flag=True, help="Show the request payload sent to the API."
)
@click.pass_context
def run_deployment_command(
    ctx: click.Context, deployment_id: str, input_query: str, show_payload: bool
) -> None:
    api_token, base_url = require_auth(ctx)
    base_url = normalize_base_url(base_url)
    url = f"{base_url}/api/v2/deployments/{deployment_id}/directAccess/generate/stream"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}",
        **get_auth_context_headers(api_token, base_url),
    }
    payload = build_agui_payload(input_query)
    if show_payload:
        click.echo(json.dumps(payload, indent=2))
    stream_agui_events(url, payload, headers)
