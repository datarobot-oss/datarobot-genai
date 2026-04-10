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

"""HTTP client helpers for remote DataRobot API commands."""

from __future__ import annotations

import json
import logging
import os
import typing
from uuid import uuid4

import click
import httpx
from colorama import Fore
from colorama import Style

logger = logging.getLogger(__name__)

_TOOL_RESULT_MAX_LEN = 1000


def _get_session_secret_key() -> str | None:
    """Read SESSION_SECRET_KEY from the environment, or None if not set."""
    return os.environ.get("SESSION_SECRET_KEY") or None


def get_auth_context_headers(api_token: str, base_url: str) -> dict[str, str]:
    """Build X-DataRobot-Authorization-Context header for CLI requests.

    Fetches user identity from DataRobot, encodes it as a JWT that
    DRAgentAGUISessionManager decodes to extract user_id.

    Returns an empty dict when SESSION_SECRET_KEY is not set, allowing
    the agent to run without auth context (e.g. local-only usage).
    """
    secret_key = _get_session_secret_key()
    if secret_key is None:
        logger.warning(
            "SESSION_SECRET_KEY is not set. Skipping auth context header. "
            "Set it if the deployment requires X-DataRobot-Authorization-Context."
        )
        return {}

    from datarobot_genai.core.utils.auth import AuthContextHeaderHandler

    resp = httpx.get(
        f"{base_url}/api/v2/account/info/",
        headers={"Authorization": f"Bearer {api_token}"},
        follow_redirects=True,
        timeout=10,
    )
    if not resp.is_success:
        raise click.ClickException(
            f"Failed to fetch user info (HTTP {resp.status_code}). "
            f"Check DATAROBOT_API_TOKEN and DATAROBOT_ENDPOINT."
        )

    raw = resp.json()
    auth_ctx = {
        "user": {"id": raw["uid"], "email": raw["email"]},
        "identities": [],
    }
    handler = AuthContextHeaderHandler(secret_key=secret_key)
    return handler.get_header(auth_ctx)


def get_local_auth_context_headers() -> dict[str, str]:
    """Build X-DataRobot-Authorization-Context header for local server requests.

    Uses SESSION_SECRET_KEY and DATAROBOT_USER_ID from env vars.
    Returns an empty dict when SESSION_SECRET_KEY is not set.
    """
    secret_key = _get_session_secret_key()
    if secret_key is None:
        return {}

    from datarobot_genai.core.utils.auth import AuthContextHeaderHandler

    user_id = os.environ.get("DATAROBOT_USER_ID", "local_cli_user")
    auth_ctx = {
        "user": {"id": user_id, "email": "local@cli"},
        "identities": [],
    }
    handler = AuthContextHeaderHandler(secret_key=secret_key)
    return handler.get_header(auth_ctx)


def require_auth(ctx: click.Context) -> tuple[str, str]:
    """Return (api_token, base_url) from the group context, or raise."""
    api_token: str | None = ctx.obj.get("api_token")
    base_url: str | None = ctx.obj.get("base_url")
    if not api_token:
        raise click.UsageError(
            "API token is required. Pass --api-token or set DATAROBOT_API_TOKEN."
        )
    if not base_url:
        raise click.UsageError("Base URL is required. Pass --base-url or set DATAROBOT_ENDPOINT.")
    return api_token, base_url


def normalize_base_url(base_url: str) -> str:
    """Strip trailing slash and /api/v2 suffix for URL construction."""
    return base_url.rstrip("/").removesuffix("/api/v2")


def build_agui_payload(user_prompt: str) -> dict[str, typing.Any]:
    """Build an AG-UI RunAgentInput payload."""
    return {
        "threadId": str(uuid4()),
        "runId": str(uuid4()),
        "state": [],
        "tools": [],
        "context": [],
        "forwardedProps": {},
        "messages": [
            {
                "id": str(uuid4()),
                "role": "user",
                "content": user_prompt,
            }
        ],
    }


def stream_agui_events(
    url: str,
    payload: dict[str, typing.Any],
    headers: dict[str, str],
) -> None:
    """POST to an AG-UI /generate/stream endpoint and print text events."""
    run_error: str | None = None
    try:
        with httpx.stream("POST", url, json=payload, headers=headers, timeout=300) as resp:
            if not resp.is_success:
                resp.read()
                raise click.ClickException(f"HTTP {resp.status_code}: {resp.text}")
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed SSE data: %s", line)
                    continue
                events = data.get("events", [data])
                for ev in events:
                    event_type = ev.get("type", "")
                    # --- Text messages (stdout) ---
                    if event_type in (
                        "TEXT_MESSAGE_CONTENT",
                        "TEXT_MESSAGE_CHUNK",
                    ):
                        click.echo(
                            f"{Fore.CYAN}{ev.get('delta', '')}{Style.RESET_ALL}",
                            nl=False,
                        )
                    elif event_type == "TEXT_MESSAGE_END":
                        click.echo("")
                    elif event_type == "TEXT_MESSAGE_START":
                        pass
                    # --- Reasoning (stderr) ---
                    elif event_type == "REASONING_MESSAGE_CONTENT":
                        click.echo(
                            f"{Fore.YELLOW}{ev.get('delta', '')}{Style.RESET_ALL}",
                            nl=False,
                            err=True,
                        )
                    elif event_type == "REASONING_MESSAGE_END":
                        click.echo("", err=True)
                    elif event_type in (
                        "REASONING_START",
                        "REASONING_MESSAGE_START",
                        "REASONING_END",
                    ):
                        pass
                    # --- Tool calls (stderr) ---
                    elif event_type == "TOOL_CALL_START":
                        name = ev.get("tool_call_name", "")
                        click.echo(
                            f"\n{Fore.MAGENTA}\u25b6 Tool Call: "
                            f"{Fore.MAGENTA}{Style.DIM}{name}{Style.RESET_ALL}",
                            err=True,
                        )
                        click.echo(
                            f"{Fore.MAGENTA}  Arguments: {Style.RESET_ALL}",
                            nl=False,
                            err=True,
                        )
                    elif event_type == "TOOL_CALL_ARGS":
                        click.echo(
                            f"{Fore.MAGENTA}{Style.DIM}{ev.get('delta', '')}{Style.RESET_ALL}",
                            nl=False,
                            err=True,
                        )
                    elif event_type == "TOOL_CALL_END":
                        click.echo("", err=True)
                    elif event_type == "TOOL_CALL_RESULT":
                        content = ev.get("content", "")
                        if len(content) > _TOOL_RESULT_MAX_LEN:
                            content = content[:_TOOL_RESULT_MAX_LEN] + "\u2026"
                        click.echo(
                            f"{Fore.MAGENTA}  Result: "
                            f"{Fore.MAGENTA}{Style.DIM}{content}{Style.RESET_ALL}\n",
                            err=True,
                        )
                    # --- Steps (stderr) ---
                    elif event_type == "STEP_STARTED":
                        step = ev.get("step_name", "")
                        click.echo(
                            f"{Style.DIM}Step: {step}{Style.RESET_ALL}",
                            err=True,
                        )
                    elif event_type == "STEP_FINISHED":
                        pass
                    # --- Run lifecycle ---
                    elif event_type == "RUN_STARTED":
                        click.echo(
                            f"{Style.DIM}Run started{Style.RESET_ALL}",
                            err=True,
                        )
                    elif event_type == "RUN_FINISHED":
                        click.echo(f"\n{Fore.GREEN}\u2705 Run finished.{Style.RESET_ALL}")
                    elif event_type == "RUN_ERROR":
                        run_error = ev.get("message", "Unknown error")
                        break
                    # --- Custom ---
                    elif event_type == "CUSTOM":
                        name = ev.get("name", "")
                        if name != "Heartbeat":
                            click.echo(
                                f"{Style.DIM}[{name}]{Style.RESET_ALL}",
                                err=True,
                            )
                    else:
                        logger.debug("Unhandled SSE event type: %s", event_type)
                if run_error:
                    break
    except click.ClickException:
        raise
    except httpx.ConnectError:
        raise click.ClickException(f"Could not connect to {url}.")
    except httpx.TimeoutException as exc:
        raise click.ClickException(f"Request timed out: {exc}")
    except httpx.HTTPError as exc:
        raise click.ClickException(f"HTTP error during streaming: {exc}")
    if run_error:
        raise click.ClickException(f"{Fore.RED}\u274c Run failed: {run_error}{Style.RESET_ALL}")
