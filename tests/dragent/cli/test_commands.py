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
from unittest.mock import patch

from click.testing import CliRunner

from datarobot_genai.dragent.cli.commands import DEFAULT_CONFIG_FILE
from datarobot_genai.dragent.cli.commands import dragent_command

_COMMANDS = "datarobot_genai.dragent.cli.commands"


# --- dragent group ---


def test_dragent_no_args_shows_help():
    # GIVEN no arguments
    # WHEN we invoke the dragent command
    result = CliRunner().invoke(dragent_command, [])
    # THEN it shows help text
    assert result.exit_code == 0
    assert "DRAgent CLI" in result.output


def test_dragent_accepts_api_token():
    # GIVEN --api-token flag
    # WHEN we invoke with --help (to avoid subcommand requirement)
    result = CliRunner().invoke(dragent_command, ["--api-token", "my-token", "--help"])
    # THEN it exits successfully
    assert result.exit_code == 0


def test_dragent_accepts_base_url():
    # GIVEN --base-url flag
    # WHEN we invoke with --help
    result = CliRunner().invoke(dragent_command, ["--base-url", "https://custom.com", "--help"])
    # THEN it exits successfully
    assert result.exit_code == 0


def test_dragent_reads_api_token_from_env(monkeypatch):
    # GIVEN DATAROBOT_API_TOKEN in the environment
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "env-token")
    # WHEN we invoke with --help
    result = CliRunner().invoke(dragent_command, ["--help"])
    # THEN it exits successfully
    assert result.exit_code == 0


# --- run-deployment ---


@patch(f"{_COMMANDS}.stream_agui_events")
@patch(f"{_COMMANDS}.get_auth_context_headers", return_value={"X-Auth": "jwt"})
def test_run_deployment_invokes_stream(mock_headers, mock_stream):
    # GIVEN valid token and deployment ID
    # WHEN we invoke run-deployment
    result = CliRunner().invoke(
        dragent_command,
        ["--api-token", "tok", "run-deployment", "--deployment-id", "dep-1", "--input", "hi"],
    )
    # THEN stream_agui_events is called with the correct URL
    assert result.exit_code == 0
    mock_stream.assert_called_once()
    call_url = mock_stream.call_args[0][0]
    assert "dep-1" in call_url
    assert "directAccess/generate/stream" in call_url


@patch(f"{_COMMANDS}.stream_agui_events")
@patch(f"{_COMMANDS}.get_auth_context_headers", return_value={})
def test_run_deployment_show_payload_prints_json(mock_headers, mock_stream):
    # GIVEN --show-payload flag
    # WHEN we invoke run-deployment
    result = CliRunner().invoke(
        dragent_command,
        [
            "--api-token",
            "tok",
            "run-deployment",
            "--deployment-id",
            "dep-1",
            "--input",
            "hi",
            "--show-payload",
        ],
    )
    # THEN the JSON payload is printed with the user message
    assert result.exit_code == 0
    parsed = json.loads(result.output.strip())
    assert "messages" in parsed
    assert parsed["messages"][0]["content"] == "hi"


def test_run_deployment_errors_without_api_token(monkeypatch):
    # GIVEN no API token
    monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)
    # WHEN we invoke run-deployment
    result = CliRunner().invoke(
        dragent_command,
        ["run-deployment", "--deployment-id", "dep-1", "--input", "hi"],
    )
    # THEN it errors with a message about the missing token
    assert result.exit_code != 0
    assert "API token is required" in result.output


def test_run_deployment_errors_without_deployment_id():
    # GIVEN no --deployment-id
    # WHEN we invoke run-deployment
    result = CliRunner().invoke(
        dragent_command,
        ["--api-token", "tok", "run-deployment", "--input", "hi"],
    )
    # THEN it errors about the missing option
    assert result.exit_code != 0
    assert "deployment-id" in result.output.lower()


def test_run_deployment_errors_without_input():
    # GIVEN no --input
    # WHEN we invoke run-deployment
    result = CliRunner().invoke(
        dragent_command,
        ["--api-token", "tok", "run-deployment", "--deployment-id", "dep-1"],
    )
    # THEN it errors about the missing option
    assert result.exit_code != 0


# --- constants ---


def test_default_config_file_value():
    assert DEFAULT_CONFIG_FILE == "agent/agent/workflow.yaml"
