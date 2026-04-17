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

from datarobot_genai.dragent.cli.commands import dragent_command

_COMMANDS = "datarobot_genai.dragent.cli.commands"


# --- query --local ---


def test_query_local_errors_without_port(monkeypatch):
    # GIVEN no --port and no AGENT_PORT env var
    monkeypatch.delenv("AGENT_PORT", raising=False)
    # WHEN we invoke query --local
    result = CliRunner().invoke(
        dragent_command,
        ["query", "--local", "--user_prompt", "hi"],
    )
    # THEN it errors about missing port
    assert result.exit_code != 0
    assert "port" in result.output.lower()


@patch(f"{_COMMANDS}.stream_agui_events")
def test_query_local_custom_port_flag(mock_stream, monkeypatch):
    # GIVEN --port flag
    monkeypatch.delenv("AGENT_PORT", raising=False)
    result = CliRunner().invoke(
        dragent_command,
        ["query", "--local", "--port", "8842", "--user_prompt", "hi"],
    )
    # THEN it uses the specified port
    assert result.exit_code == 0
    call_url = mock_stream.call_args[0][0]
    assert "localhost:8842" in call_url


@patch(f"{_COMMANDS}.stream_agui_events")
def test_query_local_custom_port_env(mock_stream, monkeypatch):
    # GIVEN AGENT_PORT env var
    monkeypatch.setenv("AGENT_PORT", "9090")
    result = CliRunner().invoke(
        dragent_command,
        ["query", "--local", "--user_prompt", "hi"],
    )
    # THEN it reads port from AGENT_PORT
    assert result.exit_code == 0
    call_url = mock_stream.call_args[0][0]
    assert "localhost:9090" in call_url


@patch(f"{_COMMANDS}.stream_agui_events")
def test_query_local_show_payload(mock_stream, monkeypatch):
    # GIVEN --show-payload with --local
    monkeypatch.setenv("AGENT_PORT", "8080")
    result = CliRunner().invoke(
        dragent_command,
        ["query", "--local", "--user_prompt", "hi", "--show-payload"],
    )
    # THEN the JSON payload is printed
    assert result.exit_code == 0
    parsed = json.loads(result.output.strip())
    assert parsed["messages"][0]["content"] == "hi"


@patch(f"{_COMMANDS}.stream_agui_events")
def test_query_local_no_auth_headers(mock_stream, monkeypatch):
    # GIVEN --local (no auth needed)
    monkeypatch.setenv("AGENT_PORT", "8080")
    monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)
    result = CliRunner().invoke(
        dragent_command,
        ["query", "--local", "--user_prompt", "hi"],
    )
    # THEN it succeeds without auth and headers have no Authorization
    assert result.exit_code == 0
    call_headers = mock_stream.call_args[0][2]
    assert "Authorization" not in call_headers


# --- query --deployment-id (remote deployment) ---


@patch(f"{_COMMANDS}.stream_agui_events")
@patch(f"{_COMMANDS}.get_auth_context_headers", return_value={"X-Auth": "jwt"})
def test_query_deployment_invokes_stream(mock_headers, mock_stream):
    # GIVEN valid token and deployment ID
    result = CliRunner().invoke(
        dragent_command,
        [
            "--api-token",
            "tok",
            "--base-url",
            "https://app.example.com",
            "query",
            "--deployment_id",
            "dep-1",
            "--user_prompt",
            "hi",
        ],
    )
    # THEN stream_agui_events is called with the deployment URL
    assert result.exit_code == 0
    mock_stream.assert_called_once()
    call_url = mock_stream.call_args[0][0]
    assert "dep-1" in call_url
    assert "directAccess/generate/stream" in call_url


@patch(f"{_COMMANDS}.stream_agui_events")
@patch(f"{_COMMANDS}.get_auth_context_headers", return_value={})
def test_query_show_payload_prints_json(mock_headers, mock_stream):
    # GIVEN --show-payload flag
    result = CliRunner().invoke(
        dragent_command,
        [
            "--api-token",
            "tok",
            "--base-url",
            "https://app.example.com",
            "query",
            "--deployment_id",
            "dep-1",
            "--user_prompt",
            "hi",
            "--show-payload",
        ],
    )
    # THEN the JSON payload is printed with the user message
    assert result.exit_code == 0
    parsed = json.loads(result.output.strip())
    assert "messages" in parsed
    assert parsed["messages"][0]["content"] == "hi"


# --- query validation ---


def test_query_errors_without_local_or_deployment_id():
    # GIVEN neither --local nor --deployment-id
    result = CliRunner().invoke(
        dragent_command,
        ["query", "--user_prompt", "hi"],
    )
    # THEN it errors
    assert result.exit_code != 0
    assert "--local" in result.output or "--deployment_id" in result.output


def test_query_errors_with_both_local_and_deployment_id():
    # GIVEN both --local and --deployment-id
    result = CliRunner().invoke(
        dragent_command,
        [
            "--api-token",
            "tok",
            "query",
            "--local",
            "--deployment_id",
            "dep-1",
            "--user_prompt",
            "hi",
        ],
    )
    # THEN it errors about mutual exclusivity
    assert result.exit_code != 0
    assert "not both" in result.output.lower()


def test_query_deployment_errors_without_api_token(monkeypatch):
    # GIVEN no API token
    monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)
    result = CliRunner().invoke(
        dragent_command,
        ["query", "--deployment_id", "dep-1", "--user_prompt", "hi"],
    )
    # THEN it errors about the missing token
    assert result.exit_code != 0
    assert "API token is required" in result.output


def test_query_errors_without_input(monkeypatch):
    # GIVEN no --user_prompt and no --completion_json (port set so we reach input validation)
    monkeypatch.setenv("AGENT_PORT", "8080")
    result = CliRunner().invoke(
        dragent_command,
        ["query", "--local"],
    )
    # THEN it errors about the missing input
    assert result.exit_code != 0
    assert "--user_prompt" in result.output or "--completion_json" in result.output


@patch(f"{_COMMANDS}.stream_agui_events")
def test_query_local_with_completion_json(mock_stream, monkeypatch, tmp_path):
    # GIVEN a completion JSON file instead of --user_prompt
    monkeypatch.setenv("AGENT_PORT", "8080")
    json_file = tmp_path / "completion.json"
    json_file.write_text("Hello from file")
    result = CliRunner().invoke(
        dragent_command,
        ["query", "--local", "--completion_json", str(json_file)],
    )
    # THEN it reads the file content and uses it as the prompt
    assert result.exit_code == 0
    call_payload = mock_stream.call_args[0][1]
    assert call_payload["messages"][0]["content"] == "Hello from file"
