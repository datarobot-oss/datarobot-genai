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
        ["query", "--local", "--input", "hi"],
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
        ["query", "--local", "--port", "8842", "--input", "hi"],
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
        ["query", "--local", "--input", "hi"],
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
        ["query", "--local", "--input", "hi", "--show-payload"],
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
        ["query", "--local", "--input", "hi"],
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
            "--deployment-id",
            "dep-1",
            "--input",
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


# --- query validation ---


def test_query_errors_without_local_or_deployment_id():
    # GIVEN neither --local nor --deployment-id
    result = CliRunner().invoke(
        dragent_command,
        ["query", "--input", "hi"],
    )
    # THEN it errors
    assert result.exit_code != 0
    assert "--local" in result.output or "--deployment-id" in result.output


def test_query_errors_with_both_local_and_deployment_id():
    # GIVEN both --local and --deployment-id
    result = CliRunner().invoke(
        dragent_command,
        [
            "--api-token",
            "tok",
            "query",
            "--local",
            "--deployment-id",
            "dep-1",
            "--input",
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
        ["query", "--deployment-id", "dep-1", "--input", "hi"],
    )
    # THEN it errors about the missing token
    assert result.exit_code != 0
    assert "API token is required" in result.output


def test_query_errors_without_input():
    # GIVEN no --input
    result = CliRunner().invoke(
        dragent_command,
        ["query", "--local"],
    )
    # THEN it errors about the missing option
    assert result.exit_code != 0
