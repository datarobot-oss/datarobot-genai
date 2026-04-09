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
from unittest.mock import MagicMock
from unittest.mock import patch

import click
import httpx
import pytest

from datarobot_genai.dragent.cli.remote import _get_session_secret_key
from datarobot_genai.dragent.cli.remote import build_agui_payload
from datarobot_genai.dragent.cli.remote import get_auth_context_headers
from datarobot_genai.dragent.cli.remote import normalize_base_url
from datarobot_genai.dragent.cli.remote import require_auth
from datarobot_genai.dragent.cli.remote import stream_agui_events

_REMOTE = "datarobot_genai.dragent.cli.remote"


def _make_click_ctx(api_token=None, base_url=None):
    """Build a Click context with api_token and base_url in obj."""
    ctx = click.Context(click.Command("test"))
    ctx.ensure_object(dict)
    ctx.obj["api_token"] = api_token
    if base_url is not None:
        ctx.obj["base_url"] = base_url
    return ctx


def _sse_lines(events):
    """Build SSE data lines from a list of event dicts."""
    return [f"data: {json.dumps(ev)}" for ev in events]


def _mock_stream_response(lines, *, is_success=True, status_code=200, text=""):
    """Build a mock httpx streaming response."""
    resp = MagicMock()
    resp.is_success = is_success
    resp.status_code = status_code
    resp.text = text
    resp.read.return_value = text.encode()
    resp.iter_lines.return_value = lines
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# --- _get_session_secret_key ---


def test_get_session_secret_key_returns_value_from_env(monkeypatch):
    # GIVEN SESSION_SECRET_KEY is set in the environment
    monkeypatch.setenv("SESSION_SECRET_KEY", "test-secret")
    # WHEN we read the key
    # THEN it returns the value
    assert _get_session_secret_key() == "test-secret"


def test_get_session_secret_key_raises_when_not_set(monkeypatch):
    # GIVEN SESSION_SECRET_KEY is not set
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)
    # WHEN we read the key
    # THEN it raises a ClickException
    with pytest.raises(click.ClickException, match="SESSION_SECRET_KEY is required"):
        _get_session_secret_key()


# --- require_auth ---


def test_require_auth_returns_token_and_url():
    # GIVEN a context with api_token and base_url
    ctx = _make_click_ctx(api_token="tok", base_url="https://example.com")
    # WHEN we require auth
    token, url = require_auth(ctx)
    # THEN both are returned
    assert token == "tok"
    assert url == "https://example.com"


def test_require_auth_raises_when_no_token():
    # GIVEN no api_token
    ctx = _make_click_ctx(api_token=None)
    # WHEN we require auth
    # THEN it raises UsageError
    with pytest.raises(click.UsageError, match="API token is required"):
        require_auth(ctx)


def test_require_auth_raises_when_no_base_url():
    # GIVEN no base_url specified
    ctx = _make_click_ctx(api_token="tok")
    # WHEN we require auth
    # THEN it raises UsageError
    with pytest.raises(click.UsageError, match="Base URL is required"):
        require_auth(ctx)


# --- build_agui_payload ---


def test_build_agui_payload_contains_user_message():
    # GIVEN a prompt string
    # WHEN we build the payload
    payload = build_agui_payload("hello world")
    # THEN it contains a user message with the prompt
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "hello world"


def test_build_agui_payload_has_required_fields():
    # GIVEN any prompt
    # WHEN we build the payload
    payload = build_agui_payload("test")
    # THEN all required AG-UI fields are present
    for field in ("threadId", "runId", "state", "tools", "context", "forwardedProps"):
        assert field in payload


# --- stream_agui_events ---


def test_stream_agui_events_prints_text_content(capsys):
    # GIVEN SSE events with text content
    events = [
        {"type": "TEXT_MESSAGE_CONTENT", "delta": "hello "},
        {"type": "TEXT_MESSAGE_CONTENT", "delta": "world"},
        {"type": "TEXT_MESSAGE_END"},
        {"type": "RUN_FINISHED"},
    ]
    resp = _mock_stream_response(_sse_lines(events))
    # WHEN we stream
    with patch(f"{_REMOTE}.httpx.stream", return_value=resp):
        stream_agui_events("http://test", {}, {})
    # THEN text content is printed
    out = capsys.readouterr().out
    assert "hello " in out
    assert "world" in out
    assert "Run finished." in out


def test_stream_agui_events_raises_on_run_error():
    # GIVEN a RUN_ERROR event
    events = [{"type": "RUN_ERROR", "message": "something broke"}]
    resp = _mock_stream_response(_sse_lines(events))
    # WHEN we stream
    # THEN it raises ClickException with the error message
    with patch(f"{_REMOTE}.httpx.stream", return_value=resp):
        with pytest.raises(click.ClickException, match="something broke"):
            stream_agui_events("http://test", {}, {})


def test_stream_agui_events_raises_on_http_error():
    # GIVEN a non-success HTTP response
    resp = _mock_stream_response(
        [], is_success=False, status_code=500, text="Internal Server Error"
    )
    # WHEN we stream
    # THEN it raises ClickException with the status code
    with patch(f"{_REMOTE}.httpx.stream", return_value=resp):
        with pytest.raises(click.ClickException, match="HTTP 500"):
            stream_agui_events("http://test", {}, {})


def test_stream_agui_events_raises_on_connect_error():
    # GIVEN a connection error
    # WHEN we stream
    # THEN it raises ClickException
    with patch(f"{_REMOTE}.httpx.stream", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(click.ClickException, match="Could not connect"):
            stream_agui_events("http://test", {}, {})


def test_stream_agui_events_raises_on_timeout():
    # GIVEN a timeout
    # WHEN we stream
    # THEN it raises ClickException
    with patch(f"{_REMOTE}.httpx.stream", side_effect=httpx.ReadTimeout("timed out")):
        with pytest.raises(click.ClickException, match="Request timed out"):
            stream_agui_events("http://test", {}, {})


def test_stream_agui_events_skips_malformed_sse_data(capsys):
    # GIVEN a mix of malformed and valid SSE lines
    lines = [
        "data: not-valid-json",
        f"data: {json.dumps({'type': 'TEXT_MESSAGE_CONTENT', 'delta': 'ok'})}",
        f"data: {json.dumps({'type': 'RUN_FINISHED'})}",
    ]
    resp = _mock_stream_response(lines)
    # WHEN we stream
    with patch(f"{_REMOTE}.httpx.stream", return_value=resp):
        stream_agui_events("http://test", {}, {})
    # THEN valid events are still printed
    out = capsys.readouterr().out
    assert "ok" in out


# --- normalize_base_url ---


def test_normalize_base_url_strips_trailing_slash():
    assert normalize_base_url("https://example.com/") == "https://example.com"


def test_normalize_base_url_strips_api_v2_suffix():
    assert normalize_base_url("https://example.com/api/v2") == "https://example.com"


def test_normalize_base_url_strips_both():
    assert normalize_base_url("https://example.com/api/v2/") == "https://example.com"


def test_normalize_base_url_noop_when_clean():
    assert normalize_base_url("https://example.com") == "https://example.com"


# --- get_auth_context_headers ---


@patch("datarobot_genai.core.utils.auth.AuthContextHeaderHandler")
@patch(f"{_REMOTE}.httpx.get")
def test_get_auth_context_headers_success(mock_get, mock_handler_cls, monkeypatch):
    # GIVEN a successful account info response and SESSION_SECRET_KEY
    monkeypatch.setenv("SESSION_SECRET_KEY", "test-secret")
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"uid": "user-123", "email": "test@example.com"}
    mock_get.return_value = mock_resp

    mock_handler = MagicMock()
    mock_handler.get_header.return_value = {"X-DataRobot-Authorization-Context": "jwt-token"}
    mock_handler_cls.return_value = mock_handler

    # WHEN we get auth context headers
    headers = get_auth_context_headers("my-token", "https://app.datarobot.com")

    # THEN it calls the API with the correct token
    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args
    assert "Bearer my-token" in str(call_kwargs)

    # AND constructs the JWT with user info
    mock_handler.get_header.assert_called_once()
    auth_ctx = mock_handler.get_header.call_args[0][0]
    assert auth_ctx["user"]["id"] == "user-123"
    assert auth_ctx["user"]["email"] == "test@example.com"

    # AND returns the header dict
    assert headers == {"X-DataRobot-Authorization-Context": "jwt-token"}


@patch(f"{_REMOTE}.httpx.get")
def test_get_auth_context_headers_api_failure(mock_get):
    # GIVEN a failed API response
    mock_resp = MagicMock()
    mock_resp.is_success = False
    mock_resp.status_code = 401
    mock_get.return_value = mock_resp

    # WHEN we get auth context headers
    # THEN it raises ClickException
    with pytest.raises(click.ClickException, match="Failed to fetch user info"):
        get_auth_context_headers("bad-token", "https://app.datarobot.com")


@patch(f"{_REMOTE}.httpx.get")
def test_get_auth_context_headers_missing_secret_key(mock_get, monkeypatch):
    # GIVEN a successful API response but no SESSION_SECRET_KEY
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"uid": "user-123", "email": "test@example.com"}
    mock_get.return_value = mock_resp

    # WHEN we get auth context headers
    # THEN it raises ClickException about missing secret key
    with pytest.raises(click.ClickException, match="SESSION_SECRET_KEY is required"):
        get_auth_context_headers("my-token", "https://app.datarobot.com")
