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


def test_get_session_secret_key_raises_when_empty(monkeypatch):
    # GIVEN SESSION_SECRET_KEY is set to empty string
    monkeypatch.setenv("SESSION_SECRET_KEY", "")
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


def test_require_auth_strips_trailing_slash():
    # GIVEN a base_url with a trailing slash
    ctx = _make_click_ctx(api_token="tok", base_url="https://example.com/")
    # WHEN we require auth
    _, url = require_auth(ctx)
    # THEN the slash is stripped
    assert url == "https://example.com"


def test_require_auth_strips_api_v2_suffix():
    # GIVEN a base_url ending with /api/v2
    ctx = _make_click_ctx(api_token="tok", base_url="https://example.com/api/v2")
    # WHEN we require auth
    _, url = require_auth(ctx)
    # THEN /api/v2 is stripped
    assert url == "https://example.com"


def test_require_auth_raises_when_no_token():
    # GIVEN no api_token
    ctx = _make_click_ctx(api_token=None)
    # WHEN we require auth
    # THEN it raises UsageError
    with pytest.raises(click.UsageError, match="API token is required"):
        require_auth(ctx)


def test_require_auth_default_base_url():
    # GIVEN no base_url specified
    ctx = _make_click_ctx(api_token="tok")
    # WHEN we require auth
    _, url = require_auth(ctx)
    # THEN the default URL is used
    assert url == "https://app.datarobot.com"


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


def test_build_agui_payload_unique_ids_per_call():
    # GIVEN two calls
    p1 = build_agui_payload("a")
    p2 = build_agui_payload("b")
    # THEN all UUIDs are unique across calls
    assert p1["threadId"] != p2["threadId"]
    assert p1["runId"] != p2["runId"]
    assert p1["messages"][0]["id"] != p2["messages"][0]["id"]


def test_build_agui_payload_is_json_serializable():
    # GIVEN a payload
    payload = build_agui_payload("test")
    # WHEN we serialize it
    # THEN it should not raise
    json.dumps(payload)


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


def test_stream_agui_events_skips_non_data_lines(capsys):
    # GIVEN non-data SSE lines (event, comment, blank)
    lines = [
        "event: message",
        ": comment",
        "",
        f"data: {json.dumps({'type': 'RUN_FINISHED'})}",
    ]
    resp = _mock_stream_response(lines)
    # WHEN we stream
    with patch(f"{_REMOTE}.httpx.stream", return_value=resp):
        stream_agui_events("http://test", {}, {})
    # THEN only data lines are processed
    out = capsys.readouterr().out
    assert "Run finished." in out
