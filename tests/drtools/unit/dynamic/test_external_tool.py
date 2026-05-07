# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the framework-agnostic ``drtools.dynamic.external_tool`` module.

These tests exercise the drtools layer directly (no drmcp / FastMCP wiring) to
demonstrate that the module is reusable from non-FastMCP MCP hosts and to lock
in the host-injection contract (``format_response`` callable, ``tool_error_class``
and ``get_outbound_headers_fn``).
"""

import inspect

import pytest
from aioresponses import CallbackResult
from aioresponses import aioresponses

from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.dynamic import external_tool
from datarobot_genai.drtools.dynamic.external_tool import REQUEST_FORWARDED_HEADERS
from datarobot_genai.drtools.dynamic.external_tool import ExternalToolRegistrationConfig
from datarobot_genai.drtools.dynamic.external_tool import _merge_outbound_headers
from datarobot_genai.drtools.dynamic.external_tool import build_external_tool_callable
from datarobot_genai.drtools.dynamic.external_tool import get_outbound_headers


def _identity_format(data: bytes, content_type: str, charset: str) -> dict[str, object]:
    """Trivial format_response — captures the response into a plain dict."""
    return {
        "data": data.decode(charset, errors="replace"),
        "content_type": content_type,
        "charset": charset,
    }


def _spec(**overrides: object) -> ExternalToolRegistrationConfig:
    base: dict[str, object] = dict(
        name="t",
        method="POST",
        base_url="https://api.example.com/",
        endpoint="thing",
        input_schema={"type": "object", "properties": {}},
    )
    base.update(overrides)
    return ExternalToolRegistrationConfig(**base)  # type: ignore[arg-type]


class TestForwardedHeadersConstant:
    def test_all_lowercase(self) -> None:
        assert REQUEST_FORWARDED_HEADERS == {h.lower() for h in REQUEST_FORWARDED_HEADERS}


class TestMergeOutboundHeaders:
    @pytest.mark.asyncio
    async def test_whitelist_passthrough(self) -> None:
        spec = _spec(headers={"X-Spec": "v"})
        out = await _merge_outbound_headers(
            spec,
            {
                "X-Agent-Id": "a",
                "X-DataRobot-Authorization-Context": "ctx",
                "Authorization": "secret",
                "Cookie": "x",
            },
        )
        assert out["X-Agent-Id"] == "a"
        assert out["X-DataRobot-Authorization-Context"] == "ctx"
        assert out["X-Spec"] == "v"
        assert "Authorization" not in out
        assert "Cookie" not in out

    @pytest.mark.asyncio
    async def test_spec_overrides_forwarded(self) -> None:
        spec = _spec(headers={"X-Agent-Id": "spec-agent"})
        out = await _merge_outbound_headers(spec, {"X-Agent-Id": "incoming"})
        assert out["X-Agent-Id"] == "spec-agent"

    @pytest.mark.asyncio
    async def test_case_insensitive_override(self) -> None:
        spec = _spec(headers={"X-Agent-Id": "spec-agent"})
        out = await _merge_outbound_headers(spec, {"x-agent-id": "lower"})
        # Casing of spec header is preserved; the lower-cased incoming was suppressed.
        assert out.get("X-Agent-Id") == "spec-agent"
        assert "x-agent-id" not in out

    @pytest.mark.asyncio
    async def test_no_spec_no_incoming(self) -> None:
        spec = _spec()
        out = await _merge_outbound_headers(spec, {})
        assert out == {}


class TestGetOutboundHeaders:
    @pytest.mark.asyncio
    async def test_uses_module_get_http_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(external_tool, "_get_http_headers", lambda: {"X-Agent-Id": "patched"})
        spec = _spec(headers={"X-Spec": "v"})
        out = await get_outbound_headers(spec)
        assert out["X-Agent-Id"] == "patched"
        assert out["X-Spec"] == "v"


class TestBuildExternalToolCallable:
    URL = "https://api.example.com/thing"

    @pytest.fixture
    def spec(self) -> ExternalToolRegistrationConfig:
        return _spec(
            input_schema={
                "type": "object",
                "properties": {
                    "query_params": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            },
        )

    @pytest.mark.asyncio
    async def test_success_invokes_format_response(
        self, spec: ExternalToolRegistrationConfig
    ) -> None:
        captured: dict[str, str] = {}

        def fmt(data: bytes, content_type: str, charset: str) -> dict[str, object]:
            captured["body"] = data.decode(charset)
            captured["content_type"] = content_type
            return {"ok": True}

        callable_fn = build_external_tool_callable(
            spec,
            format_response=fmt,
            allow_empty_schema=True,
            get_outbound_headers_fn=lambda _spec: _empty_headers(),
        )
        input_model = inspect.signature(callable_fn).parameters["inputs"].annotation

        with aioresponses() as mocked:
            mocked.post(self.URL, status=200, body='{"x":1}', content_type="application/json")
            result = await callable_fn(input_model())

        assert result == {"ok": True}
        assert captured["body"] == '{"x":1}'
        assert captured["content_type"] == "application/json"

    @pytest.mark.asyncio
    async def test_error_status_raises_tool_error_class(
        self, spec: ExternalToolRegistrationConfig
    ) -> None:
        callable_fn = build_external_tool_callable(
            spec,
            format_response=_identity_format,
            allow_empty_schema=True,
            get_outbound_headers_fn=lambda _spec: _empty_headers(),
        )
        input_model = inspect.signature(callable_fn).parameters["inputs"].annotation

        with aioresponses() as mocked:
            mocked.post(self.URL, status=400, body="bad")
            with pytest.raises(ToolError, match="HTTP 400 error from deployment"):
                await callable_fn(input_model())

    @pytest.mark.asyncio
    async def test_error_status_uses_injected_error_class(
        self, spec: ExternalToolRegistrationConfig
    ) -> None:
        class MyError(Exception):
            pass

        callable_fn = build_external_tool_callable(
            spec,
            format_response=_identity_format,
            tool_error_class=MyError,
            allow_empty_schema=True,
            get_outbound_headers_fn=lambda _spec: _empty_headers(),
        )
        input_model = inspect.signature(callable_fn).parameters["inputs"].annotation

        with aioresponses() as mocked:
            mocked.post(self.URL, status=503, body="boom", repeat=True)
            with pytest.raises(MyError, match="HTTP 503 error from deployment"):
                await callable_fn(input_model())

    @pytest.mark.asyncio
    async def test_path_params_substituted_into_url(self) -> None:
        spec = _spec(
            method="GET",
            endpoint="users/{user_id}",
            input_schema={
                "type": "object",
                "properties": {
                    "path_params": {
                        "type": "object",
                        "properties": {"user_id": {"type": "string"}},
                    }
                },
            },
        )
        callable_fn = build_external_tool_callable(
            spec,
            format_response=_identity_format,
            get_outbound_headers_fn=lambda _spec: _empty_headers(),
        )
        input_model = inspect.signature(callable_fn).parameters["inputs"].annotation

        with aioresponses() as mocked:
            mocked.get(
                "https://api.example.com/users/42",
                status=200,
                body="hi",
                content_type="text/plain",
            )
            result = await callable_fn(input_model(path_params={"user_id": "42"}))
        assert "hi" in result["data"]

    @pytest.mark.asyncio
    async def test_default_get_outbound_headers_used_when_none(
        self, spec: ExternalToolRegistrationConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # patch the drtools-level get_http_headers to verify the default flow
        monkeypatch.setattr(external_tool, "_get_http_headers", lambda: {"X-Agent-Id": "ag"})
        observed: dict[str, str] = {}

        def fmt(data: bytes, content_type: str, charset: str) -> dict[str, object]:
            return {"ok": True}

        callable_fn = build_external_tool_callable(
            spec, format_response=fmt, allow_empty_schema=True
        )
        input_model = inspect.signature(callable_fn).parameters["inputs"].annotation

        def _on_request(url: object, **kwargs: object) -> object:
            observed.update(dict(kwargs.get("headers") or {}))  # type: ignore[arg-type]
            return CallbackResult(status=200, body="ok", content_type="text/plain")

        with aioresponses() as mocked:
            mocked.post(self.URL, callback=_on_request)
            await callable_fn(input_model())

        assert observed.get("X-Agent-Id") == "ag"


async def _empty_headers() -> dict[str, str]:
    return {}
