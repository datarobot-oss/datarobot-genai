# Copyright 2025 DataRobot, Inc.
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

import pytest

from datarobot_genai.drmcp.core.dynamic_tools import register


@pytest.fixture
def incoming_headers():
    return {
        "X-Agent-Id": "agent-123",
        "X-DataRobot-Authorization-Context": "ctx-abc",
        "Unrelated": "should-not-pass",
    }


@pytest.fixture
def spec_model():
    return register.ExternalToolRegistrationConfig(
        name="test-tool",
        method="GET",
        base_url="https://example.local",
        endpoint="/",
        headers={"X-Agent-Id": "spec-agent", "Spec-Header": "spec-val"},
        # include one of the expected properties so the schema validator accepts it
        input_schema={
            "type": "object",
            "properties": {"path_params": {"type": "object", "properties": {}}},
        },
    )


@pytest.fixture
def mock_get_http_headers(monkeypatch):
    """Return a helper that replaces register.get_http_headers with a callable
    returning the provided mapping.

    Usage in tests:
        mock_get_http_headers({"Header": "value"})
    """

    def _set(headers):
        monkeypatch.setattr(register, "get_http_headers", lambda: headers)

    return _set


class TestGetOutboundHeaders:
    """Focused tests for `get_outbound_headers` merging and filtering logic."""

    @pytest.mark.asyncio
    async def test_all_request_forwarded_headers_lowercase(self):
        configured_headers = register.REQUEST_FORWARDED_HEADERS
        configured_headers_lowercased = {h.lower() for h in configured_headers}
        assert configured_headers == configured_headers_lowercased, (
            "REQUEST_FORWARDED_HEADERS must be all lowercase, to ensure case-insensitive matching."
        )

    @pytest.mark.asyncio
    async def test_whitelist_and_spec_override(
        self, mock_get_http_headers, incoming_headers, spec_model
    ):
        mock_get_http_headers(incoming_headers)

        out = await register.get_outbound_headers(spec_model)

        # Spec header should override the forwarded X-Agent-Id header
        assert out["X-Agent-Id"] == "spec-agent"
        # Forwarded whitelisted header should be present
        assert out.get("X-DataRobot-Authorization-Context") == "ctx-abc"
        # Unrelated header must not be forwarded
        assert "Unrelated" not in out
        # Spec headers that are not forwarded should still be present
        assert out.get("Spec-Header") == "spec-val"

    @pytest.mark.asyncio
    async def test_case_sensitivity_edge_case_preserves_original_keys(
        self, mock_get_http_headers, spec_model
    ):
        incoming = {"x-agent-id": "agent-lowercase"}
        mock_get_http_headers(incoming)

        out = await register.get_outbound_headers(spec_model)

        # Header names are case-insensitive per RFC; spec headers should
        # override incoming headers regardless of casing.
        assert out.get("X-Agent-Id") == "spec-agent"
        assert "x-agent-id" not in out

    @pytest.mark.asyncio
    async def test_sensitive_headers_not_forwarded_by_default(
        self, mock_get_http_headers, spec_model
    ):
        # Headers like Authorization and Cookie are not in the whitelist and
        # should not be forwarded.
        incoming = {"Authorization": "secret", "Cookie": "s=1"}
        mock_get_http_headers(incoming)

        out = await register.get_outbound_headers(spec_model)

        assert "Authorization" not in out
        assert "Cookie" not in out
        # Spec headers should still be present
        assert out.get("Spec-Header") == "spec-val"

    @pytest.mark.asyncio
    async def test_empty_incoming_uses_spec_only(self, mock_get_http_headers, spec_model):
        mock_get_http_headers({})

        out = await register.get_outbound_headers(spec_model)

        assert out == {"X-Agent-Id": "spec-agent", "Spec-Header": "spec-val"}

    @pytest.mark.asyncio
    async def test_merge_precedence_when_keys_match_casing(self, mock_get_http_headers, spec_model):
        # When key casing matches, spec headers should take precedence
        incoming = {
            "X-Agent-Id": "agent-old",
            "X-DataRobot-Authorization-Context": "ctx-old",
        }
        mock_get_http_headers(incoming)

        out = await register.get_outbound_headers(spec_model)

        # spec overrides forwarded for identical keys
        assert out["X-Agent-Id"] == "spec-agent"
        # other forwarded values remain
        assert out["X-DataRobot-Authorization-Context"] == "ctx-old"
