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

from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastapi import Request
from nat.builder.context import ContextState
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.user_info import UserInfo
from nat.runtime.session import SessionManager
from nat.runtime.user_manager import UserManager

from datarobot_genai.dragent.frontends.request import DRAgentRunAgentInput
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.frontends.session import DEFAULT_DR_AGENT_USER_ID
from datarobot_genai.dragent.frontends.session import DRAgentAGUISessionManager
from datarobot_genai.dragent.frontends.session import DRAgentUserManager
from datarobot_genai.dragent.frontends.session import _a2a_headers
from datarobot_genai.dragent.frontends.session import _build_metadata_from_headers


@pytest.fixture
def session_manager():
    """Create a DRAgentAGUISessionManager with minimal config and mocked registry."""
    config = Config(general=GeneralConfig())
    shared_builder = MagicMock()
    shared_workflow = MagicMock()

    mock_registration = MagicMock()
    mock_registration.is_per_user = False

    with patch("nat.cli.type_registry.GlobalTypeRegistry.get") as get_registry:
        get_registry.return_value.get_function.return_value = mock_registration
        manager = DRAgentAGUISessionManager(
            config=config,
            shared_builder=shared_builder,
            shared_workflow=shared_workflow,
        )
        yield manager


class TestDRAgentAGUISessionManager:
    def test_is_session_manager_subclass(self):
        assert issubclass(DRAgentAGUISessionManager, SessionManager)

    def test_get_workflow_input_schema_returns_dragent_run_agent_input(self, session_manager):
        schema = session_manager.get_workflow_input_schema()
        assert schema is DRAgentRunAgentInput

    def test_get_workflow_streaming_output_schema_returns_dragent_event_response(
        self, session_manager
    ):
        schema = session_manager.get_workflow_streaming_output_schema()
        assert schema is DRAgentEventResponse

    @pytest.mark.asyncio
    async def test_session_passes_preset_context_user_id_to_nat_session(self, session_manager):
        """Regression guard for NAT 1.6 A2A + per-user workflows without Bearer JWT.

        The A2A executor presets ``ContextState.user_id`` from ``context_id`` then calls
        ``session()`` with no arguments. NAT's ``SessionManager.session`` used to replace
        that context value with ``None``, triggering "user_id is required for per-user
        workflow". DRAgent must forward the preset into the explicit ``user_id`` kwarg.
        """
        preset = "efeb8b83-ea1c-4d63-928b-aba9927520ee"
        token = session_manager._context_state.user_id.set(preset)
        captured: dict[str, object] = {}

        @asynccontextmanager
        async def fake_nat_session(self, **kwargs: object):
            captured.update(kwargs)
            yield MagicMock()

        try:
            with patch.object(SessionManager, "session", fake_nat_session):
                async with session_manager.session():
                    pass
        finally:
            session_manager._context_state.user_id.reset(token)

        assert captured.get("user_id") == preset

    @pytest.mark.asyncio
    async def test_session_injects_preset_a2a_headers_into_nat_context(self, session_manager):
        """Regression guard: A2A HTTP headers reach Context.get().metadata.headers.

        The test captures headers from *inside* the yielded session block — the point
        in time at which auth providers actually read Context.get().metadata.headers.
        All forwarded headers (not just x-datarobot-*) should be available.
        """
        incoming = {
            "x-datarobot-custom": "tok-abc",
            "x-untrusted-claim": "claim-val",
            "authorization": "Bearer some-token",
        }

        token = _a2a_headers.set(incoming)
        captured_metadata: dict[str, object] = {}

        @asynccontextmanager
        async def fake_nat_session(self, **kwargs: object):
            yield MagicMock()

        try:
            with patch.object(SessionManager, "session", fake_nat_session):
                async with session_manager.session() as _sess:
                    context_state = ContextState.get()
                    captured_metadata["headers"] = dict(context_state._metadata.get().headers or {})
        finally:
            _a2a_headers.reset(token)

        assert captured_metadata.get("headers") == incoming

    @pytest.mark.asyncio
    async def test_session_resets_metadata_when_super_session_raises(self, session_manager):
        """_metadata ContextVar must be reset even if super().session().__aenter__ raises.

        Without a try/finally wrapping the ``async with super().session()`` block, a
        failure during per-user builder creation would leak _metadata into the task.
        """
        incoming = {"x-datarobot-custom": "tok-abc"}
        token = _a2a_headers.set(incoming)
        context_state = ContextState.get()

        @asynccontextmanager
        async def exploding_nat_session(self, **kwargs: object):
            raise RuntimeError("builder creation failed")
            yield  # noqa: unreachable — required for generator syntax

        metadata_before = context_state._metadata.get()

        try:
            with patch.object(SessionManager, "session", exploding_nat_session):
                with pytest.raises(RuntimeError, match="builder creation failed"):
                    async with session_manager.session():
                        pass  # pragma: no cover
        finally:
            _a2a_headers.reset(token)

        # _metadata must have been reset back to the value before the session call
        assert context_state._metadata.get() is metadata_before


class TestBuildMetadataFromHeaders:
    """Tests for the _build_metadata_from_headers helper."""

    def test_returns_request_attributes_with_headers(self):
        headers = {"x-datarobot-token": "tok", "authorization": "Bearer jwt"}
        attrs = _build_metadata_from_headers(headers)
        assert attrs.headers == headers

    def test_returns_empty_headers(self):
        attrs = _build_metadata_from_headers({})
        assert attrs.headers == {}


class TestDRAgentUserManager:
    """Verify DRAgentUserManager is a pure identity resolver (no workflow-keying fallback)."""

    def test_resolves_signed_auth_context(self):
        """X-DataRobot-Authorization-Context (signed) returns the user it carries."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"x-datarobot-authorization-context": "fake-jwt"}
        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user.id = "user-from-ctx"

        with patch(
            "datarobot_genai.dragent.frontends.session.AuthContextHeaderHandler.get_context",
            return_value=mock_auth_ctx,
        ):
            result = DRAgentUserManager.extract_user_from_connection(mock_request)

        assert result is not None
        assert isinstance(result, UserInfo)
        expected = UserInfo._from_session_cookie("user-from-ctx")
        assert result.get_user_id() == expected.get_user_id()

    def test_returns_none_when_no_identity(self):
        """No auth-context and no standard auth → None (default-user is a session() concern)."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.cookies = {}

        result = DRAgentUserManager.extract_user_from_connection(mock_request)

        assert result is None

    def test_subclasses_user_manager(self):
        assert issubclass(DRAgentUserManager, UserManager)


class TestSessionPerUserWorkflowFallback:
    """Verify session() defaults user_id to DEFAULT_DR_AGENT_USER_ID for per-user workflows only."""

    @pytest.mark.asyncio
    async def test_per_user_workflow_with_no_identity_uses_default(self, session_manager):
        """Per-user workflow + no identity → default-user key passes through to NAT."""
        session_manager._is_workflow_per_user = True
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.cookies = {}

        captured: dict[str, object] = {}

        @asynccontextmanager
        async def fake_nat_session(self, **kwargs: object):
            captured.update(kwargs)
            yield MagicMock()

        with patch.object(SessionManager, "session", fake_nat_session):
            async with session_manager.session(http_connection=mock_request):
                pass

        assert captured.get("user_id") == DEFAULT_DR_AGENT_USER_ID

    @pytest.mark.asyncio
    async def test_non_per_user_workflow_leaves_user_id_none(self, session_manager):
        """Non-per-user workflow → no default applied; NAT decides what to do with None."""
        session_manager._is_workflow_per_user = False
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.cookies = {}

        captured: dict[str, object] = {}

        @asynccontextmanager
        async def fake_nat_session(self, **kwargs: object):
            captured.update(kwargs)
            yield MagicMock()

        with patch.object(SessionManager, "session", fake_nat_session):
            async with session_manager.session(http_connection=mock_request):
                pass

        assert captured.get("user_id") is None

    @pytest.mark.asyncio
    async def test_dr_signed_context_resolved_in_session(self, session_manager):
        """When DR auth-context is present, the real user_id reaches NAT."""
        session_manager._is_workflow_per_user = True
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"x-datarobot-authorization-context": "fake-jwt"}
        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user.id = "real-user-1"

        captured: dict[str, object] = {}

        @asynccontextmanager
        async def fake_nat_session(self, **kwargs: object):
            captured.update(kwargs)
            yield MagicMock()

        with patch(
            "datarobot_genai.dragent.frontends.session.AuthContextHeaderHandler.get_context",
            return_value=mock_auth_ctx,
        ):
            with patch.object(SessionManager, "session", fake_nat_session):
                async with session_manager.session(http_connection=mock_request):
                    pass

        assert captured.get("user_id") == UserInfo._from_session_cookie("real-user-1").get_user_id()

    @pytest.mark.asyncio
    async def test_dr_header_reaches_user_manager_shim_through_real_nat_session(
        self, session_manager
    ):
        """End-to-end: DR header → DRAgentAGUISessionManager → real NAT session() → shim.

        ``test_dr_signed_context_resolved_in_session`` stubs ``SessionManager.session``
        and only checks the ``user_id`` kwarg, so it can't catch a regression where
        NAT's base ``session()`` fails to write that kwarg into
        ``ContextState.user_id`` (which is what the shim — and therefore
        ``auto_memory_wrapper`` — actually reads). This test runs the real
        ``super().session()`` and asserts the value that ends up on
        ``Context.get().user_manager.get_id()`` inside the session block.
        """
        # Importing installs the property(_UserManagerShim) on Context.
        from starlette.requests import Request as StarletteRequest

        from datarobot_genai.dragent.plugins import datarobot_mem0_memory  # noqa: F401

        session_manager._is_workflow_per_user = False
        session_manager._shared_workflow = MagicMock()

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/agent/run",
            "raw_path": b"/agent/run",
            "query_string": b"",
            "path_params": {},
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "headers": [(b"x-datarobot-authorization-context", b"fake-jwt")],
        }

        async def _empty_receive() -> dict[str, object]:
            return {"type": "http.request", "body": b"", "more_body": False}

        request = StarletteRequest(scope, receive=_empty_receive)

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user.id = "real-dr-user"
        expected = UserInfo._from_session_cookie("real-dr-user").get_user_id()

        captured: dict[str, object] = {}

        with patch(
            "datarobot_genai.dragent.frontends.session.AuthContextHeaderHandler.get_context",
            return_value=mock_auth_ctx,
        ):
            async with session_manager.session(http_connection=request):
                # The shim is what auto_memory_wrapper actually calls. Reading it
                # inside the session block reproduces the call site exactly.
                ctx = ContextState.get()
                captured["context_state_user_id"] = ctx.user_id.get()
                from nat.builder.context import Context as NATContext

                captured["shim_get_id"] = NATContext.get().user_manager.get_id()

        assert captured["context_state_user_id"] == expected
        assert captured["shim_get_id"] == expected
