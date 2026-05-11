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
from datarobot_genai.dragent.frontends.session import DRAgentAGUISessionManager
from datarobot_genai.dragent.frontends.session import _a2a_headers


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
        """
        incoming = {
            "x-datarobot-custom": "tok-abc",
            "x-untrusted-claim": "claim-val",
        }

        token = _a2a_headers.set(incoming)
        captured_metadata: dict[str, object] = {}

        @asynccontextmanager
        async def fake_nat_session(self, **kwargs: object):
            yield MagicMock()

        try:
            with patch.object(SessionManager, "session", fake_nat_session):
                async with session_manager.session() as _sess:
                    # Headers are injected here — after super().session() has yielded.
                    context_state = ContextState.get()
                    captured_metadata["headers"] = dict(
                        context_state._metadata.get().headers or {}
                    )
        finally:
            _a2a_headers.reset(token)

        assert captured_metadata.get("headers") == incoming


class TestUserManagerPatch:
    """Verify that the UserManager monkey-patch resolves DR auth context.

    The patch is applied at module import time by session.py's ``_patch_user_manager()``.
    These tests verify the already-applied patch.
    """

    def test_dr_auth_context_resolves_to_user_info(self):
        """UserManager returns UserInfo when DR auth context header is present."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"x-datarobot-authorization-context": "fake-jwt"}
        mock_request.cookies = {}
        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user.id = "user-abc-123"

        # Patch the handler instance captured by _patch_user_manager at import time
        with patch(
            "datarobot_genai.dragent.frontends.session.AuthContextHeaderHandler.get_context",
            return_value=mock_auth_ctx,
        ):
            result = UserManager.extract_user_from_connection(mock_request)

        assert result is not None
        assert isinstance(result, UserInfo)
        expected = UserInfo._from_session_cookie("user-abc-123")
        assert result.get_user_id() == expected.get_user_id()

    def test_falls_back_to_original_when_no_dr_header(self):
        """UserManager falls back to standard auth when no DR header is present."""
        mock_request = MagicMock(spec=Request)
        mock_request.cookies = {}
        mock_request.headers = {}

        result = UserManager.extract_user_from_connection(mock_request)

        assert result is None
