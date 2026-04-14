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
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.runtime.session import SessionManager

from datarobot_genai.dragent.frontends.request import DRAgentRunAgentInput
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.frontends.session import DRAgentAGUISessionManager
from datarobot_genai.dragent.frontends.session import _extract_user_id_from_dr_auth_context


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


@pytest.fixture
def patch_super_session():
    """Patch SessionManager.session to capture kwargs passed to it."""
    captured_kwargs = {}
    sentinel = object()

    @asynccontextmanager
    async def mock_session(self, **kwargs):
        captured_kwargs.update(kwargs)
        yield sentinel

    with patch.object(SessionManager, "session", mock_session):
        yield captured_kwargs, sentinel


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


class TestExtractUserIdFromDrAuthContext:
    def test_returns_user_id_when_auth_context_header_present(self):
        """Extract user_id from a request that has the auth context header."""
        mock_request = MagicMock(spec=Request)
        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user.id = "user-abc-123"

        with patch(
            "datarobot_genai.dragent.frontends.session._auth_context_handler"
        ) as mock_handler:
            mock_handler.get_context.return_value = mock_auth_ctx
            result = _extract_user_id_from_dr_auth_context(mock_request)

        assert result == "user-abc-123"
        mock_handler.get_context.assert_called_once_with(dict(mock_request.headers))

    def test_returns_none_when_no_auth_context_header(self):
        """Return None when the request has no auth context header."""
        mock_request = MagicMock(spec=Request)

        with patch(
            "datarobot_genai.dragent.frontends.session._auth_context_handler"
        ) as mock_handler:
            mock_handler.get_context.return_value = None
            result = _extract_user_id_from_dr_auth_context(mock_request)

        assert result is None


class TestSessionUserIdResolution:
    async def test_session_extracts_user_id_from_http_request(
        self, session_manager, patch_super_session
    ):
        """When an HTTP request has auth context, session() extracts user_id from it."""
        captured_kwargs, sentinel = patch_super_session
        mock_request = MagicMock(spec=Request)

        with patch(
            "datarobot_genai.dragent.frontends.session._extract_user_id_from_dr_auth_context",
            return_value="user-from-header",
        ) as mock_extract:
            async with session_manager.session(http_connection=mock_request) as s:
                assert s is sentinel

            mock_extract.assert_called_once_with(mock_request)

        assert captured_kwargs["user_id"] == "user-from-header"

    async def test_session_falls_back_to_context_var_for_a2a(
        self, session_manager, patch_super_session
    ):
        """When no HTTP request is provided, session() reads user_id from context var (A2A)."""
        captured_kwargs, sentinel = patch_super_session
        session_manager._context_state = MagicMock()
        session_manager._context_state.user_id.get.return_value = "user-from-ctx-var"

        async with session_manager.session() as s:
            assert s is sentinel

        assert captured_kwargs["user_id"] == "user-from-ctx-var"

    async def test_session_uses_explicit_user_id_without_extraction(
        self, session_manager, patch_super_session
    ):
        """When user_id is provided explicitly, session() skips extraction entirely."""
        captured_kwargs, sentinel = patch_super_session
        mock_request = MagicMock(spec=Request)

        with patch(
            "datarobot_genai.dragent.frontends.session._extract_user_id_from_dr_auth_context",
        ) as mock_extract:
            async with session_manager.session(
                user_id="explicit-user", http_connection=mock_request
            ) as s:
                assert s is sentinel

            mock_extract.assert_not_called()

        assert captured_kwargs["user_id"] == "explicit-user"
