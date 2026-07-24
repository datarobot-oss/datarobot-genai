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

import os
import sys
import types
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from a2a.types import AgentSkill
from a2a.types import InvalidParamsError
from a2a.utils.errors import ServerError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.user_info import UserInfo
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.routes import common_utils
from nat.front_ends.fastapi.routes import v1_chat_completions
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from pydantic import ValidationError

from datarobot_genai.dragent.cross_app_access_config import CrossApplicationAccessConfig
from datarobot_genai.dragent.cross_app_access_config import CrossAppTokenExchange
from datarobot_genai.dragent.cross_app_access_config import CrossAppTokenRequest
from datarobot_genai.dragent.frontends.a2a import CROSS_APP_EXTENSION_DESCRIPTION
from datarobot_genai.dragent.frontends.a2a import CROSS_APP_SECURITY_SCHEME_FLOW_REF
from datarobot_genai.dragent.frontends.a2a import CROSS_APP_SECURITY_SCHEME_REF
from datarobot_genai.dragent.frontends.a2a import EXTERNAL_IDENTITY_URI
from datarobot_genai.dragent.frontends.a2a import INTERNAL_IDENTITY_URI
from datarobot_genai.dragent.frontends.a2a import JWT_BEARER_GRANT_TYPE_URI
from datarobot_genai.dragent.frontends.a2a import OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE
from datarobot_genai.dragent.frontends.a2a import TOKEN_EXCHANGE_GRANT_TYPE_URI
from datarobot_genai.dragent.frontends.a2a import TOKEN_EXCHANGE_REQUESTED_TOKEN_TYPE
from datarobot_genai.dragent.frontends.a2a import create_agent_card
from datarobot_genai.dragent.frontends.a2a import get_a2a_endpoint_url
from datarobot_genai.dragent.frontends.fastapi import DATAROBOT_EXPECTED_HEALTH_ROUTES
from datarobot_genai.dragent.frontends.fastapi import DRAgentFastApiFrontEndPlugin
from datarobot_genai.dragent.frontends.fastapi import DRAgentFastApiFrontEndPluginWorker
from datarobot_genai.dragent.frontends.fastapi import _GunicornSettings
from datarobot_genai.dragent.frontends.fastapi import _patch_gunicorn_worker_timeout
from datarobot_genai.dragent.frontends.fastapi import _PerUserCompatibleAgentExecutor
from datarobot_genai.dragent.frontends.fastapi import _resolve_identity_from_headers
from datarobot_genai.dragent.frontends.register import DRAgentA2AConfig
from datarobot_genai.dragent.frontends.register import DRAgentA2AExternalConfig
from datarobot_genai.dragent.frontends.register import DRAgentFastApiFrontEndConfig
from datarobot_genai.dragent.frontends.step_adaptor import DRAgentNestedReasoningStepAdaptor


@pytest.fixture(autouse=True)
def _restore_stream_symbols(monkeypatch):
    """Restore NAT stream helpers after ``add_routes`` patches them."""
    monkeypatch.setattr(
        common_utils,
        "generate_streaming_response_as_str",
        common_utils.generate_streaming_response_as_str,
    )
    monkeypatch.setattr(
        v1_chat_completions,
        "generate_streaming_response_as_str",
        v1_chat_completions.generate_streaming_response_as_str,
    )


@pytest.fixture
def worker():
    config = Config(general=GeneralConfig())
    with patch.dict(os.environ, {"NAT_CONFIG_FILE": "unused"}):
        return DRAgentFastApiFrontEndPluginWorker(config)


@pytest.fixture
def dragent_worker():
    config = Config(
        general=GeneralConfig(
            front_end=DRAgentFastApiFrontEndConfig(
                a2a=DRAgentA2AConfig(
                    server=A2AFrontEndConfig(
                        name="Test Agent",
                        description="A test agent",
                    )
                ),
            )
        )
    )
    with patch.dict(os.environ, {"NAT_CONFIG_FILE": "unused"}):
        return DRAgentFastApiFrontEndPluginWorker(config)


@pytest.fixture
def dragent_worker_with_a2a(dragent_worker, mock_a2a_worker):
    dragent_worker._a2a_worker = mock_a2a_worker
    return dragent_worker


@pytest.fixture
def a2a_frontend_config():
    return A2AFrontEndConfig(
        name="My Agent", description="Does things", host="localhost", port=8000
    )


@pytest.fixture
def app_with_health(worker):
    """Build the FastAPI app the same way the server does, mocking WorkflowBuilder."""

    @asynccontextmanager
    async def mock_from_config(_config):
        yield MagicMock()

    with (
        patch.object(worker, "configure", new_callable=AsyncMock),
        patch.object(WorkflowBuilder, "from_config", side_effect=mock_from_config),
    ):
        yield worker.build_app()


@pytest.fixture
def mock_builder():
    builder = MagicMock()
    builder.build = AsyncMock(return_value=MagicMock())
    return builder


@pytest.fixture
def mock_a2a_worker():
    worker = MagicMock()
    worker.front_end_config = A2AFrontEndConfig(
        name="Test Agent", description="A test agent", host="localhost", port=8000
    )
    worker._generate_security_schemes = AsyncMock(return_value=(None, None))
    worker.create_a2a_server = MagicMock(
        return_value=MagicMock(build=MagicMock(return_value=FastAPI()))
    )
    worker.cleanup = AsyncMock()
    return worker


@pytest.fixture
def patch_super_add_routes():
    """Mock parent add_routes so it appends a session manager (mirrors NAT behavior)."""

    async def mock_super_add_routes(self, app, builder):
        self._session_managers.append(MagicMock())

    with patch(
        "nat.front_ends.fastapi.fastapi_front_end_plugin_worker.FastApiFrontEndPluginWorker.add_routes",
        mock_super_add_routes,
    ):
        yield


class TestDRAgentFastApiFrontEndPluginWorker:
    @pytest.mark.parametrize("path", DATAROBOT_EXPECTED_HEALTH_ROUTES)
    def test_health_routes_return_healthy_status(self, app_with_health, path):
        with TestClient(app_with_health) as client:
            response = client.get(path)
            assert response.status_code == 200, f"Expected 200 at {path}"
            assert response.json() == {"status": "healthy"}, f"Unexpected response at {path}"

    def test_step_adaptor(self, worker):
        assert isinstance(worker.get_step_adaptor(), DRAgentNestedReasoningStepAdaptor)

    def test_get_a2a_endpoint_url_default(self, worker):
        assert get_a2a_endpoint_url("localhost", 8000) == "http://localhost:8000/a2a/"

    @pytest.mark.parametrize(
        "env,expected",
        [
            (
                {
                    "MLOPS_DEPLOYMENT_ID": "abc123",
                    "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
                },
                "https://app.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/",
            ),
            (
                {
                    "MLOPS_DEPLOYMENT_ID": "abc123",
                    "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2/",
                },
                "https://app.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/",
            ),
            (
                {
                    "MLOPS_DEPLOYMENT_ID": "abc123",
                    "DATAROBOT_PUBLIC_API_ENDPOINT": "https://public.datarobot.com/api/v2",
                    "DATAROBOT_ENDPOINT": "https://internal.k8s.local/api/v2",
                },
                "https://public.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/",
            ),
        ],
    )
    def test_get_a2a_endpoint_url_deployment(self, worker, env, expected):
        with patch.dict(os.environ, env, clear=True):
            assert get_a2a_endpoint_url("localhost", 8000) == expected

    def test_get_a2a_endpoint_url_deployment_missing_endpoint_raises(self, worker):
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": "abc123"}, clear=True):
            with pytest.raises(
                ValueError, match="DATAROBOT_PUBLIC_API_ENDPOINT or DATAROBOT_ENDPOINT must be set"
            ):
                get_a2a_endpoint_url("localhost", 8000)

    async def test_add_routes_inherits_host_port_from_fastapi_config(
        self, dragent_worker, mock_builder, mock_a2a_worker
    ):
        app = FastAPI()
        nat_session_from_parent = MagicMock()

        async def mock_super_add_routes(self, _app, _builder):
            self._session_managers.append(nat_session_from_parent)

        with (
            patch(
                "nat.front_ends.fastapi.fastapi_front_end_plugin_worker.FastApiFrontEndPluginWorker.add_routes",
                mock_super_add_routes,
            ),
            patch(
                "datarobot_genai.dragent.frontends.fastapi.A2AFrontEndPluginWorker",
                return_value=mock_a2a_worker,
            ) as mock_a2a_worker_cls,
            patch(
                "datarobot_genai.dragent.frontends.fastapi.SessionManager.create",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
        ):
            await dragent_worker.add_routes(app, mock_builder)

        a2a_config_used = mock_a2a_worker_cls.call_args[0][0].general.front_end
        assert a2a_config_used.host == dragent_worker.front_end_config.host
        assert a2a_config_used.port == dragent_worker.front_end_config.port

    @pytest.mark.asyncio
    async def test_add_routes_installs_stream_error_framing(
        self, mock_builder, patch_super_add_routes
    ):
        config = Config(general=GeneralConfig(front_end=DRAgentFastApiFrontEndConfig()))
        with patch.dict(os.environ, {"NAT_CONFIG_FILE": "unused"}):
            worker = DRAgentFastApiFrontEndPluginWorker(config)

        with patch(
            "datarobot_genai.dragent.frontends.fastapi.patch_stream_error_framing"
        ) as mock_install:
            await worker.add_routes(FastAPI(), mock_builder)

        mock_install.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_routes_patches_agent_card_url(
        self, dragent_worker, mock_builder, mock_a2a_worker, patch_super_add_routes
    ):
        app = FastAPI()
        with (
            patch(
                "datarobot_genai.dragent.frontends.fastapi.A2AFrontEndPluginWorker",
                return_value=mock_a2a_worker,
            ),
            patch(
                "datarobot_genai.dragent.frontends.fastapi.SessionManager.create",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
        ):
            await dragent_worker.add_routes(app, mock_builder)
        agent_card = mock_a2a_worker.create_a2a_server.call_args[0][0]
        assert agent_card.url == "http://localhost:8000/a2a/"

    @pytest.mark.asyncio
    async def test_add_routes_mounts_a2a(
        self, dragent_worker, mock_builder, mock_a2a_worker, patch_super_add_routes
    ):
        app = FastAPI()
        with (
            patch(
                "datarobot_genai.dragent.frontends.fastapi.A2AFrontEndPluginWorker",
                return_value=mock_a2a_worker,
            ),
            patch(
                "datarobot_genai.dragent.frontends.fastapi.SessionManager.create",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
        ):
            await dragent_worker.add_routes(app, mock_builder)

        mock_a2a_worker.create_a2a_server.assert_called_once()

    async def test_add_routes_appends_session_manager(
        self, dragent_worker, mock_builder, mock_a2a_worker, patch_super_add_routes
    ):
        app = FastAPI()
        mock_session_manager = MagicMock()
        with (
            patch(
                "datarobot_genai.dragent.frontends.fastapi.A2AFrontEndPluginWorker",
                return_value=mock_a2a_worker,
            ),
            patch(
                "datarobot_genai.dragent.frontends.fastapi.SessionManager.create",
                new_callable=AsyncMock,
                return_value=mock_session_manager,
            ),
        ):
            await dragent_worker.add_routes(app, mock_builder)

        assert mock_session_manager in dragent_worker._session_managers

    async def test_add_routes_disabled(self, mock_builder, patch_super_add_routes):
        """When a2a is None (default), A2A routes are not mounted."""
        config = Config(general=GeneralConfig(front_end=DRAgentFastApiFrontEndConfig()))
        with patch.dict(os.environ, {"NAT_CONFIG_FILE": "unused"}):
            disabled_worker = DRAgentFastApiFrontEndPluginWorker(config)
        app = FastAPI()
        with patch(
            "datarobot_genai.dragent.frontends.fastapi.A2AFrontEndPluginWorker"
        ) as mock_a2a_worker_cls:
            await disabled_worker.add_routes(app, mock_builder)
            mock_a2a_worker_cls.assert_not_called()


def _expected_key(raw_user_id: str) -> str:
    """Compute the expected UUID5 workflow key for a raw DataRobot user ID."""
    return UserInfo._from_session_cookie(raw_user_id).get_user_id()


def _make_auth_ctx(user_id: str) -> MagicMock:
    """Build a mock AuthCtx with the given ``user.id``."""
    ctx = MagicMock()
    ctx.user.id = user_id
    return ctx


_AUTH_HANDLER_PATH = "datarobot_genai.dragent.frontends.fastapi._auth_handler.get_context"


class TestResolveIdentityFromHeaders:
    """Tests for the _resolve_identity_from_headers helper."""

    @pytest.fixture(autouse=True)
    def _no_real_jwt_decode(self):
        """Prevent the real _auth_handler from touching JWT secrets during tests."""
        with patch(_AUTH_HANDLER_PATH, return_value=None):
            yield

    def test_returns_none_for_none_headers(self):
        assert _resolve_identity_from_headers(None) is None

    def test_returns_none_for_empty_headers(self):
        assert _resolve_identity_from_headers({}) is None

    def test_returns_none_when_no_identity_headers(self):
        result = _resolve_identity_from_headers(
            {"authorization": "Bearer tok", "content-type": "application/json"}
        )
        assert result is None

    def test_returns_uuid5_for_valid_signed_jwt(self):
        with patch(_AUTH_HANDLER_PATH, return_value=_make_auth_ctx("dr-uid-abc")):
            result = _resolve_identity_from_headers(
                {"x-datarobot-authorization-context": "signed-jwt"}
            )
        assert result == _expected_key("dr-uid-abc")

    def test_raises_for_invalid_jwt(self):
        with pytest.raises(ServerError) as exc_info:
            _resolve_identity_from_headers({"x-datarobot-authorization-context": "garbage"})
        assert isinstance(exc_info.value.error, InvalidParamsError)
        assert exc_info.value.error.code == -32602
        assert "invalid or expired" in exc_info.value.error.message

    def test_raises_when_auth_handler_throws(self):
        """Unexpected exceptions from _auth_handler.get_context are converted to ServerError."""
        with (
            patch(_AUTH_HANDLER_PATH, side_effect=RuntimeError("key store unavailable")),
            pytest.raises(ServerError) as exc_info,
        ):
            _resolve_identity_from_headers({"x-datarobot-authorization-context": "jwt"})
        assert isinstance(exc_info.value.error, InvalidParamsError)
        assert exc_info.value.error.code == -32602

    def test_invalid_auth_context_does_not_fall_through_to_gateway_user_id(self):
        """A present but invalid auth-context JWT must not fall back to gateway user ID."""
        with pytest.raises(ServerError) as exc_info:
            _resolve_identity_from_headers(
                {
                    "x-datarobot-authorization-context": "garbage",
                    "x-datarobot-user-id": "64baa56996fb36e3eeeefc44",
                }
            )
        assert isinstance(exc_info.value.error, InvalidParamsError)
        assert exc_info.value.error.code == -32602

    def test_falls_back_to_gateway_user_id_header(self):
        result = _resolve_identity_from_headers({"x-datarobot-user-id": "64baa56996fb36e3eeeefc44"})
        assert result == _expected_key("64baa56996fb36e3eeeefc44")

    def test_auth_context_takes_precedence_over_gateway_user_id(self):
        with patch(_AUTH_HANDLER_PATH, return_value=_make_auth_ctx("auth-ctx-user")):
            result = _resolve_identity_from_headers(
                {
                    "x-datarobot-authorization-context": "signed-jwt",
                    "x-datarobot-user-id": "gateway-user",
                }
            )
        assert result == _expected_key("auth-ctx-user")
        assert result != _expected_key("gateway-user")

    def test_deterministic_same_user(self):
        with patch(_AUTH_HANDLER_PATH, return_value=_make_auth_ctx("user-xyz")):
            r1 = _resolve_identity_from_headers({"x-datarobot-authorization-context": "jwt"})
            r2 = _resolve_identity_from_headers({"x-datarobot-authorization-context": "jwt"})
        assert r1 == r2

    def test_different_users_produce_different_keys(self):
        results = []
        for uid in ("alice", "bob"):
            with patch(_AUTH_HANDLER_PATH, return_value=_make_auth_ctx(uid)):
                results.append(
                    _resolve_identity_from_headers({"x-datarobot-authorization-context": "jwt"})
                )
        assert results[0] != results[1]


class TestPerUserCompatibleAgentExecutor:
    @pytest.fixture
    def session_manager(self):
        sm = MagicMock()
        sm.config.workflow.type = "test_workflow"
        return sm

    @pytest.fixture
    def executor(self, session_manager):
        return _PerUserCompatibleAgentExecutor(session_manager)

    @pytest.fixture
    def patch_super_execute(self):
        with patch.object(
            _PerUserCompatibleAgentExecutor.__bases__[0],
            "execute",
            new_callable=AsyncMock,
        ) as mock:
            yield mock

    @pytest.fixture
    def captured_keys(self, session_manager):
        """Capture all values passed to ``_context_state.user_id.set()``."""
        keys: list[str] = []

        def capture_set(value: str) -> MagicMock:
            keys.append(value)
            return MagicMock()

        session_manager._context_state.user_id.set = capture_set
        return keys

    def _make_a2a_context(
        self,
        *,
        context_id: str | None = "ctx-1",
        headers: dict[str, str] | None = None,
    ) -> MagicMock:
        """Build a mock A2A ``RequestContext`` with optional forwarded headers."""
        ctx = MagicMock()
        ctx.context_id = context_id
        if headers is not None:
            ctx.call_context.state = {"headers": headers}
        else:
            ctx.call_context = None
        return ctx

    def test_init_sets_session_manager(self, executor, session_manager):
        assert executor.session_manager is session_manager

    async def test_execute_uses_authenticated_identity_as_workflow_key(
        self, executor, session_manager, patch_super_execute
    ):
        """When A2A headers carry a valid auth context, the per-user workflow key
        is derived from the gateway-validated identity, NOT from context_id.
        """
        context = self._make_a2a_context(
            context_id="attacker-chosen-context-id",
            headers={"X-DataRobot-Authorization-Context": "signed-jwt"},
        )
        event_queue = MagicMock()
        with patch(_AUTH_HANDLER_PATH, return_value=_make_auth_ctx("real-dr-user")):
            await executor.execute(context, event_queue)

        session_manager._context_state.user_id.set.assert_called_once_with(
            _expected_key("real-dr-user")
        )
        patch_super_execute.assert_awaited_once_with(context, event_queue)

    async def test_execute_falls_back_to_context_id_when_no_auth_headers(
        self, executor, session_manager, patch_super_execute
    ):
        """Without authenticated headers (local dev), context_id is hashed via
        _from_session_cookie for key-format consistency with authenticated paths.
        """
        context = self._make_a2a_context(context_id="local-dev-ctx-id")

        await executor.execute(context, MagicMock())

        session_manager._context_state.user_id.set.assert_called_once_with(
            _expected_key("local-dev-ctx-id")
        )

    async def test_execute_skips_user_id_injection_when_no_context_id_and_no_auth(
        self, executor, session_manager, patch_super_execute
    ):
        context = self._make_a2a_context(context_id=None)

        await executor.execute(context, MagicMock())

        session_manager._context_state.user_id.set.assert_not_called()

    async def test_execute_two_users_same_context_id_get_different_keys(
        self, session_manager, captured_keys, patch_super_execute
    ):
        """Two different authenticated users sending the same context_id must get
        different per-user workflow keys -- the core isolation guarantee.
        """
        for uid in ("alice", "bob"):
            executor = _PerUserCompatibleAgentExecutor(session_manager)
            context = self._make_a2a_context(
                context_id="shared-context-id",
                headers={"X-DataRobot-Authorization-Context": "jwt"},
            )
            with patch(_AUTH_HANDLER_PATH, return_value=_make_auth_ctx(uid)):
                await executor.execute(context, MagicMock())

        assert len(captured_keys) == 2
        assert captured_keys[0] != captured_keys[1]

    async def test_execute_same_user_different_context_id_gets_same_key(
        self, session_manager, captured_keys, patch_super_execute
    ):
        """The same authenticated user across different conversations must get the
        same per-user workflow key -- one builder per user, not per conversation.
        """
        for ctx_id in ("conversation-1", "conversation-2"):
            executor = _PerUserCompatibleAgentExecutor(session_manager)
            context = self._make_a2a_context(
                context_id=ctx_id,
                headers={"X-DataRobot-Authorization-Context": "jwt"},
            )
            with patch(_AUTH_HANDLER_PATH, return_value=_make_auth_ctx("consistent-user")):
                await executor.execute(context, MagicMock())

        assert len(captured_keys) == 2
        assert captured_keys[0] == captured_keys[1]

    async def test_execute_uses_gateway_user_id_when_no_auth_context(
        self, executor, session_manager, patch_super_execute
    ):
        """When X-DataRobot-Authorization-Context is absent but X-DataRobot-User-Id
        is present, the gateway user ID is used as the workflow key.
        """
        context = self._make_a2a_context(
            context_id="should-not-be-used",
            headers={"X-DataRobot-User-Id": "64baa56996fb36e3eeeefc44"},
        )
        with patch(_AUTH_HANDLER_PATH, return_value=None):
            await executor.execute(context, MagicMock())

        session_manager._context_state.user_id.set.assert_called_once_with(
            _expected_key("64baa56996fb36e3eeeefc44")
        )

    async def test_execute_raises_when_auth_context_invalid_instead_of_context_id_fallback(
        self, executor, session_manager, patch_super_execute
    ):
        """Invalid auth-context must fail closed; must not fall back to context_id."""
        context = self._make_a2a_context(
            context_id="must-not-be-used",
            headers={"X-DataRobot-Authorization-Context": "garbage"},
        )
        with (
            patch(_AUTH_HANDLER_PATH, return_value=None),
            pytest.raises(ServerError) as exc_info,
        ):
            await executor.execute(context, MagicMock())
        assert isinstance(exc_info.value.error, InvalidParamsError)
        assert exc_info.value.error.code == -32602

        session_manager._context_state.user_id.set.assert_not_called()
        patch_super_execute.assert_not_awaited()

    async def test_execute_resets_a2a_headers_on_identity_error(
        self, executor, session_manager, patch_super_execute
    ):
        """ContextVar for _a2a_headers must be cleaned up even when identity
        resolution raises ServerError (no ContextVar leak).
        """
        from datarobot_genai.dragent.frontends.session import _a2a_headers

        context = self._make_a2a_context(
            context_id="ctx",
            headers={"X-DataRobot-Authorization-Context": "garbage"},
        )
        sentinel = object()
        original = _a2a_headers.get(sentinel)

        with (
            patch(_AUTH_HANDLER_PATH, return_value=None),
            pytest.raises(ServerError),
        ):
            await executor.execute(context, MagicMock())

        assert _a2a_headers.get(sentinel) is original

    async def test_execute_logs_warning_on_unauthenticated_fallback(
        self, executor, session_manager, patch_super_execute
    ):
        """Falling back to context_id logs a warning for production visibility."""
        context = self._make_a2a_context(
            context_id="unauthenticated-ctx",
            headers={"some-header": "value"},
        )
        with (
            patch(_AUTH_HANDLER_PATH, return_value=None),
            patch("datarobot_genai.dragent.frontends.fastapi.logger") as mock_logger,
        ):
            await executor.execute(context, MagicMock())

        mock_logger.warning.assert_called_once()
        assert "falling back to context_id" in mock_logger.warning.call_args[0][0].lower()


class TestCreateAgentCard:
    async def test_default_skill_when_skills_empty(self, a2a_frontend_config):
        card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[])
        assert len(card.skills) == 1
        assert card.skills[0].id == "call"
        assert card.skills[0].name == "My Agent"
        assert card.skills[0].description == "Does things"

    async def test_configured_skills_used_when_present(self, a2a_frontend_config):
        skill = AgentSkill(id="summarize", name="Summarize", description="Summarizes text", tags=[])
        card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[skill])
        assert len(card.skills) == 1
        assert card.skills[0].id == "summarize"

    async def test_agent_card_fields_from_frontend_config(self):
        cfg = A2AFrontEndConfig(
            name="My Agent",
            description="Does things",
            version="2.0.0",
            host="localhost",
            port=9000,
        )
        card = await create_agent_card(cfg, cross_app_access=None, skills=[])
        assert card.name == "My Agent"
        assert card.description == "Does things"
        assert card.version == "2.0.0"
        assert card.url == "http://localhost:9000/a2a/"

    async def test_security_schemes_set_when_cross_application_access_present(
        self, a2a_frontend_config
    ):
        cross_app_access = CrossApplicationAccessConfig(
            token_endpoint_auth_method="private_key_jwt",
            token_exchange=CrossAppTokenExchange(
                trusted_issuer="https://your-org.oktapreview.com",
                audience="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7",
            ),
            token_request=CrossAppTokenRequest(
                token_url="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7/v1/token",
                audience="https://app.datarobot.com/dr_org_id/my_agent_id",
                scopes=["blog:write"],
            ),
        )
        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=cross_app_access, skills=[]
        )

        assert "oauth2" in card.security_schemes
        oauth_scheme = card.security_schemes["oauth2"].root
        assert oauth_scheme.type == "oauth2"
        assert oauth_scheme.description == OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE

        # Only client_credentials flow, no authorization_code
        assert oauth_scheme.flows.authorization_code is None
        flow = oauth_scheme.flows.client_credentials
        assert flow.token_url == "https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7/v1/token"
        assert flow.scopes == {"blog:write": "Permission: blog:write"}

        assert card.security == [{"oauth2": ["blog:write"]}]

        # JWT Bearer extension: nested params — token_url/scopes must NOT appear here
        assert card.capabilities.extensions is not None
        assert len(card.capabilities.extensions) == 1
        ext = card.capabilities.extensions[0]
        assert ext.uri == JWT_BEARER_GRANT_TYPE_URI
        assert ext.description == CROSS_APP_EXTENSION_DESCRIPTION
        assert ext.params == {
            "ref": {
                "scheme": CROSS_APP_SECURITY_SCHEME_REF,
                "flow": CROSS_APP_SECURITY_SCHEME_FLOW_REF,
            },
            "tokenEndpointAuthMethod": "private_key_jwt",
            "tokenExchange": {
                "grantType": TOKEN_EXCHANGE_GRANT_TYPE_URI,
                "requestedTokenType": TOKEN_EXCHANGE_REQUESTED_TOKEN_TYPE,
                "trustedIssuer": "https://your-org.oktapreview.com",
                "audience": "https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7",
            },
            "tokenRequest": {
                "grantType": JWT_BEARER_GRANT_TYPE_URI,
                "audience": "https://app.datarobot.com/dr_org_id/my_agent_id",
            },
        }
        # Verify OpenAPI/extension strict separation: token_url and scopes are NOT in params
        assert "token_url" not in ext.params
        assert "scopes" not in ext.params

    async def test_security_schemes_from_server_auth(self, a2a_frontend_config):
        a2a_frontend_config.server_auth = MagicMock(
            issuer_url="https://issuer.example.com",
            discovery_url=None,
            scopes=["read"],
        )
        card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[])

        oauth_scheme = card.security_schemes["oauth2"].root
        assert oauth_scheme.description == OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE
        # Only authorization_code flow, no client_credentials
        assert oauth_scheme.flows.authorization_code is not None
        assert (
            oauth_scheme.flows.authorization_code.authorization_url
            == "https://issuer.example.com/oauth/authorize"
        )
        assert (
            oauth_scheme.flows.authorization_code.token_url
            == "https://issuer.example.com/oauth/token"
        )
        assert oauth_scheme.flows.client_credentials is None
        assert card.security == [{"oauth2": ["read"]}]

    async def test_both_server_auth_and_cross_application_access(self, a2a_frontend_config):
        # server_auth → authorization_code flow
        a2a_frontend_config.server_auth = MagicMock(
            issuer_url="https://issuer.example.com",
            discovery_url=None,
            scopes=["read"],
        )

        # cross_application_access → client_credentials flow + JWT Bearer extension
        cross_app_access = CrossApplicationAccessConfig(
            token_endpoint_auth_method="private_key_jwt",
            token_exchange=CrossAppTokenExchange(
                trusted_issuer="https://your-org.oktapreview.com",
                audience="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7",
            ),
            token_request=CrossAppTokenRequest(
                token_url="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7/v1/token",
                audience="https://app.datarobot.com/dr_org_id/my_agent_id",
                scopes=["blog:write"],
            ),
        )

        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=cross_app_access, skills=[]
        )

        # Single oauth2 scheme with both flows
        assert len(card.security_schemes) == 1
        oauth_scheme = card.security_schemes["oauth2"].root
        assert oauth_scheme.description == OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE

        assert oauth_scheme.flows.authorization_code is not None
        assert (
            oauth_scheme.flows.authorization_code.authorization_url
            == "https://issuer.example.com/oauth/authorize"
        )

        assert oauth_scheme.flows.client_credentials is not None
        assert (
            oauth_scheme.flows.client_credentials.token_url
            == "https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7/v1/token"
        )

        # Merged scopes (deduplicated)
        assert card.security == [{"oauth2": ["read", "blog:write"]}]

        # Cross-app extension: nested params; token_url/scopes only under OpenAPI flows
        assert card.capabilities.extensions is not None
        ext = card.capabilities.extensions[0]
        assert ext.uri == JWT_BEARER_GRANT_TYPE_URI
        assert ext.description == CROSS_APP_EXTENSION_DESCRIPTION
        assert ext.params == {
            "ref": {
                "scheme": CROSS_APP_SECURITY_SCHEME_REF,
                "flow": CROSS_APP_SECURITY_SCHEME_FLOW_REF,
            },
            "tokenEndpointAuthMethod": "private_key_jwt",
            "tokenExchange": {
                "grantType": TOKEN_EXCHANGE_GRANT_TYPE_URI,
                "requestedTokenType": TOKEN_EXCHANGE_REQUESTED_TOKEN_TYPE,
                "trustedIssuer": "https://your-org.oktapreview.com",
                "audience": "https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7",
            },
            "tokenRequest": {
                "grantType": JWT_BEARER_GRANT_TYPE_URI,
                "audience": "https://app.datarobot.com/dr_org_id/my_agent_id",
            },
        }
        assert "token_url" not in ext.params
        assert "scopes" not in ext.params

    async def test_no_security_when_server_auth_absent(self, a2a_frontend_config):
        card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[])
        assert card.security_schemes is None
        assert card.security is None

    async def test_internal_identity_extension_when_deployment_id_set(self, a2a_frontend_config):
        """GIVEN MLOPS_DEPLOYMENT_ID is set WHEN create_agent_card is called THEN the internal
        identity extension is present with the deployment_id.
        """
        env = {
            "MLOPS_DEPLOYMENT_ID": "dep-abc123",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        }
        with patch.dict(os.environ, env):
            card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[])

        assert card.capabilities.extensions is not None
        uris = [ext.uri for ext in card.capabilities.extensions]
        assert INTERNAL_IDENTITY_URI in uris
        internal = next(e for e in card.capabilities.extensions if e.uri == INTERNAL_IDENTITY_URI)
        assert internal.required is True
        assert internal.params == {"deployment_id": "dep-abc123"}

    async def test_internal_identity_extension_when_workload_id_set(self, a2a_frontend_config):
        """GIVEN WORKLOAD_ID is set WHEN create_agent_card is called THEN the internal
        identity extension is present with the workload_id.
        """
        env = {
            "WORKLOAD_ID": "wl-abc123",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        }
        with patch.dict(os.environ, env, clear=True):
            card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[])

        assert card.capabilities.extensions is not None
        uris = [ext.uri for ext in card.capabilities.extensions]
        assert INTERNAL_IDENTITY_URI in uris
        internal = next(e for e in card.capabilities.extensions if e.uri == INTERNAL_IDENTITY_URI)
        assert internal.required is True
        assert internal.params == {"workload_id": "wl-abc123"}

    async def test_no_internal_identity_extension_in_local_dev(self, a2a_frontend_config):
        """GIVEN MLOPS_DEPLOYMENT_ID is not set WHEN create_agent_card is called THEN the
        internal identity extension is absent.
        """
        with patch.dict(os.environ, {}, clear=True):
            card = await create_agent_card(a2a_frontend_config, cross_app_access=None, skills=[])

        extensions = card.capabilities.extensions or []
        assert not any(e.uri == INTERNAL_IDENTITY_URI for e in extensions)

    async def test_external_identity_extension_when_external_id_set(self, a2a_frontend_config):
        """GIVEN external.id is provided WHEN create_agent_card is called THEN the external
        identity extension is present with the correct id.
        """
        external = DRAgentA2AExternalConfig(id="catalog-id-xyz")
        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=None, skills=[], external=external
        )

        assert card.capabilities.extensions is not None
        uris = [ext.uri for ext in card.capabilities.extensions]
        assert EXTERNAL_IDENTITY_URI in uris
        ext = next(e for e in card.capabilities.extensions if e.uri == EXTERNAL_IDENTITY_URI)
        assert ext.required is False
        assert ext.params == {"id": "catalog-id-xyz"}

    async def test_no_external_identity_extension_when_external_absent(self, a2a_frontend_config):
        """GIVEN external is None WHEN create_agent_card is called THEN no external identity
        extension is present.
        """
        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=None, skills=[], external=None
        )

        extensions = card.capabilities.extensions or []
        assert not any(e.uri == EXTERNAL_IDENTITY_URI for e in extensions)

    async def test_external_url_overrides_agent_card_url(self, a2a_frontend_config):
        """GIVEN external.url is set WHEN create_agent_card is called THEN the agent card url
        uses the external URL exactly as provided.
        """
        external = DRAgentA2AExternalConfig(url="https://custom.example.com/agent/")
        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=None, skills=[], external=external
        )

        assert card.url == "https://custom.example.com/agent/"

    async def test_external_url_used_as_provided(self, a2a_frontend_config):
        """GIVEN external.url is set without a trailing slash WHEN create_agent_card is called
        THEN the url is used exactly as provided, without modification.
        """
        external = DRAgentA2AExternalConfig(url="https://custom.example.com/agent")
        card = await create_agent_card(
            a2a_frontend_config, cross_app_access=None, skills=[], external=external
        )

        assert card.url == "https://custom.example.com/agent"

    async def test_all_extensions_combined(self, a2a_frontend_config):
        """GIVEN cross_app_access, MLOPS_DEPLOYMENT_ID, and external.id are all set WHEN
        create_agent_card is called THEN all three extensions are present.
        """
        cross_app_access = CrossApplicationAccessConfig(
            token_endpoint_auth_method="private_key_jwt",
            token_exchange=CrossAppTokenExchange(
                trusted_issuer="https://your-org.oktapreview.com",
                audience="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7",
            ),
            token_request=CrossAppTokenRequest(
                token_url="https://your-org.okta.com/oauth2/aussu3akcsQeofA0C1d7/v1/token",
                audience="https://app.datarobot.com/dr_org_id/my_agent_id",
            ),
        )
        external = DRAgentA2AExternalConfig(id="catalog-id-combined")
        env = {
            "MLOPS_DEPLOYMENT_ID": "dep-combined",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        }
        with patch.dict(os.environ, env):
            card = await create_agent_card(
                a2a_frontend_config,
                cross_app_access=cross_app_access,
                skills=[],
                external=external,
            )

        assert card.capabilities.extensions is not None
        uris = [ext.uri for ext in card.capabilities.extensions]
        assert JWT_BEARER_GRANT_TYPE_URI in uris
        assert INTERNAL_IDENTITY_URI in uris
        assert EXTERNAL_IDENTITY_URI in uris


class TestDRAgentFastApiFrontEndConfig:
    def test_is_fastapi_front_end_config(self):
        assert isinstance(DRAgentFastApiFrontEndConfig(), FastApiFrontEndConfig)

    def test_a2a_default_none(self):
        config = DRAgentFastApiFrontEndConfig()
        assert config.a2a is None

    def test_custom_a2a_fields(self):
        cross_app = CrossApplicationAccessConfig(
            token_endpoint_auth_method="private_key_jwt",
            token_exchange=CrossAppTokenExchange(
                trusted_issuer="https://id-jag.example.com",
                audience="https://idp.example.com/oauth2/ausXXX",
            ),
            token_request=CrossAppTokenRequest(
                token_url="https://idp.example.com/oauth2/v1/token",
                audience="api://my-agent",
                scopes=["agent:use"],
            ),
        )
        config = DRAgentFastApiFrontEndConfig(
            a2a=DRAgentA2AConfig(
                server=A2AFrontEndConfig(
                    name="My Agent",
                    description="Does things",
                    version="2.0.0",
                ),
                cross_application_access=cross_app,
            )
        )
        assert config.a2a.server.name == "My Agent"
        assert config.a2a.server.description == "Does things"
        assert config.a2a.server.version == "2.0.0"
        assert config.a2a.cross_application_access == cross_app

    def test_is_not_a2a_front_end_config(self):
        config = DRAgentFastApiFrontEndConfig()
        assert not isinstance(config, A2AFrontEndConfig)

    def test_a2a_enables_endpoints(self):
        config = DRAgentFastApiFrontEndConfig(a2a=DRAgentA2AConfig(server=A2AFrontEndConfig()))
        assert config.a2a is not None

    def test_a2a_external_config_optional(self):
        config = DRAgentFastApiFrontEndConfig(a2a=DRAgentA2AConfig(server=A2AFrontEndConfig()))
        assert config.a2a.external is None

    def test_a2a_with_external_config(self):
        external = DRAgentA2AExternalConfig(id="ext-id-123", url="https://external.example.com/")
        config = DRAgentFastApiFrontEndConfig(
            a2a=DRAgentA2AConfig(server=A2AFrontEndConfig(), external=external)
        )
        assert config.a2a.external.id == "ext-id-123"
        assert config.a2a.external.url == "https://external.example.com/"


class TestDRAgentFastApiFrontEndPluginWorkerCleanup:
    @pytest.mark.asyncio
    async def test_a2a_worker_cleanup_called_on_lifespan_exit(
        self, dragent_worker, mock_a2a_worker
    ):
        dragent_worker._a2a_worker = mock_a2a_worker

        parent_app = FastAPI()

        @asynccontextmanager
        async def fake_lifespan(app):
            yield

        parent_app.router.lifespan_context = fake_lifespan

        with patch.object(dragent_worker, "build_app", wraps=dragent_worker.build_app):
            with patch.object(
                type(dragent_worker).__bases__[0], "build_app", return_value=parent_app
            ):
                app = dragent_worker.build_app()

        with TestClient(app):
            mock_a2a_worker.cleanup.assert_not_awaited()
        mock_a2a_worker.cleanup.assert_awaited_once()

    async def test_cleanup_noop_when_no_a2a_worker(self, dragent_worker):
        parent_app = FastAPI()

        @asynccontextmanager
        async def fake_lifespan(app):
            yield

        parent_app.router.lifespan_context = fake_lifespan

        with patch.object(type(dragent_worker).__bases__[0], "build_app", return_value=parent_app):
            app = dragent_worker.build_app()

        with TestClient(app):
            pass  # should not raise


class TestDRAgentFastApiFrontEndPlugin:
    def test_get_worker_class(self):
        plugin = DRAgentFastApiFrontEndPlugin(full_config=Config(general=GeneralConfig()))
        assert plugin.get_worker_class() is DRAgentFastApiFrontEndPluginWorker

    def test_dask_client_initialized_to_none(self):
        """NAT's run() finally-block reads self._dask_client directly; it must exist even
        when dask is not installed so shutdown cleanup doesn't raise AttributeError.
        """
        plugin = DRAgentFastApiFrontEndPlugin(full_config=Config(general=GeneralConfig()))
        assert plugin._dask_client is None

    @pytest.mark.asyncio
    async def test_run_shutdown_without_dask_does_not_raise(self):
        """Simulate stopping the server (Ctrl+C) so NAT's run() finally-block runs its
        dask cleanup. Without the _dask_client fix this raises AttributeError when dask
        is not installed.
        """
        config = Config(general=GeneralConfig(front_end=DRAgentFastApiFrontEndConfig()))
        plugin = DRAgentFastApiFrontEndPlugin(full_config=config)

        async def fake_serve(_self):
            raise KeyboardInterrupt

        with (
            patch("datarobot_genai.dragent.workflow_paths.publish_dragent_config_file_env"),
            patch("uvicorn.Server.serve", fake_serve),
        ):
            await plugin.run()


class TestGunicornSettings:
    _ENV = "AGENT_GUNICORN_WORKER_TIMEOUT"
    _RUNTIME_PARAM_ENV = "MLOPS_RUNTIME_PARAM_AGENT_GUNICORN_WORKER_TIMEOUT"

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv(self._ENV, raising=False)
        monkeypatch.delenv(self._RUNTIME_PARAM_ENV, raising=False)

    def test_default(self):
        assert _GunicornSettings().agent_gunicorn_worker_timeout == 600

    def test_plain_env_override(self, monkeypatch):
        monkeypatch.setenv(self._ENV, "300")
        assert _GunicornSettings().agent_gunicorn_worker_timeout == 300

    def test_numeric_runtime_param_float_payload(self, monkeypatch):
        """A DataRobot numeric runtime param delivers a float payload; it coerces to int."""
        monkeypatch.setenv(self._RUNTIME_PARAM_ENV, '{"type": "numeric", "payload": 300.0}')
        assert _GunicornSettings().agent_gunicorn_worker_timeout == 300

    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv(self._ENV, "not-a-number")
        with pytest.raises(ValidationError):
            _GunicornSettings()

    def test_non_positive_raises(self, monkeypatch):
        monkeypatch.setenv(self._ENV, "0")
        with pytest.raises(ValidationError):
            _GunicornSettings()


class TestPatchGunicornWorkerTimeout:
    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv("AGENT_GUNICORN_WORKER_TIMEOUT", raising=False)
        monkeypatch.delenv("MLOPS_RUNTIME_PARAM_AGENT_GUNICORN_WORKER_TIMEOUT", raising=False)

    @pytest.fixture
    def fake_gunicorn(self, monkeypatch):
        """Install a stand-in ``gunicorn.config`` (gunicorn is not a genai dependency)."""

        class Timeout:
            default = 30

        class GracefulTimeout:
            default = 30

        config_mod = types.ModuleType("gunicorn.config")
        config_mod.Timeout = Timeout
        config_mod.GracefulTimeout = GracefulTimeout
        pkg = types.ModuleType("gunicorn")
        pkg.config = config_mod
        monkeypatch.setitem(sys.modules, "gunicorn", pkg)
        monkeypatch.setitem(sys.modules, "gunicorn.config", config_mod)
        return config_mod

    def test_applies_default(self, fake_gunicorn):
        _patch_gunicorn_worker_timeout()
        assert fake_gunicorn.Timeout.default == 600
        assert fake_gunicorn.GracefulTimeout.default == 600

    def test_applies_override(self, fake_gunicorn, monkeypatch):
        monkeypatch.setenv("AGENT_GUNICORN_WORKER_TIMEOUT", "300")
        _patch_gunicorn_worker_timeout()
        assert fake_gunicorn.Timeout.default == 300
        assert fake_gunicorn.GracefulTimeout.default == 300

    def test_noop_when_gunicorn_not_installed(self, monkeypatch):
        """The real genai env has no gunicorn; the helper must no-op, not raise."""
        monkeypatch.setitem(sys.modules, "gunicorn", None)  # forces ImportError
        _patch_gunicorn_worker_timeout()  # must not raise


class TestRunGunicornTimeoutGating:
    _SUPER_RUN = "nat.front_ends.fastapi.fastapi_front_end_plugin.FastApiFrontEndPlugin.run"
    _PATCH_FN = "datarobot_genai.dragent.frontends.fastapi._patch_gunicorn_worker_timeout"
    _PUBLISH = "datarobot_genai.dragent.workflow_paths.publish_dragent_config_file_env"

    @pytest.mark.asyncio
    async def test_run_patches_timeout_when_use_gunicorn(self):
        config = Config(
            general=GeneralConfig(front_end=DRAgentFastApiFrontEndConfig(use_gunicorn=True))
        )
        plugin = DRAgentFastApiFrontEndPlugin(full_config=config)
        with (
            patch(self._PUBLISH),
            patch(self._PATCH_FN) as mock_patch,
            patch(self._SUPER_RUN, new_callable=AsyncMock),
        ):
            await plugin.run()
        mock_patch.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_skips_patch_without_use_gunicorn(self):
        config = Config(
            general=GeneralConfig(front_end=DRAgentFastApiFrontEndConfig(use_gunicorn=False))
        )
        plugin = DRAgentFastApiFrontEndPlugin(full_config=config)
        with (
            patch(self._PUBLISH),
            patch(self._PATCH_FN) as mock_patch,
            patch(self._SUPER_RUN, new_callable=AsyncMock),
        ):
            await plugin.run()
        mock_patch.assert_not_called()
