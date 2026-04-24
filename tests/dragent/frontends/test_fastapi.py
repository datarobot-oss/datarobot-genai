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
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from a2a.types import AgentSkill
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig

from datarobot_genai.dragent.frontends.fastapi import DATAROBOT_EXPECTED_HEALTH_ROUTES
from datarobot_genai.dragent.frontends.fastapi import DRAgentFastApiFrontEndPlugin
from datarobot_genai.dragent.frontends.fastapi import DRAgentFastApiFrontEndPluginWorker
from datarobot_genai.dragent.frontends.fastapi import _PerUserCompatibleAgentExecutor
from datarobot_genai.dragent.frontends.register import DRAgentA2AConfig
from datarobot_genai.dragent.frontends.register import DRAgentFastApiFrontEndConfig
from datarobot_genai.dragent.frontends.step_adaptor import DRAgentNestedReasoningStepAdaptor


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
        cfg = A2AFrontEndConfig(host="localhost", port=8000)
        assert worker._get_a2a_endpoint_url(cfg) == "http://localhost:8000/a2a/"

    def test_get_a2a_endpoint_url_deployment(self, worker):
        cfg = A2AFrontEndConfig(host="localhost", port=8000)
        env = {
            "MLOPS_DEPLOYMENT_ID": "abc123",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        }
        with patch.dict(os.environ, env, clear=True):
            url = worker._get_a2a_endpoint_url(cfg)
        assert url == "https://app.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/"

    def test_get_a2a_endpoint_url_deployment_strips_trailing_slash(self, worker):
        cfg = A2AFrontEndConfig(host="localhost", port=8000)
        env = {
            "MLOPS_DEPLOYMENT_ID": "abc123",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2/",
        }
        with patch.dict(os.environ, env, clear=True):
            url = worker._get_a2a_endpoint_url(cfg)
        assert url == "https://app.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/"

    def test_get_a2a_endpoint_url_deployment_prefers_public_api_endpoint(self, worker):
        cfg = A2AFrontEndConfig(host="localhost", port=8000)
        env = {
            "MLOPS_DEPLOYMENT_ID": "abc123",
            "DATAROBOT_PUBLIC_API_ENDPOINT": "https://public.datarobot.com/api/v2",
            "DATAROBOT_ENDPOINT": "https://internal.k8s.local/api/v2",
        }
        with patch.dict(os.environ, env, clear=True):
            url = worker._get_a2a_endpoint_url(cfg)
        assert url == "https://public.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/"

    def test_get_a2a_endpoint_url_deployment_falls_back_to_endpoint(self, worker):
        cfg = A2AFrontEndConfig(host="localhost", port=8000)
        env = {
            "MLOPS_DEPLOYMENT_ID": "abc123",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        }
        with patch.dict(os.environ, env, clear=True):
            url = worker._get_a2a_endpoint_url(cfg)
        assert url == "https://app.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/"

    def test_get_a2a_endpoint_url_deployment_missing_endpoint_raises(self, worker):
        cfg = A2AFrontEndConfig(host="localhost", port=8000)
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": "abc123"}, clear=True):
            with pytest.raises(
                ValueError, match="DATAROBOT_PUBLIC_API_ENDPOINT or DATAROBOT_ENDPOINT must be set"
            ):
                worker._get_a2a_endpoint_url(cfg)

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

    def test_init_sets_session_manager(self, executor, session_manager):
        assert executor.session_manager is session_manager

    async def test_execute_injects_context_id_as_user_id(
        self, executor, session_manager, patch_super_execute
    ):
        context = MagicMock()
        context.context_id = "user-123"
        event_queue = MagicMock()

        await executor.execute(context, event_queue)

        session_manager._context_state.user_id.set.assert_called_once_with("user-123")
        patch_super_execute.assert_awaited_once_with(context, event_queue)

    async def test_execute_skips_user_id_injection_when_no_context_id(
        self, executor, session_manager, patch_super_execute
    ):
        context = MagicMock()
        context.context_id = None
        event_queue = MagicMock()

        await executor.execute(context, event_queue)

        session_manager._context_state.user_id.set.assert_not_called()


class TestCreateAgentCard:
    async def test_default_skill_when_skills_empty(
        self, dragent_worker_with_a2a, a2a_frontend_config
    ):
        card = await dragent_worker_with_a2a._create_agent_card(a2a_frontend_config)
        assert len(card.skills) == 1
        assert card.skills[0].id == "call"
        assert card.skills[0].name == "My Agent"
        assert card.skills[0].description == "Does things"

    async def test_configured_skills_used_when_present(
        self, dragent_worker_with_a2a, a2a_frontend_config
    ):
        skill = AgentSkill(id="summarize", name="Summarize", description="Summarizes text", tags=[])
        dragent_worker_with_a2a.front_end_config.a2a.skills = [skill]
        card = await dragent_worker_with_a2a._create_agent_card(a2a_frontend_config)
        assert len(card.skills) == 1
        assert card.skills[0].id == "summarize"

    async def test_agent_card_fields_from_frontend_config(self, dragent_worker_with_a2a):
        cfg = A2AFrontEndConfig(
            name="My Agent",
            description="Does things",
            version="2.0.0",
            host="localhost",
            port=9000,
        )
        card = await dragent_worker_with_a2a._create_agent_card(cfg)
        assert card.name == "My Agent"
        assert card.description == "Does things"
        assert card.version == "2.0.0"
        assert card.url == "http://localhost:9000/a2a/"

    async def test_security_schemes_set_when_server_auth_present(
        self, dragent_worker_with_a2a, mock_a2a_worker, a2a_frontend_config
    ):
        mock_schemes = MagicMock()
        mock_security = MagicMock()
        mock_a2a_worker._generate_security_schemes = AsyncMock(
            return_value=(mock_schemes, mock_security)
        )
        a2a_frontend_config.server_auth = MagicMock()
        with patch("datarobot_genai.dragent.frontends.fastapi.AgentCard") as mock_agent_card_cls:
            await dragent_worker_with_a2a._create_agent_card(a2a_frontend_config)
        mock_a2a_worker._generate_security_schemes.assert_awaited_once_with(
            a2a_frontend_config.server_auth
        )
        _, kwargs = mock_agent_card_cls.call_args
        assert kwargs["security_schemes"] is mock_schemes
        assert kwargs["security"] is mock_security

    async def test_no_security_when_server_auth_absent(
        self, dragent_worker_with_a2a, a2a_frontend_config
    ):
        card = await dragent_worker_with_a2a._create_agent_card(a2a_frontend_config)
        assert card.security_schemes is None
        assert card.security is None


class TestDRAgentFastApiFrontEndConfig:
    def test_is_fastapi_front_end_config(self):
        assert isinstance(DRAgentFastApiFrontEndConfig(), FastApiFrontEndConfig)

    def test_a2a_default_none(self):
        config = DRAgentFastApiFrontEndConfig()
        assert config.a2a is None

    def test_custom_a2a_fields(self):
        config = DRAgentFastApiFrontEndConfig(
            a2a=DRAgentA2AConfig(
                server=A2AFrontEndConfig(
                    name="My Agent",
                    description="Does things",
                    version="2.0.0",
                )
            )
        )
        assert config.a2a.server.name == "My Agent"
        assert config.a2a.server.description == "Does things"
        assert config.a2a.server.version == "2.0.0"

    def test_is_not_a2a_front_end_config(self):
        config = DRAgentFastApiFrontEndConfig()
        assert not isinstance(config, A2AFrontEndConfig)

    def test_a2a_enables_endpoints(self):
        config = DRAgentFastApiFrontEndConfig(a2a=DRAgentA2AConfig(server=A2AFrontEndConfig()))
        assert config.a2a is not None


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
