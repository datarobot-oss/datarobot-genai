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
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig

from datarobot_genai.dragent.frontserver import DRAgentFastApiFrontEndPlugin
from datarobot_genai.dragent.frontserver import DRAgentFastApiFrontEndPluginWorker
from datarobot_genai.dragent.register import DRAgentFastApiFrontEndConfig
from datarobot_genai.dragent.step_adaptor import DRAgentNestedReasoningStepAdaptor


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
                expose_a2a_server_endpoints=True,
                a2a=A2AFrontEndConfig(
                    name="Test Agent",
                    description="A test agent",
                ),
            )
        )
    )
    with patch.dict(os.environ, {"NAT_CONFIG_FILE": "unused"}):
        return DRAgentFastApiFrontEndPluginWorker(config)


@pytest.fixture
def app_with_health(worker):
    """Build the FastAPI app the same way the server does, mocking WorkflowBuilder."""

    async def fake_configure(app: FastAPI, builder):
        _ = builder
        await worker.add_health_route(app)

    @asynccontextmanager
    async def mock_from_config(_config):
        yield MagicMock()

    with (
        patch.object(worker, "configure", side_effect=fake_configure),
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
    worker.create_agent_card = AsyncMock(return_value=MagicMock(url="http://localhost:8000/"))
    worker.create_agent_executor = MagicMock(return_value=MagicMock())
    worker.create_a2a_server = MagicMock(
        return_value=MagicMock(build=MagicMock(return_value=FastAPI()))
    )
    worker.cleanup = AsyncMock()
    return worker


@pytest.fixture
def patch_super_add_routes():
    with patch(
        "nat.front_ends.fastapi.fastapi_front_end_plugin_worker.FastApiFrontEndPluginWorker.add_routes",
        new_callable=AsyncMock,
    ):
        yield


class TestDRAgentFastApiFrontEndPluginWorker:
    EXPECTED_HEALTH_ROUTES = ["/", "/ping", "/ping/", "/health", "/health/"]

    @pytest.mark.parametrize("path", EXPECTED_HEALTH_ROUTES)
    def test_health_routes_return_healthy_status(self, app_with_health, path):
        with TestClient(app_with_health) as client:
            response = client.get(path)
            assert response.status_code == 200, f"Expected 200 at {path}"
            assert response.json() == {"status": "healthy"}, f"Unexpected response at {path}"

    def test_step_adaptor(self, worker):
        assert isinstance(worker.get_step_adaptor(), DRAgentNestedReasoningStepAdaptor)

    def test_get_a2a_endpoint_url_default(self, worker):
        url = worker._get_a2a_endpoint_url("http://localhost:8000/")
        assert url == "http://localhost:8000/a2a/"

    def test_get_a2a_endpoint_url_strips_trailing_slash(self, worker):
        url = worker._get_a2a_endpoint_url("http://localhost:8000")
        assert url == "http://localhost:8000/a2a/"

    def test_get_a2a_endpoint_url_deployment(self, worker):
        env = {
            "MLOPS_DEPLOYMENT_ID": "abc123",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        }
        with patch.dict(os.environ, env):
            url = worker._get_a2a_endpoint_url("http://localhost:8000/")
        assert url == "https://app.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/"

    def test_get_a2a_endpoint_url_deployment_strips_trailing_slash(self, worker):
        env = {
            "MLOPS_DEPLOYMENT_ID": "abc123",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2/",
        }
        with patch.dict(os.environ, env):
            url = worker._get_a2a_endpoint_url("http://localhost:8000/")
        assert url == "https://app.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/"

    def test_get_a2a_endpoint_url_deployment_missing_endpoint_raises(self, worker):
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": "abc123"}, clear=False):
            os.environ.pop("DATAROBOT_ENDPOINT", None)
            with pytest.raises(ValueError, match="DATAROBOT_ENDPOINT must be set"):
                worker._get_a2a_endpoint_url("http://localhost:8000/")

    async def test_add_routes_inherits_host_port_from_fastapi_config(
        self, dragent_worker, mock_builder, mock_a2a_worker, patch_super_add_routes
    ):
        app = FastAPI()
        with patch(
            "datarobot_genai.dragent.frontserver.A2AFrontEndPluginWorker",
            return_value=mock_a2a_worker,
        ) as mock_a2a_worker_cls:
            await dragent_worker.add_routes(app, mock_builder)

        a2a_config_used = mock_a2a_worker_cls.call_args[0][0].general.front_end
        assert a2a_config_used.host == dragent_worker.front_end_config.host
        assert a2a_config_used.port == dragent_worker.front_end_config.port

    @pytest.mark.asyncio
    async def test_add_routes_patches_agent_card_url(
        self, dragent_worker, mock_builder, mock_a2a_worker, patch_super_add_routes
    ):
        app = FastAPI()
        mock_a2a_worker.create_agent_card.return_value = MagicMock(url="http://localhost:8000/")
        with patch(
            "datarobot_genai.dragent.frontserver.A2AFrontEndPluginWorker",
            return_value=mock_a2a_worker,
        ):
            await dragent_worker.add_routes(app, mock_builder)
        assert mock_a2a_worker.create_agent_card.return_value.url == "http://localhost:8000/a2a/"

    @pytest.mark.asyncio
    async def test_add_routes_mounts_a2a(
        self, dragent_worker, mock_builder, mock_a2a_worker, patch_super_add_routes
    ):
        app = FastAPI()
        with patch(
            "datarobot_genai.dragent.frontserver.A2AFrontEndPluginWorker",
            return_value=mock_a2a_worker,
        ):
            await dragent_worker.add_routes(app, mock_builder)

        mock_a2a_worker.create_agent_card.assert_awaited_once()
        mock_a2a_worker.create_agent_executor.assert_called_once()
        mock_a2a_worker.create_a2a_server.assert_called_once()

    async def test_add_routes_disabled(self, mock_builder, patch_super_add_routes):
        """When expose_a2a_server_endpoints is False (default), A2A routes are not mounted."""
        config = Config(
            general=GeneralConfig(
                front_end=DRAgentFastApiFrontEndConfig(expose_a2a_server_endpoints=False)
            )
        )
        with patch.dict(os.environ, {"NAT_CONFIG_FILE": "unused"}):
            disabled_worker = DRAgentFastApiFrontEndPluginWorker(config)
        app = FastAPI()
        with patch(
            "datarobot_genai.dragent.frontserver.A2AFrontEndPluginWorker"
        ) as mock_a2a_worker_cls:
            await disabled_worker.add_routes(app, mock_builder)
            mock_a2a_worker_cls.assert_not_called()


class TestDRAgentFastApiFrontEndConfig:
    def test_is_fastapi_front_end_config(self):
        assert isinstance(DRAgentFastApiFrontEndConfig(), FastApiFrontEndConfig)

    def test_has_nested_a2a_config(self):
        config = DRAgentFastApiFrontEndConfig()
        assert isinstance(config.a2a, A2AFrontEndConfig)

    def test_custom_a2a_fields(self):
        config = DRAgentFastApiFrontEndConfig(
            a2a=A2AFrontEndConfig(
                name="My Agent",
                description="Does things",
                version="2.0.0",
            )
        )
        assert config.a2a.name == "My Agent"
        assert config.a2a.description == "Does things"
        assert config.a2a.version == "2.0.0"

    def test_is_not_a2a_front_end_config(self):
        config = DRAgentFastApiFrontEndConfig()
        assert not isinstance(config, A2AFrontEndConfig)

    def test_expose_a2a_server_endpoints_default_false(self):
        config = DRAgentFastApiFrontEndConfig()
        assert config.expose_a2a_server_endpoints is False

    def test_expose_a2a_server_endpoints_can_be_enabled(self):
        config = DRAgentFastApiFrontEndConfig(expose_a2a_server_endpoints=True)
        assert config.expose_a2a_server_endpoints is True


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
