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
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig

from datarobot_genai.dragent.frontserver import DRAgentFastApiFrontEndPluginWorker


@pytest.fixture
def worker():
    config = Config(general=GeneralConfig())
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


class TestDRAgentFastApiFrontEndPluginWorker:
    EXPECTED_HEALTH_ROUTES = ["/", "/ping", "/ping/", "/health", "/health/"]

    def test_health_routes_return_healthy_status(self, app_with_health):
        with TestClient(app_with_health) as client:
            for path in self.EXPECTED_HEALTH_ROUTES:
                response = client.get(path)
                assert response.status_code == 200, f"Expected 200 at {path}"
                assert response.json() == {"status": "healthy"}, f"Unexpected response at {path}"
