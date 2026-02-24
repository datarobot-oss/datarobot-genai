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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from datarobot_genai.dragent.frontserver import DRAgentFastApiFrontEndPluginWorker
from datarobot_genai.dragent.session import DRAgentAGUISessionManager
from datarobot_genai.dragent.step_adaptor import DRAgentNestedReasoningStepAdaptor


class TestDRAgentFastApiFrontEndPluginWorker:
    @pytest.mark.asyncio
    async def test_add_health_route_registers_all_paths(self, worker):
        app = FastAPI()
        await worker.add_health_route(app)

        routes = {route.path for route in app.routes}
        for path in ['/', '/ping', '/ping/', '/health', '/health/']:
            assert path in routes, f"Expected health route at {path}"

    @pytest.mark.asyncio
    async def test_add_health_route_returns_healthy_status(self, worker):
        app = FastAPI()
        await worker.add_health_route(app)

        client = TestClient(app)
        for path in ['/', '/ping', '/ping/', '/health', '/health/']:
            response = client.get(path)
            assert response.status_code == 200, f"Expected 200 at {path}"
            assert response.json() == {"status": "healthy"}, f"Unexpected response at {path}"
