# Copyright 2025 DataRobot, Inc. and its affiliates.
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

import logging
import os

from fastapi import FastAPI
from nat.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import SessionManager
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.plugins.a2a.server.front_end_plugin_worker import A2AFrontEndPluginWorker
from nat.runtime.loader import WorkflowBuilder
from pydantic import BaseModel
from pydantic import Field

from datarobot_genai.dragent.a2a_config import A2AConfig
from datarobot_genai.dragent.session import DRAgentAGUISessionManager
from datarobot_genai.dragent.step_adaptor import DRAgentNestedReasoningStepAdaptor

DATAROBOT_EXPECTED_HEALTH_ROUTES = ["/", "/ping", "/ping/", "/health", "/health/"]

logger = logging.getLogger(__name__)


class DRAgentFastApiFrontEndPluginWorker(FastApiFrontEndPluginWorker):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._a2a_worker: A2AFrontEndPluginWorker | None = None

    def get_step_adaptor(self) -> StepAdaptor:
        return DRAgentNestedReasoningStepAdaptor(self.front_end_config.step_adaptor)

    async def _create_session_manager(
        self, builder: WorkflowBuilder, entry_function: str | None = None
    ) -> SessionManager:
        """Create and register a SessionManager."""
        sm = await DRAgentAGUISessionManager.create(
            config=self._config, shared_builder=builder, entry_function=entry_function
        )
        self._session_managers.append(sm)

        return sm

    def _get_a2a_endpoint_url(self, default_url: str) -> str:
        """Construct the A2A endpoint URL.

        In a DataRobot deployment, uses the deployment's direct access URL.
        Otherwise, appends the /a2a/ mount path to the default base URL.
        """
        mlops_deployment_id = os.getenv("MLOPS_DEPLOYMENT_ID", "")
        if mlops_deployment_id:
            datarobot_endpoint = os.getenv("DATAROBOT_ENDPOINT", "")
            if not datarobot_endpoint:
                raise ValueError("DATAROBOT_ENDPOINT must be set when MLOPS_DEPLOYMENT_ID is set")
            return f"{datarobot_endpoint.rstrip('/')}/deployments/{mlops_deployment_id}/directAccess/a2a/"
        return default_url.rstrip("/") + "/a2a/"

    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder) -> None:
        await super().add_routes(app, builder)

        if not A2AConfig().expose_a2a_server_endpoints:
            logger.info("A2A server endpoints are disabled")
            return

        workflow = await builder.build()

        # A2AFrontEndPluginWorker reads config.general.front_end to get its front_end_config,
        # so we must pass it a full Config with the A2AFrontEndConfig substituted in.
        nat_config = self._config.model_copy(
            update={
                "general": self._config.general.model_copy(
                    update={"front_end": self.front_end_config.a2a}
                )
            }
        )
        self._a2a_worker = A2AFrontEndPluginWorker(nat_config)

        agent_card = await self._a2a_worker.create_agent_card(workflow)
        agent_card.url = self._get_a2a_endpoint_url(agent_card.url)
        agent_executor = self._a2a_worker.create_agent_executor(workflow, builder)
        a2a_server = self._a2a_worker.create_a2a_server(agent_card, agent_executor)

        a2a_app = a2a_server.build()
        app.mount("/a2a", a2a_app)
        app.add_event_handler("shutdown", self._cleanup_a2a_worker)

        logger.info("Mounted A2A server endpoints at /a2a")
        logger.info("The A2A agent card can be accessed at: /a2a/.well-known/agent-card.json")

    async def _cleanup_a2a_worker(self) -> None:
        """Clean up A2A worker resources on shutdown."""
        if self._a2a_worker is not None:
            await self._a2a_worker.cleanup()
            logger.info("A2A worker resources cleaned up")

    async def add_health_route(self, app: FastAPI) -> None:
        """Add a health check endpoint to the FastAPI app."""

        class HealthResponse(BaseModel):
            status: str = Field(description="Health status of the server")

        async def health_check() -> HealthResponse:
            """Health check endpoint for liveness/readiness probes."""
            return HealthResponse(status="healthy")

        for path in DATAROBOT_EXPECTED_HEALTH_ROUTES:
            app.add_api_route(
                path=path,
                endpoint=health_check,
                methods=["GET"],
                response_model=HealthResponse,
                description="Health check endpoint for liveness/readiness probes",
                tags=["Health"],
                responses={
                    200: {
                        "description": "Server is healthy",
                        "content": {"application/json": {"example": {"status": "healthy"}}},
                    }
                },
            )

            logger.info(f"Added health check endpoint at {path}")


class DRAgentFastApiFrontEndPlugin(FastApiFrontEndPlugin):
    def get_worker_class(self) -> type[FastApiFrontEndPluginWorker]:
        return DRAgentFastApiFrontEndPluginWorker
