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

from fastapi import FastAPI
from fastapi.routing import APIRoute
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import SessionManager
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.runtime.loader import WorkflowBuilder
from pydantic import BaseModel
from pydantic import Field

from datarobot_genai.dragent.session import DRAgentAGUISessionManager
from datarobot_genai.dragent.step_adaptor import DRAgentNestedReasoningStepAdaptor

DATAROBOT_EXPECTED_HEALTH_ROUTES = ["/", "/ping", "/ping/", "/health", "/health/"]

logger = logging.getLogger(__name__)


DATAROBOT_CHAT_COMPLETIONS_PATH = "/chat/completions"
NAT_OPENAI_API_V1_PATH_ATTR = "openai_api_v1_path"


class DRAgentFastApiFrontEndPluginWorker(FastApiFrontEndPluginWorker):
    def get_step_adaptor(self) -> StepAdaptor:
        return DRAgentNestedReasoningStepAdaptor(self.front_end_config.step_adaptor)

    async def add_route(
        self,
        app: FastAPI,
        endpoint: FastApiFrontEndConfig.EndpointBase,
        session_manager: SessionManager,
    ) -> None:
        """Register routes via NAT, then also mount at /chat/completions for DataRobot deployments."""
        await super().add_route(app, endpoint, session_manager)

        # DataRobot's gateway strips the /{org_id}/{deployment_id} prefix and forwards
        # requests to /chat/completions (without the /v1 prefix that NAT uses by default).
        # Find the handler NAT registered at openai_api_v1_path and re-register it at
        # /chat/completions so DataRobot can reach it.
        v1_path = getattr(endpoint, NAT_OPENAI_API_V1_PATH_ATTR, None)
        if v1_path and v1_path != DATAROBOT_CHAT_COMPLETIONS_PATH:
            v1_route = next(
                (r for r in app.routes if isinstance(r, APIRoute) and r.path == v1_path),
                None,
            )
            if v1_route:
                app.add_api_route(
                    path=DATAROBOT_CHAT_COMPLETIONS_PATH,
                    endpoint=v1_route.endpoint,
                    methods=list(v1_route.methods or ["POST"]),
                    response_model=v1_route.response_model,
                    description=v1_route.description,
                    responses=v1_route.responses,
                )
                logger.info(
                    f"Added DataRobot chat/completions alias: {DATAROBOT_CHAT_COMPLETIONS_PATH} -> {v1_path}"
                )

    async def _create_session_manager(
        self, builder: WorkflowBuilder, entry_function: str | None = None
    ) -> SessionManager:
        """Create and register a SessionManager."""
        sm = await DRAgentAGUISessionManager.create(
            config=self._config, shared_builder=builder, entry_function=entry_function
        )
        self._session_managers.append(sm)

        return sm

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
