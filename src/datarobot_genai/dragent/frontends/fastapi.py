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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from nat.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import SessionManager
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from nat.plugins.a2a.server.front_end_plugin_worker import A2AFrontEndPluginWorker
from nat.runtime.loader import WorkflowBuilder
from pydantic import BaseModel
from pydantic import Field

from datarobot_genai.core.utils.logging import setup_logging

from .a2a import A2A_MOUNT_PATH
from .a2a import OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE  # noqa: F401 (re-exported)
from .a2a import RFC8693_GRANT_TYPE_URI  # noqa: F401 (re-exported)
from .a2a import RFC8693_SECURITY_SCHEME_FLOW_REF  # noqa: F401 (re-exported)
from .a2a import RFC8693_SECURITY_SCHEME_REF  # noqa: F401 (re-exported)
from .a2a import RFC8693_TOKEN_EXCHANGE_EXTENSION_DESCRIPTION  # noqa: F401 (re-exported)
from .a2a import (
    PerUserCompatibleAgentExecutor as _PerUserCompatibleAgentExecutor,  # noqa: F401 (re-exported)
)
from .a2a import create_agent_card
from .session import DRAgentAGUISessionManager
from .step_adaptor import DRAgentNestedReasoningStepAdaptor

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

    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder) -> None:
        await super().add_routes(app, builder)
        if self.front_end_config.a2a:
            await self._add_a2a_routes(app, builder, self.front_end_config.a2a)

    async def _add_a2a_routes(
        self, app: FastAPI, builder: WorkflowBuilder, a2a_config: A2AFrontEndConfig
    ) -> None:
        # A2AFrontEndPluginWorker reads config.general.front_end to get its front_end_config.
        # Pass a full Config with the A2AFrontEndConfig substituted in, and inherit host/port
        # from the FastAPI front end so the agent card URL matches where the app is mounted.
        a2a_config = a2a_config.server.model_copy(
            update={"host": self.front_end_config.host, "port": self.front_end_config.port}
        )
        nat_config = self._config.model_copy(
            update={"general": self._config.general.model_copy(update={"front_end": a2a_config})}
        )
        self._a2a_worker = A2AFrontEndPluginWorker(nat_config)

        token_exchange = (
            self.front_end_config.a2a.oauth_token_exchange if self.front_end_config.a2a else None
        )
        skills = self.front_end_config.a2a.skills if self.front_end_config.a2a else []

        agent_card = await create_agent_card(
            frontend_config=self._a2a_worker.front_end_config,
            token_exchange=token_exchange,
            skills=skills,
        )
        session_manager = await DRAgentAGUISessionManager.create(
            config=self._config,
            shared_builder=builder,
            max_concurrency=self._a2a_worker.max_concurrency,
        )
        self._session_managers.append(session_manager)
        agent_executor = _PerUserCompatibleAgentExecutor(session_manager)

        a2a_server = self._a2a_worker.create_a2a_server(agent_card, agent_executor)
        a2a_app = a2a_server.build()

        app.mount(f"/{A2A_MOUNT_PATH}", a2a_app)

        logger.info(f"A2A endpoint URL: {agent_card.url}")
        logger.info(f"A2A agent card URL: {agent_card.url}.well-known/agent-card.json")

    def build_app(self) -> FastAPI:
        """Build the FastAPI app, wrapping the parent lifespan to clean up the A2A worker."""
        app = super().build_app()

        # Register DataRobot health routes (/, /ping, /ping/, /health, /health/).
        # NAT 1.6 no longer calls self.add_health_route() so we register here.
        self._register_health_routes(app)

        # app.router.lifespan_context is the lifespan set by the parent's build_app().
        # We wrap it to ensure the A2A worker's httpx client is closed on shutdown.
        # (app.add_event_handler("shutdown", ...) is silently ignored when a lifespan is set.)
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(lifespan_app: FastAPI) -> AsyncIterator[None]:
            async with parent_lifespan(lifespan_app):
                yield
            if self._a2a_worker is not None:
                await self._a2a_worker.cleanup()
                logger.info("A2A worker resources cleaned up")

        app.router.lifespan_context = lifespan

        setup_logging()
        return app

    def _register_health_routes(self, app: FastAPI) -> None:
        """Register DataRobot health check endpoints."""

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
