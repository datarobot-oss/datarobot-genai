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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentSkill
from fastapi import FastAPI
from nat.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import SessionManager
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.plugins.a2a.server.agent_executor_adapter import NATWorkflowAgentExecutor
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from nat.plugins.a2a.server.front_end_plugin_worker import A2AFrontEndPluginWorker
from nat.runtime.loader import WorkflowBuilder
from pydantic import BaseModel
from pydantic import Field

from datarobot_genai.dragent.session import DRAgentAGUISessionManager
from datarobot_genai.dragent.step_adaptor import DRAgentNestedReasoningStepAdaptor

DATAROBOT_EXPECTED_HEALTH_ROUTES = ["/", "/ping", "/ping/", "/health", "/health/"]
A2A_MOUNT_PATH = "a2a"

logger = logging.getLogger(__name__)


class _PerUserCompatibleAgentExecutor(NATWorkflowAgentExecutor):
    """Subclass of NATWorkflowAgentExecutor that supports per-user workflows.

    Two problems with the parent class for per-user workflows:

    1. ``__init__`` accesses ``session_manager.workflow`` which raises ``ValueError``
       for per-user workflows.  We bypass it and log via ``config.workflow.type`` instead.

    2. ``execute`` calls ``self.session_manager.session()`` with no ``user_id``, which
       raises ``ValueError`` for per-user workflows.  We override it to pass the A2A
       ``context_id`` as the ``user_id``, giving each conversation its own isolated
       per-user workflow instance.
    """

    def __init__(self, session_manager: SessionManager) -> None:
        # Bypass parent __init__ to avoid session_manager.workflow access,
        # which raises ValueError for per-user workflows. Log via config instead.
        self.session_manager = session_manager
        logger.info(
            "Initialized NATWorkflowAgentExecutor (message-only) for workflow: %s",
            session_manager.config.workflow.type,
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[override]
        # Inject the A2A context_id as user_id before delegating to the parent execute.
        # The parent calls self.session_manager.session() with no user_id, which raises
        # ValueError for per-user workflows.  Setting the context var here means the
        # SessionManager's _get_user_id_from_context() will find it automatically.
        if context.context_id:
            self.session_manager._context_state.user_id.set(context.context_id)
        await super().execute(context, event_queue)


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

    async def _create_agent_card(self, frontend_config: A2AFrontEndConfig) -> AgentCard:
        assert self._a2a_worker is not None
        security_schemes = None
        security = None
        if frontend_config.server_auth:
            security_schemes, security = await self._a2a_worker._generate_security_schemes(
                frontend_config.server_auth
            )
        agent_card = AgentCard(
            name=frontend_config.name,
            description=frontend_config.description,
            url=self._get_a2a_endpoint_url(frontend_config),
            version=frontend_config.version,
            default_input_modes=frontend_config.default_input_modes,
            default_output_modes=frontend_config.default_output_modes,
            capabilities=AgentCapabilities(
                streaming=frontend_config.capabilities.streaming,
                push_notifications=frontend_config.capabilities.push_notifications,
            ),
            skills=list(self.front_end_config.a2a.skills)
            or [
                AgentSkill(
                    id="call",
                    name=frontend_config.name,
                    description=frontend_config.description,
                    tags=[],
                    examples=[],
                )
            ],
            security_schemes=security_schemes,
            security=security,
        )
        return agent_card

    def _get_a2a_endpoint_url(self, frontend_config: A2AFrontEndConfig) -> str:
        """Construct the A2A endpoint URL.

        In a DataRobot deployment, uses the deployment's direct access URL.
        Otherwise, appends the /a2a/ mount path to the default base URL.
        """
        mlops_deployment_id = os.getenv("MLOPS_DEPLOYMENT_ID", "")
        if mlops_deployment_id:
            datarobot_endpoint = os.getenv("DATAROBOT_ENDPOINT", "")
            if not datarobot_endpoint:
                raise ValueError("DATAROBOT_ENDPOINT must be set when MLOPS_DEPLOYMENT_ID is set")
            base = datarobot_endpoint.rstrip("/")
            return f"{base}/deployments/{mlops_deployment_id}/directAccess/{A2A_MOUNT_PATH}/"
        return f"http://{frontend_config.host}:{frontend_config.port}/{A2A_MOUNT_PATH}/"

    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder) -> None:
        await super().add_routes(app, builder)

        if self.front_end_config.a2a is None:
            logger.info("A2A server endpoints are disabled")
            return

        # A2AFrontEndPluginWorker reads config.general.front_end to get its front_end_config.
        # We must pass it a full Config with the A2AFrontEndConfig substituted in.
        # We also inherit host/port from the FastAPI config so the agent card URL is gets mounted
        # under the correct endpoint.
        a2a_config = self.front_end_config.a2a.server.model_copy(
            update={"host": self.front_end_config.host, "port": self.front_end_config.port}
        )
        nat_config = self._config.model_copy(
            update={"general": self._config.general.model_copy(update={"front_end": a2a_config})}
        )
        self._a2a_worker = A2AFrontEndPluginWorker(nat_config)

        agent_card = await self._create_agent_card(self._a2a_worker.front_end_config)
        session_manager = await SessionManager.create(
            config=self._config,
            shared_builder=builder,
            max_concurrency=self._a2a_worker.max_concurrency,
        )
        agent_executor = _PerUserCompatibleAgentExecutor(session_manager)

        a2a_server = self._a2a_worker.create_a2a_server(agent_card, agent_executor)
        a2a_app = a2a_server.build()

        app.mount(f"/{A2A_MOUNT_PATH}", a2a_app)

        logger.info(f"A2A endpoint URL: {agent_card.url}")
        logger.info(f"A2A agent card URL: {agent_card.url}.well-known/agent-card.json")

    def build_app(self) -> FastAPI:
        """Build the FastAPI app, wrapping the parent lifespan to clean up the A2A worker."""
        app = super().build_app()

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
        return app

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
