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

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import InvalidParamsError
from a2a.utils.errors import ServerError
from fastapi import FastAPI
from nat.data_models.user_info import UserInfo
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

from datarobot_genai.core.utils.logging import setup_logging

from .a2a import A2A_MOUNT_PATH
from .a2a import create_agent_card
from .session import DRAgentAGUISessionManager
from .session import _a2a_headers
from .session import _auth_handler
from .step_adaptor import DRAgentNestedReasoningStepAdaptor

DATAROBOT_EXPECTED_HEALTH_ROUTES = ["/", "/ping", "/ping/", "/health", "/health/"]

logger = logging.getLogger(__name__)


_AUTH_CONTEXT_HEADER = "x-datarobot-authorization-context"
_GATEWAY_USER_ID_HEADER = "x-datarobot-user-id"
_INVALID_AUTH_CONTEXT_MSG = (
    "X-DataRobot-Authorization-Context header is present but invalid or expired"
)


def _resolve_identity_from_headers(headers: dict[str, str] | None) -> str | None:
    """Extract gateway-validated user identity from A2A-forwarded headers.

    Resolution order (first match wins):

    1. ``X-DataRobot-Authorization-Context`` -- signed JWT forwarded by
       components in the agent application template.  Decoded via
       :data:`_auth_handler` and hashed through
       ``UserInfo._from_session_cookie`` to produce the same UUID5 workflow
       key as the AG-UI path.  When this header is present but validation
       fails, raises :class:`~a2a.utils.errors.ServerError` with
       :class:`~a2a.types.InvalidParamsError` (no fall-through to other headers
       or ``context_id``).
    2. ``X-DataRobot-User-Id`` -- raw DataRobot user ID injected by the API
       gateway, tied to the API-key owner.  Used only when the auth-context
       header is absent.  Same ``_from_session_cookie`` transform is applied
       for key-format consistency.
    3. ``None`` -- no gateway-provided identity (local dev).

    Returns ``None`` when *headers* are absent or contain no recognised
    identity header.
    """
    if not headers:
        return None

    if _AUTH_CONTEXT_HEADER in headers:
        try:
            auth_ctx = _auth_handler.get_context(headers)
        except Exception:
            logger.warning("Failed to decode auth-context header", exc_info=True)
            auth_ctx = None
        if auth_ctx is None:
            raise ServerError(error=InvalidParamsError(message=_INVALID_AUTH_CONTEXT_MSG))
        return UserInfo._from_session_cookie(auth_ctx.user.id).get_user_id()

    raw_user_id = headers.get(_GATEWAY_USER_ID_HEADER)
    if raw_user_id:
        return UserInfo._from_session_cookie(raw_user_id).get_user_id()

    return None


class _PerUserCompatibleAgentExecutor(NATWorkflowAgentExecutor):
    """Subclass of NATWorkflowAgentExecutor that supports per-user workflows.

    Three problems with the parent class for per-user workflows:

    1. ``__init__`` accesses ``session_manager.workflow`` which raises ``ValueError``
       for per-user workflows.  We bypass it and log via ``config.workflow.type`` instead.

    2. ``execute`` calls ``self.session_manager.session()`` with no ``user_id``. NAT 1.6+
       would overwrite a preset ``ContextState.user_id`` with ``None``. We resolve the
       gateway-validated identity from forwarded A2A headers and set *that* on the context
       var before delegating; :class:`DRAgentAGUISessionManager` merges it into the
       ``user_id`` argument so each user gets their own per-user workflow instance.
       When no authenticated identity is available (local dev), we fall back to the A2A
       ``context_id``.

    3. ``execute`` does not forward the incoming A2A HTTP request headers into the NAT
       context.  We forward all headers from the A2A call context so that auth
       providers can read whichever headers they need (e.g. ``x-datarobot-*``,
       ``Authorization``) via ``Context.get().metadata.headers``.
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
        # Forward incoming A2A HTTP headers so DRAgentAGUISessionManager.session()
        # can inject them into NAT context metadata.  Auth providers pick the specific
        # headers they need (e.g. x-datarobot-external-access-token, Authorization).
        # Extracted first because identity resolution reads these headers.
        normalised_headers: dict[str, str] | None = None
        token_headers = None
        if context.call_context and isinstance(context.call_context.state, dict):
            raw_headers = context.call_context.state.get("headers")
            if raw_headers and isinstance(raw_headers, dict):
                normalised_headers = {k.lower(): v for k, v in raw_headers.items()}
                token_headers = _a2a_headers.set(normalised_headers)

        # Identity resolution must happen *before* super().execute() so that a
        # ServerError(InvalidParamsError) propagates directly.  The parent's
        # execute() has a catch-all that re-wraps exceptions as InternalError.
        token = None
        try:
            workflow_key = _resolve_identity_from_headers(normalised_headers)
            if workflow_key is None and context.context_id:
                workflow_key = UserInfo._from_session_cookie(context.context_id).get_user_id()
                logger.warning(
                    "No authenticated identity in A2A headers; falling back to context_id "
                    "for per-user workflow key. This is expected in local dev but should not "
                    "occur in production behind the DataRobot gateway."
                )

            if workflow_key:
                token = self.session_manager._context_state.user_id.set(workflow_key)

            await super().execute(context, event_queue)
        finally:
            if token is not None:
                self.session_manager._context_state.user_id.reset(token)
            if token_headers is not None:
                _a2a_headers.reset(token_headers)


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

        cross_app_access = (
            self.front_end_config.a2a.cross_application_access
            if self.front_end_config.a2a
            else None
        )
        skills = self.front_end_config.a2a.skills if self.front_end_config.a2a else []
        external = self.front_end_config.a2a.external if self.front_end_config.a2a else None

        agent_card = await create_agent_card(
            frontend_config=self._a2a_worker.front_end_config,
            cross_app_access=cross_app_access,
            skills=skills,
            external=external,
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
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # NAT's FastApiFrontEndPlugin.run() finally-block accesses self._dask_client
        # directly, but that attribute is only set lazily by the dask_client property.
        # When dask isn't installed it is never set, so shutdown raises AttributeError.
        self._dask_client = None

    async def run(self) -> None:
        # Resolve ``workflow.yaml`` before NAT builds the app (gunicorn calls ``get_app()`` in
        # the parent process, which initializes middleware including datarobot_moderation).
        from datarobot_genai.dragent.workflow_paths import publish_dragent_config_file_env

        publish_dragent_config_file_env()
        await super().run()

    def get_worker_class(self) -> type[FastApiFrontEndPluginWorker]:
        return DRAgentFastApiFrontEndPluginWorker
