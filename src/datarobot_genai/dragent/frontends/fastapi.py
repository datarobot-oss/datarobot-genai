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
from a2a.types import AgentExtension
from a2a.types import AgentSkill
from a2a.types import AuthorizationCodeOAuthFlow
from a2a.types import ClientCredentialsOAuthFlow
from a2a.types import OAuth2SecurityScheme
from a2a.types import OAuthFlows
from a2a.types import SecurityScheme
from fastapi import FastAPI
from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
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

from .server_auth import OAuth2TokenExchangeConfig
from .session import DRAgentAGUISessionManager
from .step_adaptor import DRAgentNestedReasoningStepAdaptor

DATAROBOT_EXPECTED_HEALTH_ROUTES = ["/", "/ping", "/ping/", "/health", "/health/"]
A2A_MOUNT_PATH = "a2a"


OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE = (
    "OAuth 2.0 authorization utilizing RFC 8693 Token Exchange. Clients must "
    "supply a valid internal passport JWT as the subject token. Refer to the "
    "capabilities.extensions block for strict token exchange parameters and "
    "audience specifications."
)

# params.ref points at securitySchemes.oauth2 + clientCredentials for SDK binding.
RFC8693_GRANT_TYPE_URI = "urn:ietf:params:oauth:grant-type:token-exchange"
RFC8693_SECURITY_SCHEME_REF = "oauth2"  # The key in securitySchemes
RFC8693_SECURITY_SCHEME_FLOW_REF = "clientCredentials"  # The exact flow being overridden
RFC8693_TOKEN_EXCHANGE_EXTENSION_DESCRIPTION = (
    "Two-Step RFC 8693 Token Exchange execution parameters."
)


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
        token = None
        if context.context_id:
            token = self.session_manager._context_state.user_id.set(context.context_id)
        try:
            await super().execute(context, event_queue)
        finally:
            if token is not None:
                self.session_manager._context_state.user_id.reset(token)


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

    @staticmethod
    async def _oauth_flow_from_server_auth(
        server_auth: OAuth2ResourceServerConfig,
    ) -> tuple[AuthorizationCodeOAuthFlow, list[str]]:
        """Build the authorization_code OAuth2 flow and scopes for NAT ``server_auth``."""
        auth_url, token_url = await DRAgentFastApiFrontEndPluginWorker._resolve_oauth_endpoints(
            server_auth
        )
        flow = AuthorizationCodeOAuthFlow(
            authorization_url=auth_url,
            token_url=token_url,
            scopes={scope: f"Permission: {scope}" for scope in server_auth.scopes},
        )
        return flow, list(server_auth.scopes)

    @staticmethod
    def _oauth_flow_from_token_exchange(
        config: OAuth2TokenExchangeConfig,
    ) -> tuple[ClientCredentialsOAuthFlow, list[str]]:
        """Build the client_credentials flow and scopes for ``a2a.oauth_token_exchange``.

        Token URL and scopes live only here (OpenAPI-compatible). RFC 8693 second-phase
        requirements are signaled separately via :meth:`_token_exchange_capability_extension`.
        """
        flow = ClientCredentialsOAuthFlow(
            token_url=config.token_url,
            scopes={scope: f"Permission: {scope}" for scope in config.scopes},
        )
        return flow, list(config.scopes)

    @staticmethod
    def _token_exchange_capability_extension(
        config: OAuth2TokenExchangeConfig,
    ) -> list[AgentExtension]:
        """Build the RFC 8693 extension for the agent card (two-step flow).

        OpenAPI ``token_url`` / ``scopes`` remain on ``securitySchemes.oauth2.flows``.
        ``params`` carries Step 1 (passport) and Step 2 (Okta exchange) only, plus a
        ``ref`` to the client-credentials flow for unambiguous SDK binding.
        """
        params = {
            "ref": {
                "scheme": RFC8693_SECURITY_SCHEME_REF,
                "flow": RFC8693_SECURITY_SCHEME_FLOW_REF,
            },
            "passport_requirement": config.passport_requirement.model_dump(),
            "exchange_payload": config.exchange_payload.model_dump(),
        }
        return [
            AgentExtension(
                uri=RFC8693_GRANT_TYPE_URI,
                description=RFC8693_TOKEN_EXCHANGE_EXTENSION_DESCRIPTION,
                params=params,
            )
        ]

    async def _build_security_schemes(
        self, frontend_config: A2AFrontEndConfig
    ) -> tuple[
        dict[str, SecurityScheme] | None,
        list[dict[str, list[str]]] | None,
        list[AgentExtension] | None,
    ]:
        """Build A2A security schemes from the frontend configuration.

        Supports two independent auth sources that are merged into a single
        ``oauth2`` security scheme with separate flows:

        * ``server_auth`` (OAuth2ResourceServerConfig) → authorization_code flow.
          Endpoint URLs are resolved via OIDC discovery or derived from the issuer URL.
        * ``a2a.oauth_token_exchange`` (OAuth2TokenExchangeConfig) → client_credentials flow.
          Used for RFC 8693 second-phase token acquisition. Configured on the DataRobot
          ``a2a`` block, not on NAT's A2A frontend model, so it stays stable if NAT's
          config types change.

        Returns
        -------
            Tuple of (security_schemes, security requirements, capability extensions),
            all ``None`` when neither auth source is configured.
        """
        server_auth = frontend_config.server_auth
        a2a_cfg = self.front_end_config.a2a
        token_exchange = a2a_cfg.oauth_token_exchange if a2a_cfg else None

        if not server_auth and not token_exchange:
            return None, None, None

        auth_code_flow, server_auth_scopes = (
            await self._oauth_flow_from_server_auth(server_auth) if server_auth else (None, [])
        )
        client_creds_flow, token_exchange_scopes = (
            self._oauth_flow_from_token_exchange(token_exchange) if token_exchange else (None, [])
        )
        extensions = (
            self._token_exchange_capability_extension(token_exchange) if token_exchange else None
        )

        all_scopes = list(dict.fromkeys(server_auth_scopes + token_exchange_scopes))
        security_schemes = {
            "oauth2": SecurityScheme(
                root=OAuth2SecurityScheme(
                    type="oauth2",
                    description=OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE,
                    flows=OAuthFlows(
                        authorization_code=auth_code_flow,
                        client_credentials=client_creds_flow,
                    ),
                )
            )
        }
        return security_schemes, [{"oauth2": all_scopes}], extensions

    @staticmethod
    async def _resolve_oauth_endpoints(
        server_auth_config: OAuth2ResourceServerConfig,
    ) -> tuple[str, str]:
        """Resolve authorization and token URLs from OAuth2ResourceServerConfig.

        Uses OIDC discovery when available, otherwise derives URLs from issuer_url.
        """
        import httpx

        if server_auth_config.discovery_url:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(server_auth_config.discovery_url, timeout=5.0)
                    response.raise_for_status()
                    metadata = response.json()
                    auth_url = metadata.get("authorization_endpoint")
                    token_url = metadata.get("token_endpoint")
                    if auth_url and token_url:
                        logger.info(
                            "Resolved OAuth endpoints via discovery: %s",
                            server_auth_config.discovery_url,
                        )
                        return auth_url, token_url
            except Exception as e:
                logger.warning("Failed to discover OAuth endpoints: %s", e)

        issuer = server_auth_config.issuer_url.rstrip("/")
        auth_url = f"{issuer}/oauth/authorize"
        token_url = f"{issuer}/oauth/token"
        logger.info("Using derived OAuth endpoints from issuer: %s", issuer)
        return auth_url, token_url

    async def _create_agent_card(self, frontend_config: A2AFrontEndConfig) -> AgentCard:
        security_schemes, security, extensions = await self._build_security_schemes(frontend_config)

        if self.front_end_config.a2a.skills:
            skills = self.front_end_config.a2a.skills
        else:
            skills = [
                AgentSkill(
                    id="call",
                    name=frontend_config.name,
                    description=frontend_config.description,
                    tags=[],
                    examples=[],
                )
            ]
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
                extensions=extensions,
            ),
            skills=skills,
            security_schemes=security_schemes or None,
            security=security or None,
        )
        return agent_card

    def _get_a2a_endpoint_url(self, frontend_config: A2AFrontEndConfig) -> str:
        """Construct the A2A endpoint URL.

        In a DataRobot deployment, uses the deployment's direct access URL.
        Otherwise, appends the /a2a/ mount path to the default base URL.
        """
        mlops_deployment_id = os.getenv("MLOPS_DEPLOYMENT_ID", "")
        if mlops_deployment_id:
            # Prefer DATAROBOT_PUBLIC_API_ENDPOINT over DATAROBOT_ENDPOINT because on-prem
            # deployments often set DATAROBOT_ENDPOINT to an internal k8s URL, while
            # DATAROBOT_PUBLIC_API_ENDPOINT holds the externally reachable URL needed here
            # to construct a publicly accessible agent-card URL.
            datarobot_endpoint = os.getenv("DATAROBOT_PUBLIC_API_ENDPOINT") or os.getenv(
                "DATAROBOT_ENDPOINT", ""
            )
            if not datarobot_endpoint:
                raise ValueError(
                    "DATAROBOT_PUBLIC_API_ENDPOINT or DATAROBOT_ENDPOINT must be set "
                    "when MLOPS_DEPLOYMENT_ID is set"
                )
            base = datarobot_endpoint.rstrip("/")
            return f"{base}/deployments/{mlops_deployment_id}/directAccess/{A2A_MOUNT_PATH}/"
        return f"http://{frontend_config.host}:{frontend_config.port}/{A2A_MOUNT_PATH}/"

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

        agent_card = await self._create_agent_card(self._a2a_worker.front_end_config)
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
