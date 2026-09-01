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
import sys
import types
from contextlib import asynccontextmanager
from contextlib import contextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from a2a.types import InvalidParamsError
from a2a.utils.errors import ServerError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from pydantic import ValidationError

from datarobot_genai.dragent.cross_app_access_config import CrossApplicationAccessConfig
from datarobot_genai.dragent.cross_app_access_config import CrossAppTokenExchange
from datarobot_genai.dragent.cross_app_access_config import CrossAppTokenRequest
from datarobot_genai.dragent.frontends.claim_validation import GeneralOAuthClaimValidationMiddleware
from datarobot_genai.dragent.frontends.fastapi import DATAROBOT_EXPECTED_HEALTH_ROUTES
from datarobot_genai.dragent.frontends.fastapi import DRAgentFastApiFrontEndPlugin
from datarobot_genai.dragent.frontends.fastapi import DRAgentFastApiFrontEndPluginWorker
from datarobot_genai.dragent.frontends.fastapi import _GunicornSettings
from datarobot_genai.dragent.frontends.fastapi import DATAROBOT_MODEL_MONITORING_HEADER
from datarobot_genai.dragent.frontends.fastapi import _patch_gunicorn_worker_timeout
from datarobot_genai.dragent.frontends.fastapi import _PerUserCompatibleAgentExecutor
from datarobot_genai.dragent.frontends.register import DRAgentA2AConfig
from datarobot_genai.dragent.frontends.register import DRAgentA2AExternalConfig
from datarobot_genai.dragent.frontends.register import DRAgentFastApiFrontEndConfig
from datarobot_genai.dragent.frontends.step_adaptor import DRAgentNestedReasoningStepAdaptor

from ..helpers import make_jwt
from .helpers import AUTH_HANDLER_PATH
from .helpers import expected_workflow_key
from .helpers import make_auth_ctx


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


def test_build_app_instruments_fastapi(worker):
    """build_app wires OTel FastAPI instrumentation onto the served app."""

    @asynccontextmanager
    async def mock_from_config(_config):
        yield MagicMock()

    with (
        patch.object(worker, "configure", new_callable=AsyncMock),
        patch.object(WorkflowBuilder, "from_config", side_effect=mock_from_config),
        patch(
            "datarobot_genai.dragent.frontends.fastapi._instrument_fastapi_app"
        ) as mock_instrument,
    ):
        app = worker.build_app()
    mock_instrument.assert_called_once_with(app)


def test_instrument_fastapi_app_excludes_streaming_and_probe_spans():
    """_instrument_fastapi_app drops per-SSE-chunk send spans and health probes."""
    pytest.importorskip("opentelemetry.instrumentation.fastapi")
    from datarobot_genai.dragent.frontends.fastapi import _instrument_fastapi_app

    app = FastAPI()
    with patch(
        "opentelemetry.instrumentation.fastapi.FastAPIInstrumentor.instrument_app"
    ) as mock_instr:
        _instrument_fastapi_app(app)
    mock_instr.assert_called_once()
    kwargs = mock_instr.call_args.kwargs
    assert kwargs["exclude_spans"] == ["receive", "send"]

    from opentelemetry.util.http import parse_excluded_urls

    excluded = parse_excluded_urls(kwargs["excluded_urls"])
    dep, model = "6a6a20b7fb870c8f3ea97011", "6a6a207a102de64dbe013214"
    # probes are dropped: bare root, mount-prefixed root, health, ping
    for url in ("http://h/", f"http://h/{dep}/{model}/", "http://h/health", "http://h/ping"):
        assert excluded.url_disabled(url), url
    # named endpoints keep their server span
    for url in (
        "http://h/a2a/",
        f"http://h/{dep}/{model}/a2a/",
        "http://h/v1/chat/completions",
        f"http://h/{dep}/{model}/chat/completions",
    ):
        assert not excluded.url_disabled(url), url


class TestDRAgentFastApiFrontEndPluginWorker:
    @pytest.mark.parametrize("path", DATAROBOT_EXPECTED_HEALTH_ROUTES)
    def test_health_routes_return_healthy_status(self, app_with_health, path):
        with TestClient(app_with_health) as client:
            response = client.get(path)
            assert response.status_code == 200, f"Expected 200 at {path}"
            assert response.json() == {"status": "healthy"}, f"Unexpected response at {path}"

    def test_agent_manifest_route_returns_declared_structure(self, app_with_health):
        """Served alongside the health routes by the same build_app() call - proves
        the route is actually wired up, not just that build_agent_manifest() works
        in isolation (covered separately in test_agent_manifest.py).
        """
        with TestClient(app_with_health) as client:
            response = client.get("/.well-known/agent-manifest.json")

        assert response.status_code == 200
        body = response.json()
        assert body["components"] == []
        # `worker`'s Config has no workflow= set, so it defaults to NAT's
        # EmptyFunctionConfig - same empty-config shape asserted directly
        # against build_agent_manifest() in test_agent_manifest.py.
        assert body["root_agent"]["type"] == "EmptyFunctionConfig"

    def test_step_adaptor(self, worker):
        assert isinstance(worker.get_step_adaptor(), DRAgentNestedReasoningStepAdaptor)

    def test_model_monitoring_header_disabled_by_default(self, app_with_health):
        """MODEL_MONITORING_HEADER_ENABLED defaults off; no unwanted header.

        See the module-level comment on ``DATAROBOT_MODEL_MONITORING_HEADER`` in fastapi.py.
        """
        with TestClient(app_with_health) as client:
            response = client.get("/health")
            assert DATAROBOT_MODEL_MONITORING_HEADER not in response.headers

    def test_model_monitoring_header_enabled_via_env_var(self, worker):
        """Setting MODEL_MONITORING_HEADER_ENABLED=true adds the header to the
        OpenAI-compatible chat-completions routes, but not to unrelated routes like health.

        See the module-level comment on ``DATAROBOT_MODEL_MONITORING_HEADER`` in fastapi.py.
        """

        @asynccontextmanager
        async def mock_from_config(_config):
            yield MagicMock()

        with (
            patch.dict(os.environ, {"MODEL_MONITORING_HEADER_ENABLED": "true"}),
            patch.object(worker, "configure", new_callable=AsyncMock),
            patch.object(WorkflowBuilder, "from_config", side_effect=mock_from_config),
        ):
            app = worker.build_app()
            with TestClient(app) as client:
                # No chat route is actually mounted (configure() is mocked out), but the
                # middleware matches on path alone, so the 404 response still carries it.
                chat_response = client.get("/chat")
                assert (
                    chat_response.headers[DATAROBOT_MODEL_MONITORING_HEADER]
                    == "true"
                )

                health_response = client.get("/health")
                assert DATAROBOT_MODEL_MONITORING_HEADER not in health_response.headers

    def test_chat_completion_paths_built_from_config(self, worker):
        """_chat_completion_paths() derives paths from front_end_config, not hardcoded strings."""
        paths = worker._chat_completion_paths()
        assert paths == {
            "/v1/chat/completions",
            "/v1/chat",
            "/v1/chat/stream",
            "/chat",
            "/chat/stream",
        }

    def test_chat_completion_paths_excludes_legacy_when_disabled(self):
        config = Config(
            general=GeneralConfig(
                front_end=DRAgentFastApiFrontEndConfig(disable_legacy_routes=True),
            )
        )
        with patch.dict(os.environ, {"NAT_CONFIG_FILE": "unused"}):
            worker = DRAgentFastApiFrontEndPluginWorker(config)
        paths = worker._chat_completion_paths()
        assert paths == {"/chat/completions"}

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
                "datarobot_genai.dragent.frontends.fastapi.DRAgentA2AFrontEndPluginWorker",
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
                "datarobot_genai.dragent.frontends.fastapi.DRAgentA2AFrontEndPluginWorker",
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
                "datarobot_genai.dragent.frontends.fastapi.DRAgentA2AFrontEndPluginWorker",
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
                "datarobot_genai.dragent.frontends.fastapi.DRAgentA2AFrontEndPluginWorker",
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
            "datarobot_genai.dragent.frontends.fastapi.DRAgentA2AFrontEndPluginWorker"
        ) as mock_a2a_worker_cls:
            await disabled_worker.add_routes(app, mock_builder)
            mock_a2a_worker_cls.assert_not_called()


class TestInboundAudienceValidation:
    """Wiring of the L2 audience check onto both the A2A app and the serving app."""

    @staticmethod
    def _worker(
        audience: str | None, *, opted_in: bool = True, with_xaa: bool | None = None
    ) -> DRAgentFastApiFrontEndPluginWorker:
        """Build a worker whose a2a config advertises XAA with the given audience.

        ``with_xaa`` defaults to ``audience is not None``, so ``_worker(None)`` means no
        ``cross_application_access`` block at all; pass it explicitly for the block-present
        but audience-unset case.  ``opted_in`` drives ``a2a.oauth_claim_validation``.
        """
        cross_app = None
        if with_xaa if with_xaa is not None else audience is not None:
            cross_app = CrossApplicationAccessConfig(
                token_exchange=CrossAppTokenExchange(
                    trusted_issuer="https://your-org.okta.com",
                    audience="https://your-org.okta.com/oauth2/ausXXX",
                ),
                token_request=CrossAppTokenRequest(
                    token_url="https://your-org.okta.com/oauth2/ausXXX/v1/token",
                    audience=audience,
                ),
            )
        config = Config(
            general=GeneralConfig(
                front_end=DRAgentFastApiFrontEndConfig(
                    a2a=DRAgentA2AConfig(
                        server=A2AFrontEndConfig(name="Test Agent", description="A test agent"),
                        oauth_claim_validation=opted_in,
                        cross_application_access=cross_app,
                    ),
                )
            )
        )
        with patch.dict(os.environ, {"NAT_CONFIG_FILE": "unused"}):
            return DRAgentFastApiFrontEndPluginWorker(config)

    def test_not_installed_without_the_opt_in(self):
        """GIVEN an audience but no oauth_claim_validation THEN nothing is enforced.

        Opt-in means opt-in: filling in the XAA audience must not start enforcing on its own.
        """
        with self._built_app(self._worker("api://my-agent", opted_in=False)) as app:
            assert self._claim_middleware(app) == []

    def test_opt_in_without_an_audience_fails_loudly(self):
        """GIVEN the opt-in but nothing to enforce THEN startup fails.

        Silently running an agent that believes it validates is the worst outcome.
        """
        for worker in (
            self._worker(None, opted_in=True, with_xaa=True),  # block present, audience unset
            self._worker(None, opted_in=True, with_xaa=False),  # no block at all
        ):
            with pytest.raises(ValueError, match="oauth_claim_validation is true"):
                with self._built_app(worker):
                    pass

    def test_flag_defaults_off(self):
        """An a2a block that says nothing about it does not enforce."""
        config = DRAgentA2AConfig(server=A2AFrontEndConfig())
        assert config.oauth_claim_validation is False

    def test_resolve_expected_audience_from_cross_app_access(self):
        """GIVEN token_request.audience is set THEN it is the expected inbound audience."""
        worker = self._worker("api://my-agent")
        assert worker._resolve_expected_audience() == "api://my-agent"

    def test_resolve_expected_audience_none_without_cross_app_access(self):
        """GIVEN no cross_application_access block THEN no audience is expected."""
        assert self._worker(None)._resolve_expected_audience() is None

    def test_resolve_expected_audience_none_when_audience_unset(self):
        """GIVEN XAA configured but token_request.audience unset THEN no audience is expected."""
        worker = self._worker("api://my-agent")
        worker.front_end_config.a2a.cross_application_access.token_request.audience = None
        assert worker._resolve_expected_audience() is None

    def test_resolve_expected_audience_none_without_a2a(self):
        """GIVEN a2a is not configured at all THEN no audience is expected."""
        config = Config(general=GeneralConfig(front_end=DRAgentFastApiFrontEndConfig()))
        with patch.dict(os.environ, {"NAT_CONFIG_FILE": "unused"}):
            worker = DRAgentFastApiFrontEndPluginWorker(config)
        assert worker._resolve_expected_audience() is None

    async def _add_routes(self, worker, mock_builder, mock_a2a_worker) -> FastAPI:
        app = FastAPI()
        with (
            patch(
                "datarobot_genai.dragent.frontends.fastapi.DRAgentA2AFrontEndPluginWorker",
                return_value=mock_a2a_worker,
            ),
            patch(
                "datarobot_genai.dragent.frontends.fastapi.SessionManager.create",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
        ):
            await worker.add_routes(app, mock_builder)
        return mock_a2a_worker.create_a2a_server.return_value.build.return_value

    async def test_a2a_app_is_not_separately_guarded(
        self, mock_builder, mock_a2a_worker, patch_super_add_routes
    ):
        """GIVEN an expected audience THEN the mounted A2A app gets no instance of its own.

        The served app's instance already covers ``/a2a``; a second would decode twice.
        """
        worker = self._worker("api://my-agent")
        a2a_app = await self._add_routes(worker, mock_builder, mock_a2a_worker)
        assert self._claim_middleware(a2a_app) == []

    @staticmethod
    def _claim_middleware(app: FastAPI) -> list:
        return [
            middleware
            for middleware in app.user_middleware
            if middleware.cls is GeneralOAuthClaimValidationMiddleware
        ]

    @staticmethod
    @contextmanager
    def _built_app(worker: DRAgentFastApiFrontEndPluginWorker):
        """Build the served app the way the server does, stubbing workflow construction.

        Yields inside the patches because the lifespan runs on ``TestClient.__enter__``,
        not at ``build_app()``.
        """

        @asynccontextmanager
        async def mock_from_config(_config):
            yield MagicMock()

        with (
            patch.object(worker, "configure", new_callable=AsyncMock),
            patch.object(WorkflowBuilder, "from_config", side_effect=mock_from_config),
        ):
            yield worker.build_app()

    def test_build_app_installs_one_instance_when_audience_configured(self):
        """GIVEN an expected audience THEN exactly one instance guards the served app.

        Also covers why this lives in build_app: add_routes runs inside the lifespan, where
        Starlette has frozen the middleware stack.
        """
        with self._built_app(self._worker("api://my-agent")) as app:
            installed = self._claim_middleware(app)
        assert len(installed) == 1
        assert installed[0].kwargs["expected_audience"] == "api://my-agent"

    def test_build_app_installs_nothing_by_default(self):
        """GIVEN neither the opt-in nor an audience THEN no route is guarded.

        The default state for an agent that has not asked for L2 checks.
        """
        with self._built_app(self._worker(None, opted_in=False)) as app:
            assert self._claim_middleware(app) == []

    def test_serving_route_rejects_a_token_naming_another_agent(self):
        """GIVEN a wrong-audience token on a non-A2A route THEN it is rejected.

        NAT copies inbound headers into the workflow context on every route.
        """
        token = make_jwt(aud="api://another-agent")
        with self._built_app(self._worker("api://my-agent")) as app, TestClient(app) as client:
            response = client.get("/health", headers={"x-datarobot-external-access-token": token})
        assert response.status_code == 401

    def test_serving_route_accepts_a_token_naming_this_agent(self):
        token = make_jwt(aud="api://my-agent")
        with self._built_app(self._worker("api://my-agent")) as app, TestClient(app) as client:
            response = client.get("/health", headers={"x-datarobot-external-access-token": token})
        assert response.status_code == 200

    def test_serving_route_without_an_idp_token_still_works(self):
        """GIVEN validation is enabled but no IdP token is sent THEN nothing breaks.

        The DataRobot-API-token path must be unaffected.
        """
        with self._built_app(self._worker("api://my-agent")) as app, TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert (
                client.get(
                    "/health", headers={"authorization": "Bearer NjRiYWE1Njk5NmZiMzZlM2Vl"}
                ).status_code
                == 200
            )


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

    @pytest.fixture
    def captured_keys(self, session_manager):
        """Capture all values passed to ``_context_state.user_id.set()``."""
        keys: list[str] = []

        def capture_set(value: str) -> MagicMock:
            keys.append(value)
            return MagicMock()

        session_manager._context_state.user_id.set = capture_set
        return keys

    def _make_a2a_context(
        self,
        *,
        context_id: str | None = "ctx-1",
        headers: dict[str, str] | None = None,
    ) -> MagicMock:
        """Build a mock A2A ``RequestContext`` with optional forwarded headers."""
        ctx = MagicMock()
        ctx.context_id = context_id
        if headers is not None:
            ctx.call_context.state = {"headers": headers}
        else:
            ctx.call_context = None
        return ctx

    def test_init_sets_session_manager(self, executor, session_manager):
        assert executor.session_manager is session_manager

    async def test_execute_uses_authenticated_identity_as_workflow_key(
        self, executor, session_manager, patch_super_execute
    ):
        """When A2A headers carry a valid auth context, the per-user workflow key
        is derived from the gateway-validated identity, NOT from context_id.
        """
        context = self._make_a2a_context(
            context_id="attacker-chosen-context-id",
            headers={"X-DataRobot-Authorization-Context": "signed-jwt"},
        )
        event_queue = MagicMock()
        with patch(AUTH_HANDLER_PATH, return_value=make_auth_ctx("real-dr-user")):
            await executor.execute(context, event_queue)

        session_manager._context_state.user_id.set.assert_called_once_with(
            expected_workflow_key("real-dr-user")
        )
        patch_super_execute.assert_awaited_once_with(context, event_queue)

    async def test_execute_falls_back_to_context_id_when_no_auth_headers(
        self, executor, session_manager, patch_super_execute
    ):
        """Without authenticated headers (local dev), context_id is hashed via
        _from_session_cookie for key-format consistency with authenticated paths.
        """
        context = self._make_a2a_context(context_id="local-dev-ctx-id")

        await executor.execute(context, MagicMock())

        session_manager._context_state.user_id.set.assert_called_once_with(
            expected_workflow_key("local-dev-ctx-id")
        )

    async def test_execute_skips_user_id_injection_when_no_context_id_and_no_auth(
        self, executor, session_manager, patch_super_execute
    ):
        context = self._make_a2a_context(context_id=None)

        await executor.execute(context, MagicMock())

        session_manager._context_state.user_id.set.assert_not_called()

    async def test_execute_two_users_same_context_id_get_different_keys(
        self, session_manager, captured_keys, patch_super_execute
    ):
        """Two different authenticated users sending the same context_id must get
        different per-user workflow keys -- the core isolation guarantee.
        """
        for uid in ("alice", "bob"):
            executor = _PerUserCompatibleAgentExecutor(session_manager)
            context = self._make_a2a_context(
                context_id="shared-context-id",
                headers={"X-DataRobot-Authorization-Context": "jwt"},
            )
            with patch(AUTH_HANDLER_PATH, return_value=make_auth_ctx(uid)):
                await executor.execute(context, MagicMock())

        assert len(captured_keys) == 2
        assert captured_keys[0] != captured_keys[1]

    async def test_execute_same_user_different_context_id_gets_same_key(
        self, session_manager, captured_keys, patch_super_execute
    ):
        """The same authenticated user across different conversations must get the
        same per-user workflow key -- one builder per user, not per conversation.
        """
        for ctx_id in ("conversation-1", "conversation-2"):
            executor = _PerUserCompatibleAgentExecutor(session_manager)
            context = self._make_a2a_context(
                context_id=ctx_id,
                headers={"X-DataRobot-Authorization-Context": "jwt"},
            )
            with patch(AUTH_HANDLER_PATH, return_value=make_auth_ctx("consistent-user")):
                await executor.execute(context, MagicMock())

        assert len(captured_keys) == 2
        assert captured_keys[0] == captured_keys[1]

    async def test_execute_uses_gateway_user_id_when_no_auth_context(
        self, executor, session_manager, patch_super_execute
    ):
        """When X-DataRobot-Authorization-Context is absent but X-DataRobot-User-Id
        is present, the gateway user ID is used as the workflow key.
        """
        context = self._make_a2a_context(
            context_id="should-not-be-used",
            headers={"X-DataRobot-User-Id": "64baa56996fb36e3eeeefc44"},
        )
        with patch(AUTH_HANDLER_PATH, return_value=None):
            await executor.execute(context, MagicMock())

        session_manager._context_state.user_id.set.assert_called_once_with(
            expected_workflow_key("64baa56996fb36e3eeeefc44")
        )

    async def test_execute_raises_when_auth_context_invalid_instead_of_context_id_fallback(
        self, executor, session_manager, patch_super_execute
    ):
        """Invalid auth-context must fail closed; must not fall back to context_id."""
        context = self._make_a2a_context(
            context_id="must-not-be-used",
            headers={"X-DataRobot-Authorization-Context": "garbage"},
        )
        with (
            patch(AUTH_HANDLER_PATH, return_value=None),
            pytest.raises(ServerError) as exc_info,
        ):
            await executor.execute(context, MagicMock())
        assert isinstance(exc_info.value.error, InvalidParamsError)
        assert exc_info.value.error.code == -32602

        session_manager._context_state.user_id.set.assert_not_called()
        patch_super_execute.assert_not_awaited()

    async def test_execute_resets_a2a_headers_on_identity_error(
        self, executor, session_manager, patch_super_execute
    ):
        """ContextVar for _a2a_headers must be cleaned up even when identity
        resolution raises ServerError (no ContextVar leak).
        """
        from datarobot_genai.dragent.frontends.session import _a2a_headers

        context = self._make_a2a_context(
            context_id="ctx",
            headers={"X-DataRobot-Authorization-Context": "garbage"},
        )
        sentinel = object()
        original = _a2a_headers.get(sentinel)

        with (
            patch(AUTH_HANDLER_PATH, return_value=None),
            pytest.raises(ServerError),
        ):
            await executor.execute(context, MagicMock())

        assert _a2a_headers.get(sentinel) is original

    async def test_execute_logs_warning_on_unauthenticated_fallback(
        self, executor, session_manager, patch_super_execute
    ):
        """Falling back to context_id logs a warning for production visibility."""
        context = self._make_a2a_context(
            context_id="unauthenticated-ctx",
            headers={"some-header": "value"},
        )
        with (
            patch(AUTH_HANDLER_PATH, return_value=None),
            patch("datarobot_genai.dragent.frontends.fastapi.logger") as mock_logger,
        ):
            await executor.execute(context, MagicMock())

        mock_logger.warning.assert_called_once()
        assert "falling back to context_id" in mock_logger.warning.call_args[0][0].lower()


class TestDRAgentFastApiFrontEndConfig:
    def test_is_fastapi_front_end_config(self):
        assert isinstance(DRAgentFastApiFrontEndConfig(), FastApiFrontEndConfig)

    def test_a2a_default_none(self):
        config = DRAgentFastApiFrontEndConfig()
        assert config.a2a is None

    def test_custom_a2a_fields(self):
        cross_app = CrossApplicationAccessConfig(
            token_endpoint_auth_method="private_key_jwt",
            token_exchange=CrossAppTokenExchange(
                trusted_issuer="https://id-jag.example.com",
                audience="https://idp.example.com/oauth2/ausXXX",
            ),
            token_request=CrossAppTokenRequest(
                token_url="https://idp.example.com/oauth2/v1/token",
                audience="api://my-agent",
                scopes=["agent:use"],
            ),
        )
        config = DRAgentFastApiFrontEndConfig(
            a2a=DRAgentA2AConfig(
                server=A2AFrontEndConfig(
                    name="My Agent",
                    description="Does things",
                    version="2.0.0",
                ),
                cross_application_access=cross_app,
            )
        )
        assert config.a2a.server.name == "My Agent"
        assert config.a2a.server.description == "Does things"
        assert config.a2a.server.version == "2.0.0"
        assert config.a2a.cross_application_access == cross_app

    def test_is_not_a2a_front_end_config(self):
        config = DRAgentFastApiFrontEndConfig()
        assert not isinstance(config, A2AFrontEndConfig)

    def test_a2a_enables_endpoints(self):
        config = DRAgentFastApiFrontEndConfig(a2a=DRAgentA2AConfig(server=A2AFrontEndConfig()))
        assert config.a2a is not None

    def test_a2a_external_config_optional(self):
        config = DRAgentFastApiFrontEndConfig(a2a=DRAgentA2AConfig(server=A2AFrontEndConfig()))
        assert config.a2a.external is None

    def test_a2a_with_external_config(self):
        external = DRAgentA2AExternalConfig(id="ext-id-123", url="https://external.example.com/")
        config = DRAgentFastApiFrontEndConfig(
            a2a=DRAgentA2AConfig(server=A2AFrontEndConfig(), external=external)
        )
        assert config.a2a.external.id == "ext-id-123"
        assert config.a2a.external.url == "https://external.example.com/"

    def test_a2a_enable_unauthenticated_well_known_route_defaults_false(self):
        config = DRAgentFastApiFrontEndConfig(a2a=DRAgentA2AConfig(server=A2AFrontEndConfig()))
        assert config.a2a.enable_unauthenticated_well_known_route is False

    def test_a2a_enable_unauthenticated_well_known_route_can_be_enabled(self):
        config = DRAgentFastApiFrontEndConfig(
            a2a=DRAgentA2AConfig(
                server=A2AFrontEndConfig(),
                enable_unauthenticated_well_known_route=True,
            )
        )
        assert config.a2a.enable_unauthenticated_well_known_route is True


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

    def test_dask_client_initialized_to_none(self):
        """NAT's run() finally-block reads self._dask_client directly; it must exist even
        when dask is not installed so shutdown cleanup doesn't raise AttributeError.
        """
        plugin = DRAgentFastApiFrontEndPlugin(full_config=Config(general=GeneralConfig()))
        assert plugin._dask_client is None

    @pytest.mark.asyncio
    async def test_run_shutdown_without_dask_does_not_raise(self):
        """Simulate stopping the server (Ctrl+C) so NAT's run() finally-block runs its
        dask cleanup. Without the _dask_client fix this raises AttributeError when dask
        is not installed.
        """
        config = Config(general=GeneralConfig(front_end=DRAgentFastApiFrontEndConfig()))
        plugin = DRAgentFastApiFrontEndPlugin(full_config=config)

        async def fake_serve(_self):
            raise KeyboardInterrupt

        with (
            patch("datarobot_genai.dragent.workflow_paths.publish_dragent_config_file_env"),
            patch("uvicorn.Server.serve", fake_serve),
        ):
            await plugin.run()


class TestGunicornSettings:
    _ENV = "AGENT_GUNICORN_WORKER_TIMEOUT"
    _RUNTIME_PARAM_ENV = "MLOPS_RUNTIME_PARAM_AGENT_GUNICORN_WORKER_TIMEOUT"

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv(self._ENV, raising=False)
        monkeypatch.delenv(self._RUNTIME_PARAM_ENV, raising=False)

    def test_default(self):
        assert _GunicornSettings().agent_gunicorn_worker_timeout == 600

    def test_plain_env_override(self, monkeypatch):
        monkeypatch.setenv(self._ENV, "300")
        assert _GunicornSettings().agent_gunicorn_worker_timeout == 300

    def test_numeric_runtime_param_float_payload(self, monkeypatch):
        """A DataRobot numeric runtime param delivers a float payload; it coerces to int."""
        monkeypatch.setenv(self._RUNTIME_PARAM_ENV, '{"type": "numeric", "payload": 300.0}')
        assert _GunicornSettings().agent_gunicorn_worker_timeout == 300

    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv(self._ENV, "not-a-number")
        with pytest.raises(ValidationError):
            _GunicornSettings()

    def test_non_positive_raises(self, monkeypatch):
        monkeypatch.setenv(self._ENV, "0")
        with pytest.raises(ValidationError):
            _GunicornSettings()


class TestPatchGunicornWorkerTimeout:
    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv("AGENT_GUNICORN_WORKER_TIMEOUT", raising=False)
        monkeypatch.delenv("MLOPS_RUNTIME_PARAM_AGENT_GUNICORN_WORKER_TIMEOUT", raising=False)

    @pytest.fixture
    def fake_gunicorn(self, monkeypatch):
        """Install a stand-in ``gunicorn.config`` (gunicorn is not a genai dependency)."""

        class Timeout:
            default = 30

        class GracefulTimeout:
            default = 30

        config_mod = types.ModuleType("gunicorn.config")
        config_mod.Timeout = Timeout
        config_mod.GracefulTimeout = GracefulTimeout
        pkg = types.ModuleType("gunicorn")
        pkg.config = config_mod
        monkeypatch.setitem(sys.modules, "gunicorn", pkg)
        monkeypatch.setitem(sys.modules, "gunicorn.config", config_mod)
        return config_mod

    def test_applies_default(self, fake_gunicorn):
        _patch_gunicorn_worker_timeout()
        assert fake_gunicorn.Timeout.default == 600
        assert fake_gunicorn.GracefulTimeout.default == 600

    def test_applies_override(self, fake_gunicorn, monkeypatch):
        monkeypatch.setenv("AGENT_GUNICORN_WORKER_TIMEOUT", "300")
        _patch_gunicorn_worker_timeout()
        assert fake_gunicorn.Timeout.default == 300
        assert fake_gunicorn.GracefulTimeout.default == 300

    def test_noop_when_gunicorn_not_installed(self, monkeypatch):
        """The real genai env has no gunicorn; the helper must no-op, not raise."""
        monkeypatch.setitem(sys.modules, "gunicorn", None)  # forces ImportError
        _patch_gunicorn_worker_timeout()  # must not raise


class TestRunGunicornTimeoutGating:
    _SUPER_RUN = "nat.front_ends.fastapi.fastapi_front_end_plugin.FastApiFrontEndPlugin.run"
    _PATCH_FN = "datarobot_genai.dragent.frontends.fastapi._patch_gunicorn_worker_timeout"
    _PUBLISH = "datarobot_genai.dragent.workflow_paths.publish_dragent_config_file_env"

    @pytest.mark.asyncio
    async def test_run_patches_timeout_when_use_gunicorn(self):
        config = Config(
            general=GeneralConfig(front_end=DRAgentFastApiFrontEndConfig(use_gunicorn=True))
        )
        plugin = DRAgentFastApiFrontEndPlugin(full_config=config)
        with (
            patch(self._PUBLISH),
            patch(self._PATCH_FN) as mock_patch,
            patch(self._SUPER_RUN, new_callable=AsyncMock),
        ):
            await plugin.run()
        mock_patch.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_skips_patch_without_use_gunicorn(self):
        config = Config(
            general=GeneralConfig(front_end=DRAgentFastApiFrontEndConfig(use_gunicorn=False))
        )
        plugin = DRAgentFastApiFrontEndPlugin(full_config=config)
        with (
            patch(self._PUBLISH),
            patch(self._PATCH_FN) as mock_patch,
            patch(self._SUPER_RUN, new_callable=AsyncMock),
        ):
            await plugin.run()
        mock_patch.assert_not_called()
