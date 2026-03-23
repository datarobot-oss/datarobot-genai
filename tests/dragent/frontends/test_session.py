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

import asyncio
from datetime import datetime
from datetime import timedelta
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.runtime.session import SessionManager

from datarobot_genai.dragent.frontends.request import DRAgentRunAgentInput
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.frontends.session import DRAgentAGUISessionManager

_BUILDER_PATCH = "nat.builder.per_user_workflow_builder.PerUserWorkflowBuilder"


class FakeBuilder:
    """Async-context-manager stand-in for ``PerUserWorkflowBuilder``.

    Records which asyncio task called ``__aenter__`` / ``__aexit__`` and
    exposes an :attr:`exited` event so tests can synchronise on teardown.
    """

    def __init__(self, workflow=None, *, populate_error=None):
        self.workflow = workflow or MagicMock()
        self.populate_error = populate_error
        self.enter_task_id: int | None = None
        self.exit_task_id: int | None = None
        self.exited = asyncio.Event()

    async def __aenter__(self):
        self.enter_task_id = id(asyncio.current_task())
        return self

    async def __aexit__(self, *exc):
        self.exit_task_id = id(asyncio.current_task())
        self.exited.set()

    async def populate_builder(self, config):
        if self.populate_error:
            raise self.populate_error

    async def build(self, entry_function=None):
        return self.workflow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_manager():
    """Create a DRAgentAGUISessionManager with minimal config and mocked registry."""
    config = Config(general=GeneralConfig())
    shared_builder = MagicMock()
    shared_workflow = MagicMock()

    mock_registration = MagicMock()
    mock_registration.is_per_user = False

    with patch("nat.cli.type_registry.GlobalTypeRegistry.get") as get_registry:
        get_registry.return_value.get_function.return_value = mock_registration
        manager = DRAgentAGUISessionManager(
            config=config,
            shared_builder=shared_builder,
            shared_workflow=shared_workflow,
        )
        yield manager


@pytest.fixture
def per_user_session_manager():
    """DRAgentAGUISessionManager configured for per-user workflows."""
    mock_reg = MagicMock(
        is_per_user=True,
        per_user_function_input_schema=MagicMock(),
        per_user_function_single_output_schema=MagicMock(),
        per_user_function_streaming_output_schema=MagicMock(),
    )
    with patch("nat.cli.type_registry.GlobalTypeRegistry.get") as g:
        g.return_value.get_function.return_value = mock_reg
        yield DRAgentAGUISessionManager(
            config=Config(general=GeneralConfig()),
            shared_builder=MagicMock(),
        )


# ---------------------------------------------------------------------------
# Schema override tests
# ---------------------------------------------------------------------------


class TestDRAgentAGUISessionManager:
    def test_is_session_manager_subclass(self):
        assert issubclass(DRAgentAGUISessionManager, SessionManager)

    def test_get_workflow_input_schema_returns_dragent_run_agent_input(self, session_manager):
        schema = session_manager.get_workflow_input_schema()
        assert schema is DRAgentRunAgentInput

    def test_get_workflow_streaming_output_schema_returns_dragent_event_response(
        self, session_manager
    ):
        schema = session_manager.get_workflow_streaming_output_schema()
        assert schema is DRAgentEventResponse


# ---------------------------------------------------------------------------
# Cross-task-safe per-user builder lifecycle tests
# ---------------------------------------------------------------------------


class TestPerUserBuilderLifecycle:
    """Verify that __aenter__ and __aexit__ always run in the same task."""

    @staticmethod
    async def _create(mgr, user_id, **builder_kw):
        """Patch ``PerUserWorkflowBuilder`` and create a per-user builder.

        Returns ``(builder, workflow, fake)`` where *fake* is the
        :class:`FakeBuilder` instance used inside the lifecycle task.
        """
        fake = FakeBuilder(**builder_kw)
        with patch(_BUILDER_PATCH, return_value=fake):
            builder, workflow = await mgr._get_or_create_per_user_builder(user_id)
        return builder, workflow, fake

    async def test_aenter_and_aexit_run_in_same_task(self, per_user_session_manager):
        builder, _, fake = await self._create(per_user_session_manager, "u1")

        # Trigger the replaced __aexit__ (same path upstream cleanup uses).
        await builder.__aexit__(None, None, None)

        assert fake.enter_task_id is not None
        assert fake.enter_task_id == fake.exit_task_id

    async def test_returns_cached_builder(self, per_user_session_manager):
        b1, w1, _ = await self._create(per_user_session_manager, "u2")
        # Second call hits the cache — no patch needed.
        b2, w2 = await per_user_session_manager._get_or_create_per_user_builder("u2")

        assert (b1, w1) == (b2, w2)
        await per_user_session_manager.shutdown()

    async def test_upstream_cleanup_triggers_aexit(self, per_user_session_manager):
        _, _, fake = await self._create(per_user_session_manager, "u3")

        # Mark the builder as expired.
        info = per_user_session_manager._per_user_builders["u3"]
        info.last_activity = datetime.now() - timedelta(hours=2)

        cleaned = await per_user_session_manager._cleanup_inactive_per_user_builders()

        assert cleaned == 1
        assert "u3" not in per_user_session_manager._per_user_builders
        await asyncio.wait_for(fake.exited.wait(), timeout=5.0)

    async def test_cleanup_skips_active_builders(self, per_user_session_manager):
        _, _, _ = await self._create(per_user_session_manager, "u4")

        info = per_user_session_manager._per_user_builders["u4"]
        info.ref_count = 1
        info.last_activity = datetime.now() - timedelta(hours=2)

        cleaned = await per_user_session_manager._cleanup_inactive_per_user_builders()
        assert cleaned == 0
        assert "u4" in per_user_session_manager._per_user_builders

        info.ref_count = 0
        await per_user_session_manager.shutdown()

    async def test_shutdown_exits_all_builders(self, per_user_session_manager):
        fakes = {}
        for uid in ("ua", "ub"):
            _, _, fake = await self._create(per_user_session_manager, uid)
            fakes[uid] = fake

        await per_user_session_manager.shutdown()

        assert all(f.exited.is_set() for f in fakes.values())
        assert not per_user_session_manager._per_user_builders

    async def test_creation_failure_propagates(self, per_user_session_manager):
        with pytest.raises(RuntimeError, match="populate failed"):
            await self._create(
                per_user_session_manager,
                "uf",
                populate_error=RuntimeError("populate failed"),
            )

        assert "uf" not in per_user_session_manager._per_user_builders
