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

from __future__ import annotations

import asyncio
import logging
import typing
from contextlib import nullcontext
from datetime import datetime

from fastapi import Request
from nat.runtime.session import PerUserBuilderInfo
from nat.runtime.session import SessionManager
from pydantic import BaseModel

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler

from .request import DRAgentRunAgentInput
from .response import DRAgentEventResponse

if typing.TYPE_CHECKING:
    from nat.builder.per_user_workflow_builder import PerUserWorkflowBuilder
    from nat.builder.workflow import Workflow

logger = logging.getLogger(__name__)

_auth_context_handler = AuthContextHeaderHandler()

# Default timeout (seconds) for waiting on a lifecycle task to finish teardown.
_LIFECYCLE_TASK_TIMEOUT = 30.0


class DRAgentAGUISessionManager(SessionManager):
    """Session manager with cross-task-safe per-user builder lifecycle management.

    The upstream :class:`SessionManager` calls ``builder.__aenter__()`` in an HTTP
    request task but ``builder.__aexit__()`` in a periodic background cleanup task.
    The MCP streamable-HTTP client uses *anyio* cancel scopes internally, and *anyio*
    forbids exiting a cancel scope from a task different to the one that entered it,
    causing a ``RuntimeError`` during idle-session cleanup.

    This subclass fixes the problem by running each per-user builder's full lifecycle
    (``__aenter__`` → populate → build → … → ``__aexit__``) inside a **dedicated
    asyncio task**, and replacing the builder's ``__aexit__`` on the instance so that
    the upstream ``_cleanup_inactive_per_user_builders`` and ``shutdown`` transparently
    signal the lifecycle task instead of calling ``__aexit__`` cross-task.
    """

    def get_workflow_input_schema(self) -> type[BaseModel]:
        """Get workflow input schema for OpenAPI documentation."""
        return DRAgentRunAgentInput

    def get_workflow_streaming_output_schema(self) -> type[BaseModel]:
        """Get workflow streaming output schema for OpenAPI documentation."""
        return DRAgentEventResponse

    # ------------------------------------------------------------------
    # Per-user builder lifecycle — runs enter/exit in one dedicated task
    # ------------------------------------------------------------------

    async def _get_or_create_per_user_builder(
        self, user_id: str
    ) -> tuple[PerUserWorkflowBuilder, Workflow]:
        """Create or return the cached per-user builder for *user_id*.

        Overrides the base implementation so that ``builder.__aenter__()`` and
        all resource setup happen inside a **dedicated asyncio task**.  That task
        lives until cleanup or shutdown calls the builder's ``__aexit__``
        (replaced on the instance to signal the task) — so the real
        ``__aexit__`` always runs in the *same* task as ``__aenter__``,
        avoiding the cross-task cancel-scope error from anyio.
        """
        from nat.builder.per_user_workflow_builder import PerUserWorkflowBuilder  # noqa: PLC0415

        async with self._per_user_builders_lock:
            if user_id in self._per_user_builders:
                builder_info = self._per_user_builders[user_id]
                builder_info.last_activity = datetime.now()
                return builder_info.builder, builder_info.workflow

            logger.info(
                "Creating per-user builder for user=%s, entry_function=%s",
                user_id,
                self._entry_function,
            )

            builder = PerUserWorkflowBuilder(user_id=user_id, shared_builder=self._shared_builder)

            # Coordination primitives shared with the lifecycle task.
            ready_event = asyncio.Event()
            shutdown_event = asyncio.Event()
            loop = asyncio.get_running_loop()
            result_future: asyncio.Future[Workflow] = loop.create_future()

            async def _builder_lifecycle() -> None:
                """Run the full builder lifecycle in a single task.

                This ensures ``__aenter__`` and ``__aexit__`` share the same
                asyncio task so that anyio cancel scopes remain valid.
                """
                try:
                    await builder.__aenter__()
                    try:
                        await builder.populate_builder(self._config)
                        workflow = await builder.build(entry_function=self._entry_function)
                        result_future.set_result(workflow)
                    except BaseException as exc:
                        if not result_future.done():
                            result_future.set_exception(exc)
                        raise
                    finally:
                        ready_event.set()

                    # Keep the task alive until cleanup/shutdown signals.
                    await shutdown_event.wait()

                except BaseException:
                    if not ready_event.is_set():
                        ready_event.set()
                finally:
                    try:
                        # Call the *real* class-level __aexit__ (not our
                        # instance replacement below) — in this same task.
                        await type(builder).__aexit__(builder, None, None, None)
                    except Exception:
                        logger.exception(
                            "Error during builder teardown for user %s",
                            user_id,
                        )

            task = asyncio.create_task(_builder_lifecycle(), name=f"builder-lifecycle-{user_id}")

            # Wait for the lifecycle task to finish setup.
            await ready_event.wait()

            try:
                workflow = result_future.result()
            except BaseException:
                # Wait for the lifecycle task to finish its own cleanup.
                try:
                    await asyncio.wait_for(task, timeout=_LIFECYCLE_TASK_TIMEOUT)
                except Exception:
                    pass
                raise

            # Replace __aexit__ on the *instance* so that the upstream
            # _cleanup_inactive_per_user_builders() and shutdown() — which
            # call ``builder_info.builder.__aexit__(…)`` — transparently
            # signal the lifecycle task instead of calling __aexit__
            # cross-task.
            async def _signal_shutdown(*_exc_details: object) -> None:
                shutdown_event.set()
                try:
                    await asyncio.wait_for(task, timeout=_LIFECYCLE_TASK_TIMEOUT)
                except TimeoutError:
                    logger.warning(
                        "Lifecycle task for user %s did not finish in time; cancelling",
                        user_id,
                    )
                    task.cancel()

            builder.__aexit__ = _signal_shutdown  # type: ignore[method-assign]

            # Create per-user semaphore for concurrency control.
            if self._max_concurrency > 0:
                per_user_semaphore: asyncio.Semaphore | nullcontext = asyncio.Semaphore(
                    self._max_concurrency
                )
            else:
                per_user_semaphore = nullcontext()

            builder_info = PerUserBuilderInfo(
                builder=builder,
                workflow=workflow,
                semaphore=per_user_semaphore,
                last_activity=datetime.now(),
                ref_count=0,
                lock=asyncio.Lock(),
            )
            self._per_user_builders[user_id] = builder_info

            logger.info(
                "Created per-user builder for user=%s (total users: %d)",
                user_id,
                len(self._per_user_builders),
            )
            return builder_info.builder, builder_info.workflow

    # ------------------------------------------------------------------
    # HTTP metadata extraction
    # ------------------------------------------------------------------

    def set_metadata_from_http_request(self, request: Request) -> None:
        """Extend base metadata extraction to also set user_id from DataRobot auth context.

        The DataRobot UI does not set the ``nat-session`` cookie that NAT normally uses
        to identify users for per-user workflows.  Instead, DataRobot passes user identity
        via the ``X-DataRobot-Authorization-Context`` JWT header.  We decode that header
        and use the contained user ID so that per-user workflows receive a stable identity.
        """
        super().set_metadata_from_http_request(request)

        # Only attempt extraction if user_id wasn't already set (e.g. via nat-session cookie).
        if self._context_state.user_id.get():
            return

        auth_ctx = _auth_context_handler.get_context(dict(request.headers))
        if auth_ctx is not None:
            user_id = auth_ctx.user.id
            self._context_state.user_id.set(user_id)
            logger.debug("Set user_id from DataRobot auth context: %s", user_id)
