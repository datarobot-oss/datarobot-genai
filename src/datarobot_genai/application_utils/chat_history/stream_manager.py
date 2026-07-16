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

"""Disconnect survival and cancellation for the AG-UI storage agent.

:class:`StreamPersistenceManager` wraps an :class:`~.ag_ui_storage.AGUIStorageAgent`
with the *outer* half of a two-level detachment:

* **Level A (this module)** — for each run, a background *producer* task iterates
  the storage agent's event stream and drains it into an unbounded
  :class:`queue.Queue`.  Because the queue is unbounded, the producer never
  blocks on a slow (or absent) client, so persistence runs to completion even if
  the client disconnects.  A :class:`RunHandle` exposes a separate client-facing
  :meth:`RunHandle.events` generator that reads that queue.
* **Level B (the storage agent)** — an internal :class:`asyncio.Queue` consumer
  that folds the same stream into persisted chat history with a guaranteed final
  flush.

The manager is *instance-scoped*: it owns an in-process registry of in-flight
runs keyed by ``(thread_id, run_id)`` (Decision 2 in the plan) — there is no
module-global state.  :meth:`StreamPersistenceManager.cancel` (and the
equivalent :meth:`RunHandle.cancel`) cancels the producer task, which propagates
:class:`asyncio.CancelledError` into the storage agent's ``run`` so its existing
interrupt finalisation flips still-active records to ``interrupted``.

The producer's ``finally`` **always** enqueues a :class:`NoMoreEvents` sentinel
so a client reading :meth:`RunHandle.events` can never hang, and — when the
producer dies without a terminal event — synthesises a
:class:`~ag_ui.core.RunErrorEvent` so the client stops waiting for one.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from collections.abc import AsyncGenerator
from collections.abc import Callable
from functools import partial
from typing import Generic
from typing import ParamSpec

from ag_ui.core import BaseEvent
from ag_ui.core import RunAgentInput
from ag_ui.core import RunErrorEvent
from ag_ui.core import RunFinishedEvent

from datarobot_genai.application_utils.chat_history.ag_ui_storage import AGUIStorageAgent
from datarobot_genai.application_utils.chat_history.ag_ui_storage import ErrorCodes

logger = logging.getLogger(__name__)

#: ParamSpec capturing the trailing arguments the agent factory (and therefore
#: :meth:`StreamPersistenceManager.run`) accepts.
P = ParamSpec("P")


class NoMoreEvents:
    """Sentinel enqueued exactly once when a run's producer terminates.

    A client reading :meth:`RunHandle.events` stops when it dequeues this
    marker; the producer's ``finally`` block guarantees it is always enqueued
    (even on cancellation or failure), so the client can never hang.
    """


class RunHandle:
    """A handle to one in-flight run started by :class:`StreamPersistenceManager`.

    Attributes
    ----------
    thread_id : str
        The AG-UI thread id of the run.
    run_id : str
        The AG-UI run id of the run.
    """

    def __init__(
        self,
        *,
        thread_id: str,
        run_id: str,
        event_queue: queue.Queue[BaseEvent | NoMoreEvents],
        task: asyncio.Task[None],
        cancel_fn: Callable[[], bool],
        poll_interval: float,
    ) -> None:
        """Initialise the handle.

        Parameters
        ----------
        thread_id : str
            The AG-UI thread id of the run.
        run_id : str
            The AG-UI run id of the run.
        event_queue : queue.Queue[BaseEvent | NoMoreEvents]
            The unbounded queue the producer drains into.
        task : asyncio.Task[None]
            The background producer task.
        cancel_fn : Callable[[], bool]
            A zero-argument callable that cancels this run and returns ``True``
            when a live run was found (``False`` if it had already finished).
        poll_interval : float
            Seconds :meth:`events` sleeps between polls of an empty queue.
        """
        self.thread_id = thread_id
        self.run_id = run_id
        self._queue = event_queue
        self._task = task
        self._cancel_fn = cancel_fn
        self._poll_interval = poll_interval

    async def events(self) -> AsyncGenerator[BaseEvent, None]:
        """Yield the run's events until the terminating sentinel.

        Reads the unbounded producer queue, sleeping briefly when it is empty.
        The generator ends when it dequeues :class:`NoMoreEvents`; because the
        producer always enqueues that sentinel, this can never hang.  A consumer
        may stop iterating at any time (a client disconnect) — the producer
        keeps draining and persisting regardless.

        Yields
        ------
        ag_ui.core.BaseEvent
            Each event the producer forwarded, in order.
        """
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(self._poll_interval)
                continue
            if isinstance(item, NoMoreEvents):
                break
            yield item

    def cancel(self) -> bool:
        """Cancel this run.

        Returns
        -------
        bool
            ``True`` if a live run was found and cancellation requested;
            ``False`` if the run had already finished.
        """
        return self._cancel_fn()

    async def wait(self) -> None:
        """Wait until the producer finishes (the stream is fully drained and persisted).

        Returns after the storage agent's run — including its guaranteed final
        flush and, on cancellation, interrupt finalisation — has completed.  This
        never raises for a cancelled or failed run: the producer captures those
        outcomes internally (a failure is surfaced as a synthesised
        :class:`~ag_ui.core.RunErrorEvent`).
        """
        await asyncio.wait({self._task})


class StreamPersistenceManager(Generic[P]):
    """Run AG-UI storage agents so their output survives client disconnects.

    The manager builds a fresh :class:`~.ag_ui_storage.AGUIStorageAgent` per run
    from an injected factory, then spawns a background producer task that drains
    the agent's stream into an unbounded queue.  It owns the instance-scoped
    registry of in-flight runs, keyed by ``(thread_id, run_id)``, used by
    :meth:`cancel`.
    """

    def __init__(
        self,
        agent_factory: Callable[P, AGUIStorageAgent],
        *,
        poll_interval: float = 0.01,
    ) -> None:
        """Initialise the manager.

        Parameters
        ----------
        agent_factory : Callable[..., AGUIStorageAgent]
            Builds the storage agent for a run.  It is invoked with the trailing
            arguments passed to :meth:`run`, so a run can select the inner agent
            and repositories it needs (e.g. per-request headers or user).
        poll_interval : float
            Seconds :meth:`RunHandle.events` sleeps between polls of an empty
            queue.
        """
        self._agent_factory = agent_factory
        self._poll_interval = poll_interval
        self._runs: dict[
            tuple[str, str], tuple[queue.Queue[BaseEvent | NoMoreEvents], asyncio.Task[None]]
        ] = {}

    async def run(self, input: RunAgentInput, *args: P.args, **kwargs: P.kwargs) -> RunHandle:
        """Start a run and return a :class:`RunHandle`.

        Spawns a producer task that builds the storage agent (via the factory,
        with *args* / *kwargs*), iterates its stream, and drains every event into
        an unbounded queue.  The run is registered under ``(thread_id, run_id)``
        for the lifetime of the producer; the producer unregisters itself when it
        finishes.

        Parameters
        ----------
        input : ag_ui.core.RunAgentInput
            The AG-UI run input; its ``thread_id`` / ``run_id`` key the registry.
        *args, **kwargs
            Forwarded verbatim to the agent factory.

        Returns
        -------
        RunHandle
            A handle exposing the run's event stream, cancellation and a
            completion await.
        """
        event_queue: queue.Queue[BaseEvent | NoMoreEvents] = queue.Queue()
        key = (input.thread_id, input.run_id)

        async def _produce() -> None:
            saw_terminal = False
            try:
                agent = self._agent_factory(*args, **kwargs)
                async for event in agent.run(input):
                    if isinstance(event, (RunFinishedEvent, RunErrorEvent)):
                        saw_terminal = True
                    event_queue.put(event)
            except asyncio.CancelledError:
                # Propagated into the storage agent's ``run``, whose finaliser
                # marks still-active records ``interrupted``.  Re-raise so the
                # task ends cancelled; the ``finally`` still terminates the queue.
                raise
            except Exception as exc:
                logger.exception("Stream producer failed for run %s", key)
                # The client only stops waiting on RUN_FINISHED / RUN_ERROR; if
                # the producer died without one, synthesise a terminal error so
                # the run does not hang forever on the client side.
                if not saw_terminal:
                    event_queue.put(
                        RunErrorEvent(
                            message=str(exc) or "Agent stream failed",
                            code=ErrorCodes.INTERNAL_ERROR.value,
                        )
                    )
            finally:
                # ALWAYS terminate the client-facing queue and unregister, even
                # when the producer was cancelled or raised — otherwise
                # ``RunHandle.events`` loops forever.  Only drop the registry
                # entry when it is still *this* run's: a later run that reused the
                # same ``(thread_id, run_id)`` key must not be unregistered by an
                # earlier run finishing.
                event_queue.put(NoMoreEvents())
                if self._runs.get(key, (None, None))[0] is event_queue:
                    self._runs.pop(key, None)

        task = asyncio.create_task(_produce())
        self._runs[key] = (event_queue, task)
        return RunHandle(
            thread_id=input.thread_id,
            run_id=input.run_id,
            event_queue=event_queue,
            task=task,
            cancel_fn=partial(self.cancel, input.thread_id, input.run_id),
            poll_interval=self._poll_interval,
        )

    def cancel(self, thread_id: str, run_id: str) -> bool:
        """Cancel the run keyed by ``(thread_id, run_id)``.

        Cancels the producer task, propagating :class:`asyncio.CancelledError`
        into the storage agent's ``run`` so its interrupt finalisation marks
        still-active records ``interrupted``.

        Parameters
        ----------
        thread_id : str
            The run's AG-UI thread id.
        run_id : str
            The run's AG-UI run id.

        Returns
        -------
        bool
            ``True`` if a live run was found and cancellation requested;
            ``False`` when no matching run exists (already finished or unknown).
        """
        entry = self._runs.get((thread_id, run_id))
        if entry is None:
            return False
        _, task = entry
        task.cancel()
        return True
