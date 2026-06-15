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

"""Disable crewai's kickoff-outputs SQLite storage for stateless serving.

crewai's ``Crew`` persists every task output to an on-disk WAL-mode SQLite
database via ``_task_output_handler`` (a ``TaskOutputStorageHandler``), built
unconditionally as a ``PrivateAttr`` at Crew construction. There is no public
option to turn it off. Each operation runs ``with sqlite3.connect(...) as
conn``; the stdlib sqlite3 context manager commits but never closes, so every
kickoff strands db/-wal/-shm file descriptors until cyclic GC happens to run.
In a long-lived ``nat dragent serve`` process the fd table fills and the next
open fails with ``OSError: [Errno 24] Too many open files``.

That storage exists solely to back ``Crew.replay(task_id)``. The dragent
request path is stateless -- every request builds and kicks off a fresh crew
and never replays -- so the storage is pure liability (local disk state, not
shared-safe for horizontal scaling). Rather than patch crewai or stdlib
``sqlite3``, we replace the handler with an in-process no-op so no database is
ever opened. Replay simply returns nothing (which we never call).

Two entry points cover the two ways the integration produces a crew:

* :class:`StatelessCrew` -- a ``Crew`` subclass that overrides the
  ``_task_output_handler`` ``PrivateAttr`` default factory so the real
  sqlite-backed handler is *never constructed*. Used by the base ``crew``
  property. This opens zero SQLite connections, even across many fresh
  per-request constructions.
* :func:`neutralize_kickoff_storage` -- a post-construction swap for crews we
  receive already built (the ``datarobot_agent_class_from_crew`` singleton, or
  a subclass that overrides ``crew`` to return a plain ``crewai.Crew``).

Both rely only on crewai's public ``Crew`` subclassing surface and the
documented ``TaskOutputStorageHandler`` method contract -- no stdlib
monkeypatch and no crewai version pin.
"""

from __future__ import annotations

import logging
from typing import Any

import pydantic
from crewai import Crew
from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler

logger = logging.getLogger(__name__)


class _NoOpTaskOutputHandler(TaskOutputStorageHandler):
    """Drop-in for ``TaskOutputStorageHandler`` that persists nothing.

    Subclasses the real handler so it satisfies the ``Crew`` ``PrivateAttr``
    type, but skips ``super().__init__`` so it never builds the sqlite-backed
    storage (the real ``__init__`` constructs a ``KickoffTaskOutputsSQLiteStorage``,
    which opens and initializes the WAL-mode db). Implements the full surface
    ``Crew`` calls (``update``, ``reset``, ``load``) plus ``add`` for parity.
    ``load`` returns an empty list, so a stray ``Crew.replay`` degrades to
    crewai's normal "task not found" rather than crashing.
    """

    def __init__(self) -> None:
        # Intentionally do NOT call super().__init__(): that would construct
        # the sqlite-backed storage and open a connection we never close.
        pass

    def update(self, task_index: int, log: dict[str, Any]) -> None:
        return None

    def add(self, *args: Any, **kwargs: Any) -> None:
        return None

    def reset(self) -> None:
        return None

    def load(self) -> list[dict[str, Any]]:
        return []


class StatelessCrew(Crew):
    """``Crew`` whose kickoff-outputs handler never touches disk.

    Overrides the ``_task_output_handler`` private attribute's default factory
    so the real sqlite-backed :class:`TaskOutputStorageHandler` is never built.
    Construct this instead of :class:`crewai.Crew` wherever the integration
    builds a crew that will be kicked off in the stateless serving path.

    Why a subclass instead of just swapping ``_task_output_handler`` after
    construction (as :func:`neutralize_kickoff_storage` does): the real handler
    is built by this private attribute's ``default_factory`` *inside*
    ``Crew.__init__`` -- so by the time you could swap it, the db has already
    been opened and the file already created. crewai also ignores private attrs
    passed to ``Crew(...)``, so you cannot inject the no-op at construction. And
    because the ``crew`` property builds a fresh crew per request, that
    construction-time open would recur every request -- still leaking ~3 fds/req
    and still writing the local file. Overriding the factory prevents the real
    handler from ever being built, so no connection is opened and no file is
    created.
    """

    _task_output_handler: TaskOutputStorageHandler = pydantic.PrivateAttr(
        default_factory=_NoOpTaskOutputHandler
    )


def neutralize_kickoff_storage(crew: Crew) -> Crew:
    """Replace an already-built ``crew``'s kickoff-outputs handler with a no-op.

    Use this for crews the integration does not construct itself (e.g. the
    pre-built crew passed to ``datarobot_agent_class_from_crew``). Prefer
    :class:`StatelessCrew` when constructing the crew directly, since that
    avoids building the real sqlite handler at all.

    Safe to call on any crew. If a future crewai release renames or drops the
    private ``_task_output_handler`` attribute, this degrades to a logged
    no-op instead of raising, leaving the crew functionally unchanged.
    """
    if hasattr(crew, "_task_output_handler"):
        crew._task_output_handler = _NoOpTaskOutputHandler()  # type: ignore[assignment]
    else:
        logger.debug(
            "crew has no _task_output_handler attribute; kickoff-storage "
            "neutralization skipped (crewai layout changed)"
        )
    return crew
