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

"""Deterministically close sqlite connections leaked by crewai storages.

crewai's sqlite-backed storages (kickoff task outputs, flow persistence) run
every operation as ``with sqlite3.connect(...) as conn``. Python's sqlite3
context manager only commits or rolls back -- it never closes -- so each call
strands a WAL-mode connection (up to 3 fds for the db/-wal/-shm files) until
cyclic GC happens to run. In a long-lived serving process the fd table fills
up and the next file open fails with ``OSError: [Errno 24] Too many open
files``.

Until this is fixed upstream, ``apply()`` swaps the ``sqlite3`` reference
inside the affected crewai modules for a delegate whose ``connect()`` result
additionally closes the connection when its with-block exits. Modules that
are missing or already patched are skipped, so the patch degrades to a no-op
on crewai versions with a different layout.

Upgrading crewai does not avoid this. The leak is still present in the latest
release (1.14.7), whose storages run the same unclosed
``with sqlite3.connect(...)``, so the patch is needed regardless of version.
That release is also unreachable in this stack anyway: crewai 1.14.7 pins
``aiofiles~=24.1.0`` while ``nvidia-nat-crewai`` (via ``nvidia-nat-core``)
requires ``aiofiles>=25.1``, and the non-overlapping ranges leave the lock
unsatisfiable on Python 3.13. crewai 1.13.0 carries no aiofiles pin and
resolves cleanly, which is why we stay on it.
"""

import importlib
import logging
import sqlite3
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)

_TARGET_MODULES = (
    "crewai.memory.storage.kickoff_task_outputs_storage",
    "crewai.flow.persistence.sqlite",
)


class _ClosingConnection:
    """sqlite3.Connection wrapper that closes after commit/rollback on exit."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self._conn.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        try:
            return bool(self._conn.__exit__(exc_type, exc, tb))
        finally:
            self._conn.close()

    def __getattr__(self, name: str) -> Any:
        # Direct (non with-block) usage keeps working against the real
        # connection, with the same lifetime semantics as unpatched code.
        return getattr(self._conn, name)


class _ClosingSqlite3:
    """sqlite3 module stand-in whose connect() closes on with-block exit."""

    def connect(self, *args: Any, **kwargs: Any) -> _ClosingConnection:
        return _ClosingConnection(sqlite3.connect(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlite3, name)


def apply() -> None:
    """Patch the known leaking crewai modules; safe to call more than once."""
    for module_name in _TARGET_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            logger.debug("sqlite fd patch: %s not importable, skipping", module_name)
            continue
        current = getattr(module, "sqlite3", None)
        if current is sqlite3:
            module.sqlite3 = _ClosingSqlite3()  # type: ignore[attr-defined]
        elif not isinstance(current, _ClosingSqlite3):
            logger.warning(
                "sqlite fd patch: %s.sqlite3 is unexpectedly %r, skipping",
                module_name,
                current,
            )
