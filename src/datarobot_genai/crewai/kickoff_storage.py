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

crewai's ``Crew`` opens an on-disk SQLite db via ``_task_output_handler`` on
every kickoff, using ``with sqlite3.connect(...)`` blocks that commit but never
close -- so a long-lived ``nat dragent serve`` leaks fds until
``[Errno 24] Too many open files``. It can't be disabled and only backs
``Crew.replay()``, which we never use.

:func:`neutralize_kickoff_storage` swaps that handler for an in-process no-op so
no further database connections are opened. It is applied to every crew built
via :func:`datarobot_genai.crewai.agent.datarobot_agent_class_from_crew`. No
stdlib monkeypatch, no crewai version pin.
"""

from __future__ import annotations

import logging
from typing import Any

from crewai import Crew
from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler

logger = logging.getLogger(__name__)


class _NoOpTaskOutputHandler(TaskOutputStorageHandler):
    """No-op ``TaskOutputStorageHandler``: same interface, opens no sqlite db.

    Skips ``super().__init__`` (which would build the sqlite storage); ``load``
    returns ``[]`` so a stray ``Crew.replay`` degrades gracefully.
    """

    def __init__(self) -> None:
        # Skip super().__init__(): it builds the sqlite storage we want to avoid.
        pass

    def update(self, task_index: int, log: dict[str, Any]) -> None:
        return None

    def add(self, *args: Any, **kwargs: Any) -> None:
        return None

    def reset(self) -> None:
        return None

    def load(self) -> list[dict[str, Any]]:
        return []


def neutralize_kickoff_storage(crew: Crew) -> Crew:
    """Replace a crew's kickoff-outputs handler with the no-op (in place).

    Safe to call on any crew. If a future crewai release renames or drops the
    private ``_task_output_handler`` attribute, this degrades to a logged no-op
    instead of raising, leaving the crew functionally unchanged.
    """
    if hasattr(crew, "_task_output_handler"):
        crew._task_output_handler = _NoOpTaskOutputHandler()  # type: ignore[assignment]
    else:
        logger.debug(
            "crew has no _task_output_handler attribute; kickoff-storage "
            "neutralization skipped (crewai layout changed)"
        )
    return crew
