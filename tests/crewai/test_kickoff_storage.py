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

"""Tests for crewai kickoff-outputs storage neutralization.

crewai's ``Crew`` persists every task output to an on-disk WAL-mode SQLite db
whose ``with sqlite3.connect(...)`` blocks never close, leaking file descriptors
until garbage collection. The integration replaces that handler with an
in-process no-op so a long-lived serve process opens no database and cannot
exhaust its fd table. These tests pin that the no-op is wired in and satisfies
the contract crewai relies on (no sqlite-backed ``storage`` is ever built).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("crewai")

# A crewai Agent eagerly builds an LLM at construction; a dummy key lets the
# Agent construct (the LLM is never invoked here -- only storage wiring is checked).
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-not-used")
os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-4.1-mini")

from crewai import Agent  # noqa: E402
from crewai import Crew  # noqa: E402
from crewai import Task  # noqa: E402
from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler  # noqa: E402

from datarobot_genai.crewai._kickoff_storage import _NoOpTaskOutputHandler  # noqa: E402
from datarobot_genai.crewai._kickoff_storage import neutralize_kickoff_storage  # noqa: E402


def _make_task() -> Task:
    agent = Agent(role="r", goal="g", backstory="b")
    return Task(description="d", expected_output="e", agent=agent)


def test_noop_handler_is_a_task_output_storage_handler() -> None:
    """The no-op must satisfy crewai's PrivateAttr type and open no sqlite db."""
    handler = _NoOpTaskOutputHandler()
    assert isinstance(handler, TaskOutputStorageHandler)
    # The real handler builds a sqlite-backed `storage`; the no-op must not.
    assert not hasattr(handler, "storage")
    assert handler.load() == []


def test_neutralize_swaps_handler_on_prebuilt_crew() -> None:
    """For crews we receive already built, the swap removes the sqlite handler."""
    task = _make_task()
    crew = Crew(agents=[task.agent], tasks=[task], verbose=False)
    assert isinstance(crew._task_output_handler, TaskOutputStorageHandler)
    assert hasattr(crew._task_output_handler, "storage")  # real sqlite storage

    returned = neutralize_kickoff_storage(crew)
    assert returned is crew
    assert isinstance(crew._task_output_handler, _NoOpTaskOutputHandler)
    assert not hasattr(crew._task_output_handler, "storage")


def test_neutralize_is_safe_when_attr_missing() -> None:
    """Degrades to a logged no-op if crewai drops the private attribute."""

    class _NoHandler:
        pass

    obj = _NoHandler()
    # Should not raise even though there is no _task_output_handler.
    assert neutralize_kickoff_storage(obj) is obj  # type: ignore[arg-type]


def test_crewai_version_is_validated() -> None:
    """Sentinel for the crewai version this neutralization was validated against.

    crewai cannot currently move past the 1.13.x line (1.14.7 is blocked by an
    aiofiles conflict with nvidia-nat-core). If this fails, crewai changed --
    re-verify that ``_task_output_handler`` is still the kickoff-outputs sqlite
    handler and that the no-op subclass still satisfies its type.
    """
    import crewai

    major, minor = (int(part) for part in crewai.__version__.split(".")[:2])
    assert (major, minor) == (1, 13), (
        f"crewai is {crewai.__version__}, but kickoff-storage neutralization "
        "was validated against 1.13.x. Re-verify _kickoff_storage.py."
    )
