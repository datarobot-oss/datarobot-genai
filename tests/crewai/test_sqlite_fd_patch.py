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

import gc
import os
import sqlite3

import pytest

pytest.importorskip("crewai")

from crewai.memory.storage.kickoff_task_outputs_storage import KickoffTaskOutputsSQLiteStorage


def _open_fds() -> int:
    return len(os.listdir("/dev/fd"))


@pytest.fixture
def storage(tmp_path):
    import datarobot_genai.crewai  # noqa: F401  applies the sqlite fd patch on import

    return KickoffTaskOutputsSQLiteStorage(db_path=str(tmp_path / "kickoff.db"))


def test_kickoff_storage_does_not_leak_fds(storage):
    # Warm up so WAL sidecar files and lazy allocations exist before measuring.
    storage.delete_all()
    storage.load()

    gc.disable()
    try:
        gc.collect()
        before = _open_fds()
        for _ in range(20):
            storage.delete_all()
            storage.load()
        after = _open_fds()
    finally:
        gc.enable()

    assert after - before == 0


def test_patched_connect_commits_on_success(storage, tmp_path):
    db = str(tmp_path / "kickoff.db")
    from crewai.memory.storage import kickoff_task_outputs_storage as target

    with target.sqlite3.connect(db, timeout=30) as conn:
        conn.execute(
            "INSERT INTO latest_kickoff_task_outputs"
            " (task_id, expected_output, output, task_index, inputs, was_replayed)"
            " VALUES ('t1', 'x', '{}', 0, '{}', 0)"
        )

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")  # connection must be closed after the with-block

    verify = sqlite3.connect(db)
    try:
        rows = verify.execute("SELECT task_id FROM latest_kickoff_task_outputs").fetchall()
    finally:
        verify.close()
    assert rows == [("t1",)]


def test_patched_connect_rolls_back_on_error(storage, tmp_path):
    db = str(tmp_path / "kickoff.db")
    from crewai.memory.storage import kickoff_task_outputs_storage as target

    with pytest.raises(RuntimeError):
        with target.sqlite3.connect(db, timeout=30) as conn:
            conn.execute(
                "INSERT INTO latest_kickoff_task_outputs"
                " (task_id, expected_output, output, task_index, inputs, was_replayed)"
                " VALUES ('t2', 'x', '{}', 0, '{}', 0)"
            )
            raise RuntimeError("boom")

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")  # closed even when the body raised

    verify = sqlite3.connect(db)
    try:
        rows = verify.execute(
            "SELECT task_id FROM latest_kickoff_task_outputs WHERE task_id = 't2'"
        ).fetchall()
    finally:
        verify.close()
    assert rows == []  # the failed transaction must have been rolled back


def test_crewai_version_is_validated_for_patch():
    """Sentinel guarding the crewai version the fd-leak patch was validated against.

    crewai cannot currently move past the 1.13.x line: 1.14.7 still leaks and is
    blocked by an aiofiles conflict with nvidia-nat-core (see the
    ``_sqlite_fd_patch`` module docstring). If this fails, crewai changed --
    re-verify the patch still closes the leaked connections and refresh the
    docstring's upgrade notes.
    """
    import crewai

    major, minor = (int(part) for part in crewai.__version__.split(".")[:2])
    assert (major, minor) == (1, 13), (
        f"crewai is {crewai.__version__}, but the fd-leak patch was validated "
        "against 1.13.x. Re-verify datarobot_genai.crewai._sqlite_fd_patch and "
        "whether the upstream leak / aiofiles block has changed."
    )
