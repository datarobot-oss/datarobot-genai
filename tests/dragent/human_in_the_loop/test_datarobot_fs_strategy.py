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
import json
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fsspec import AbstractFileSystem

from datarobot_genai.dragent.human_in_the_loop.datarobot_fs_strategy import (
    DataRobotFileSystemHumanInTheLoopStorageStrategy,
)


@contextmanager
def immediate_backoff_retry() -> Iterator[None]:
    """Patch backoff.expo so retries happen immediately in tests."""

    def zero_backoff_wait(**_kwargs: object) -> object:
        yield  # Advance past initial .send() call, like backoff.expo.
        while True:
            yield 0

    with patch(
        "datarobot_genai.dragent.human_in_the_loop.datarobot_fs_strategy.backoff.expo",
        zero_backoff_wait,
    ):
        yield


class FakeDataRobotFileSystem(AbstractFileSystem):
    def __init__(self, catalog_id: str, remote_files: dict[str, object]):
        self._remote_files = {}
        self._catalog_id = catalog_id

    def exists(self, path: str) -> bool:
        return path in self._remote_files

    def put(self, local_path: str, remote_path: str) -> None:
        if not remote_path.startswith(f"dr://{self._catalog_id}"):
            raise ValueError(f"Invalid remote path: {remote_path}")
        self._remote_files[remote_path] = json.load(open(local_path))

    def mv(self, src: str, dst: str) -> None:
        # Thats an expectation that we do mv from remote path to local path
        assert src.startswith(f"dr://{self._catalog_id}")
        if src not in self._remote_files:
            raise FileNotFoundError(f"File not found: {src}")
        with open(dst, "w") as f:
            json.dump(self._remote_files[src], f)
        del self._remote_files[src]

    def cp(self, src: str, dst: str) -> None:
        # Thats an expectation that we do cp from remote path to local path
        assert src.startswith(f"dr://{self._catalog_id}")
        if src not in self._remote_files:
            raise FileNotFoundError(f"File not found: {src}")
        with open(dst, "w") as f:
            json.dump(self._remote_files[src], f)


class FakeFailingDataRobotFileSystem(FakeDataRobotFileSystem):
    def __init__(self, n_failures: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_failures = n_failures
        self._exists_failures = 0
        self._mv_failures = 0
        self._cp_failures = 0

    def exists(self, path: str) -> bool:
        if self._exists_failures < self._n_failures:
            self._exists_failures += 1
            raise Exception("Failed to exists file")
        return super().exists(path)

    def mv(self, src: str, dst: str) -> None:
        if self._mv_failures < self._n_failures:
            self._mv_failures += 1
            raise Exception("Failed to mv file")
        super().mv(src, dst)

    def cp(self, src: str, dst: str) -> None:
        if self._cp_failures < self._n_failures:
            self._cp_failures += 1
            raise Exception("Failed to cp file")
        super().cp(src, dst)


@pytest.fixture
def request_id() -> str:
    return "request-1"


@pytest.fixture
def catalog_id() -> str:
    return "catalog-123"


@pytest.fixture
def response_object() -> object:
    return {"response": "test"}


async def test_provide_human_input_stores_object(
    catalog_id: str, response_object: object, request_id: str
) -> None:
    # GIVEN a DataRobotFileSystem with a certain catalog_id existing
    # Which may fail on exists, mv or cp operations
    fs = FakeDataRobotFileSystem(catalog_id, {})

    # GIVEN a DataRobotFileSystemHumanInTheLoopStorageStrategy
    strategy = DataRobotFileSystemHumanInTheLoopStorageStrategy(
        fs, catalog_id, max_wait=0.03, polling_interval=0.01
    )

    # WHEN human input is provided
    await strategy.provide_human_input(request_id, response_object)

    # THEN the object is stored in the DataRobotFileSystem at the correct remote path
    remote_path = f"dr://{catalog_id}/human_in_the_loop/{request_id}.json"
    assert fs.exists(remote_path)

    # THEN the object can be retrieved
    retrieved_object = await strategy.wait_for_human_input(request_id)
    assert retrieved_object == response_object


@pytest.mark.parametrize("n_failures", [0, 1])
@pytest.mark.parametrize("cleanup", [True, False])
async def test_wait_for_human_input_retrieves_object(
    cleanup: bool, n_failures: int, catalog_id: str, response_object: object, request_id: str
) -> None:
    # GIVEN a DataRobotFileSystem with a certain catalog_id existing
    # Which may fail on exists, mv or cp operations
    fs = FakeFailingDataRobotFileSystem(n_failures, catalog_id, {})

    # GIVEN a DataRobotFileSystemHumanInTheLoopStorageStrategy
    with immediate_backoff_retry():
        strategy = DataRobotFileSystemHumanInTheLoopStorageStrategy(
            fs, catalog_id, max_wait=0.03, polling_interval=0.01, cleanup=cleanup
        )

    # GIVEN response is provided later
    async def provide_response() -> None:
        await asyncio.sleep(0.01)
        await strategy.provide_human_input(request_id, response_object)

    asyncio.create_task(provide_response())

    # WHEN human input is retrieved
    retrieved_object = await strategy.wait_for_human_input(request_id)

    # THEN the expected object is retrieved
    assert retrieved_object == response_object

    remote_path = f"dr://{catalog_id}/human_in_the_loop/{request_id}.json"
    if cleanup:
        # THEN the object is deleted from the DataRobotFileSystem
        assert not fs.exists(remote_path)
    else:
        # THEN the object is still in the DataRobotFileSystem
        assert fs.exists(remote_path)


async def test_wait_for_human_input_raises_timeout(catalog_id: str, request_id: str) -> None:
    # GIVEN a DataRobotFileSystem with a certain catalog_id existing
    fs = FakeDataRobotFileSystem(catalog_id, {})

    # GIVEN a DataRobotFileSystemHumanInTheLoopStorageStrategy
    strategy = DataRobotFileSystemHumanInTheLoopStorageStrategy(
        fs, catalog_id, max_wait=0.02, polling_interval=0.01
    )

    # GIVEN no response is provided
    with pytest.raises(TimeoutError, match="Max wait time reached"):
        await strategy.wait_for_human_input(request_id)
