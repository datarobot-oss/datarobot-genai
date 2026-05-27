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

import json
import os
import tempfile
import time
from typing import TypeVar

import backoff
from datarobot.fs import DataRobotFileSystem
from fsspec.asyn import asyncio

from .strategy import HumanInTheLoopStorageStrategy

T = TypeVar("T")


class DataRobotFileSystemHumanInTheLoopStorageStrategy(HumanInTheLoopStorageStrategy[T]):
    def __init__(
        self,
        fs: DataRobotFileSystem,
        catalog_id: str,
        polling_interval: float = 1.0,
        max_wait: float = 600.0,
        cleanup: bool = True,
    ):
        self.fs = fs
        self.catalog_id = catalog_id
        self.polling_interval = polling_interval
        self.max_wait = max_wait
        self.cleanup = cleanup

        # Decorate in init method to use max_wait; do not retry on timeout.
        self.wait_for_human_input = backoff.on_exception(  # type: ignore[method-assign]
            backoff.expo,
            Exception,
            max_time=self.max_wait,
            giveup=lambda exc: isinstance(exc, TimeoutError),
        )(self.wait_for_human_input)

    def _get_file_path(self, id: str) -> str:
        return os.path.join(f"dr://{self.catalog_id}", "human_in_the_loop", f"{id}.json")

    async def wait_for_human_input(self, id: str) -> T:
        file_path = self._get_file_path(id)
        start_time = time.time()
        while time.time() - start_time < self.max_wait:
            if self.fs.exists(file_path):
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as f:
                    # Explicitly move to delete original after reading
                    # and store disk space
                    if self.cleanup:
                        self.fs.mv(file_path, f.name)
                    else:
                        self.fs.cp(file_path, f.name)
                    f.seek(0)
                    return json.load(f)
            await asyncio.sleep(self.polling_interval)
        raise TimeoutError("Max wait time reached")

    async def provide_human_input(self, id: str, obj: T) -> None:
        file_path = self._get_file_path(id)
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as f:
            json.dump(obj, f)
            f.flush()
            self.fs.put(f.name, file_path)
