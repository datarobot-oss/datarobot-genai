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
import time
from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import TypeVar

T = TypeVar("T")


class HumanInTheLoopStorageStrategy(ABC, Generic[T]):
    """Strategy for storing and retrieving human input in the loop."""

    @abstractmethod
    async def wait_for_human_input(self, id: str) -> T:
        raise NotImplementedError()

    @abstractmethod
    async def provide_human_input(self, id: str, object: T) -> None:
        raise NotImplementedError()


class InMemoryHumanInTheLoopStorageStrategy(HumanInTheLoopStorageStrategy[T]):
    def __init__(self, polling_interval: float = 1.0, max_wait: float = 600.0):
        self.id_to_object: dict[str, T] = {}
        self.polling_interval = polling_interval
        self.max_wait = max_wait

    async def wait_for_human_input(self, id: str) -> T:
        start_time = time.time()
        while time.time() - start_time < self.max_wait:
            if id in self.id_to_object:
                return self.id_to_object[id]
            await asyncio.sleep(self.polling_interval)

        raise TimeoutError("Max wait time reached")

    async def provide_human_input(self, id: str, object: T) -> None:
        self.id_to_object[id] = object
