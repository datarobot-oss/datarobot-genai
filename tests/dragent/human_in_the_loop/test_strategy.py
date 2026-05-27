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

import pytest

from datarobot_genai.dragent.human_in_the_loop.strategy import InMemoryHumanInTheLoopStorageStrategy


async def test_provide_human_input_stores_object() -> None:
    strategy = InMemoryHumanInTheLoopStorageStrategy[str]()

    await strategy.provide_human_input("request-1", "approved")

    assert strategy.id_to_object["request-1"] == "approved"


async def test_wait_for_human_input_returns_when_input_already_provided() -> None:
    strategy = InMemoryHumanInTheLoopStorageStrategy[int]()
    await strategy.provide_human_input("request-1", 42)

    result = await strategy.wait_for_human_input("request-1")

    assert result == 42


async def test_wait_for_human_input_waits_until_input_provided() -> None:
    strategy = InMemoryHumanInTheLoopStorageStrategy[str](polling_interval=0.01)

    async def provide_after_delay() -> None:
        await asyncio.sleep(0.05)
        await strategy.provide_human_input("request-1", "human response")

    provider_task = asyncio.create_task(provide_after_delay())

    result = await strategy.wait_for_human_input("request-1")

    await provider_task
    assert result == "human response"


async def test_wait_for_human_input_raises_timeout() -> None:
    strategy = InMemoryHumanInTheLoopStorageStrategy[str](
        polling_interval=0.05,
        max_wait=0.01,
    )

    with pytest.raises(TimeoutError, match="Max wait time reached"):
        await strategy.wait_for_human_input("request-1")


async def test_ids_are_independent() -> None:
    strategy = InMemoryHumanInTheLoopStorageStrategy[str]()

    await strategy.provide_human_input("request-1", "first")
    await strategy.provide_human_input("request-2", "second")

    assert await strategy.wait_for_human_input("request-1") == "first"
    assert await strategy.wait_for_human_input("request-2") == "second"
