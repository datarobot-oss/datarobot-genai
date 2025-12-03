# Copyright 2025 DataRobot, Inc. and its affiliates.
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
import queue
import threading
from collections.abc import AsyncIterator
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import TypeVar
from unittest.mock import ANY

import pytest
from datarobot.core.config import DataRobotAppFrameworkBaseSettings

from datarobot_genai.core.chat.responses import to_custom_model_streaming_response_old
from datarobot_genai.nat.agent import NatAgent


class Config(DataRobotAppFrameworkBaseSettings):
    """
    Finds variables in the priority order of: env
    variables (including Runtime Parameters), .env, file_secrets, then
    Pulumi output variables.
    """

    datarobot_endpoint: str = "https://app.datarobot.com/api/v2"
    datarobot_api_token: str | None = None


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def workflow_path():
    return Path(__file__).parent / "workflow.yaml"


@pytest.fixture
def agent(workflow_path, config):
    return NatAgent(
        workflow_path=workflow_path,
        api_key=config.datarobot_api_token,
        api_base=config.datarobot_endpoint,
    )


async def test_run_method(agent):
    # Call the run method with test inputs
    completion_create_params = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "AI"}],
        "environment_var": True,
    }
    result, pipeline_interactions, usage = await agent.invoke(completion_create_params)

    assert result
    assert isinstance(result, str)
    assert pipeline_interactions
    assert usage["completion_tokens"] > 0
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] > 0


async def test_run_method_streaming(agent):
    # Call the run method with test inputs
    completion_create_params = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "AI"}],
        "environment_var": True,
        "stream": True,
    }
    streaming_response_iterator = await agent.invoke(completion_create_params)

    async for (
        result,
        pipeline_interactions,
        usage,
    ) in streaming_response_iterator:
        assert isinstance(result, str)
        assert usage == {
            "completion_tokens": ANY,
            "prompt_tokens": ANY,
            "total_tokens": ANY,
        }
    # Final chunk has the total usage and pipeline interactions
    assert usage["completion_tokens"] > 0
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] > 0
    assert pipeline_interactions


T = TypeVar("T")


def async_gen_to_sync_thread(async_iterator: AsyncIterator[T]) -> Iterator[T]:
    """
    Runs an async iterator in a separate thread and provides a sync iterator.
    """
    # A thread-safe queue for communication
    sync_queue: queue.Queue[Any] = queue.Queue()
    # A sentinel object to signal the end of the async generator
    SENTINEL = object()

    async def run_async_to_queue():
        """The coroutine that runs in the separate thread's event loop."""
        try:
            async for item in await async_iterator:
                sync_queue.put(item)
        except Exception as e:
            # Put the exception on the queue to be re-raised in the main thread
            sync_queue.put(e)
        finally:
            # Signal the end of iteration
            sync_queue.put(SENTINEL)

    def thread_target():
        """The entry point for the separate thread."""
        # Create a new event loop for this new thread and run the coroutine
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_async_to_queue())
        loop.close()

    # Start the background thread
    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()

    # The main thread consumes items synchronously
    while True:
        item = sync_queue.get()
        if item is SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def test_custom_model_streaming_response(agent):
    # Call the run method with test inputs
    completion_create_params = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "AI"}],
        "environment_var": True,
        "stream": True,
    }

    streaming_response_iterator = async_gen_to_sync_thread(
        agent.invoke(completion_create_params=completion_create_params)
    )

    for response in to_custom_model_streaming_response_old(
        streaming_response_iterator, model=completion_create_params.get("model")
    ):
        result = response.choices[0].delta.content
        usage = response.usage
        pipeline_interactions = response.pipeline_interactions
        if result:
            assert isinstance(result, str)
        if usage:
            assert isinstance(usage.completion_tokens, int)
            assert isinstance(usage.prompt_tokens, int)
            assert isinstance(usage.total_tokens, int)
    # Final chunk has the total usage and pipeline interactions
    assert usage.completion_tokens > 0
    assert usage.prompt_tokens > 0
    assert usage.total_tokens > 0
    assert pipeline_interactions
