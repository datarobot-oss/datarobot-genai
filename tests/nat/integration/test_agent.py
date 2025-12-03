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
from concurrent.futures import ThreadPoolExecutor
from collections.abc import AsyncIterator, Iterator
from typing import TypeVar

from pathlib import Path
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


def async_iterable_to_sync_iterable(async_iterator: AsyncIterator[T]) -> Iterator[T]:
    """
    Converts an async iterator to a sync iterator using asyncio.Runner (Python 3.11+).
    """
    with asyncio.Runner() as runner:
        while True:
            try:
                # Run the next iteration within the persistent event loop
                result = runner.run(anext(async_iterator))
                yield result
            except StopAsyncIteration:
                # The async generator is exhausted
                break
            except Exception as e:
                # Handle other exceptions as needed
                raise e


async def test_custom_model_streaming_response(agent):
    # Call the run method with test inputs
    completion_create_params = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "AI"}],
        "environment_var": True,
        "stream": True,
    }

    streaming_response_iterator = async_iterable_to_sync_iterable(
        await agent.invoke(completion_create_params=completion_create_params)
    )

    for response in streaming_response_iterator:
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
