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
from pathlib import Path
from unittest.mock import ANY

import pytest
from datarobot.core.config import DataRobotAppFrameworkBaseSettings

from datarobot_genai.core.chat.responses import to_custom_model_streaming_response
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


async def test_custom_model_streaming_response(agent):
    # Call the run method with test inputs
    completion_create_params = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "AI"}],
        "environment_var": True,
        "stream": True,
    }

    thread_pool_executor = ThreadPoolExecutor(1)
    event_loop = asyncio.new_event_loop()
    thread_pool_executor.submit(asyncio.set_event_loop, event_loop).result()

    def invoke_with_auth_context():  # type: ignore[no-untyped-def]
        return event_loop.run_until_complete(
            agent.invoke(completion_create_params=completion_create_params)
        )

    result = thread_pool_executor.submit(invoke_with_auth_context).result()

    streaming_response_iterator = to_custom_model_streaming_response(
        thread_pool_executor,
        event_loop,
        result,
        model=completion_create_params.get("model"),
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
