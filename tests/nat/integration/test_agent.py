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

from pathlib import Path
from unittest.mock import ANY

import pytest
from datarobot.core.config import DataRobotAppFrameworkBaseSettings

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
    assert usage == {
        "completion_tokens": ANY,
        "prompt_tokens": ANY,
        "total_tokens": ANY,
    }
