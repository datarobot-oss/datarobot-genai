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

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from datarobot_genai.nat.agent import NatAgent


@pytest.fixture
def workflow_path():
    return Path("some_path") / "workflow.yaml"


@pytest.fixture
def agent(workflow_path):
    return NatAgent(workflow_path=workflow_path)


@patch.dict(os.environ, {}, clear=True)
def test_init_with_additional_kwargs(workflow_path):
    """Test initialization with additional keyword arguments."""
    # Setup
    additional_kwargs = {"extra_param1": "value1", "extra_param2": 42}

    # Execute
    agent = NatAgent(workflow_path=workflow_path, **additional_kwargs)

    # Verify that the extra parameters don't create attributes
    with pytest.raises(AttributeError):
        _ = agent.extra_param1


async def test_run_method(agent, workflow_path):
    # Patch the run_nat_workflow method
    with patch.object(NatAgent, "run_nat_workflow", return_value="success"):
        # Call the run method with test inputs
        completion_create_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Artificial Intelligence"}],
            "environment_var": True,
        }
        result, pipeline_interactions, usage = await agent.invoke(completion_create_params)

        # Verify run_nat_workflow was called with the right inputs
        agent.run_nat_workflow.assert_called_once_with(
            workflow_path,
            "Artificial Intelligence",
        )

        assert result == "success"
        assert pipeline_interactions is None
        assert usage == {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
