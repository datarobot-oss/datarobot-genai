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

from unittest.mock import Mock

from datarobot_genai.llama_index.workflow_runtime import apply_agent_workflow_injected_params
from datarobot_genai.llama_index.workflow_runtime import apply_function_agent_injected_params


def test_apply_agent_workflow_injected_params_applies_known_keys() -> None:
    wf = Mock()
    wf.early_stopping_method = "force"
    apply_agent_workflow_injected_params(
        wf,
        {"timeout": 42.0, "verbose": True, "early_stopping_method": "generate"},
    )
    assert wf._timeout == 42.0
    assert wf._verbose is True
    assert wf.early_stopping_method == "generate"


def test_apply_agent_workflow_injected_params_none_is_noop() -> None:
    wf = Mock()
    apply_agent_workflow_injected_params(wf, None)
    assert wf.mock_calls == []


def test_apply_function_agent_injected_params_applies_known_keys() -> None:
    agent = Mock()
    apply_function_agent_injected_params(
        agent,
        {
            "timeout": 3.0,
            "verbose": False,
            "streaming": True,
            "early_stopping_method": "generate",
            "initial_tool_choice": "auto",
            "allow_parallel_tool_calls": False,
        },
    )
    assert agent._timeout == 3.0
    assert agent._verbose is False
    assert agent.streaming is True
    assert agent.early_stopping_method == "generate"
    assert agent.initial_tool_choice == "auto"
    assert agent.allow_parallel_tool_calls is False


def test_apply_function_agent_injected_params_skips_none_values() -> None:
    class _Agent:
        pass

    agent = _Agent()
    apply_function_agent_injected_params(
        agent,
        {"timeout": None, "streaming": True},
    )
    assert not hasattr(agent, "_timeout")
    assert agent.streaming is True
