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

from nat.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig

from datarobot_genai.dragent.per_user_tool_calling_agent import PerUserToolCallAgentWorkflowConfig


class TestPerUserToolCallAgentWorkflowConfig:
    def test_is_subclass_of_tool_call_agent_workflow_config(self):
        assert issubclass(PerUserToolCallAgentWorkflowConfig, ToolCallAgentWorkflowConfig)

    def test_registered_name(self):
        assert PerUserToolCallAgentWorkflowConfig.static_type() == "per_user_tool_calling_agent"

    def test_inherits_all_fields_from_parent(self):
        parent_fields = set(ToolCallAgentWorkflowConfig.model_fields) - {"type"}
        child_fields = set(PerUserToolCallAgentWorkflowConfig.model_fields) - {"type"}
        assert parent_fields == child_fields

    def test_default_instantiation(self):
        config = PerUserToolCallAgentWorkflowConfig(llm_name="gpt-4o")
        assert config is not None

    def test_registered_as_per_user_function(self):
        """Importing the module registers a per-user function; verify no exception is raised."""
        import datarobot_genai.dragent.per_user_tool_calling_agent  # noqa: F401
