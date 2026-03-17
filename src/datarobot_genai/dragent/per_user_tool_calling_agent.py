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

"""Per-user variant of the NAT tool_calling_agent workflow.

The built-in ``tool_calling_agent`` is registered as a *shared* workflow, which
means NAT's dependency validator forbids it from referencing per-user function
groups such as ``a2a_client``.  This module registers an identical workflow under
the name ``per_user_tool_calling_agent`` using ``register_per_user_function`` so
that per-user function groups can be used while still benefiting from OpenAI-style
structured tool calling (``bind_tools``).

``tool_calling_agent_workflow.__wrapped__`` is the raw async generator function
before ``@register_function`` wrapped it with ``asynccontextmanager``.
``register_per_user_function`` re-wraps it with ``asynccontextmanager`` internally,
so no implementation needs to be duplicated here.
"""

from nat.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from nat.agent.tool_calling_agent.register import tool_calling_agent_workflow
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse


class PerUserToolCallAgentWorkflowConfig(
    ToolCallAgentWorkflowConfig,
    name="per_user_tool_calling_agent",  # type: ignore[call-arg]
):
    """Per-user version of tool_calling_agent.

    Identical to ``tool_calling_agent`` in every way except that it is instantiated
    once per user, which allows per-user function groups (e.g. ``a2a_client``) to be
    listed in ``tool_names``.
    """

    pass  # Inherits all fields from ToolCallAgentWorkflowConfig


# Re-register the raw build function (.__wrapped__ is the original async generator
# before asynccontextmanager wrapped it) as a per-user function under the new config type.
register_per_user_function(
    config_type=PerUserToolCallAgentWorkflowConfig,
    input_type=ChatRequest,
    single_output_type=ChatResponse,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)(tool_calling_agent_workflow.__wrapped__)
