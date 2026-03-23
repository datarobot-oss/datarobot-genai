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


class AgentConfig:
    pass


# api_key: str | None = (None,)
# api_base: str | None = (None,)
# model: str | None = (None,)
# verbose: bool | str | None = (True,)
# timeout: int | None = (90,)
# authorization_context: dict[str, Any] | None = (None,)
# forwarded_headers: dict[str, str] | None = (None,)
# max_history_messages: int | None = (None,)

# name: str | None = Field(
#     default=None,
#     description="Optional display name for this function. Used in tracing and observability.",
# )
# middleware: list[str] = Field(
#     default_factory=list,
#     description="List of function middleware names to apply to this function in order",
# )
# description: str = Field(
#     default="ReAct Agent Workflow", description="The description of this functions use."
# )
# workflow_alias: str | None = Field(
#     default=None,
#     description=(
#         "The alias of the workflow. Useful when the agent is configured as a workflow "
#         "and needs to expose a customized name as a tool."
#     ),
# )
# llm_name: LLMRef = Field(description="The LLM model to use with the agent.")
# verbose: bool = Field(default=False, description="Set the verbosity of the agent's logging.")
# description: str = Field(description="The description of this function's use.")
# log_response_max_chars: PositiveInt = Field(
#     default=1000,
#     description="Maximum number of characters to display in logs when logging responses.",
# )
