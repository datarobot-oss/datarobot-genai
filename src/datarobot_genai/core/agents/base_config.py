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


from datarobot.core.config import DataRobotAppFrameworkBaseSettings


class DRAgentConfig(DataRobotAppFrameworkBaseSettings):
    """Base Agent Config class for Agents with common fields.

    These can be overridden globally via environment variable.
    """

    api_key: str | None = None
    api_base: str | None = None
    model: str | None = None
    timeout: int = 90
    use_datarobot_llm_gateway: bool = False
    mcp_deployment_id: str | None = None
    external_mcp_url: str | None = None
    verbose: bool = False
