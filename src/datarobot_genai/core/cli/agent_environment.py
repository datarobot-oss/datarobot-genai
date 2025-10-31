# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from datarobot_genai.core.cli.agent_kernel import AgentKernel


class AgentEnvironment:
    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
    ):
        self.api_token = os.environ.get("DATAROBOT_API_TOKEN") or api_token
        if not self.api_token:
            raise ValueError(
                "Missing DataRobot API token. Set the DATAROBOT_API_TOKEN "
                "environment variable or provide it explicitly."
            )
        self.base_url = (
            os.environ.get("DATAROBOT_ENDPOINT") or base_url or "https://app.datarobot.com"
        )
        if not self.base_url:
            raise ValueError(
                "Missing DataRobot endpoint. Set the DATAROBOT_ENDPOINT environment "
                "variable or provide it explicitly."
            )
        self.base_url = self.base_url.replace("/api/v2", "")

    @property
    def interface(self) -> AgentKernel:
        return AgentKernel(
            api_token=str(self.api_token),
            base_url=str(self.base_url),
        )
