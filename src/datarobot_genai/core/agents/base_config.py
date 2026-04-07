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
from nat.data_models.agent import AgentBaseConfig

from datarobot_genai.core.memory.base import BaseMemoryClient

DEFAULT_MAX_HISTORY_MESSAGES = 20


class BaseAgentConfig(DataRobotAppFrameworkBaseSettings, AgentBaseConfig):
    api_key: str | None = None
    api_base: str | None = None
    model: str | None = None
    timeout: int | None = 90
    forwarded_headers: dict[str, str] | None = None
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
    memory_client: BaseMemoryClient | None = None
