# Copyright 2026 DataRobot, Inc.
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
from enum import Enum
from enum import auto


class LRSEnvVarIsNotSetError(Exception):
    """Exception raised when env var is missing in LRS container."""


class LRSEnvVars(Enum):
    MLOPS_DEPLOYMENT_ID = auto()

    def to_env_name(self) -> str:
        return self.name

    def get_os_env_value(self) -> str:
        env_value = os.getenv(self.to_env_name())
        if not env_value:
            error_message = f"Env var {self} is not assigned in the LRS container"
            raise LRSEnvVarIsNotSetError(error_message)
        return env_value
