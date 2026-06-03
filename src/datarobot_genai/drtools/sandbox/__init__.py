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

"""Sandbox abstraction for offloading code execution to isolated containers.

The :class:`Sandbox` Protocol is the only thing call sites should depend on,
allowing implementations to swap freely (workload-api for production, and
future backends — e.g. a local-Docker dev/test sandbox) without code churn.
"""

from datarobot_genai.drtools.sandbox.base import Sandbox
from datarobot_genai.drtools.sandbox.base import SandboxError
from datarobot_genai.drtools.sandbox.base import SandboxResult
from datarobot_genai.drtools.sandbox.base import SandboxSecurityContext
from datarobot_genai.drtools.sandbox.base import SandboxTimeout
from datarobot_genai.drtools.sandbox.tools import MCP_SANDBOX_FEATURE_FLAG
from datarobot_genai.drtools.sandbox.tools import execute_code
from datarobot_genai.drtools.sandbox.workload import DataRobotWorkloadSandbox

__all__ = [
    "MCP_SANDBOX_FEATURE_FLAG",
    "DataRobotWorkloadSandbox",
    "Sandbox",
    "SandboxError",
    "SandboxResult",
    "SandboxSecurityContext",
    "SandboxTimeout",
    "execute_code",
]
