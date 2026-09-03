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
from __future__ import annotations

import os
from enum import Enum
from enum import auto


class MCPDeploymentType(Enum):
    MLOPS = auto()
    WORKLOAD = auto()


class DeploymentConfig(Enum):
    WORKLOAD_ID = auto()
    MLOPS_DEPLOYMENT_ID = auto()
    DATAROBOT_PUBLIC_API_ENDPOINT = auto()
    DATAROBOT_ENDPOINT = auto()
    DR_WORKLOAD_EXTERNAL_URL_HOST = auto()

    def get_from_os_env(self) -> str | None:
        return os.getenv(self.name, "").strip() or None

    @staticmethod
    def get_datarobot_public_api_endpoint() -> str | None:
        """
        ``DATAROBOT_PUBLIC_API_ENDPOINT`` wins over ``DATAROBOT_ENDPOINT``
        An on-prem install commonly points ``DATAROBOT_ENDPOINT`` at an internal
        cluster address. A resource identifier built from it is one no external
        client can reach, and RFC 9728 §7.3 has the client compare the metadata URL
        it fetched against ``resource``, so a wrong value fails discovery outright.

        When neither variable is set this returns ``None`` rather than guessing a
        default host. The value is published as this server's identity, and a
        confidently wrong identity is worse than an absent one.
        """
        return (
            DeploymentConfig.DATAROBOT_PUBLIC_API_ENDPOINT.get_from_os_env()
            or DeploymentConfig.DATAROBOT_ENDPOINT.get_from_os_env()
        )
