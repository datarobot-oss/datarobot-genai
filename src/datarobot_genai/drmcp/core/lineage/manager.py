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
from datarobot_genai.drmcp.core.clients import get_api_client as get_datarobot_client
from datarobot_genai.drmcp.core.feature_flags import FeatureFlag
from datarobot_genai.drmcp.core.lineage.enums import LRSEnvVars


class LineageManager:
    def __init__(self) -> None:
        self.datarobot_client = get_datarobot_client()
        self.feature_flag_enabled = FeatureFlag.create("ENABLE_MCP_TOOLS_GALLERY_SUPPORT").enabled
        self.custom_model_deployment_id = LRSEnvVars.MLOPS_DEPLOYMENT_ID.get_os_env_value()
        self.custom_model_version_id = LRSEnvVars.MLOPS_MODEL_ID.get_os_env_value()
