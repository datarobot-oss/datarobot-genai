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
from datarobot_genai.drmcpbase.datarobot_services.client import DataRobotClientWithAsyncAPI


async def is_mcp_tools_gallery_support_enabled(
    datarobot_api_endpoint: str,
    datarobot_user_bear_token: str,
) -> bool:
    async with DataRobotClientWithAsyncAPI(datarobot_api_endpoint) as datarobot_client:
        return await datarobot_client._is_feature_flag_enabled(
            "ENABLE_MCP_TOOLS_GALLERY_SUPPORT",
            datarobot_user_bear_token,
        )


async def get_feature_flag_enablement_with_existing_datarobot_client(
    datarobot_api_client: DataRobotClientWithAsyncAPI,
    datarobot_user_bear_token: str,
    feature_flag_name: str,
) -> bool:
    return await datarobot_api_client._is_feature_flag_enabled(
        feature_flag_name,
        datarobot_user_bear_token,
    )
