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
from async_lru import alru_cache

from datarobot_genai.drmcpbase.datarobot_services.feature_flags import (
    is_mcp_tools_gallery_support_enabled,
)
from datarobot_genai.drmcputils.credentials import get_credentials


class FeatureFlag:
    @staticmethod
    @alru_cache(maxsize=1)
    async def is_mcp_tools_gallery_support_enabled_for_static_mcp_container_user() -> bool:
        credentials = get_credentials()
        dr_api_endpoint = credentials.datarobot.datarobot_endpoint
        dr_api_token_of_static_account_in_mcp_container = credentials.datarobot.datarobot_api_token

        return await is_mcp_tools_gallery_support_enabled(
            dr_api_endpoint, dr_api_token_of_static_account_in_mcp_container
        )
