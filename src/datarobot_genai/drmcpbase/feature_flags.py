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
import logging

from datarobot_genai.drmcpbase.auth.enums import DataRobotBearerHeaderEnum
from datarobot_genai.drmcpbase.auth.exceptions import InvalidBearerTokenError
from datarobot_genai.drmcpbase.auth.exceptions import (
    NoDataRobotBearerTokenFoundInRequestContextError,
)
from datarobot_genai.drmcpbase.auth.exceptions import NoHeadersFoundInRequestContextError
from datarobot_genai.drmcpbase.datarobot_services.client import DataRobotClientWithAsyncAPI
from datarobot_genai.drmcpbase.datarobot_services.feature_flags import (
    get_feature_flag_enablement_with_existing_datarobot_client,
)

logger = logging.getLogger(__name__)


class FeatureFlagEvaluation:
    @staticmethod
    async def is_mcp_tools_gallery_support_enabled_for_user_in_mcp_request(
        datarobot_api_client: DataRobotClientWithAsyncAPI,
    ) -> bool:
        try:
            datarobot_api_token = (
                DataRobotBearerHeaderEnum.X_DATAROBOT_AUTHORIZATION.get_from_mcp_request()
            )
            return await get_feature_flag_enablement_with_existing_datarobot_client(
                datarobot_api_client,
                datarobot_api_token,
                "ENABLE_MCP_TOOLS_GALLERY_SUPPORT",
            )
        except (
            NoDataRobotBearerTokenFoundInRequestContextError,
            NoHeadersFoundInRequestContextError,
            InvalidBearerTokenError,
        ):
            logger.info(
                "No bearer token or valid header is found. Feature flag is evaluated to False."
            )
            return False


async def check_mcp_tools_gallery_support(
    datarobot_api_client: DataRobotClientWithAsyncAPI | None,
) -> bool:
    """Return True when the MCP tools-gallery feature is enabled for the request's user.

    Shared entry point for everything gated on ``ENABLE_MCP_TOOLS_GALLERY_SUPPORT``: the
    tool providers (which pass their lifespan client) and the global-mcp ``/toolGallery/tools/``
    route gate (which passes a short-lived client). None-safe — returns False when the
    client is not yet available (e.g. a request that races a provider's lifespan), so
    callers no longer need a separate ``is_datarobot_api_client_initialized`` guard.
    Otherwise it delegates to
    :meth:`FeatureFlagEvaluation.is_mcp_tools_gallery_support_enabled_for_user_in_mcp_request`,
    which reads the user's bearer token from the request context and fails closed to False.
    """
    if datarobot_api_client is None:
        return False
    return await FeatureFlagEvaluation.is_mcp_tools_gallery_support_enabled_for_user_in_mcp_request(
        datarobot_api_client
    )
