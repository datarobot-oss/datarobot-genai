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
from typing import Any

import httpx

from datarobot_genai.dragent.http_client import get_retriable_async_http_client
from datarobot_genai.dragent.plugins.xaa_auth import XAAParams
from datarobot_genai.dragent.plugins.xaa_auth import XAAStepOneTokenExchangeParams
from datarobot_genai.dragent.plugins.xaa_auth import XAAStepTwoTokenRequestParams


def parse_xaa_params_from_mcp_auth_server_metadata(
    mcp_auth_server_metadata: dict[str, Any],
) -> XAAParams:
    xaa_metadata = mcp_auth_server_metadata["urn:datarobot:nat_mcp_xaa_client"]
    token_exchange_metata = xaa_metadata["tokenExchange"]
    token_request_metata = xaa_metadata["tokenRequest"]

    return XAAParams(
        XAAStepOneTokenExchangeParams(
            token_exchange_metata["trustedIssuer"],
            token_exchange_metata["audience"],
        ),
        XAAStepTwoTokenRequestParams(
            token_request_metata["tokenUrl"],
            token_request_metata["audience"],
            token_request_metata["scopes"],
        ),
    )


async def get_xaa_param_from_mcp_auth_server_metadata(
    mcp_auth_server_metadata_url: str,
) -> XAAParams:
    async with get_retriable_async_http_client() as http_client:
        try:
            resp = await http_client.get(mcp_auth_server_metadata_url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Failed to fetch MCP auth server metadata from "
                f"{mcp_auth_server_metadata_url}: {exc}"
            )

    return parse_xaa_params_from_mcp_auth_server_metadata(resp.json())
