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
from dataclasses import dataclass
from functools import lru_cache

from datarobot_genai.drmcp.core.clients import (
    setup_and_return_dr_api_client_with_static_config_in_container,
)


@dataclass
class FeatureFlag:
    name: str
    enabled: bool

    @classmethod
    def create(cls, feature_flag_name: str) -> "FeatureFlag":
        client = setup_and_return_dr_api_client_with_static_config_in_container()
        flags_json = {"entitlements": [{"name": feature_flag_name}]}
        response = client.post("entitlements/evaluate/", json=flags_json)

        json_response = response.json()
        has_entitlement_return = "entitlements" in json_response
        return cls(
            name=feature_flag_name,
            enabled=bool(json_response["entitlements"][0]["value"])
            if has_entitlement_return
            else False,
        )

    @classmethod
    @lru_cache(maxsize=1)
    def is_mcp_tools_gallery_support_enabled(cls) -> bool:
        return cls.create("ENABLE_MCP_TOOLS_GALLERY_SUPPORT").enabled
