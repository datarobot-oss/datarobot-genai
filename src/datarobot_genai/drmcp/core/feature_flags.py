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

from datarobot_genai.drmcp.core.clients import get_api_client as get_datarobot_client


@dataclass
class FeatureFlag:
    name: str
    enabled: bool

    @classmethod
    def create(cls, feature_flag_name: str) -> "FeatureFlag":
        client = get_datarobot_client()
        flags_json = {"entitlements": [{"name": feature_flag_name}]}
        response = client.post("entitlements/evaluate/", json=flags_json)

        feature_flag_info = response.json()["entitlements"][0]
        return cls(
            name=feature_flag_info["name"],
            enabled=bool(feature_flag_info["value"]),
        )
