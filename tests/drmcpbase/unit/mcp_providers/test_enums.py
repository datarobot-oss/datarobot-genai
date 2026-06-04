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
import pytest

from datarobot_genai.drmcpbase.mcp_providers.enums import ProviderNamespace


class TestProviderNamespace:
    @pytest.mark.parametrize("enum_item", [enum_item for enum_item in ProviderNamespace])
    def test_to_value(self, enum_item: ProviderNamespace) -> None:
        expected_enum_values = {
            ProviderNamespace.DATAROBOT_USER_MCP: "datarobot-user-mcp",
        }
        assert enum_item.to_qualified_value() == expected_enum_values[enum_item]
