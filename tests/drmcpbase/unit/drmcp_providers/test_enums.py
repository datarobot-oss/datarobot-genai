# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# DataRobot, Inc. Confidential.
#
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
#
# The copyright notice above does not evidence any actual or intended
# publication of such source code.
import pytest

from datarobot_genai.drmcpbase.drmcp_providers.enums import ProviderNamespace


class TestProviderNamespace:
    @pytest.mark.parametrize("enum_item", [enum_item for enum_item in ProviderNamespace])
    def test_to_value(self, enum_item: ProviderNamespace) -> None:
        expected_enum_values = {
            ProviderNamespace.DATAROBOT_USER_MCP: "datarobot-user-mcp",
        }
        assert enum_item.to_qualified_value() == expected_enum_values[enum_item]
