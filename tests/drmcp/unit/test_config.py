# Copyright 2025 DataRobot, Inc.
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
import os
from typing import get_args
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core import config as config_module
from datarobot_genai.drmcp.core.config import MCPServerConfig
from datarobot_genai.drmcp.core.mcp_instance import TaggedFastMCP


def test_config_defaults() -> None:
    """Test that the important default configuration values are as expected."""
    # Clear any environment variables that might override defaults
    with patch.dict(os.environ, clear=True):
        # Clear the cached config instance to ensure we get a fresh one
        config_module._config = None

        # Create a new config instance without loading from .env file
        config = MCPServerConfig(_env_file=None)

        # Dynamic tools registration should be disabled by default
        # as it can cause startup delays and is not always desired.
        assert config.mcp_server_register_dynamic_tools_on_startup is False

        # The default behavior for duplicate tool registrations is aligned
        # with FastMCP default.
        assert config.tool_registration_duplicate_behavior == "warn"

        # Clean up the cached config after the test
        config_module._config = None


class TestDuplicateBehavior:
    def test_allowed_duplicate_behaviors(self) -> None:
        """Test that the allowed duplicate behaviors are as expecte d."""
        expected_behaviors = {"error", "warn", "ignore", "replace"}

        # Get the type annotation from the model field to check if the
        # allowed values match, if there are any changes in the future,
        # please review the tool registration logic to ensure it still works
        # as intended.
        field_info = MCPServerConfig.model_fields["tool_registration_duplicate_behavior"]
        allowed_behaviors = set(get_args(field_info.annotation))

        assert expected_behaviors == allowed_behaviors

    @pytest.mark.parametrize("value", ["error", "warn", "replace", "ignore"])
    def test_setting_is_propagated_correctly(self, value) -> None:
        """Test that setting the duplicate behavior is propagated correctly to the tool manager."""
        env_var = "MCP_SERVER_TOOL_REGISTRATION_DUPLICATE_BEHAVIOR"

        with patch.dict(os.environ, {env_var: value}, clear=False):
            # Create a fresh config instance that reads from the updated
            # environment without affecting the global mcp instance
            # shared across tests.
            config = MCPServerConfig()

            test_mcp = TaggedFastMCP(
                name=config.mcp_server_name,
                port=config.mcp_server_port,
                log_level=config.mcp_server_log_level,
                host=config.mcp_server_host,
                stateless_http=True,
                on_duplicate_tools=config.tool_registration_duplicate_behavior,
            )

            # Verify the setting was applied correctly
            assert test_mcp._tool_manager.duplicate_behavior == value
