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

"""Tests for drtools.core.tool_metadata — the registry + gallery UI metadata view."""

from unittest.mock import patch

from datarobot_genai.drtools.core.tool_metadata import get_tool_ui_metadata

REGISTRY = "datarobot_genai.drtools.core.tool_metadata.get_registered_tools"


class TestGetToolUiMetadata:
    """``get_tool_ui_metadata`` is injected into the tools-gallery route as the provider."""

    def test_reads_private_keys_from_registry(self) -> None:
        def my_tool() -> None:
            pass

        fake_registry = [
            (
                my_tool,
                {
                    "name": "my_tool",
                    "display_name": "My Tool",
                    "description_ui": "Does a thing.",
                    "auth_provider": "jira",
                },
            )
        ]
        with patch(REGISTRY, return_value=fake_registry):
            lookup = get_tool_ui_metadata()
        assert lookup["my_tool"] == {
            "display_name": "My Tool",
            "description_ui": "Does a thing.",
            "auth_provider": "jira",
        }

    def test_falls_back_to_func_name_when_no_name_key(self) -> None:
        def fallback_tool() -> None:
            pass

        with patch(REGISTRY, return_value=[(fallback_tool, {"display_name": "X"})]):
            lookup = get_tool_ui_metadata()
        assert "fallback_tool" in lookup

    def test_absent_ui_keys_become_none(self) -> None:
        def bare_tool() -> None:
            pass

        with patch(REGISTRY, return_value=[(bare_tool, {"name": "bare_tool"})]):
            lookup = get_tool_ui_metadata()
        assert lookup["bare_tool"] == {
            "display_name": None,
            "description_ui": None,
            "auth_provider": None,
        }

    def test_empty_registry_returns_empty_lookup(self) -> None:
        with patch(REGISTRY, return_value=[]):
            assert get_tool_ui_metadata() == {}
