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

"""Tests for GitHub tools dynamic registration."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.tools.github.register import MANIFEST_SCHEMA
from datarobot_genai.drmcp.tools.github.register import _register_tools
from datarobot_genai.drmcp.tools.github.register import load_manifest
from datarobot_genai.drmcp.tools.github.register import register_github_tools
from datarobot_genai.drmcp.tools.github.register import update_manifest_cache


class TestLoadManifest:
    """Test load_manifest function."""

    def test_load_valid_manifest(self, tmp_path: Path) -> None:
        """Test loading a valid manifest file."""
        manifest_data = {
            "$schema": MANIFEST_SCHEMA,
            "metadata": {"tool_count": 2},
            "tools": [
                {"name": "get_me", "description": "Get user", "inputSchema": {}, "enabled": True},
                {
                    "name": "create_issue",
                    "description": "Create issue",
                    "inputSchema": {},
                    "enabled": True,
                },
            ],
        }
        manifest_path = tmp_path / "github_tools.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        tools = load_manifest(manifest_path)
        assert len(tools) == 2
        assert tools[0]["name"] == "get_me"
        assert tools[1]["name"] == "create_issue"

    def test_load_manifest_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised when manifest doesn't exist."""
        manifest_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_manifest(manifest_path)

    def test_load_manifest_warns_on_schema_mismatch(self, tmp_path: Path) -> None:
        """Test that a warning is logged when schema doesn't match."""
        manifest_data = {
            "$schema": "wrong_schema",
            "tools": [{"name": "test", "description": "test", "inputSchema": {}}],
        }
        manifest_path = tmp_path / "github_tools.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        # Should not raise, but should log a warning
        tools = load_manifest(manifest_path)
        assert len(tools) == 1


class TestUpdateManifestCache:
    """Test update_manifest_cache function."""

    def test_creates_manifest_file(self, tmp_path: Path) -> None:
        """Test that manifest file is created."""
        tool_defs = [
            {"name": "get_me", "description": "Get user", "inputSchema": {}},
            {"name": "create_issue", "description": "Create issue", "inputSchema": {}},
        ]
        manifest_path = tmp_path / "github_tools.json"

        update_manifest_cache(tool_defs, manifest_path)

        assert manifest_path.exists()
        with open(manifest_path) as f:
            data = json.load(f)
        assert data["$schema"] == MANIFEST_SCHEMA
        assert len(data["tools"]) == 2
        assert data["metadata"]["tool_count"] == 2

    def test_preserves_enabled_status(self, tmp_path: Path) -> None:
        """Test that enabled status is preserved from existing manifest."""
        # Create initial manifest with one tool disabled
        existing_manifest = {
            "$schema": MANIFEST_SCHEMA,
            "tools": [
                {"name": "get_me", "enabled": True},
                {"name": "create_issue", "enabled": False},
            ],
        }
        manifest_path = tmp_path / "github_tools.json"
        with open(manifest_path, "w") as f:
            json.dump(existing_manifest, f)

        # Update with new tool definitions
        new_tools = [
            {"name": "get_me", "description": "Get user", "inputSchema": {}},
            {"name": "create_issue", "description": "Create issue", "inputSchema": {}},
            {"name": "new_tool", "description": "New tool", "inputSchema": {}},
        ]

        update_manifest_cache(new_tools, manifest_path)

        with open(manifest_path) as f:
            data = json.load(f)

        # Check enabled status preserved
        tools_by_name = {t["name"]: t for t in data["tools"]}
        assert tools_by_name["get_me"]["enabled"] is True
        assert tools_by_name["create_issue"]["enabled"] is False
        assert tools_by_name["new_tool"]["enabled"] is True  # Default for new tools


class TestRegisterTools:
    """Test _register_tools function."""

    def test_registers_enabled_tools(self) -> None:
        """Test that enabled tools are registered."""
        tool_defs = [
            {"name": "get_me", "description": "Get user", "inputSchema": {}, "enabled": True},
            {
                "name": "create_issue",
                "description": "Create issue",
                "inputSchema": {},
                "enabled": True,
            },
        ]

        with patch("datarobot_genai.drmcp.tools.github.register.mcp") as mock_mcp:
            count = _register_tools(tool_defs)

        assert count == 2
        assert mock_mcp.add_tool.call_count == 2

    def test_skips_disabled_tools(self) -> None:
        """Test that disabled tools are not registered."""
        tool_defs = [
            {"name": "get_me", "description": "Get user", "inputSchema": {}, "enabled": True},
            {
                "name": "dangerous_tool",
                "description": "Dangerous",
                "inputSchema": {},
                "enabled": False,
            },
        ]

        with patch("datarobot_genai.drmcp.tools.github.register.mcp") as mock_mcp:
            count = _register_tools(tool_defs)

        assert count == 1
        assert mock_mcp.add_tool.call_count == 1


class TestRegisterGitHubTools:
    """Test register_github_tools function."""

    def test_loads_from_manifest(self, tmp_path: Path) -> None:
        """Test that tools are loaded from manifest."""
        manifest_data = {
            "$schema": MANIFEST_SCHEMA,
            "tools": [
                {"name": "get_me", "description": "Get user", "inputSchema": {}, "enabled": True},
            ],
        }
        manifest_path = tmp_path / "github_tools.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        with patch("datarobot_genai.drmcp.tools.github.register.mcp"):
            count = register_github_tools(manifest_path)

        assert count == 1

    def test_returns_zero_when_manifest_not_found(self, tmp_path: Path) -> None:
        """Test that 0 is returned when manifest is not found."""
        manifest_path = tmp_path / "nonexistent.json"

        count = register_github_tools(manifest_path)

        assert count == 0

    def test_returns_zero_on_invalid_manifest(self, tmp_path: Path) -> None:
        """Test that 0 is returned when manifest is invalid JSON."""
        manifest_path = tmp_path / "github_tools.json"
        with open(manifest_path, "w") as f:
            f.write("not valid json")

        count = register_github_tools(manifest_path)

        assert count == 0
