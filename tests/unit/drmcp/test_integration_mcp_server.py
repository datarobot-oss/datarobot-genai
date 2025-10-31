"""Tests for integration_mcp_server.py."""

from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import datarobot_genai.drmcp.test_utils.integration_mcp_server as integration_module
from datarobot_genai.drmcp.test_utils.integration_mcp_server import detect_user_modules
from datarobot_genai.drmcp.test_utils.integration_mcp_server import main


class TestIntegrationMCPServer:
    """Test cases for integration_mcp_server.py."""

    def test_detect_user_modules_no_app_directory(self):
        """Test detect_user_modules when no app directory exists."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/some/path")

            # Mock that app directories don't exist
            with patch("pathlib.Path.exists", return_value=False):
                result = detect_user_modules()
                assert result is None

    @patch("datarobot_genai.drmcp.test_utils.integration_mcp_server.create_mcp_server")
    @patch("datarobot_genai.drmcp.test_utils.integration_mcp_server.detect_user_modules")
    def test_main_with_user_modules(self, mock_detect, mock_create_server):
        """Test main function when user modules are detected."""
        # Mock user components
        mock_config_factory = Mock()
        mock_credentials_factory = Mock()
        mock_lifecycle = Mock()
        mock_module_paths = [("path1", "module1"), ("path2", "module2")]

        mock_detect.return_value = (
            mock_config_factory,
            mock_credentials_factory,
            mock_lifecycle,
            mock_module_paths,
        )

        mock_server = Mock()
        mock_create_server.return_value = mock_server

        main()

        mock_create_server.assert_called_once_with(
            config_factory=mock_config_factory,
            credentials_factory=mock_credentials_factory,
            lifecycle=mock_lifecycle,
            additional_module_paths=mock_module_paths,
            transport="stdio",
        )
        mock_server.run.assert_called_once()

    @patch("datarobot_genai.drmcp.test_utils.integration_mcp_server.create_mcp_server")
    @patch("datarobot_genai.drmcp.test_utils.integration_mcp_server.detect_user_modules")
    def test_main_without_user_modules(self, mock_detect, mock_create_server):
        """Test main function when no user modules are detected."""
        mock_detect.return_value = None

        mock_server = Mock()
        mock_create_server.return_value = mock_server

        main()

        mock_create_server.assert_called_once_with(transport="stdio")
        mock_server.run.assert_called_once()

    def test_main_module_execution(self):
        """Test that the module can be executed as main."""
        # Check that main function exists
        assert hasattr(integration_module, "main")
        assert callable(integration_module.main)

        # Check that detect_user_modules function exists
        assert hasattr(integration_module, "detect_user_modules")
        assert callable(integration_module.detect_user_modules)

    def test_detect_user_modules_basic_functionality(self):
        """Test basic functionality of detect_user_modules."""
        # Test with no app directory
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/some/path")

            with patch("pathlib.Path.exists", return_value=False):
                result = detect_user_modules()
                assert result is None

    def test_detect_user_modules_path_search(self):
        """Test that detect_user_modules searches multiple directories."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/some/path")

            # Mock that no app directories exist
            with patch("pathlib.Path.exists", return_value=False):
                result = detect_user_modules()
                assert result is None
