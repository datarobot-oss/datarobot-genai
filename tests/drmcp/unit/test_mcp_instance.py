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

from collections.abc import Iterator
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import NotFoundError

from datarobot_genai.drmcp.core.enums import DataRobotMCPToolCategory
from datarobot_genai.drmcp.core.mcp_instance import DataRobotMCP
from datarobot_genai.drmcp.core.mcp_instance import PromptInitArguments
from datarobot_genai.drmcp.core.mcp_instance import ResourceInitArguments
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_integration_tool
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_prompt
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_resource
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.core.mcp_instance import update_mcp_tool_init_args_with_tool_category


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.drmcp.core.mcp_instance"


class TestDataRobotMCPInstanceAdditional:
    """Additional test cases for DataRobotMCP class."""

    @pytest.mark.asyncio
    async def test_set_deployment_mapping_updates_existing(self):
        """Test that set_deployment_mapping updates existing mapping and removes old tool."""
        mcp = DataRobotMCP()
        mcp._deployments_map = {"deployment1": "old_tool"}

        with patch.object(mcp, "remove_tool") as mock_remove_tool:
            await mcp.set_deployment_mapping("deployment1", "new_tool")

            assert mcp._deployments_map["deployment1"] == "new_tool"
            mock_remove_tool.assert_called_once_with("old_tool")

    @pytest.mark.asyncio
    async def test_set_deployment_mapping_handles_remove_tool_not_found(self):
        """Test that set_deployment_mapping handles NotFoundError when removing old tool."""
        mcp = DataRobotMCP()
        mcp._deployments_map = {"deployment1": "old_tool"}

        with patch.object(mcp, "remove_tool", side_effect=NotFoundError("Tool not found")):
            # Should not raise an exception
            await mcp.set_deployment_mapping("deployment1", "new_tool")

            assert mcp._deployments_map["deployment1"] == "new_tool"

    @pytest.mark.asyncio
    async def test_set_deployment_mapping_new_deployment(self):
        """Test that set_deployment_mapping works for new deployment."""
        mcp = DataRobotMCP()
        mcp._deployments_map = {}

        await mcp.set_deployment_mapping("deployment1", "new_tool")

        assert mcp._deployments_map["deployment1"] == "new_tool"

    @pytest.mark.asyncio
    async def test_set_deployment_mapping_same_tool(self):
        """Test that set_deployment_mapping works when mapping to same tool."""
        mcp = DataRobotMCP()
        mcp._deployments_map = {"deployment1": "existing_tool"}

        with patch.object(mcp, "remove_tool") as mock_remove_tool:
            await mcp.set_deployment_mapping("deployment1", "existing_tool")

            assert mcp._deployments_map["deployment1"] == "existing_tool"
            # Should not call remove_tool when mapping to same tool
            mock_remove_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_deployment_mapping_existing(self):
        """Test that remove_deployment_mapping removes existing mapping."""
        mcp = DataRobotMCP()
        mcp._deployments_map = {"deployment1": "tool1", "deployment2": "tool2"}

        await mcp.remove_deployment_mapping("deployment1")

        assert "deployment1" not in mcp._deployments_map
        assert "deployment2" in mcp._deployments_map

    @pytest.mark.asyncio
    async def test_remove_deployment_mapping_nonexistent(self):
        """Test that remove_deployment_mapping handles nonexistent deployment."""
        mcp = DataRobotMCP()
        mcp._deployments_map = {"deployment1": "tool1"}

        # Should not raise an exception
        await mcp.remove_deployment_mapping("nonexistent")

        assert mcp._deployments_map == {"deployment1": "tool1"}

    @pytest.mark.asyncio
    async def test_remove_deployment_mapping_empty_map(self):
        """Test that remove_deployment_mapping works with empty map."""
        mcp = DataRobotMCP()
        mcp._deployments_map = {}

        # Should not raise an exception
        await mcp.remove_deployment_mapping("deployment1")

        assert mcp._deployments_map == {}

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.mcp_instance.logger")
    async def test_set_deployment_mapping_logs_debug_message(self, mock_logger):
        """Test that set_deployment_mapping logs debug message when updating existing mapping."""
        mcp = DataRobotMCP()
        mcp._deployments_map = {"deployment1": "old_tool"}

        with patch.object(mcp, "remove_tool"):
            await mcp.set_deployment_mapping("deployment1", "new_tool")

            mock_logger.debug.assert_called_with(
                "Deployment ID deployment1 already mapped to old_tool, updating to new_tool"
            )

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.mcp_instance.logger")
    async def test_set_deployment_mapping_logs_remove_tool_not_found(self, mock_logger):
        """Test that set_deployment_mapping logs debug message when remove_tool raises NotFoundError."""  # noqa: E501
        mcp = DataRobotMCP()
        mcp._deployments_map = {"deployment1": "old_tool"}

        with patch.object(mcp, "remove_tool", side_effect=NotFoundError("Tool not found")):
            await mcp.set_deployment_mapping("deployment1", "new_tool")

            mock_logger.debug.assert_called_with(
                "Tool old_tool not found in registry, skipping removal"
            )


class TestMCPToolDecorator:
    @pytest.fixture
    def mock_dr_mcp_tool(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.dr_mcp_tool") as mock_decorator:
            yield mock_decorator

    @pytest.fixture
    def mock_dr_mcp_extras(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.dr_mcp_extras") as mock_decorator:
            yield mock_decorator

    @pytest.fixture
    def mock_datarobot_mcp_server_tool(self) -> Iterator[Mock]:
        with patch.object(DataRobotMCP, "tool") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_update_mcp_tool_init_args_with_tool_category(
        self, module_under_test: str
    ) -> Iterator[Mock]:
        with patch(
            f"{module_under_test}.update_mcp_tool_init_args_with_tool_category"
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_mcp_tool_callable(self) -> Iterator[Mock]:
        yield Mock(return_value=Mock())

    def test_update_mcp_tool_init_args_raises_error_if_tool_category_reserved_field_is_overridden(
        self,
    ) -> None:
        mock_tool_init_args = {"meta": {"tool_category": Mock()}}
        with pytest.raises(ValueError):
            update_mcp_tool_init_args_with_tool_category(Mock(), **mock_tool_init_args)

    @pytest.mark.parametrize(
        "tool_init_args",
        [{}, {"meta": {"1ewqa": "adfa"}}, {"meta": None}],
    )
    def test_update_mcp_tool_init_args_with_tool_category(
        self,
        tool_init_args: dict[str, str | None],
    ) -> None:
        expected_tool_category = Mock()
        updated_tool_init_args = update_mcp_tool_init_args_with_tool_category(
            expected_tool_category,
            **tool_init_args,
        )

        original_meta_arg = tool_init_args.get("meta") or {}
        assert updated_tool_init_args["meta"] == original_meta_arg | {
            "tool_category": expected_tool_category.name
        }

    def test_dr_mcp_tool(
        self,
        mock_mcp_tool_callable: Mock,
        mock_dr_mcp_extras: Mock,
        mock_datarobot_mcp_server_tool: Mock,
        mock_update_mcp_tool_init_args_with_tool_category: Mock,
    ) -> None:
        mock_mcp_tool_callable_args = {}
        decorator = dr_mcp_tool()
        decorator(mock_mcp_tool_callable)(**mock_mcp_tool_callable_args)

        mock_dr_mcp_extras.assert_called_with()
        mock_dr_mcp_extras_decorator = mock_dr_mcp_extras.return_value
        dr_mcp_extras_decorator_call_args = mock_dr_mcp_extras_decorator.call_args.args
        (inner_wrapper_func,) = dr_mcp_extras_decorator_call_args
        assert inner_wrapper_func.__qualname__ == "dr_mcp_tool.<locals>.decorator.<locals>.wrapper"

        mock_datarobot_mcp_server_tool.assert_called_once_with(
            **mock_update_mcp_tool_init_args_with_tool_category.return_value
        )
        mock_mcp_server_tool_callable = mock_datarobot_mcp_server_tool.return_value
        mock_mcp_server_tool_callable.assert_called_once_with(
            mock_dr_mcp_extras_decorator.return_value
        )

    @pytest.mark.usefixtures(
        "mock_dr_mcp_extras",
    )
    def test_dr_mcp_tool_decorator_with_tool_type_setup(
        self,
        mock_mcp_tool_callable: Mock,
        mock_datarobot_mcp_server_tool: Mock,
        mock_update_mcp_tool_init_args_with_tool_category: Mock,
    ) -> None:
        mock_mcp_tool_callable_args = {}
        mock_tool_type = Mock()
        decorator = dr_mcp_tool(tool_category=mock_tool_type)
        decorator(mock_mcp_tool_callable)(**mock_mcp_tool_callable_args)

        mock_update_mcp_tool_init_args_with_tool_category.assert_called_once_with(
            mock_tool_type,
            **mock_mcp_tool_callable_args,
        )
        mock_datarobot_mcp_server_tool.assert_called_once_with(
            **mock_update_mcp_tool_init_args_with_tool_category.return_value,
        )

    def test_dr_mcp_integration_tool(
        self,
        mock_mcp_tool_callable: Mock,
        mock_dr_mcp_tool: Mock,
    ) -> None:
        expected_kwarg_key = "sadfa"
        expected_kwarg_value = "23rew"
        mock_mcp_tool_callable_args = {expected_kwarg_key: expected_kwarg_value}
        decorator = dr_mcp_integration_tool(**mock_mcp_tool_callable_args)
        decorator(mock_mcp_tool_callable)(**mock_mcp_tool_callable_args)

        assert not mock_dr_mcp_tool.call_args.args
        assert mock_dr_mcp_tool.call_args.kwargs == {
            "tool_category": DataRobotMCPToolCategory.INTEGRATION_TOOL,
            expected_kwarg_key: expected_kwarg_value,
        }


class TestPromptInitArguments:
    def test_prompt_init_arguments_fail_when_setting_reserved_prompt_category(self) -> None:
        with pytest.raises(ValueError):
            PromptInitArguments(meta={"prompt_category": Mock()})

    def test_to_dict(self) -> None:
        prompt_init_args = PromptInitArguments()
        assert prompt_init_args.to_dict() == {
            "name": None,
            "title": None,
            "description": None,
            "icons": None,
            "tags": None,
            "enabled": None,
            "meta": None,
        }

    @pytest.mark.parametrize(
        "meta_arg",
        [{}, None, {"1ewqa": "adfa"}],
    )
    def test_set_prompt_category(self, meta_arg: dict[str, str | None] | None):
        prompt_init_args = PromptInitArguments(meta=meta_arg)

        expected_prompt_category = Mock()
        prompt_init_args.set_prompt_category(expected_prompt_category)

        original_meta_arg = meta_arg or {}
        expected_meta_arg = original_meta_arg | {"prompt_category": expected_prompt_category.name}
        assert prompt_init_args.meta == expected_meta_arg


class TestResourceInitArguments:
    def test_resource_init_arguments_fail_when_setting_reserved_resource_category(self) -> None:
        with pytest.raises(ValueError):
            ResourceInitArguments(uri="dsafds", meta={"resource_category": Mock()})

    def test_to_dict(self) -> None:
        expected_uri = "sdafsd"
        resource_init_args = ResourceInitArguments(uri=expected_uri)
        assert resource_init_args.to_dict() == {
            "uri": expected_uri,
            "name": None,
            "title": None,
            "description": None,
            "icons": None,
            "mime_type": None,
            "tags": None,
            "enabled": None,
            "annotations": None,
            "meta": None,
        }

    @pytest.mark.parametrize(
        "meta_arg",
        [{}, None, {"1ewqa": "adfa"}],
    )
    def test_set_resource_category(self, meta_arg: dict[str, str | None] | None):
        resource_init_args = ResourceInitArguments(uri="dafds", meta=meta_arg)

        expected_resource_category = Mock()
        resource_init_args.set_resource_category(expected_resource_category)

        original_meta_arg = meta_arg or {}
        expected_meta_arg = original_meta_arg | {
            "resource_category": expected_resource_category.name
        }
        assert resource_init_args.meta == expected_meta_arg


class TestMCPPromptDecorator:
    @pytest.fixture
    def mock_mcp_prompt_callable(self) -> Iterator[Mock]:
        yield Mock(return_value=Mock())

    @pytest.fixture
    def mock_datarobot_mcp_server_prompt(self) -> Iterator[Mock]:
        with patch.object(DataRobotMCP, "prompt") as mock_func:
            yield mock_func

    def test_dr_mcp_prompt(
        self,
        mock_mcp_prompt_callable: Mock,
        mock_datarobot_mcp_server_prompt: Mock,
    ) -> None:
        mock_mcp_prompt_callable_args = {}
        mock_prompt_category = Mock()
        mock_prompt_init_args = Mock()
        mock_prompt_init_args.to_dict.return_value = {}
        decorator = dr_mcp_prompt(mock_prompt_category, mock_prompt_init_args)
        decorator(mock_mcp_prompt_callable)(**mock_mcp_prompt_callable_args)

        mock_prompt_init_args.set_prompt_category.assert_called_once_with(mock_prompt_category)
        mock_datarobot_mcp_server_prompt.assert_called_once_with(
            **mock_prompt_init_args.to_dict.return_value,
        )
        mock_datarobot_mcp_server_prompt_func = mock_datarobot_mcp_server_prompt.return_value
        call_args = mock_datarobot_mcp_server_prompt_func.call_args.args
        (inner_wrapper_func,) = call_args
        assert (
            inner_wrapper_func.__qualname__
            == "dr_mcp_prompt.<locals>.prompt_decorator.<locals>._inner_decorator"
        )


class TestMCPResourceDecorator:
    @pytest.fixture
    def mock_mcp_resource_callable(self) -> Iterator[Mock]:
        yield Mock(return_value=Mock())

    @pytest.fixture
    def mock_datarobot_mcp_server_resource(self) -> Iterator[Mock]:
        with patch.object(DataRobotMCP, "resource") as mock_func:
            yield mock_func

    def test_dr_mcp_prompt(
        self,
        mock_mcp_resource_callable: Mock,
        mock_datarobot_mcp_server_resource: Mock,
    ) -> None:
        mock_mcp_resource_callable_args = {}
        mock_resource_category = Mock()
        mock_resource_init_args = Mock()
        mock_resource_init_args.to_dict.return_value = {}
        decorator = dr_mcp_resource(mock_resource_init_args, mock_resource_category)
        decorator(mock_mcp_resource_callable)(**mock_mcp_resource_callable_args)

        mock_resource_init_args.set_resource_category.assert_called_once_with(
            mock_resource_category
        )
        mock_datarobot_mcp_server_resource.assert_called_once_with(
            **mock_resource_init_args.to_dict.return_value,
        )
        mock_datarobot_mcp_server_resource_func = mock_datarobot_mcp_server_resource.return_value
        call_args = mock_datarobot_mcp_server_resource_func.call_args.args
        (inner_wrapper_func,) = call_args
        assert (
            inner_wrapper_func.__qualname__
            == "dr_mcp_resource.<locals>.resource_decorator.<locals>._inner_decorator"
        )
