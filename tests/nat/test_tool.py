# Copyright 2025 DataRobot, Inc. and its affiliates.
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

import asyncio
import inspect
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.nat.tool import _sync_to_async
from datarobot_genai.nat.tool import nat_tool


class TestSyncToAsync:
    """Tests for _sync_to_async."""

    def test_returns_coroutine_function(self):
        def sync_fn(x: int) -> int:
            return x + 1

        wrapped = _sync_to_async(sync_fn)
        assert inspect.iscoroutinefunction(wrapped)

    def test_preserves_annotations(self):
        def sync_fn(x: int) -> str:
            return str(x)

        wrapped = _sync_to_async(sync_fn)
        assert wrapped.__annotations__["x"] is int
        assert wrapped.__annotations__["return"] is str

    def test_preserves_name_and_doc(self):
        def sync_fn(x: int) -> int:
            """Double the input."""
            return x * 2

        wrapped = _sync_to_async(sync_fn)
        assert wrapped.__name__ == "sync_fn"
        assert wrapped.__doc__ == "Double the input."

    @pytest.mark.asyncio
    async def test_wrapper_returns_same_result(self):
        def sync_fn(x: int, y: int) -> int:
            return x + y

        wrapped = _sync_to_async(sync_fn)
        result = await wrapped(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_wrapper_accepts_kwargs(self):
        def sync_fn(a: int, b: int) -> int:
            return a - b

        wrapped = _sync_to_async(sync_fn)
        result = await wrapped(a=10, b=3)
        assert result == 7


class TestNatTool:
    """Tests for nat_tool decorator."""

    def test_returns_original_function(self):
        def my_sync_tool(x: int) -> int:
            return x

        result = nat_tool(my_sync_tool, "my_tool", description="Does nothing")
        assert result is my_sync_tool

    def test_async_function_returned_as_is(self):
        async def my_async_tool(x: int) -> int:
            return x

        result = nat_tool(my_async_tool, "my_async_tool")
        assert result is my_async_tool

    @patch("datarobot_genai.nat.tool.FunctionInfo")
    @patch("datarobot_genai.nat.tool._sync_to_async")
    @patch("datarobot_genai.nat.tool.register_function")
    def test_sync_function_causes_sync_to_async_call(
        self,
        mock_register: MagicMock,
        mock_sync_to_async: MagicMock,
        mock_function_info: MagicMock,
    ):
        async def fake_async(x: str) -> str:
            return x

        mock_sync_to_async.return_value = fake_async

        def sync_tool(msg: str) -> str:
            return msg

        nat_tool(sync_tool, "echo", description="Echoes")

        # Wrapper is only run when NAT invokes it; run it here to trigger from_fn
        wrapper = mock_register.return_value.call_args[0][0]
        config = MagicMock()
        builder = MagicMock()

        async def run_wrapper():
            gen = wrapper(config, builder)
            return await gen.__anext__()

        asyncio.run(run_wrapper())

        mock_sync_to_async.assert_called_once_with(sync_tool)
        mock_function_info.from_fn.assert_called_once()
        call_kw = mock_function_info.from_fn.call_args.kwargs
        assert call_kw["fn"] is fake_async
        assert call_kw["description"] == "Echoes"

    @patch("datarobot_genai.nat.tool.FunctionInfo")
    @patch("datarobot_genai.nat.tool._sync_to_async")
    @patch("datarobot_genai.nat.tool.register_function")
    def test_async_function_does_not_call_sync_to_async(
        self,
        mock_register: MagicMock,
        mock_sync_to_async: MagicMock,
        mock_function_info: MagicMock,
    ):
        async def async_tool(msg: str) -> str:
            return msg

        nat_tool(async_tool, "async_echo", description="Async echo")

        # Run the registered wrapper so from_fn is invoked
        wrapper = mock_register.return_value.call_args[0][0]
        asyncio.run(wrapper(MagicMock(), MagicMock()).__anext__())

        mock_sync_to_async.assert_not_called()
        mock_function_info.from_fn.assert_called_once_with(
            fn=async_tool,
            description="Async echo",
        )

    @patch("datarobot_genai.nat.tool.FunctionInfo")
    @patch("datarobot_genai.nat.tool.register_function")
    def test_description_none_passed_through(
        self, mock_register: MagicMock, mock_function_info: MagicMock
    ):
        async def async_tool(x: int) -> int:
            return x

        nat_tool(async_tool, "no_desc")

        # Run the registered wrapper so from_fn is invoked
        wrapper = mock_register.return_value.call_args[0][0]
        asyncio.run(wrapper(MagicMock(), MagicMock()).__anext__())

        mock_function_info.from_fn.assert_called_once()
        assert mock_function_info.from_fn.call_args.kwargs["description"] is None

    @patch("datarobot_genai.nat.tool.FunctionInfo")
    @patch("datarobot_genai.nat.tool.register_function")
    def test_register_function_called_with_config_type(
        self, mock_register: MagicMock, mock_function_info: MagicMock
    ):
        from nat.data_models.function import FunctionBaseConfig

        async def async_tool(x: int) -> int:
            return x

        nat_tool(async_tool, "my_tool_name")

        call_args = mock_register.call_args
        config_type = call_args.kwargs["config_type"]
        assert issubclass(config_type, FunctionBaseConfig)
