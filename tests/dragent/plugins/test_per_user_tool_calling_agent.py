# Copyright 2026 DataRobot, Inc. and its affiliates.
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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nat.builder.function_info import FunctionInfo
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig

from datarobot_genai.dragent.plugins.per_user_tool_calling_agent import (
    PerUserToolCallAgentWorkflowConfig,
)
from datarobot_genai.dragent.plugins.per_user_tool_calling_agent import _per_user_tool_calling_agent


class TestPerUserToolCallAgentWorkflowConfig:
    def test_is_subclass_of_tool_call_agent_workflow_config(self):
        assert issubclass(PerUserToolCallAgentWorkflowConfig, ToolCallAgentWorkflowConfig)

    def test_registered_name(self):
        assert PerUserToolCallAgentWorkflowConfig.static_type() == "per_user_tool_calling_agent"

    def test_inherits_all_fields_from_parent(self):
        parent_fields = set(ToolCallAgentWorkflowConfig.model_fields) - {"type"}
        child_fields = set(PerUserToolCallAgentWorkflowConfig.model_fields) - {"type"}
        assert parent_fields == child_fields

    def test_default_instantiation(self):
        config = PerUserToolCallAgentWorkflowConfig(llm_name="gpt-4o")
        assert config is not None

    def test_registered_as_per_user_function(self):
        """Importing the module registers a per-user function; verify no exception is raised."""
        import datarobot_genai.dragent.plugins.per_user_tool_calling_agent  # noqa: F401


def _make_fn_info(stream_fn=MagicMock()):
    return FunctionInfo.create(
        single_fn=AsyncMock(),
        stream_fn=stream_fn,
        description="test",
    )


class TestPerUserToolCallingAgentWrapper:
    @pytest.mark.asyncio
    async def test_wraps_stream_fn_when_present(self):
        """When stream_fn is provided, the wrapper yields a FunctionInfo with wrapped stream."""
        original_fn_info = _make_fn_info(stream_fn=AsyncMock())

        async def fake_gen(config, builder):
            yield original_fn_info

        with patch(
            "datarobot_genai.dragent.plugins.per_user_tool_calling_agent"
            ".tool_calling_agent_workflow"
        ) as mock_workflow:
            mock_workflow.__wrapped__ = fake_gen
            config = MagicMock()
            builder = MagicMock()

            gen = _per_user_tool_calling_agent(config, builder)
            result = await gen.__anext__()

            assert isinstance(result, FunctionInfo)
            # The stream_fn should be wrapped, not the original
            assert result.stream_fn is not original_fn_info.stream_fn
            assert result.single_fn is original_fn_info.single_fn
            assert result.description == original_fn_info.description

            await gen.aclose()

    @pytest.mark.asyncio
    async def test_yields_fn_info_unchanged_when_stream_fn_is_none(self):
        """When stream_fn is None, the wrapper yields fn_info as-is."""
        original_fn_info = MagicMock(spec=FunctionInfo)
        original_fn_info.stream_fn = None

        async def fake_gen(config, builder):
            yield original_fn_info

        with patch(
            "datarobot_genai.dragent.plugins.per_user_tool_calling_agent"
            ".tool_calling_agent_workflow"
        ) as mock_workflow:
            mock_workflow.__wrapped__ = fake_gen
            config = MagicMock()
            builder = MagicMock()

            gen = _per_user_tool_calling_agent(config, builder)
            result = await gen.__anext__()

            assert result is original_fn_info
            await gen.aclose()

    @pytest.mark.asyncio
    async def test_original_generator_is_closed_on_exit(self):
        """The original generator must be closed via aclose() in the finally block."""
        original_fn_info = _make_fn_info(stream_fn=AsyncMock())
        closed = False

        async def fake_gen(config, builder):
            nonlocal closed
            try:
                yield original_fn_info
            finally:
                closed = True

        with patch(
            "datarobot_genai.dragent.plugins.per_user_tool_calling_agent"
            ".tool_calling_agent_workflow"
        ) as mock_workflow:
            mock_workflow.__wrapped__ = fake_gen
            config = MagicMock()
            builder = MagicMock()

            gen = _per_user_tool_calling_agent(config, builder)
            await gen.__anext__()
            await gen.aclose()

            assert closed
