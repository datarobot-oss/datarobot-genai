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

from typing import Any

import pytest
from mcp.types import CallToolResult
from mcp.types import ListToolsResult
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session


@pytest.mark.asyncio
class TestMCPToolsIntegration:
    """Integration tests for MCP tools."""

    async def test_model_tools(self, classification_project: dict[str, Any]) -> None:
        """Complete integration test for ModelTools through MCP."""
        async with integration_test_mcp_session() as session:
            # 1 Test listing available tools
            tools_result: ListToolsResult = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]

            assert "get_best_model" in tool_names
            assert "score_dataset_with_model" in tool_names

            # 2 Test getting best model with specified metric
            result: CallToolResult = await session.call_tool(
                "get_best_model",
                {
                    "project_id": classification_project["project"].id,
                    "metric": "AUC",
                },
            )

            assert not result.isError
            assert len(result.content) > 0
            assert isinstance(result.content[0], TextContent)

            result_text = (
                result.content[0].text
                if hasattr(result.content[0], "text")
                else str(result.content[0])
            )
            assert "Best model:" in result_text, f"Result text: {result_text}"
            assert "Keras Text Convolutional Neural Network Classifier" in result_text, (
                f"Result text: {result_text}"
            )  # Actual model type
            assert "AUC: 0.88" in result_text, f"Result text: {result_text}"  # Actual AUC value

            # 3 Test getting best model without specifying metric
            result = await session.call_tool(
                "get_best_model",
                {
                    "project_id": classification_project["project"].id,
                },
            )

            assert not result.isError
            result_text = (
                result.content[0].text
                if hasattr(result.content[0], "text")
                else str(result.content[0])
            )
            assert "Best model:" in result_text, f"Result text: {result_text}"
            assert "Keras Text Convolutional Neural Network Classifier" in result_text, (
                f"Result text: {result_text}"
            )  # This is the actual model type

            # 4 Test error handling for nonexistent project
            result = await session.call_tool(
                "get_best_model", {"project_id": "nonexistent_project"}
            )

            assert result.isError
            result_text = (
                result.content[0].text
                if hasattr(result.content[0], "text")
                else str(result.content[0])
            )
            assert (
                "Error in get_best_model: ClientError: 404 client error: {'message': 'Not Found'}"
            ) in result_text, f"Result text: {result_text}"

            # 5 Test metric-based sorting of models
            result = await session.call_tool(
                "get_best_model",
                {
                    "project_id": classification_project["project"].id,
                    "metric": "LogLoss",
                },
            )

            result_text = (
                result.content[0].text
                if hasattr(result.content[0], "text")
                else str(result.content[0])
            )
            assert "Best model:" in result_text, f"Result text: {result_text}"
            assert "Keras Text Convolutional Neural Network Classifier" in result_text, (
                f"Result text: {result_text}"
            )
            assert "LogLoss: 0.56" in result_text, (
                f"Result text: {result_text}"
            )  # Actual LogLoss value

            # 6 Test scoring dataset with specified model none existent model or dataset
            result = await session.call_tool(
                "score_dataset_with_model",
                {
                    "project_id": classification_project["project"].id,
                    "model_id": "standalone_model",
                    "dataset_url": "https://example.com/dataset.csv",
                },
            )

            assert result.isError
            result_text = (
                result.content[0].text
                if hasattr(result.content[0], "text")
                else str(result.content[0])
            )
            assert (
                "Error in "
                "score_dataset_with_model: ClientError: 404 client error: "
                "{'message': 'Not Found'}" in result_text
            ), f"Result text: {result_text}"
