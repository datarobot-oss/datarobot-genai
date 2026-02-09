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

import json
from typing import Any

import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session


@pytest.mark.asyncio
class TestMCPDeploymentInfoIntegration:
    """Integration tests for MCP deployment info tools (multiclass project)."""

    async def test_get_deployment_features_and_template(
        self, classification_project: dict[str, Any]
    ) -> None:
        """Integration test for get_deployment_features and generate_prediction_data_template
        on a multiclass deployment.
        """
        async with integration_test_mcp_session() as session:
            deployment_id = classification_project["deployment_id"]
            # Test get_deployment_features
            result = await session.call_tool(
                "get_deployment_features",
                {"deployment_id": deployment_id},
            )
            assert not result.isError, (
                f"get_deployment_features failed: {result.content[0].text}"  # type: ignore[union-attr]
            )
            result_content = result.content[0]
            assert isinstance(result_content, TextContent)
            assert "error" not in result_content.text.lower()

            # Test generate_prediction_data_template (returns JSON structured content)
            result = await session.call_tool(
                "generate_prediction_data_template",
                {"deployment_id": deployment_id, "n_rows": 3},
            )
            assert not result.isError, (
                f"generate_prediction_data_template failed: {result.content[0].text}"  # type: ignore[union-attr]
            )
            result_content = result.content[0]
            assert isinstance(result_content, TextContent)
            data = json.loads(result_content.text)
            assert data["deployment_id"] == deployment_id
            assert "Keras Text Convolutional Neural Network Classifier" in data["model_type"]
            assert data["target"] == "sentiment"
            assert data["total_features"] == 2
            assert "template_data" in data
            assert len(data["template_data"]) == 3
            assert "text_review" in data["template_data"][0]
            assert "product_category" in data["template_data"][0]
