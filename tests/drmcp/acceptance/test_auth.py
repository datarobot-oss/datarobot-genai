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

import inspect
import os
from typing import Any
from unittest.mock import patch

import jwt
import pytest

# Note: get_auth_context_user_info tool is defined in datarobot_genai.drmcp.tools.test_tools
from datarobot_genai.drmcp.test_utils.mcp_utils_ete import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY
from datarobot_genai.drmcp.test_utils.tool_base_ete import ETETestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolBaseE2E
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolCallTestExpectations


@pytest.fixture(scope="session")
def secret_key() -> str:
    """Return a test secret key for JWT signing."""
    return "acceptance-test-secret-key-12345"


@pytest.fixture(scope="session")
def auth_context_data() -> dict[str, Any]:
    """Return sample authorization context data matching AuthCtx structure."""
    return {
        "user": {
            "id": "acc_user_456",
            "name": "Acceptance Test User",
            "email": "acceptance@example.com",
        },
        "identities": [
            {
                "id": "identity_789",
                "type": "user",
                "provider_type": "google",
                "provider_user_id": "google_user_123",
            }
        ],
        "metadata": {"session_id": "acceptance_session_999"},
    }


@pytest.fixture(scope="session")
def auth_token(auth_context_data: dict[str, Any], secret_key: str) -> str:
    """Generate a valid JWT token from auth context data using PyJWT."""
    return jwt.encode(auth_context_data, secret_key, algorithm="HS256")


@pytest.fixture(scope="session")
def expectations_for_auth_context_tool_success(
    auth_context_data: dict[str, Any],
) -> ETETestExpectations:
    """Return expected results when tool is called with valid auth token."""
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="get_auth_context_user_info",
                parameters={},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            auth_context_data["user"]["id"],
            auth_context_data["user"]["name"],
            auth_context_data["user"]["email"],
        ],
    )


@pytest.mark.asyncio
class TestAuthContextE2E(ToolBaseE2E):
    """End-to-end acceptance tests for OAuth middleware and auth context propagation."""

    @pytest.mark.parametrize(
        "prompt",
        [
            """Please help me check my user info within auth context using
            available tools. I need to know my user ID, name, and email.
            """
        ],
    )
    async def test_auth_context_tool_with_valid_token(
        self,
        openai_llm_client: Any,
        expectations_for_auth_context_tool_success: ETETestExpectations,
        prompt: str,
        auth_token: str,
        secret_key: str,
    ) -> None:
        """Test OAuth middleware processes auth header and tool accesses auth context."""
        # Set the secret key in environment so the server can validate tokens
        with patch.dict(os.environ, {"SESSION_SECRET_KEY": secret_key}, clear=False):
            # Create MCP session with auth headers
            async with ete_test_mcp_session(
                additional_headers={"X-DataRobot-Authorization-Context": auth_token}
            ) as session:
                await self._run_test_with_expectations(
                    prompt,
                    expectations_for_auth_context_tool_success,
                    openai_llm_client,
                    session,
                    (
                        inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
                        if inspect.currentframe()
                        else "test_auth_context_tool_with_valid_token"
                    ),
                )
