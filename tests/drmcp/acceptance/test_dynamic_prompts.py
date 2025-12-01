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


# @pytest.mark.asyncio
# class TestDeploymentE2E(ToolBaseE2E):
#     """End-to-end tests for deployment-related functionality."""
#
#     @pytest.mark.parametrize(
#         "prompt",
#         [
#             """
#         I'm working on a machine learning project and I need to list all the deployments I
#         have access to. Can you help me list all the deployments?
#         """
#         ],
#     )
#     async def test_list_deployments_success(
#         self,
#         openai_llm_client: Any,
#         expectations_for_list_deployments_success: ETETestExpectations,
#         prompt: str,
#     ) -> None:
#         async with ete_test_mcp_session() as session:
#             await self._run_test_with_expectations(
#                 prompt,
#                 expectations_for_list_deployments_success,
#                 openai_llm_client,
#                 session,
#                 (
#                     inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
#                     if inspect.currentframe()
#                     else "test_list_deployments_success"
#                 ),
#             )
