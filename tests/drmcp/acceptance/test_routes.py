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
from contextlib import contextmanager
from http import HTTPStatus
from uuid import uuid4

import pytest

from datarobot_genai.drmcp.test_utils.mcp_utils_ete import ete_test_http_session
from tests.drmcp.integration.helper import create_prompt_template
from tests.drmcp.integration.helper import delete_prompt_template
from tests.drmcp.integration.helper import get_or_create_prompt_template_version


@contextmanager
def _create_prompt_template_with_versions() -> Iterator[tuple[str, str, str, str]]:
    # Helper to create data and then delete it
    new_prompt_api_name = str(uuid4())
    new_prompt_api = create_prompt_template(new_prompt_api_name)
    new_prompt_api_id = new_prompt_api["id"]
    first_version = get_or_create_prompt_template_version(
        new_prompt_api_id, "prompt text 1", variables=[], headers_auth_only=True
    )
    second_version = get_or_create_prompt_template_version(
        new_prompt_api_id, "prompt text 2", variables=[], headers_auth_only=True
    )
    third_version = get_or_create_prompt_template_version(
        new_prompt_api_id, "prompt text 3", variables=[], headers_auth_only=True
    )
    yield new_prompt_api_id, first_version["id"], second_version["id"], third_version["id"]
    delete_prompt_template(new_prompt_api_id)


@pytest.mark.asyncio
class TestCustomRoutesE2E:
    """End-to-end tests for custom routes."""

    async def test_prompt_templates_custom_routes(self) -> None:
        """End-to-end test for prompt templates custom routes."""
        async with ete_test_http_session() as session:
            # Setup -- create prompt template with 2 versions
            with _create_prompt_template_with_versions() as (
                prompt_template_id,
                first_version_id,
                second_version_id,
                third_version_id,
            ):
                # Test list
                resp = await session.get("/registeredPrompts")
                assert resp.status == HTTPStatus.OK
                jsoned = await resp.json()
                available_prompt_ids = {
                    prompt["promptTemplateId"] for prompt in jsoned["promptTemplates"]
                }
                assert prompt_template_id not in available_prompt_ids

                # Test adding new prompt (first -- "old" -- version)
                resp = await session.put(
                    f"/registeredPrompts/{prompt_template_id}",
                    params={"promptTemplateVersionId": first_version_id},
                )
                assert resp.status == HTTPStatus.CREATED
                jsoned = await resp.json()
                assert jsoned["promptTemplateId"] == prompt_template_id
                assert jsoned["promptTemplateVersionId"] == first_version_id

                # Test list after adding new prompt
                resp = await session.get("/registeredPrompts")
                assert resp.status == HTTPStatus.OK
                jsoned = await resp.json()
                available_prompt_ids = {
                    (prompt["promptTemplateId"], prompt["promptTemplateVersionId"])
                    for prompt in jsoned["promptTemplates"]
                }
                assert (prompt_template_id, first_version_id) in available_prompt_ids
                assert (prompt_template_id, second_version_id) not in available_prompt_ids
                assert (prompt_template_id, third_version_id) not in available_prompt_ids

                # Test update (add newer version of already existing)
                # should add new (2nd) and remove previous one (1st)
                resp = await session.put(
                    f"/registeredPrompts/{prompt_template_id}",
                    params={"promptTemplateVersionId": second_version_id},
                )
                assert resp.status == HTTPStatus.CREATED
                jsoned = await resp.json()
                assert jsoned["promptTemplateId"] == prompt_template_id
                assert jsoned["promptTemplateVersionId"] == second_version_id

                # Test list after updating prompt
                resp = await session.get("/registeredPrompts")
                assert resp.status == HTTPStatus.OK
                jsoned = await resp.json()
                available_prompt_ids = {
                    (prompt["promptTemplateId"], prompt["promptTemplateVersionId"])
                    for prompt in jsoned["promptTemplates"]
                }
                assert (prompt_template_id, first_version_id) not in available_prompt_ids
                assert (prompt_template_id, second_version_id) in available_prompt_ids
                assert (prompt_template_id, third_version_id) not in available_prompt_ids

                # Test refresh (should update to the newest -- 3rd -- version)
                resp = await session.put("/registeredPrompts")
                assert resp.status == HTTPStatus.OK

                # Test list after refresh
                resp = await session.get("/registeredPrompts")
                assert resp.status == HTTPStatus.OK
                jsoned = await resp.json()
                available_prompt_ids = {
                    (prompt["promptTemplateId"], prompt["promptTemplateVersionId"])
                    for prompt in jsoned["promptTemplates"]
                }
                assert (prompt_template_id, first_version_id) not in available_prompt_ids
                assert (prompt_template_id, second_version_id) not in available_prompt_ids
                assert (prompt_template_id, third_version_id) in available_prompt_ids

                # Test delete
                resp = await session.delete(f"/registeredPrompts/{prompt_template_id}")
                assert resp.status == HTTPStatus.OK

                # Test list after delete -- should "totally" remove this prompt template id
                resp = await session.get("/registeredPrompts")
                assert resp.status == HTTPStatus.OK
                jsoned = await resp.json()
                available_prompt_ids = {
                    (prompt["promptTemplateId"], prompt["promptTemplateVersionId"])
                    for prompt in jsoned["promptTemplates"]
                }
                assert (prompt_template_id, first_version_id) not in available_prompt_ids
                assert (prompt_template_id, second_version_id) not in available_prompt_ids
                assert (prompt_template_id, third_version_id) not in available_prompt_ids

    async def test_metadata_route(self) -> None:
        """End-to-end test for metadata route."""
        async with ete_test_http_session() as session:
            # Test GET /metadata
            resp = await session.get("/metadata")
            assert resp.status == HTTPStatus.OK
            jsoned = await resp.json()

            # Verify response structure
            assert "tools" in jsoned
            assert "prompts" in jsoned
            assert "resources" in jsoned
            assert "config" in jsoned

            # Verify tools structure
            tools = jsoned["tools"]
            assert "items" in tools
            assert "count" in tools
            assert isinstance(tools["items"], list)
            assert isinstance(tools["count"], int)
            assert tools["count"] == len(tools["items"])

            # Verify each tool has name and tags
            for tool in tools["items"]:
                assert "name" in tool
                assert "tags" in tool
                assert isinstance(tool["name"], str)
                assert isinstance(tool["tags"], list)
                # Tags should be sorted (as per implementation)
                assert tool["tags"] == sorted(tool["tags"])

            # Verify prompts structure
            prompts = jsoned["prompts"]
            assert "items" in prompts
            assert "count" in prompts
            assert isinstance(prompts["items"], list)
            assert isinstance(prompts["count"], int)
            assert prompts["count"] == len(prompts["items"])

            # Verify each prompt has name and tags
            for prompt in prompts["items"]:
                assert "name" in prompt
                assert "tags" in prompt
                assert isinstance(prompt["name"], str)
                assert isinstance(prompt["tags"], list)
                # Tags should be sorted (as per implementation)
                assert prompt["tags"] == sorted(prompt["tags"])

            # Verify resources structure
            resources = jsoned["resources"]
            assert "items" in resources
            assert "count" in resources
            assert isinstance(resources["items"], list)
            assert isinstance(resources["count"], int)
            assert resources["count"] == len(resources["items"])

            # Verify each resource has name and tags
            for resource in resources["items"]:
                assert "name" in resource
                assert "tags" in resource
                assert isinstance(resource["name"], str)
                assert isinstance(resource["tags"], list)
                # Tags should be sorted (as per implementation)
                assert resource["tags"] == sorted(resource["tags"])

            # Verify config structure
            config = jsoned["config"]
            assert "server" in config
            assert "features" in config
            assert "tool_config" in config

            # Verify server config
            server_config = config["server"]
            assert "name" in server_config
            assert "port" in server_config
            assert "log_level" in server_config
            assert "app_log_level" in server_config
            assert "mount_path" in server_config
            assert isinstance(server_config["name"], str)
            assert isinstance(server_config["port"], int)
            assert isinstance(server_config["log_level"], str)
            assert isinstance(server_config["app_log_level"], str)
            assert isinstance(server_config["mount_path"], str)

            # Verify features config
            features_config = config["features"]
            assert "register_dynamic_tools_on_startup" in features_config
            assert "register_dynamic_prompts_on_startup" in features_config
            assert "tool_registration_allow_empty_schema" in features_config
            assert "tool_registration_duplicate_behavior" in features_config
            assert "prompt_registration_duplicate_behavior" in features_config
            assert isinstance(features_config["register_dynamic_tools_on_startup"], bool)
            assert isinstance(features_config["register_dynamic_prompts_on_startup"], bool)
            assert isinstance(features_config["tool_registration_allow_empty_schema"], bool)
            assert isinstance(features_config["tool_registration_duplicate_behavior"], str)
            assert isinstance(features_config["prompt_registration_duplicate_behavior"], str)

            # Verify tool_config structure
            tool_config = config["tool_config"]
            assert isinstance(tool_config, dict)
            # Verify expected tool types are present
            expected_tool_types = ["predictive", "jira", "confluence", "gdrive", "microsoft_graph"]
            for tool_type in expected_tool_types:
                assert tool_type in tool_config
                tool_type_config = tool_config[tool_type]
                assert "enabled" in tool_type_config
                assert "oauth_required" in tool_type_config
                assert "oauth_configured" in tool_type_config
                assert isinstance(tool_type_config["enabled"], bool)
                assert isinstance(tool_type_config["oauth_required"], bool)
                # oauth_configured can be None or bool
                assert tool_type_config["oauth_configured"] is None or isinstance(
                    tool_type_config["oauth_configured"], bool
                )

    async def test_metadata_route_with_prompt_registration(self) -> None:
        """Test metadata route reflects prompt registration changes."""
        async with ete_test_http_session() as session:
            with _create_prompt_template_with_versions() as (
                prompt_template_id,
                first_version_id,
                second_version_id,
                third_version_id,
            ):
                # Get initial metadata
                resp = await session.get("/metadata")
                assert resp.status == HTTPStatus.OK
                jsoned = await resp.json()
                initial_prompts_count = jsoned["prompts"]["count"]
                initial_prompt_names = {p["name"] for p in jsoned["prompts"]["items"]}

                # Register a prompt
                resp = await session.put(
                    f"/registeredPrompts/{prompt_template_id}",
                    params={"promptTemplateVersionId": first_version_id},
                )
                assert resp.status == HTTPStatus.CREATED

                # Get metadata after registration
                resp = await session.get("/metadata")
                assert resp.status == HTTPStatus.OK
                jsoned = await resp.json()
                updated_prompts_count = jsoned["prompts"]["count"]
                updated_prompt_names = {p["name"] for p in jsoned["prompts"]["items"]}

                # Verify prompt count increased
                assert updated_prompts_count == initial_prompts_count + 1

                # Verify the new prompt name is in the list
                # The prompt name should be derived from the prompt template
                assert len(updated_prompt_names - initial_prompt_names) == 1

                # Verify all prompts still have correct structure
                for prompt in jsoned["prompts"]["items"]:
                    assert "name" in prompt
                    assert "tags" in prompt
                    assert isinstance(prompt["tags"], list)

    async def test_metadata_route_tool_tags(self) -> None:
        """Test that metadata route correctly returns tool tags."""
        async with ete_test_http_session() as session:
            resp = await session.get("/metadata")
            assert resp.status == HTTPStatus.OK
            jsoned = await resp.json()

            # Verify tools have tags (even if empty)
            for tool in jsoned["tools"]["items"]:
                assert "tags" in tool
                assert isinstance(tool["tags"], list)
                # Tags should be sorted
                assert tool["tags"] == sorted(tool["tags"])

            # If there are tools, verify at least one has a name
            if jsoned["tools"]["count"] > 0:
                assert any(tool["name"] for tool in jsoned["tools"]["items"])

    async def test_metadata_route_config_values(self) -> None:
        """Test that metadata route returns valid config values."""
        async with ete_test_http_session() as session:
            resp = await session.get("/metadata")
            assert resp.status == HTTPStatus.OK
            jsoned = await resp.json()

            config = jsoned["config"]

            # Verify server config has valid values
            server = config["server"]
            assert server["port"] > 0
            assert server["port"] < 65536
            assert server["log_level"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            assert server["app_log_level"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

            # Verify duplicate behavior values
            features = config["features"]
            assert features["tool_registration_duplicate_behavior"] in [
                "warn",
                "replace",
                "error",
                "ignore",
            ]
            assert features["prompt_registration_duplicate_behavior"] in [
                "warn",
                "replace",
                "error",
                "ignore",
            ]

            # Verify tool config values are consistent
            tool_config = config["tool_config"]
            for tool_type, tool_info in tool_config.items():
                # If OAuth is not required, oauth_configured should be None
                if not tool_info["oauth_required"]:
                    assert tool_info["oauth_configured"] is None
                # If OAuth is required, oauth_configured should be a bool
                else:
                    assert isinstance(tool_info["oauth_configured"], bool)
