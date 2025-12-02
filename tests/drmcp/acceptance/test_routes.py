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
def _create_prompt_template_with_versions() -> Iterator[tuple[str, str, str]]:
    # Helper to create data and then delete it
    new_prompt_api_name = str(uuid4())
    new_prompt_api = create_prompt_template(new_prompt_api_name)
    new_prompt_api_id = new_prompt_api["id"]
    first_version = get_or_create_prompt_template_version(
        new_prompt_api_id, "prompt text 1", variables=[]
    )
    second_version = get_or_create_prompt_template_version(
        new_prompt_api_id, "prompt text 2", variables=[]
    )
    yield new_prompt_api_id, first_version["id"], second_version["id"]
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

                # Test refresh (should update to the newest version)
                resp = await session.put("/registeredPrompts")
                assert resp.status == HTTPStatus.NO_CONTENT

                # Test list after refresh
                resp = await session.get("/registeredPrompts")
                assert resp.status == HTTPStatus.OK
                jsoned = await resp.json()
                available_prompt_ids = {
                    (prompt["promptTemplateId"], prompt["promptTemplateVersionId"])
                    for prompt in jsoned["promptTemplates"]
                }
                assert (prompt_template_id, first_version_id) not in available_prompt_ids
                assert (prompt_template_id, second_version_id) in available_prompt_ids

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
