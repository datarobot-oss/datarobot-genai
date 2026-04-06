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

import os
import uuid

import pytest

from datarobot_genai.core.memory.mem0client import Mem0Client

pytestmark = pytest.mark.skipif(
    not os.getenv("MEM0_API_KEY"),
    reason="MEM0_API_KEY is required for live mem0 integration tests.",
)


@pytest.mark.asyncio
async def test_mem0_client_initializes_with_live_api_key() -> None:
    # GIVEN a live mem0 API key
    client = Mem0Client(api_key=os.environ["MEM0_API_KEY"])

    # WHEN the client is created
    # THEN mem0 validation succeeds and a user id is derived
    assert client._memory.user_id
    assert client._memory.user_email


@pytest.mark.asyncio
async def test_mem0_store_and_retrieve_do_not_error() -> None:
    # GIVEN a live mem0 client and unique scoped metadata
    client = Mem0Client(api_key=os.environ["MEM0_API_KEY"])
    unique_id = uuid.uuid4().hex
    agent_id = "LangGraphMem0IntegrationTest"
    app_id = "tests.core.memory"
    user_message = f"LangGraph memory token {unique_id}"

    # WHEN a memory is stored and then retrieved through the wrapper
    await client.store(
        user_message=user_message,
        agent_id=agent_id,
        app_id=app_id,
        attributes={"thread_id": unique_id},
    )
    retrieved = await client.retrieve(
        prompt=user_message,
        agent_id=agent_id,
        app_id=app_id,
        attributes={"thread_id": unique_id},
    )

    # THEN both live mem0 operations complete successfully
    assert isinstance(retrieved, str)
