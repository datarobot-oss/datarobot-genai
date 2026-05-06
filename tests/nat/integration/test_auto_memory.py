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

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.function import FunctionBaseConfig
from nat.runtime.session import SessionManager

import datarobot_genai.nat.datarobot_mem0_memory  # noqa: F401
from datarobot_genai.core.memory.mem0client import Mem0Client
from datarobot_genai.nat.helpers import load_workflow

WORKFLOW_WITH_MEMORY_PATH = Path(__file__).parent / "workflow_with_memory.yaml"


class AutoMemoryProbeAgentConfig(  # type: ignore[call-arg]
    FunctionBaseConfig,
    name="auto_memory_probe_agent",
):
    """Probe function used by the auto-memory integration test."""


async def _auto_memory_probe_agent(chat_request: ChatRequest) -> str:
    system_messages = [
        message.content
        for message in chat_request.messages
        if "system" in str(message.role).lower()
    ]
    return "\n".join(system_messages) or "NO_MEMORY_CONTEXT"


@register_function(
    config_type=AutoMemoryProbeAgentConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def auto_memory_probe_agent(
    config: AutoMemoryProbeAgentConfig, builder: Builder
) -> AsyncGenerator[FunctionInfo, None]:
    yield FunctionInfo.from_fn(
        _auto_memory_probe_agent,
        description="Returns memory context injected by NAT's auto-memory wrapper.",
    )


@pytest.fixture
def mem0_api_key() -> str:
    api_key = os.environ.get("MEM0_API_KEY")
    if not api_key:
        pytest.skip("requires MEM0_API_KEY for real Mem0 integration")
    return api_key


@pytest.fixture
async def workflow_with_memory(mem0_api_key: str) -> AsyncGenerator[SessionManager, None]:
    async with load_workflow(WORKFLOW_WITH_MEMORY_PATH) as workflow:
        yield workflow


async def test_auto_memory_agent_wrapper_round_trips_with_real_mem0(
    workflow_with_memory: SessionManager,
    mem0_api_key: str,
) -> None:
    # GIVEN a workflow.yaml with a real Mem0-backed NAT memory provider and auto-memory wrapper.
    test_id = uuid.uuid4().hex
    user_id = f"nat-auto-memory-{test_id}"
    secret_code = f"DRMEM-{test_id}"
    first_message = f"My NAT auto-memory integration secret code is {secret_code}."
    recall_message = "What is my NAT auto-memory integration secret code?"

    async def run_memory_workflow(message: str) -> str:
        async with workflow_with_memory.session(user_id=user_id) as session:
            async with session.run(message) as runner:
                return await runner.result(to_type=str)

    try:
        # WHEN the wrapped workflow sees one message to store.
        await run_memory_workflow(first_message)

        # THEN a later turn retrieves the real Mem0 memory and injects it as context.
        last_response = ""
        for _ in range(10):
            last_response = await run_memory_workflow(recall_message)
            if secret_code in last_response:
                break
            await asyncio.sleep(2)
        else:
            pytest.fail(
                f"Mem0 did not return expected memory text. Last response: {last_response!r}"
            )
    finally:
        await Mem0Client(api_key=mem0_api_key)._memory.delete_all(user_id=user_id)
