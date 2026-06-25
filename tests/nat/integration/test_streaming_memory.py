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
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated

import pytest
from ag_ui.core import RunAgentInput
from ag_ui.core import SystemMessage as AgUiSystemMessage
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import UserMessage as AgUiUserMessage
from ag_ui.core.events import EventType
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.builder.function_info import Streaming
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.function import FunctionBaseConfig
from nat.runtime.session import SessionManager

import datarobot_genai.dragent.plugins.datarobot_mem0_memory  # noqa: F401
import datarobot_genai.dragent.plugins.streaming_memory_agent  # noqa: F401
from datarobot_genai.core.memory.mem0client import Mem0Client
from datarobot_genai.dragent.frontends.converters import aggregate_dragent_event_responses
from datarobot_genai.dragent.frontends.converters import convert_dragent_event_response_to_str
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.plugins import datarobot_mem0_memory
from datarobot_genai.dragent.plugins.datarobot_mem0_memory import Config as Mem0Settings
from datarobot_genai.nat.helpers import load_workflow

WORKFLOW_WITH_STREAMING_MEMORY_PATH = Path(__file__).parent / "workflow_with_streaming_memory.yaml"

# The probe yields the injected system text in this many chunks so the test can
# also verify that streaming_memory_agent really streams discrete events instead
# of collapsing them (which is the whole reason it exists vs auto_memory_agent).
_PROBE_CHUNK_COUNT = 3
_NO_MEMORY_SENTINEL = "NO_MEMORY_CONTEXT"


class StreamingMemoryProbeAgentConfig(  # type: ignore[call-arg]
    FunctionBaseConfig,
    name="streaming_memory_probe_agent",
):
    """Per-user probe used by the streaming-memory integration test."""


def _chunk_text(text: str, n: int) -> list[str]:
    """Split text into n roughly-equal chunks (with no chunk left empty)."""
    if not text or n <= 1:
        return [text]
    size = max(1, (len(text) + n - 1) // n)
    return [text[i : i + size] for i in range(0, len(text), size)]


async def _probe_stream(
    input_message: RunAgentInput,
) -> Annotated[
    AsyncGenerator[DRAgentEventResponse, None],
    Streaming(convert=aggregate_dragent_event_responses),
]:
    """Echo any injected system-message text back as multiple stream events.

    streaming_memory_agent's step 2 inserts an ``AgUiSystemMessage`` carrying
    the Mem0 search results immediately before the last user message. The
    probe reads that content (or returns a sentinel) so the test can assert
    that memory was retrieved and injected on the second turn.
    """
    system_text = "\n".join(
        str(m.content)
        for m in input_message.messages
        if isinstance(m, AgUiSystemMessage) and m.content
    )
    payload = system_text or _NO_MEMORY_SENTINEL
    message_id = str(uuid.uuid4())
    for piece in _chunk_text(payload, _PROBE_CHUNK_COUNT):
        if not piece:
            continue
        yield DRAgentEventResponse(
            events=[
                TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=piece,
                )
            ],
        )


async def _probe_stream_to_str(
    responses: AsyncGenerator[DRAgentEventResponse],
) -> str:
    aggregated = aggregate_dragent_event_responses([r async for r in responses])
    return convert_dragent_event_response_to_str(aggregated)


@register_per_user_function(  # type: ignore[untyped-decorator]
    config_type=StreamingMemoryProbeAgentConfig,
    input_type=RunAgentInput,
    streaming_output_type=DRAgentEventResponse,
)
async def streaming_memory_probe_agent(
    config: StreamingMemoryProbeAgentConfig, builder: Builder
) -> AsyncGenerator[FunctionInfo, None]:
    yield FunctionInfo.create(
        stream_fn=_probe_stream,
        stream_to_single_fn=_probe_stream_to_str,
        description=("Echoes back system messages injected by the streaming-memory wrapper."),
    )


def _build_run_agent_input(user_text: str) -> RunAgentInput:
    return RunAgentInput(
        thread_id=f"thread-{uuid.uuid4().hex}",
        run_id=f"run-{uuid.uuid4().hex}",
        state={},
        messages=[AgUiUserMessage(id=str(uuid.uuid4()), content=user_text)],
        tools=[],
        context=[],
        forwarded_props={},
    )


@pytest.fixture
def mem0_api_key() -> str:
    api_key = Mem0Settings().mem0_api_key
    if not api_key:
        pytest.skip("requires MEM0_API_KEY for real Mem0 integration")
    return api_key


@pytest.fixture
async def workflow_with_streaming_memory(
    mem0_api_key: str,
) -> AsyncGenerator[SessionManager, None]:
    async with load_workflow(WORKFLOW_WITH_STREAMING_MEMORY_PATH) as workflow:
        yield workflow


async def test_streaming_memory_agent_wrapper_round_trips_with_real_mem0(
    workflow_with_streaming_memory: SessionManager,
    mem0_api_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN a workflow.yaml with a real Mem0-backed NAT memory provider and the
    # streaming_memory_agent wrapper. The provider's ``_UserManagerShim`` reads
    # ``Context.user_id``, which in production is populated by
    # ``DRAgentAGUISessionManager`` from the DR signed auth header. Stub the
    # shim to a stable per-test UUID so the round-trip proves memories are
    # scoped per user rather than globally per api key.
    test_id = uuid.uuid4().hex
    dr_memory_user_uuid = f"nat-streaming-memory-{test_id}"
    monkeypatch.setattr(
        datarobot_mem0_memory._UserManagerShim,
        "get_id",
        lambda self: dr_memory_user_uuid,
    )

    secret_code = f"DRSTREAM-{test_id}"
    first_message = f"My NAT streaming-memory integration secret code is {secret_code}."
    recall_message = "What is my NAT streaming-memory integration secret code?"

    async def run_memory_workflow(message: str) -> str:
        run_agent_input = _build_run_agent_input(message)
        async with workflow_with_streaming_memory.session(user_id=dr_memory_user_uuid) as session:
            async with session.run(run_agent_input) as runner:
                return await runner.result(to_type=str)

    async def stream_memory_workflow(message: str) -> list[DRAgentEventResponse]:
        run_agent_input = _build_run_agent_input(message)
        async with workflow_with_streaming_memory.session(user_id=dr_memory_user_uuid) as session:
            async with session.run(run_agent_input) as runner:
                events: list[DRAgentEventResponse] = []
                async for event in runner.result_stream(to_type=DRAgentEventResponse):
                    events.append(event)
                return events

    mem0 = Mem0Client(api_key=mem0_api_key)._memory
    api_key_user_id = mem0.user_id
    assert api_key_user_id != dr_memory_user_uuid

    try:
        # WHEN the wrapped workflow sees one message to store.
        await run_memory_workflow(first_message)

        # THEN a later turn retrieves the real Mem0 memory and injects it as
        # context. New per-user user_ids need a few seconds for Mem0 to index
        # the first write, so the budget is generous.
        last_response = ""
        for _ in range(20):
            last_response = await run_memory_workflow(recall_message)
            if secret_code in last_response:
                break
            await asyncio.sleep(3)
        else:
            pytest.fail(
                f"Mem0 did not return expected memory text. Last response: {last_response!r}"
            )

        # AND the wrapper actually streams: the probe yields multiple
        # DRAgentEventResponse events, and result_stream surfaces each as it
        # arrives instead of collapsing to one. This is the behavioral
        # difference vs auto_memory_agent.
        stream_events = await stream_memory_workflow(recall_message)
        assert len(stream_events) > 1, (
            f"Expected streaming_memory_agent to surface multiple events; "
            f"got {len(stream_events)}: {stream_events!r}"
        )
        collapsed = convert_dragent_event_response_to_str(
            aggregate_dragent_event_responses(stream_events)
        )
        assert secret_code in collapsed, (
            f"Streamed events should still contain the recalled secret. "
            f"Collapsed text: {collapsed!r}"
        )

        # AND the memory is scoped to the DR-user UUID, NOT the api-key-derived
        # fallback. Recall succeeded, so indexing is done — this lookup is
        # immediate.
        user_hits = await mem0.get_all(
            filters={"AND": [{"user_id": dr_memory_user_uuid}]},
            page=1,
            page_size=50,
        )
        assert any(
            secret_code in (item.get("memory") or "") for item in user_hits.get("results", []) or []
        ), (
            f"Memory not found under user_id={dr_memory_user_uuid!r}; the shim "
            f"should have propagated the stubbed user uuid through the wrapper."
        )
    finally:
        # Scope is per-user, so delete_all by the stubbed uuid is safe — it
        # only wipes this test's memories.
        await mem0.delete_all(user_id=dr_memory_user_uuid)
