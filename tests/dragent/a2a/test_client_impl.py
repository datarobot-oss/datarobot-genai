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
#
# Tests that the a2a_client tools (get_task, cancel_task, send_message, send_message_streaming)
# still return what they returned on a2a-sdk 0.3: v0.3 pydantic objects, and for events a bare
# Message or a (Task, update | None) tuple.

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from a2a.compat.v0_3 import types as v03
from a2a.types import Artifact
from a2a.types import Message
from a2a.types import Part
from a2a.types import Role
from a2a.types import StreamResponse
from a2a.types import Task
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent

from datarobot_genai.dragent.a2a.client.client_config import A2AClientConfig
from datarobot_genai.dragent.a2a.client.client_impl import A2AClientFunctionGroup
from datarobot_genai.dragent.a2a.client.client_impl import CancelTaskInput
from datarobot_genai.dragent.a2a.client.client_impl import GetTaskInput
from datarobot_genai.dragent.a2a.client.client_impl import SendMessageInput

_TASK = Task(
    id="task-1",
    context_id="ctx-1",
    status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
)


def _events() -> list[StreamResponse]:
    artifact = Artifact(artifact_id="a1", parts=[Part(text="Hello")])
    return [
        StreamResponse(task=_TASK),
        StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="task-1", context_id="ctx-1", artifact=artifact
            )
        ),
        StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="task-1",
                context_id="ctx-1",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            )
        ),
    ]


def _group(events: list[StreamResponse]) -> A2AClientFunctionGroup:
    async def _stream(*_args: object) -> AsyncIterator[StreamResponse]:
        for event in events:
            yield event

    base_client = MagicMock()
    base_client.get_task = AsyncMock(return_value=_TASK)
    base_client.cancel_task = AsyncMock(return_value=_TASK)
    base_client.send_message = _stream
    base_client.send_message_streaming = _stream
    group = _unentered_group()
    group._client = base_client
    return group


def _unentered_group() -> A2AClientFunctionGroup:
    return A2AClientFunctionGroup(A2AClientConfig(url="http://agent.example.com"), MagicMock())


async def test_get_task_returns_a_v0_3_task() -> None:
    # GIVEN a client whose task is a protobuf Task
    group = _group([])

    # WHEN the get_task tool runs
    result = await group._wrap_get_task(GetTaskInput(task_id="task-1"))

    # THEN it returns the v0.3 pydantic Task, as it did on a2a-sdk 0.3
    assert isinstance(result, v03.Task)
    assert result.id == "task-1"
    assert result.status.state == v03.TaskState.working


async def test_cancel_task_returns_a_v0_3_task() -> None:
    # GIVEN a client whose task is a protobuf Task
    group = _group([])

    # WHEN the cancel_task tool runs
    result = await group._wrap_cancel_task(CancelTaskInput(task_id="task-1"))

    # THEN it returns the v0.3 pydantic Task
    assert isinstance(result, v03.Task)
    assert result.context_id == "ctx-1"


async def test_send_message_returns_task_and_update_tuples() -> None:
    # GIVEN a task reply: the task, an artifact update, then a final status update
    group = _group(_events())

    # WHEN the send_message tool runs
    result = await group._send_message_advanced(SendMessageInput(query="hi"))

    # THEN each event is a (Task, update | None) tuple, with the task accumulated so far
    assert [type(update) for _, update in result] == [
        type(None),
        v03.TaskArtifactUpdateEvent,
        v03.TaskStatusUpdateEvent,
    ]
    assert all(isinstance(task, v03.Task) for task, _ in result)
    assert result[0][0].artifacts is None
    assert result[1][0].artifacts[0].parts[0].root.text == "Hello"
    assert result[2][0].status.state == v03.TaskState.completed


async def test_send_message_returns_a_message_reply_as_a_v0_3_message() -> None:
    # GIVEN a reply that is a plain message
    reply = Message(role=Role.ROLE_AGENT, parts=[Part(text="pong")], message_id="m-1")
    group = _group([StreamResponse(message=reply)])

    # WHEN the send_message tool runs
    result = await group._send_message_advanced(SendMessageInput(query="ping"))

    # THEN it is a bare v0.3 Message, not a tuple
    assert len(result) == 1
    assert isinstance(result[0], v03.Message)
    assert result[0].parts[0].root.text == "pong"
    assert result[0].role == v03.Role.agent


async def test_send_message_streaming_yields_the_same_shapes() -> None:
    # GIVEN a task reply
    group = _group(_events())

    # WHEN the streaming tool runs
    result = [event async for event in group._send_message_streaming(SendMessageInput(query="hi"))]

    # THEN it yields the same (Task, update | None) tuples
    assert len(result) == 3
    assert all(isinstance(task, v03.Task) for task, _ in result)
    assert isinstance(result[-1][1], v03.TaskStatusUpdateEvent)
    assert result[-1][1].final is True


@pytest.mark.parametrize(
    "method,params",
    [
        ("_wrap_get_task", GetTaskInput(task_id="t")),
        ("_wrap_cancel_task", CancelTaskInput(task_id="t")),
        ("_send_message_advanced", SendMessageInput(query="hi")),
    ],
)
async def test_tools_need_an_entered_group(method: str, params: object) -> None:
    # GIVEN a group that was never entered, so it has no client
    group = _unentered_group()

    # WHEN a tool runs
    # THEN it fails clearly
    with pytest.raises(RuntimeError, match="not initialized"):
        await getattr(group, method)(params)
