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
# The tools of dragent's A2A client (get_task, cancel_task, send_message,
# send_message_streaming) return v0.3 objects: pydantic models, and for events either a bare
# Message or a (Task, update | None) tuple. Agent workflows are written against these shapes.

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from datarobot_genai.dragent.a2a.client.client_config import A2AClientConfig
from datarobot_genai.dragent.a2a.client.client_impl import A2AClientFunctionGroup
from datarobot_genai.dragent.a2a.client.client_impl import CancelTaskInput
from datarobot_genai.dragent.a2a.client.client_impl import GetTaskInput
from datarobot_genai.dragent.a2a.client.client_impl import SendMessageInput

from .sdk_adapter import client_message_event
from .sdk_adapter import client_task
from .sdk_adapter import client_task_events
from .sdk_adapter import v03


def _unentered_group() -> A2AClientFunctionGroup:
    return A2AClientFunctionGroup(A2AClientConfig(url="http://agent.example.com"), MagicMock())


def _group(events: list[Any]) -> A2AClientFunctionGroup:
    async def _stream(*_args: object) -> AsyncIterator[Any]:
        for event in events:
            yield event

    base_client = MagicMock()
    base_client.get_task = AsyncMock(return_value=client_task())
    base_client.cancel_task = AsyncMock(return_value=client_task())
    base_client.send_message = _stream
    base_client.send_message_streaming = _stream
    group = _unentered_group()
    group._client = base_client
    return group


async def test_get_task_returns_a_v0_3_task() -> None:
    # GIVEN a client that holds a working task
    group = _group([])

    # WHEN the get_task tool runs
    result = await group._wrap_get_task(GetTaskInput(task_id="task-1"))

    # THEN it returns a v0.3 Task
    assert isinstance(result, v03.Task)
    assert result.id == "task-1"
    assert result.status.state == v03.TaskState.working


async def test_cancel_task_returns_a_v0_3_task() -> None:
    # GIVEN a client that holds a working task
    group = _group([])

    # WHEN the cancel_task tool runs
    result = await group._wrap_cancel_task(CancelTaskInput(task_id="task-1"))

    # THEN it returns a v0.3 Task
    assert isinstance(result, v03.Task)
    assert result.context_id == "ctx-1"


async def test_send_message_returns_task_and_update_tuples() -> None:
    # GIVEN a task reply: the task, an artifact update, then a final status update
    group = _group(client_task_events())

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
    group = _group([client_message_event("pong")])

    # WHEN the send_message tool runs
    result = await group._send_message_advanced(SendMessageInput(query="ping"))

    # THEN it is a bare v0.3 Message, not a tuple
    assert len(result) == 1
    assert isinstance(result[0], v03.Message)
    assert result[0].parts[0].root.text == "pong"
    assert result[0].role == v03.Role.agent


async def test_send_message_streaming_yields_the_same_shapes() -> None:
    # GIVEN a task reply
    group = _group(client_task_events())

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
