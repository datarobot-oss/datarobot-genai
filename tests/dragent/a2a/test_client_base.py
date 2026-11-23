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
# Tests for the event handling of dragent's A2A client on a2a-sdk 1.x: rebuilding a task
# from streamed events and extracting the reply text from them.

from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest
from a2a.types import Artifact
from a2a.types import Message
from a2a.types import Part
from a2a.types import Role
from a2a.types import SendMessageRequest
from a2a.types import StreamResponse
from a2a.types import Task
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent

from datarobot_genai.dragent.a2a.client.client_base import A2ABaseClient
from datarobot_genai.dragent.a2a.client.client_base import assemble_task

_TASK_ID = "task-1"
_CONTEXT_ID = "ctx-1"


def _agent_message(text: str) -> Message:
    return Message(role=Role.ROLE_AGENT, parts=[Part(text=text)], message_id="m-agent")


def _task_event(
    state: int = TaskState.TASK_STATE_WORKING,
    *,
    status_text: str | None = None,
    artifacts: list[Artifact] | None = None,
    history: list[Message] | None = None,
) -> StreamResponse:
    status = TaskStatus(state=state)
    if status_text is not None:
        status.message.CopyFrom(_agent_message(status_text))
    task = Task(id=_TASK_ID, context_id=_CONTEXT_ID, status=status)
    task.artifacts.extend(artifacts or [])
    task.history.extend(history or [])
    return StreamResponse(task=task)


def _status_event(state: int, text: str | None = None) -> StreamResponse:
    status = TaskStatus(state=state)
    if text is not None:
        status.message.CopyFrom(_agent_message(text))
    return StreamResponse(
        status_update=TaskStatusUpdateEvent(task_id=_TASK_ID, context_id=_CONTEXT_ID, status=status)
    )


def _artifact(artifact_id: str, *texts: str) -> Artifact:
    return Artifact(artifact_id=artifact_id, parts=[Part(text=text) for text in texts])


def _artifact_event(artifact: Artifact, *, append: bool = False) -> StreamResponse:
    return StreamResponse(
        artifact_update=TaskArtifactUpdateEvent(
            task_id=_TASK_ID, context_id=_CONTEXT_ID, artifact=artifact, append=append
        )
    )


def _texts(artifact: Artifact) -> list[str]:
    return [part.text for part in artifact.parts]


@pytest.fixture
def client() -> A2ABaseClient:
    return A2ABaseClient(base_url="http://agent.example.com")


class TestAssembleTask:
    def test_status_update_replaces_status_and_records_its_message(self) -> None:
        # GIVEN a task whose status carries a message, then a status update
        events = [
            _task_event(TaskState.TASK_STATE_WORKING, status_text="thinking"),
            _status_event(TaskState.TASK_STATE_COMPLETED, "done"),
        ]

        # WHEN assembling
        task = assemble_task(events)

        # THEN the status is the update's, and its message joins the history, as a2a-sdk 0.3's
        # client recorded it
        assert task is not None
        assert task.id == _TASK_ID
        assert task.status.state == TaskState.TASK_STATE_COMPLETED
        assert task.status.message.parts[0].text == "done"
        assert [m.parts[0].text for m in task.history] == ["done"]

    def test_status_update_merges_its_metadata_into_the_task(self) -> None:
        # GIVEN a task with metadata, then a status update that carries metadata of its own
        first = _task_event()
        first.task.metadata.update({"source": "agent", "step": "start"})
        update = _status_event(TaskState.TASK_STATE_WORKING)
        update.status_update.metadata.update({"step": "planning"})

        # WHEN assembling
        task = assemble_task([first, update])

        # THEN the update's metadata is merged in, overriding shared keys, as a2a-sdk 0.3's
        # client merged it
        assert task is not None
        assert dict(task.metadata) == {"source": "agent", "step": "planning"}

    def test_status_update_without_metadata_keeps_the_task_metadata(self) -> None:
        # GIVEN a task with metadata, then a status update without any
        first = _task_event()
        first.task.metadata.update({"source": "agent"})

        # WHEN assembling
        task = assemble_task([first, _status_event(TaskState.TASK_STATE_COMPLETED)])

        # THEN the task's metadata is untouched
        assert task is not None
        assert dict(task.metadata) == {"source": "agent"}

    def test_artifact_update_with_append_extends_existing_artifact(self) -> None:
        # GIVEN an artifact followed by an appending update for the same id
        events = [
            _task_event(),
            _artifact_event(_artifact("a1", "Hello")),
            _artifact_event(_artifact("a1", "world"), append=True),
        ]

        # WHEN assembling
        task = assemble_task(events)

        # THEN the parts are concatenated in one artifact
        assert task is not None
        assert len(task.artifacts) == 1
        assert _texts(task.artifacts[0]) == ["Hello", "world"]

    def test_artifact_update_without_append_replaces_existing_artifact(self) -> None:
        # GIVEN an artifact followed by a non-appending update for the same id
        events = [
            _task_event(),
            _artifact_event(_artifact("a1", "old")),
            _artifact_event(_artifact("a1", "new")),
        ]

        # WHEN assembling
        task = assemble_task(events)

        # THEN the artifact content is replaced
        assert task is not None
        assert len(task.artifacts) == 1
        assert _texts(task.artifacts[0]) == ["new"]


class TestExtractTextFromEvents:
    def test_final_message_returns_its_text(self, client: A2ABaseClient) -> None:
        # GIVEN a reply that ends in a message
        events = [_task_event(), StreamResponse(message=_agent_message("Direct answer"))]

        # WHEN extracting text
        text = client.extract_text_from_events(events)

        # THEN the message text is returned
        assert text == "Direct answer"

    def test_task_reply_reads_artifacts(self, client: A2ABaseClient) -> None:
        # GIVEN a streamed task that completes with an artifact
        events = [
            _task_event(),
            _artifact_event(_artifact("a1", "Hello")),
            _artifact_event(_artifact("a1", "world"), append=True),
            _status_event(TaskState.TASK_STATE_COMPLETED),
        ]

        # WHEN extracting text
        text = client.extract_text_from_events(events)

        # THEN the artifact text is returned
        assert text == "Hello world"

    def test_completed_task_answer_only_in_final_status_message(
        self, client: A2ABaseClient
    ) -> None:
        # GIVEN a task that completes with no artifacts, answering in the status message
        events = [
            _task_event(TaskState.TASK_STATE_WORKING, status_text="thinking"),
            _status_event(TaskState.TASK_STATE_COMPLETED, "The answer"),
        ]

        # WHEN extracting text
        text = client.extract_text_from_events(events)

        # THEN the final status message is the answer, not the earlier progress message
        assert text == "The answer"


class TestExtractTextFromTask:
    def test_task_without_status_still_reads_artifacts(self, client: A2ABaseClient) -> None:
        # GIVEN a task that carries no status at all, only an artifact
        task = Task(id=_TASK_ID, artifacts=[_artifact("a1", "Hello")])

        # WHEN extracting text
        text = client.extract_text_from_task(task)

        # THEN it is not mistaken for an unfinished task, and the artifact is read
        assert text == "Hello"

    def test_failed_task_reports_its_message(self, client: A2ABaseClient) -> None:
        # GIVEN a failed task whose status explains why
        task = _task_event(TaskState.TASK_STATE_FAILED, status_text="Tool exploded").task

        # WHEN extracting text
        text = client.extract_text_from_task(task)

        # THEN the explanation is returned as plain text
        assert text == "Task failed: Tool exploded"

    def test_failed_task_without_message_says_unknown_error(self, client: A2ABaseClient) -> None:
        # GIVEN a failed task with no status message
        task = _task_event(TaskState.TASK_STATE_FAILED).task

        # WHEN extracting text
        text = client.extract_text_from_task(task)

        # THEN the fallback text is used
        assert text == "Task failed: Unknown error"

    def test_unfinished_task_names_its_state(self, client: A2ABaseClient) -> None:
        # GIVEN a task that is still working
        task = _task_event(TaskState.TASK_STATE_WORKING).task

        # WHEN extracting text
        text = client.extract_text_from_task(task)

        # THEN the state is shown by name, not as a number
        assert text == "Task in progress (state: TASK_STATE_WORKING)"


class TestSendMessage:
    @staticmethod
    def _client_with_sdk(events: list[StreamResponse]) -> tuple[A2ABaseClient, MagicMock]:
        sdk_client = MagicMock()
        calls: list[SendMessageRequest] = []

        async def _send(request: SendMessageRequest) -> AsyncIterator[StreamResponse]:
            calls.append(request)
            for event in events:
                yield event

        sdk_client.send_message.side_effect = _send
        sdk_client.requests = calls
        a2a_client = A2ABaseClient(base_url="http://agent.example.com")
        a2a_client._client = sdk_client
        return a2a_client, sdk_client

    async def test_builds_request_without_ids_by_default(self) -> None:
        # GIVEN a client wrapping an SDK client
        client, sdk_client = self._client_with_sdk([])

        # WHEN sending a message with no task or context id
        _ = [event async for event in client.send_message("hello")]

        # THEN the request holds a user message with that text and no ids
        (request,) = sdk_client.requests
        assert isinstance(request, SendMessageRequest)
        assert request.message.role == Role.ROLE_USER
        assert [part.text for part in request.message.parts] == ["hello"]
        assert request.message.message_id
        assert request.message.task_id == ""
        assert request.message.context_id == ""

    async def test_includes_task_and_context_ids_when_given(self) -> None:
        # GIVEN a client wrapping an SDK client
        client, sdk_client = self._client_with_sdk([])

        # WHEN sending a message that continues a task
        _ = [event async for event in client.send_message("again", task_id="t-9", context_id="c-9")]

        # THEN the ids are on the message
        (request,) = sdk_client.requests
        assert request.message.task_id == "t-9"
        assert request.message.context_id == "c-9"

    async def test_yields_the_sdk_events(self) -> None:
        # GIVEN an SDK client that streams two events
        events = [_task_event(), StreamResponse(message=_agent_message("done"))]
        client, _ = self._client_with_sdk(events)

        # WHEN sending a message
        received = [event async for event in client.send_message("hello")]

        # THEN the events are passed through unchanged
        assert received == events
