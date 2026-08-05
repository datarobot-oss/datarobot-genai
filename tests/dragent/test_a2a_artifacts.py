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

"""Tests for A2A Tasks, Artifacts, Files and Images.

No live infrastructure required: the NAT session manager and A2A event queue are
stubbed, so these exercise inbound part parsing, artifact construction, wire-format
serialisation, the task lifecycle, response-shape selection, builder loading, and
the per-user composition contract.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import pytest
from a2a.types import DataPart
from a2a.types import FilePart
from a2a.types import FileWithBytes
from a2a.types import FileWithUri
from a2a.types import Message
from a2a.types import Part
from a2a.types import Role
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskStatusUpdateEvent
from a2a.types import TextPart
from a2a.utils.errors import ServerError

from datarobot_genai.dragent.a2a_artifacts import A2ARequestInputs
from datarobot_genai.dragent.a2a_artifacts import ArtifactBuilder
from datarobot_genai.dragent.a2a_artifacts import InboundFile
from datarobot_genai.dragent.a2a_artifacts import OutboundArtifact
from datarobot_genai.dragent.a2a_artifacts import TaskArtifactAgentExecutor
from datarobot_genai.dragent.a2a_artifacts import data_artifact
from datarobot_genai.dragent.a2a_artifacts import extract_request_inputs
from datarobot_genai.dragent.a2a_artifacts import file_artifact
from datarobot_genai.dragent.a2a_artifacts import file_uri_artifact
from datarobot_genai.dragent.a2a_artifacts import load_artifact_builder
from datarobot_genai.dragent.a2a_artifacts import mixed_artifact
from datarobot_genai.dragent.a2a_artifacts import text_artifact

PNG_BYTES = b"\x89PNG\r\n\x1a\n-opaque-test-bytes"


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _FakeContext:
    """Stand-in for a2a's RequestContext (only the read fields are provided)."""

    def __init__(self, message: Message | None, current_task: Any = None) -> None:
        self.message = message
        self.current_task = current_task
        self.context_id = "ctx-1"
        self.task_id = "task-1"

    def get_user_input(self) -> str:
        return ""


class _RecordingQueue:
    """Captures the events an executor publishes."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: object) -> None:
        self.events.append(event)


class _StubSessionManager:
    """Minimal SessionManager: records the query, returns a canned result."""

    def __init__(self, result: str = "canned result", raises: bool = False) -> None:
        self.result = result
        self.raises = raises
        self.last_query: str | None = None
        _type = SimpleNamespace(type="stub-workflow")
        self.config = SimpleNamespace(workflow=_type)
        self.workflow = SimpleNamespace(config=SimpleNamespace(workflow=_type))

    def session(self) -> Any:
        outer = self

        class _SessionCM:
            async def __aenter__(self) -> Any:
                class _Session:
                    def run(self, query: str) -> Any:
                        outer.last_query = query

                        class _RunCM:
                            async def __aenter__(self) -> Any:
                                class _Runner:
                                    async def result(self, to_type: type = str) -> str:
                                        if outer.raises:
                                            raise RuntimeError("workflow exploded")
                                        return outer.result

                                return _Runner()

                            async def __aexit__(self, *a: object) -> bool:
                                return False

                        return _RunCM()

                return _Session()

            async def __aexit__(self, *a: object) -> bool:
                return False

        return _SessionCM()


class TwoArtifactBuilder:
    """Builder that always produces a text artifact and a file artifact."""

    async def build_artifacts(
        self, inputs: A2ARequestInputs, response_text: str
    ) -> list[OutboundArtifact]:
        return [
            text_artifact("summary", response_text),
            file_artifact("chart.png", PNG_BYTES, "image/png"),
        ]


class EmptyBuilder:
    """Builder that reports nothing task-shaped."""

    async def build_artifacts(
        self, inputs: A2ARequestInputs, response_text: str
    ) -> list[OutboundArtifact]:
        return []


class EchoAttachmentsBuilder:
    """Builder that proves inbound files reached the application."""

    async def build_artifacts(
        self, inputs: A2ARequestInputs, response_text: str
    ) -> list[OutboundArtifact]:
        return [
            data_artifact(
                "echo-of-your-attachments",
                {"files": [f.describe() for f in inputs.files]},
            )
        ]


class NotABuilder:
    """Deliberately missing ``build_artifacts``."""


def _message(*parts: Part) -> Message:
    return Message(role=Role.user, parts=list(parts), message_id="m1")


def _executor(
    session_manager: _StubSessionManager,
    builder: ArtifactBuilder | None = None,
    task_mode: str = "auto",
) -> TaskArtifactAgentExecutor:
    return TaskArtifactAgentExecutor(
        session_manager,
        builder=builder,
        task_mode=task_mode,  # type: ignore[arg-type]
    )


def _states(queue: _RecordingQueue) -> list[str]:
    return [e.status.state.value for e in queue.events if isinstance(e, TaskStatusUpdateEvent)]


def _artifacts(queue: _RecordingQueue) -> list[TaskArtifactUpdateEvent]:
    return [e for e in queue.events if isinstance(e, TaskArtifactUpdateEvent)]


# ---------------------------------------------------------------------------
# Inbound parsing -- the half NAT's get_user_input() discards
# ---------------------------------------------------------------------------


class TestExtractRequestInputs:
    def test_extracts_text(self) -> None:
        # GIVEN a request carrying a single text part
        ctx = _FakeContext(_message(Part(root=TextPart(text="hello"))))
        # WHEN the request is parsed
        inputs = extract_request_inputs(ctx)  # type: ignore[arg-type]
        # THEN the text is available and no attachments are reported
        assert inputs.text == "hello"
        assert not inputs.has_attachments

    def test_joins_multiple_text_parts(self) -> None:
        # GIVEN two text parts
        ctx = _FakeContext(
            _message(Part(root=TextPart(text="one")), Part(root=TextPart(text="two")))
        )
        # WHEN parsed
        inputs = extract_request_inputs(ctx)  # type: ignore[arg-type]
        # THEN they are newline-joined in order
        assert inputs.text == "one\ntwo"

    def test_extracts_inline_file_and_decodes_base64(self) -> None:
        # GIVEN an inline file part carrying base64 bytes
        encoded = base64.b64encode(b"hello bytes").decode("ascii")
        ctx = _FakeContext(
            _message(
                Part(
                    root=FilePart(
                        file=FileWithBytes(bytes=encoded, mime_type="text/plain", name="a.txt")
                    )
                )
            )
        )
        # WHEN parsed
        inputs = extract_request_inputs(ctx)  # type: ignore[arg-type]
        # THEN the bytes are decoded and reported as inline
        assert len(inputs.files) == 1
        file = inputs.files[0]
        assert file.content == b"hello bytes"
        assert file.is_inline
        assert file.size == len(b"hello bytes")
        assert file.name == "a.txt"

    def test_extracts_uri_file(self) -> None:
        # GIVEN a file part carrying a URI rather than bytes
        ctx = _FakeContext(
            _message(
                Part(
                    root=FilePart(
                        file=FileWithUri(
                            uri="https://example.com/x.csv", mime_type="text/csv", name="x.csv"
                        )
                    )
                )
            )
        )
        # WHEN parsed
        inputs = extract_request_inputs(ctx)  # type: ignore[arg-type]
        # THEN the reference is preserved and nothing is inlined
        file = inputs.files[0]
        assert file.uri == "https://example.com/x.csv"
        assert not file.is_inline
        assert file.size == 0

    def test_extracts_data_part(self) -> None:
        # GIVEN a structured data part
        ctx = _FakeContext(_message(Part(root=DataPart(data={"k": 1}))))
        # WHEN parsed
        inputs = extract_request_inputs(ctx)  # type: ignore[arg-type]
        # THEN the payload is captured
        assert inputs.data == [{"k": 1}]
        assert inputs.has_attachments

    def test_all_kinds_together(self) -> None:
        # GIVEN a request mixing text, file and data parts
        encoded = base64.b64encode(b"img").decode("ascii")
        ctx = _FakeContext(
            _message(
                Part(root=TextPart(text="describe this")),
                Part(root=FilePart(file=FileWithBytes(bytes=encoded, mime_type="image/png"))),
                Part(root=DataPart(data={"threshold": 0.5})),
            )
        )
        # WHEN parsed
        inputs = extract_request_inputs(ctx)  # type: ignore[arg-type]
        # THEN each kind lands in its own bucket
        assert inputs.text == "describe this"
        assert inputs.files[0].content == b"img"
        assert inputs.data == [{"threshold": 0.5}]

    def test_bad_base64_is_tolerated(self) -> None:
        # GIVEN a file part whose payload is not valid base64
        ctx = _FakeContext(
            _message(
                Part(root=FilePart(file=FileWithBytes(bytes="!!!not base64!!!", name="bad.bin")))
            )
        )
        # WHEN parsed
        inputs = extract_request_inputs(ctx)  # type: ignore[arg-type]
        # THEN the attachment is degraded rather than failing the whole request
        assert len(inputs.files) == 1
        assert inputs.files[0].content is None

    def test_base64_with_newlines_is_decoded(self) -> None:
        # GIVEN base64 wrapped across lines, as pretty-printers emit
        raw = b"a longer payload that will wrap when encoded" * 2
        encoded = base64.encodebytes(raw).decode("ascii")
        ctx = _FakeContext(
            _message(Part(root=FilePart(file=FileWithBytes(bytes=encoded, name="w.bin"))))
        )
        # WHEN parsed
        inputs = extract_request_inputs(ctx)  # type: ignore[arg-type]
        # THEN the lenient pass recovers the bytes
        assert inputs.files[0].content == raw

    def test_urlsafe_base64_is_decoded(self) -> None:
        # GIVEN a payload encoded with the base64url alphabet
        raw = bytes(range(256))
        encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        ctx = _FakeContext(
            _message(Part(root=FilePart(file=FileWithBytes(bytes=encoded, name="u.bin"))))
        )
        # WHEN parsed
        inputs = extract_request_inputs(ctx)  # type: ignore[arg-type]
        # THEN the base64url fallback recovers the bytes
        assert inputs.files[0].content == raw

    def test_empty_message_is_safe(self) -> None:
        # GIVEN a context with no message at all
        ctx = _FakeContext(None)
        # WHEN parsed
        inputs = extract_request_inputs(ctx)  # type: ignore[arg-type]
        # THEN an empty result is returned rather than raising
        assert inputs.text == ""
        assert inputs.files == []
        assert inputs.data == []


class TestInboundFile:
    def test_describe_reports_inline_size(self) -> None:
        # GIVEN an inline file
        file = InboundFile(name="a.png", mime_type="image/png", content=b"1234")
        # WHEN described
        described = file.describe()
        # THEN the summary names it and reports the byte count
        assert "a.png" in described
        assert "4 bytes inline" in described

    def test_describe_reports_uri(self) -> None:
        # GIVEN a URI reference with no name
        file = InboundFile(name=None, mime_type=None, uri="https://example.com/a")
        # WHEN described
        described = file.describe()
        # THEN a placeholder name is used and the uri is shown
        assert "<unnamed>" in described
        assert "uri=https://example.com/a" in described


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


class TestArtifactHelpers:
    def test_text_artifact(self) -> None:
        # GIVEN a text artifact
        result = text_artifact("summary", "hello")
        # WHEN inspected
        # THEN it carries one TextPart with the supplied text
        assert result.name == "summary"
        assert result.parts[0].root.text == "hello"

    def test_data_artifact(self) -> None:
        # GIVEN a data artifact
        result = data_artifact("analysis", {"n": 3})
        # WHEN inspected
        # THEN the structured payload round-trips
        assert result.parts[0].root.data == {"n": 3}

    def test_file_artifact_encodes_base64(self) -> None:
        # GIVEN raw bytes
        result = file_artifact("chart.png", b"\x00\x01", "image/png")
        # WHEN inspected
        file = result.parts[0].root.file
        # THEN the bytes are base64-encoded and the mime type preserved
        assert base64.b64decode(file.bytes) == b"\x00\x01"
        assert file.mime_type == "image/png"
        assert file.name == "chart.png"

    def test_file_artifact_filename_can_differ_from_artifact_name(self) -> None:
        # GIVEN an explicit filename
        result = file_artifact("report", b"x", "text/csv", filename="q3.csv")
        # WHEN inspected
        # THEN the artifact name and the file name are independent
        assert result.name == "report"
        assert result.parts[0].root.file.name == "q3.csv"

    def test_file_uri_artifact_inlines_nothing(self) -> None:
        # GIVEN a URI artifact
        result = file_uri_artifact("big.csv", "https://example.com/big.csv", "text/csv")
        # WHEN inspected
        file = result.parts[0].root.file
        # THEN it references the uri and carries no inline bytes
        assert file.uri == "https://example.com/big.csv"
        assert getattr(file, "bytes", None) is None

    def test_mixed_artifact_ordering(self) -> None:
        # GIVEN a multi-part artifact with text, data and a file
        result = mixed_artifact(
            "bundle",
            text="summary",
            data={"k": 1},
            files=[("a.bin", b"xy", "application/octet-stream")],
        )
        # WHEN the part kinds are read in order
        kinds = [p.root.kind for p in result.parts]
        # THEN text precedes data precedes file
        assert kinds == ["text", "data", "file"]

    def test_metadata_is_carried(self) -> None:
        # GIVEN an artifact built with metadata
        result = text_artifact("s", "t", {"source": "unit-test"})
        # WHEN inspected
        # THEN the metadata is preserved for publication
        assert result.metadata == {"source": "unit-test"}


class TestWireFormat:
    def test_file_part_serialises_to_camel_case(self) -> None:
        # GIVEN a file artifact
        result = file_artifact("a.png", b"\x00", "image/png")
        # WHEN serialised the way the A2A transport does
        dumped = result.parts[0].model_dump(by_alias=True, exclude_none=True)
        # THEN the file payload uses the wire's camelCase key
        assert dumped["kind"] == "file"
        assert "mimeType" in dumped["file"]
        assert "mime_type" not in dumped["file"]

    def test_message_wire_keys(self) -> None:
        # GIVEN a message carrying a text part
        message = _message(Part(root=TextPart(text="hi")))
        # WHEN serialised by alias
        dumped = message.model_dump(by_alias=True, exclude_none=True)
        # THEN identifiers use camelCase and the discriminator is present
        assert dumped["kind"] == "message"
        assert "messageId" in dumped


# ---------------------------------------------------------------------------
# Response shape: task_mode
# ---------------------------------------------------------------------------


class TestTaskModeAlways:
    @pytest.mark.asyncio
    async def test_emits_submitted_working_artifacts_completed(self) -> None:
        # GIVEN an executor in ``always`` mode with a builder producing two artifacts
        sm = _StubSessionManager("AAPL up 1.2%")
        queue = _RecordingQueue()
        ctx = _FakeContext(_message(Part(root=TextPart(text="how is AAPL?"))))
        # WHEN it executes
        await _executor(sm, TwoArtifactBuilder(), "always").execute(ctx, queue)  # type: ignore[arg-type]
        # THEN a Task is opened, driven working -> completed, and both artifacts emitted
        assert queue.events[0].status.state.value == "submitted"
        assert _states(queue) == ["working", "completed"]
        assert len(_artifacts(queue)) == 2
        assert sm.last_query == "how is AAPL?"

    @pytest.mark.asyncio
    async def test_falls_back_to_text_artifact_when_builder_is_empty(self) -> None:
        # GIVEN ``always`` mode with a builder that produces nothing
        sm = _StubSessionManager("only words")
        queue = _RecordingQueue()
        ctx = _FakeContext(_message(Part(root=TextPart(text="q"))))
        # WHEN it executes
        await _executor(sm, EmptyBuilder(), "always").execute(ctx, queue)  # type: ignore[arg-type]
        # THEN the task still completes, carrying the response as one text artifact
        assert _states(queue)[-1] == "completed"
        assert len(_artifacts(queue)) == 1

    @pytest.mark.asyncio
    async def test_failure_publishes_failed_state_and_raises(self) -> None:
        # GIVEN a workflow that raises
        sm = _StubSessionManager(raises=True)
        queue = _RecordingQueue()
        ctx = _FakeContext(_message(Part(root=TextPart(text="boom"))))
        # WHEN it executes
        with pytest.raises(ServerError):
            await _executor(sm, TwoArtifactBuilder(), "always").execute(ctx, queue)  # type: ignore[arg-type]
        # THEN the task reaches a terminal failed state so the caller stops waiting
        assert "failed" in _states(queue)

    @pytest.mark.asyncio
    async def test_file_only_request_is_allowed(self) -> None:
        # GIVEN a request with a file but no text, which NAT would reject
        encoded = base64.b64encode(PNG_BYTES).decode("ascii")
        sm = _StubSessionManager("described")
        queue = _RecordingQueue()
        ctx = _FakeContext(
            _message(Part(root=FilePart(file=FileWithBytes(bytes=encoded, mime_type="image/png"))))
        )
        # WHEN it executes
        await _executor(sm, TwoArtifactBuilder(), "always").execute(ctx, queue)  # type: ignore[arg-type]
        # THEN the task completes and the workflow was invoked with an empty query
        assert _states(queue)[-1] == "completed"
        assert sm.last_query == ""

    @pytest.mark.asyncio
    async def test_inbound_files_reach_the_builder(self) -> None:
        # GIVEN a request carrying a file and a builder that echoes attachments
        encoded = base64.b64encode(b"1234").decode("ascii")
        queue = _RecordingQueue()
        ctx = _FakeContext(
            _message(
                Part(root=TextPart(text="echo")),
                Part(root=FilePart(file=FileWithBytes(bytes=encoded, name="in.bin"))),
            )
        )
        # WHEN it executes
        await _executor(_StubSessionManager(), EchoAttachmentsBuilder(), "always").execute(  # type: ignore[arg-type]
            ctx, queue
        )
        # THEN the application saw the decoded attachment
        payload = _artifacts(queue)[0].artifact.parts[0].root.data
        assert "in.bin" in payload["files"][0]
        assert "4 bytes inline" in payload["files"][0]

    @pytest.mark.asyncio
    async def test_missing_message_raises_invalid_params(self) -> None:
        # GIVEN a context with no message
        ctx = _FakeContext(None)
        # WHEN it executes
        with pytest.raises(ServerError) as exc:
            await _executor(_StubSessionManager(), TwoArtifactBuilder(), "always").execute(  # type: ignore[arg-type]
                ctx, _RecordingQueue()
            )
        # THEN the caller is told the params were invalid, not that the server failed
        assert exc.value.error.__class__.__name__ == "InvalidParamsError"

    @pytest.mark.asyncio
    async def test_empty_textpart_is_rejected_cleanly(self) -> None:
        # GIVEN a message whose only part is empty text
        ctx = _FakeContext(_message(Part(root=TextPart(text=""))))
        # WHEN it executes
        with pytest.raises(ServerError) as exc:
            await _executor(_StubSessionManager(), TwoArtifactBuilder(), "always").execute(  # type: ignore[arg-type]
                ctx, _RecordingQueue()
            )
        # THEN the ValueError from new_task() is translated, not leaked
        assert exc.value.error.__class__.__name__ == "InvalidParamsError"


class TestTaskModeAuto:
    @pytest.mark.asyncio
    async def test_returns_message_when_no_artifacts(self) -> None:
        # GIVEN the default ``auto`` mode and a builder producing nothing
        sm = _StubSessionManager("just words")
        queue = _RecordingQueue()
        ctx = _FakeContext(_message(Part(root=TextPart(text="hi"))))
        # WHEN it executes
        await _executor(sm, EmptyBuilder()).execute(ctx, queue)  # type: ignore[arg-type]
        # THEN a plain Message is returned and no Task is opened
        assert len(queue.events) == 1
        assert isinstance(queue.events[0], Message)
        assert queue.events[0].parts[0].root.text == "just words"
        assert _states(queue) == []

    @pytest.mark.asyncio
    async def test_returns_message_when_no_builder_configured(self) -> None:
        # GIVEN no builder at all
        queue = _RecordingQueue()
        ctx = _FakeContext(_message(Part(root=TextPart(text="hi"))))
        # WHEN it executes
        await _executor(_StubSessionManager("plain")).execute(ctx, queue)  # type: ignore[arg-type]
        # THEN behaviour matches a message-only agent
        assert isinstance(queue.events[0], Message)
        assert _artifacts(queue) == []

    @pytest.mark.asyncio
    async def test_returns_task_when_artifacts_exist(self) -> None:
        # GIVEN ``auto`` mode and a builder producing artifacts
        queue = _RecordingQueue()
        ctx = _FakeContext(_message(Part(root=TextPart(text="how is AAPL?"))))
        # WHEN it executes
        await _executor(_StubSessionManager(), TwoArtifactBuilder()).execute(ctx, queue)  # type: ignore[arg-type]
        # THEN a Task carrying both artifacts is completed
        assert len(_artifacts(queue)) == 2
        assert _states(queue)[-1] == "completed"

    @pytest.mark.asyncio
    async def test_no_working_event_because_outcome_decides_shape(self) -> None:
        # GIVEN ``auto`` mode, where the response shape depends on the outcome
        queue = _RecordingQueue()
        ctx = _FakeContext(_message(Part(root=TextPart(text="q"))))
        # WHEN it executes
        await _executor(_StubSessionManager(), TwoArtifactBuilder()).execute(ctx, queue)  # type: ignore[arg-type]
        # THEN no ``working`` progress event is emitted -- the documented trade-off
        # of deciding after the workflow has run
        assert "working" not in _states(queue)

    @pytest.mark.asyncio
    async def test_workflow_failure_raises_server_error(self) -> None:
        # GIVEN a workflow that raises, before any task exists
        sm = _StubSessionManager(raises=True)
        # WHEN it executes in ``auto`` mode
        with pytest.raises(ServerError):
            await _executor(sm, TwoArtifactBuilder()).execute(  # type: ignore[arg-type]
                _FakeContext(_message(Part(root=TextPart(text="boom")))), _RecordingQueue()
            )
        # THEN the caller receives a ServerError rather than a bare exception


class TestTaskModeNever:
    @pytest.mark.asyncio
    async def test_returns_message_even_when_artifacts_exist(self) -> None:
        # GIVEN ``never`` mode with a builder that would produce artifacts
        queue = _RecordingQueue()
        ctx = _FakeContext(_message(Part(root=TextPart(text="hi"))))
        # WHEN it executes
        await _executor(_StubSessionManager("text only"), TwoArtifactBuilder(), "never").execute(  # type: ignore[arg-type]
            ctx, queue
        )
        # THEN the artifacts are suppressed and a Message is returned
        assert isinstance(queue.events[0], Message)
        assert _artifacts(queue) == []
        assert _states(queue) == []

    def test_auto_is_the_default(self) -> None:
        # GIVEN an executor constructed without an explicit mode
        executor = TaskArtifactAgentExecutor(_StubSessionManager())
        # WHEN the mode is read
        # THEN it defaults to ``auto``, which is non-breaking for existing callers
        assert executor.task_mode == "auto"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancel:
    @pytest.mark.asyncio
    async def test_no_task_raises_task_not_found(self) -> None:
        # GIVEN a cancellation request with no associated task
        ctx = _FakeContext(_message(Part(root=TextPart(text="x"))), current_task=None)
        # WHEN cancel is called
        with pytest.raises(ServerError) as exc:
            await _executor(_StubSessionManager(), TwoArtifactBuilder()).cancel(  # type: ignore[arg-type]
                ctx, _RecordingQueue()
            )
        # THEN the caller is told there is nothing to cancel
        assert exc.value.error.__class__.__name__ == "TaskNotFoundError"


# ---------------------------------------------------------------------------
# Builder loading
# ---------------------------------------------------------------------------


class TestLoadArtifactBuilder:
    def test_loads_a_valid_builder(self) -> None:
        # GIVEN a dotted path to a conforming builder
        path = f"{TwoArtifactBuilder.__module__}.TwoArtifactBuilder"
        # WHEN it is loaded
        builder = load_artifact_builder(path)
        # THEN an instance satisfying the protocol is returned
        assert isinstance(builder, TwoArtifactBuilder)
        assert isinstance(builder, ArtifactBuilder)

    def test_rejects_class_without_build_artifacts(self) -> None:
        # GIVEN a class that does not implement the protocol
        path = f"{NotABuilder.__module__}.NotABuilder"
        # WHEN it is loaded
        with pytest.raises(ValueError, match="build_artifacts"):
            load_artifact_builder(path)
        # THEN the misconfiguration is reported at startup, not at request time

    def test_rejects_path_without_module(self) -> None:
        # GIVEN a bare name with no module component
        # WHEN it is loaded
        with pytest.raises(ValueError, match="dotted path"):
            load_artifact_builder("NoModule")
        # THEN the malformed path is rejected

    def test_rejects_unimportable_module(self) -> None:
        # GIVEN a module that does not exist
        # WHEN it is loaded
        with pytest.raises(ValueError, match="Could not import"):
            load_artifact_builder("datarobot_genai.does_not_exist.Thing")
        # THEN the import failure is surfaced with the offending path

    def test_rejects_non_class_target(self) -> None:
        # GIVEN a dotted path naming a function rather than a class
        # WHEN it is loaded
        with pytest.raises(ValueError, match="not a class"):
            load_artifact_builder("datarobot_genai.dragent.a2a_artifacts.text_artifact")
        # THEN it is rejected


# ---------------------------------------------------------------------------
# Composition contract with the per-user executor
# ---------------------------------------------------------------------------


class TestPerUserComposition:
    def test_per_user_executor_exposes_a_single_seam(self) -> None:
        # GIVEN the per-user executor
        from datarobot_genai.dragent.frontends.fastapi import _PerUserCompatibleAgentExecutor

        # WHEN its overridable seam is inspected
        # THEN subclasses can change the response without touching execute(), so the
        # identity/header setup cannot be bypassed
        assert hasattr(_PerUserCompatibleAgentExecutor, "_run_request")

    def test_task_artifact_executor_inherits_execute_unchanged(self) -> None:
        # GIVEN the composed task/artifact executor
        from datarobot_genai.dragent.frontends.fastapi import _PerUserCompatibleAgentExecutor
        from datarobot_genai.dragent.frontends.fastapi import _TaskArtifactPerUserExecutor

        # WHEN execute and the seam are compared
        # THEN execute() is inherited verbatim (auth context preserved) and only the
        # seam is overridden -- there is no base-ordering requirement to get wrong
        assert _TaskArtifactPerUserExecutor.execute is _PerUserCompatibleAgentExecutor.execute
        assert (
            _TaskArtifactPerUserExecutor._run_request
            is not _PerUserCompatibleAgentExecutor._run_request
        )

    @pytest.mark.asyncio
    async def test_seam_delegates_to_the_task_artifact_executor(self) -> None:
        # GIVEN a composed executor with a builder
        from datarobot_genai.dragent.frontends.fastapi import _TaskArtifactPerUserExecutor

        executor = _TaskArtifactPerUserExecutor(
            _StubSessionManager(),  # type: ignore[arg-type]
            builder=TwoArtifactBuilder(),
            task_mode="always",
        )
        queue = _RecordingQueue()
        ctx = _FakeContext(_message(Part(root=TextPart(text="q"))))
        # WHEN the seam is invoked directly
        await executor._run_request(ctx, queue)  # type: ignore[arg-type]
        # THEN the task/artifact lifecycle runs
        assert _states(queue)[-1] == "completed"
        assert len(_artifacts(queue)) == 2
