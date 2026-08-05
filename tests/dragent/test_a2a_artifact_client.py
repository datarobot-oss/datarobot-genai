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

"""Tests for the consuming half of A2A artifacts.

Covers putting non-text parts on the wire and recovering them from response
events -- the mirror of ``test_a2a_artifacts.py``, which covers producing them.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from a2a.types import Artifact
from a2a.types import DataPart
from a2a.types import FilePart
from a2a.types import FileWithBytes
from a2a.types import FileWithUri
from a2a.types import Message
from a2a.types import Part
from a2a.types import Role
from a2a.types import Task
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TextPart

from datarobot_genai.dragent.a2a_artifact_client import OutboundFile
from datarobot_genai.dragent.a2a_artifact_client import build_client_message
from datarobot_genai.dragent.a2a_artifact_client import build_send_message_payload
from datarobot_genai.dragent.a2a_artifact_client import iter_artifacts
from datarobot_genai.dragent.a2a_artifact_client import save_task_files
from datarobot_genai.dragent.a2a_artifact_client import summarize_task

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-image-payload"


def _task(
    artifacts: list[Artifact] | None = None,
    state: TaskState = TaskState.completed,
    task_id: str = "task-1",
) -> Task:
    """Build a Task carrying the given artifacts."""
    return Task(
        id=task_id,
        context_id="ctx-1",
        status=TaskStatus(state=state),
        artifacts=artifacts or [],
    )


def _artifact(name: str, parts: list[Part], artifact_id: str = "a-1") -> Artifact:
    """Build an Artifact with an explicit id, for de-duplication tests."""
    return Artifact(artifact_id=artifact_id, name=name, parts=parts)


def _message(text: str = "hello") -> Message:
    """Build a bare agent Message, the artifact-free response shape."""
    return Message(
        role=Role.agent, parts=[Part(root=TextPart(text=text))], message_id="m-1"
    )


class TestOutboundFile:
    def test_inlines_content_as_base64(self) -> None:
        # GIVEN a file carrying raw bytes
        outbound = OutboundFile("chart.png", "image/png", content=PNG_BYTES)
        # WHEN converted to a typed part
        part = outbound.to_part()
        # THEN the bytes are base64-encoded into a FileWithBytes
        assert isinstance(part.root, FilePart)
        assert isinstance(part.root.file, FileWithBytes)
        assert base64.b64decode(part.root.file.bytes) == PNG_BYTES

    def test_references_uri_without_inlining(self) -> None:
        # GIVEN a file carrying a URI
        outbound = OutboundFile("doc.html", "text/html", uri="https://example.com/d")
        # WHEN converted to a typed part
        part = outbound.to_part()
        # THEN it is a FileWithUri and nothing is inlined
        assert isinstance(part.root.file, FileWithUri)
        assert part.root.file.uri == "https://example.com/d"
        assert getattr(part.root.file, "bytes", None) is None

    def test_wire_form_uses_camel_case_mime_type(self) -> None:
        # GIVEN a file with raw bytes
        outbound = OutboundFile("r.csv", "text/csv", content=b"a,b\n")
        # WHEN rendered to the raw JSON-RPC form
        wire = outbound.to_wire()
        # THEN the key is camelCase, per the A2A wire convention
        assert wire["mimeType"] == "text/csv"
        assert "mime_type" not in wire

    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({}, "neither content nor uri"),
            ({"content": b"x", "uri": "https://e.com"}, "both content and uri"),
        ],
    )
    def test_rejects_ambiguous_representations(
        self, kwargs: dict[str, Any], reason: str
    ) -> None:
        # GIVEN neither or both of the two mutually exclusive representations
        # WHEN the file is constructed
        # THEN it fails loudly rather than silently picking one
        with pytest.raises(ValueError, match="exactly one of"):
            OutboundFile("f", "application/octet-stream", **kwargs)

    def test_accepts_a_legitimate_zero_byte_file(self) -> None:
        # GIVEN an empty file, which is falsy but not unset
        outbound = OutboundFile("empty.txt", "text/plain", content=b"")
        # WHEN converted
        part = outbound.to_part()
        # THEN it is treated as inline content, not as "no content supplied"
        assert base64.b64decode(part.root.file.bytes) == b""


class TestBuildClientMessage:
    def test_carries_every_part_kind_in_order(self) -> None:
        # GIVEN text, a file and structured data
        message = build_client_message(
            text="analyse",
            files=[OutboundFile("in.png", "image/png", content=PNG_BYTES)],
            data={"threshold": 0.5},
        )
        # WHEN the parts are inspected
        kinds = [p.root.kind for p in message.parts]
        # THEN all three are present, text first
        assert kinds == ["text", "file", "data"]
        assert message.role == Role.user

    def test_omits_empty_text_but_keeps_empty_data(self) -> None:
        # GIVEN no text and a deliberately empty data payload
        message = build_client_message(data={})
        # WHEN the parts are inspected
        # THEN the empty dict still produces a DataPart -- {} is a value, not "unset"
        assert [p.root.kind for p in message.parts] == ["data"]

    def test_threads_task_and_context_ids(self) -> None:
        # GIVEN ids for an in-flight conversation
        message = build_client_message(text="more", task_id="t-9", context_id="c-9")
        # WHEN the message is built
        # THEN both are carried, so the turn continues rather than starting fresh
        assert message.task_id == "t-9"
        assert message.context_id == "c-9"

    def test_rejects_a_message_with_no_parts(self) -> None:
        # GIVEN nothing to send
        # WHEN a message is requested
        # THEN it fails rather than sending an empty message the peer must reject
        with pytest.raises(ValueError, match="at least one part"):
            build_client_message()

    def test_generates_a_unique_message_id_per_call(self) -> None:
        # GIVEN two identical sends
        first = build_client_message(text="same")
        second = build_client_message(text="same")
        # WHEN their ids are compared
        # THEN they differ, so the peer cannot treat the second as a replay
        assert first.message_id != second.message_id


class TestBuildSendMessagePayload:
    def test_matches_the_json_rpc_envelope(self) -> None:
        # GIVEN a text and data send
        payload = build_send_message_payload(text="hi", data={"a": 1}, request_id=7)
        # WHEN the envelope is inspected
        # THEN it is a well-formed message/send request
        assert payload["jsonrpc"] == "2.0"
        assert payload["id"] == 7
        assert payload["method"] == "message/send"
        assert payload["params"]["message"]["kind"] == "message"

    def test_is_json_serialisable(self) -> None:
        # GIVEN a payload including binary content
        payload = build_send_message_payload(
            files=[OutboundFile("a.png", "image/png", content=PNG_BYTES)]
        )
        # WHEN serialised
        # THEN it round-trips, because bytes were base64-encoded
        assert json.loads(json.dumps(payload)) == payload

    def test_rejects_a_payload_with_no_parts(self) -> None:
        # GIVEN nothing to send
        # WHEN a payload is requested
        # THEN it fails, consistent with build_client_message
        with pytest.raises(ValueError, match="at least one part"):
            build_send_message_payload()


class TestIterArtifacts:
    def test_collects_artifacts_from_a_client_event_tuple(self) -> None:
        # GIVEN a ClientEvent tuple, the shape the sdk yields for tasks
        task = _task([_artifact("summary", [Part(root=TextPart(text="ok"))])])
        # WHEN artifacts are collected
        artifacts = iter_artifacts([(task, None)])
        # THEN the tuple is unwrapped and the artifact found
        assert [a.name for a in artifacts] == ["summary"]

    def test_collects_artifacts_from_a_bare_task(self) -> None:
        # GIVEN a Task not wrapped in a tuple
        task = _task([_artifact("summary", [Part(root=TextPart(text="ok"))])])
        # WHEN artifacts are collected
        # THEN both event shapes are handled
        assert len(iter_artifacts([task])) == 1

    def test_returns_nothing_for_a_bare_message(self) -> None:
        # GIVEN a Message response, which cannot carry artifacts
        # WHEN artifacts are collected
        # THEN the result is empty rather than an error
        assert iter_artifacts([_message()]) == []

    def test_deduplicates_by_artifact_id(self) -> None:
        # GIVEN a task and an update event referencing the same artifact
        artifact = _artifact("chart", [Part(root=TextPart(text="x"))], artifact_id="dup")
        task = _task([artifact])
        # WHEN artifacts are collected across both events
        artifacts = iter_artifacts([(task, None), (task, None)])
        # THEN it appears once, not twice
        assert len(artifacts) == 1

    def test_preserves_distinct_artifacts(self) -> None:
        # GIVEN two different artifacts
        task = _task(
            [
                _artifact("one", [Part(root=TextPart(text="1"))], artifact_id="a"),
                _artifact("two", [Part(root=TextPart(text="2"))], artifact_id="b"),
            ]
        )
        # WHEN collected
        # THEN both survive, in order
        assert [a.name for a in iter_artifacts([task])] == ["one", "two"]

    def test_handles_no_events(self) -> None:
        # GIVEN an empty event list
        # WHEN artifacts are collected
        # THEN the result is empty
        assert iter_artifacts([]) == []


class TestSummarizeTask:
    def test_reports_state_and_every_part_kind(self) -> None:
        # GIVEN a completed task with all three part kinds
        task = _task(
            [
                _artifact(
                    "deliverable",
                    [
                        Part(root=TextPart(text="the answer")),
                        Part(root=DataPart(data={"k": "v"})),
                        Part(
                            root=FilePart(
                                file=FileWithBytes(
                                    bytes=base64.b64encode(PNG_BYTES).decode(),
                                    mime_type="image/png",
                                    name="chart.png",
                                )
                            )
                        ),
                    ],
                )
            ]
        )
        # WHEN summarised
        summary = summarize_task([(task, None)])
        # THEN state, artifact name and each part kind are all reported
        assert "state=completed" in summary
        assert "1 artifact(s)" in summary
        assert "deliverable" in summary
        assert "text: the answer" in summary
        assert '"k": "v"' in summary
        assert f"chart.png (image/png) {len(PNG_BYTES):,} bytes inline" in summary

    def test_reports_a_uri_file_without_a_size(self) -> None:
        # GIVEN a file sent by reference
        task = _task(
            [
                _artifact(
                    "ref",
                    [
                        Part(
                            root=FilePart(
                                file=FileWithUri(
                                    uri="https://example.com/d",
                                    mime_type="text/html",
                                    name="d.html",
                                )
                            )
                        )
                    ],
                )
            ]
        )
        # WHEN summarised
        summary = summarize_task([task])
        # THEN the URI is shown and no byte count is claimed
        assert "uri=https://example.com/d" in summary
        assert "bytes inline" not in summary

    def test_truncates_long_text_previews(self) -> None:
        # GIVEN an artifact with a very long text part
        task = _task([_artifact("big", [Part(root=TextPart(text="x" * 5000))])])
        # WHEN summarised
        summary = summarize_task([task])
        # THEN the preview is capped, so a tool result cannot flood an LLM context
        assert len(summary) < 500

    def test_distinguishes_a_message_response(self) -> None:
        # GIVEN a bare Message
        # WHEN summarised
        summary = summarize_task([_message()])
        # THEN it says so explicitly, rather than reporting an empty task
        assert "bare Message" in summary

    def test_reports_a_task_with_no_artifacts(self) -> None:
        # GIVEN a completed task that produced nothing
        # WHEN summarised
        summary = summarize_task([_task([])])
        # THEN state is reported and the absence is explicit
        assert "state=completed" in summary
        assert "No artifacts returned." in summary

    def test_handles_no_events(self) -> None:
        # GIVEN no events at all
        # WHEN summarised
        # THEN it degrades to a readable message instead of raising
        assert summarize_task([]) == "No response events."


class TestSaveTaskFiles:
    def test_writes_inline_files_with_original_bytes(self, tmp_path: Any) -> None:
        # GIVEN an artifact carrying an inline file
        task = _task(
            [
                _artifact(
                    "out",
                    [
                        Part(
                            root=FilePart(
                                file=FileWithBytes(
                                    bytes=base64.b64encode(PNG_BYTES).decode(),
                                    mime_type="image/png",
                                    name="chart.png",
                                )
                            )
                        )
                    ],
                )
            ]
        )
        # WHEN saved
        written = save_task_files([task], tmp_path)
        # THEN the file exists with byte-identical content
        assert [p.name for p in written] == ["chart.png"]
        assert written[0].read_bytes() == PNG_BYTES

    def test_skips_uri_files(self, tmp_path: Any) -> None:
        # GIVEN a file sent by reference, which carries no bytes to write
        task = _task(
            [
                _artifact(
                    "ref",
                    [
                        Part(
                            root=FilePart(
                                file=FileWithUri(
                                    uri="https://example.com/x",
                                    mime_type="text/html",
                                    name="x.html",
                                )
                            )
                        )
                    ],
                )
            ]
        )
        # WHEN saved
        # THEN nothing is written -- fetching a URI is the caller's problem, since
        # it carries none of the A2A call's authentication
        assert save_task_files([task], tmp_path) == []

    def test_flattens_paths_to_prevent_traversal(self, tmp_path: Any) -> None:
        # GIVEN a hostile artifact filename attempting directory traversal
        task = _task(
            [
                _artifact(
                    "evil",
                    [
                        Part(
                            root=FilePart(
                                file=FileWithBytes(
                                    bytes=base64.b64encode(b"pwned").decode(),
                                    mime_type="text/plain",
                                    name="../../etc/passwd",
                                )
                            )
                        )
                    ],
                )
            ]
        )
        # WHEN saved
        written = save_task_files([task], tmp_path)
        # THEN the file lands inside the output directory under its basename only
        assert written[0].parent == tmp_path
        assert written[0].name == "passwd"

    def test_creates_the_output_directory(self, tmp_path: Any) -> None:
        # GIVEN a target directory that does not exist
        target = tmp_path / "nested" / "out"
        task = _task(
            [
                _artifact(
                    "out",
                    [
                        Part(
                            root=FilePart(
                                file=FileWithBytes(
                                    bytes=base64.b64encode(b"x").decode(),
                                    mime_type="text/plain",
                                    name="a.txt",
                                )
                            )
                        )
                    ],
                )
            ]
        )
        # WHEN saved
        save_task_files([task], target)
        # THEN the directory was created rather than raising
        assert target.is_dir()

    def test_skips_malformed_base64_without_failing_the_batch(
        self, tmp_path: Any
    ) -> None:
        # GIVEN one corrupt file and one valid file
        task = _task(
            [
                _artifact(
                    "mixed",
                    [
                        Part(
                            root=FilePart(
                                file=FileWithBytes(
                                    bytes="!!!not-base64!!!",
                                    mime_type="application/octet-stream",
                                    name="bad.bin",
                                )
                            )
                        ),
                        Part(
                            root=FilePart(
                                file=FileWithBytes(
                                    bytes=base64.b64encode(b"good").decode(),
                                    mime_type="text/plain",
                                    name="good.txt",
                                )
                            )
                        ),
                    ],
                )
            ]
        )
        # WHEN saved
        written = save_task_files([task], tmp_path)
        # THEN the good file still lands: one bad attachment cannot fail the rest
        assert [p.name for p in written] == ["good.txt"]
