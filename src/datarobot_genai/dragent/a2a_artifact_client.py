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

"""Client-side A2A Files, Images, Data and Artifacts -- the consuming half.

:mod:`datarobot_genai.dragent.a2a_artifacts` lets an agent **return** Tasks and
Artifacts.  This module lets an agent **send** files and **read** artifacts back,
which is the same upstream gap mirrored on the caller's side of the hop:

============  ==========================================================  ==============================
Direction     NAT flattens at                                             Consequence
============  ==========================================================  ==============================
Sending       ``A2ABaseClient.send_message(message_text: str)``            builds a text-only ``Message``
Reading       ``A2ABaseClient.extract_text_from_events()``                 artifacts silently discarded
============  ==========================================================  ==============================

Neither is a protocol limitation.  The raw ``events`` NAT yields already carry
every artifact; what is missing is a way to *put* non-text parts on the wire and
helpers to *get* the non-text parts back off it.  This module supplies both.

Usage
-----
Enable the artifact-aware functions on an authenticated A2A client::

    function_groups:
      finance_agent:
        _type: authenticated_a2a_client
        registry:
          external_id: agent-finance
        auth_provider: okta_auth
        artifact_client: true

That registers two extra functions alongside NAT's text-only ``call``:

``send_with_attachments``
    Sends text plus optional files and structured data, and returns a rendered
    report of the Task and every artifact.
``get_task_artifacts``
    Re-reads the artifacts of a known ``task_id``.

For programmatic access, use the readers directly::

    from datarobot_genai.dragent.a2a_artifact_client import (
        OutboundFile, iter_artifacts, save_task_files,
    )

    events = await group.send_parts(
        text="analyse this",
        files=[OutboundFile("in.csv", "text/csv", content=csv_bytes)],
    )
    for artifact in iter_artifacts(events):
        ...
    paths = save_task_files(events, "/tmp/out")

Rendering vs access
-------------------
:func:`summarize_task` renders artifacts as **text**, because a NAT function's
return value becomes an LLM tool result.  That is a presentation choice, not a
protocol one -- prefer :func:`iter_artifacts` and :func:`save_task_files` when
code, rather than a model, consumes the result.

.. note::
   This module bridges an upstream gap.  If NAT gains native multi-part send and
   artifact reads, prefer those; the surface here is intentionally small so
   migration stays cheap.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from a2a.types import DataPart
from a2a.types import FilePart
from a2a.types import FileWithBytes
from a2a.types import FileWithUri
from a2a.types import Message
from a2a.types import Part
from a2a.types import Role
from a2a.types import TextPart

logger = logging.getLogger(__name__)

__all__ = [
    "OutboundFile",
    "build_client_message",
    "build_send_message_payload",
    "outbound_file_from_uri",
    "iter_artifacts",
    "save_task_files",
    "summarize_task",
]

# Cap the number of bytes a rendered summary will echo per text part, so a large
# artifact cannot blow an LLM's context window through a tool result.
_TEXT_PREVIEW_CHARS = 120


# ---------------------------------------------------------------------------
# Outbound: putting non-text parts on the wire
# ---------------------------------------------------------------------------


@dataclass
class OutboundFile:
    """A file to attach to an outbound A2A message.

    Set :attr:`content` to inline the bytes (A2A ``FileWithBytes``) **or**
    :attr:`uri` to send a reference (``FileWithUri``) -- the two are mutually
    exclusive in the protocol.

    Inline is simpler but base64 inflates the payload by roughly a third, so
    prefer :attr:`uri` for large files. Note that a URI carries no A2A
    authentication: the receiving agent must be able to fetch it independently,
    which usually means a pre-signed URL.

    Attributes:
        name: Filename advertised to the receiving agent.
        mime_type: MIME type, e.g. ``image/png``.
        content: Raw bytes to inline. Base64-encoded for you.
        uri: Remote location, when referencing rather than inlining.

    Raises:
        ValueError: If neither or both of ``content`` and ``uri`` are set.
    """

    name: str
    mime_type: str
    content: bytes | None = None
    uri: str | None = None

    def __post_init__(self) -> None:
        # Compare against None rather than truthiness, so a legitimate zero-byte
        # file (content=b"") is not mistaken for "unset".
        if (self.content is None) == (self.uri is None):
            raise ValueError(
                f"OutboundFile {self.name!r}: set exactly one of 'content' or 'uri' "
                "(A2A FileWithBytes and FileWithUri are mutually exclusive)."
            )

    def to_part(self) -> Part:
        """Return this file as a typed A2A ``FilePart``.

        Returns:
            A ``Part`` wrapping either ``FileWithBytes`` or ``FileWithUri``.
        """
        file_obj: FileWithBytes | FileWithUri
        if self.content is not None:
            file_obj = FileWithBytes(
                bytes=base64.b64encode(self.content).decode("ascii"),
                mime_type=self.mime_type,
                name=self.name,
            )
        else:
            file_obj = FileWithUri(
                uri=self.uri or "", mime_type=self.mime_type, name=self.name
            )
        return Part(root=FilePart(file=file_obj))

    def to_wire(self) -> dict[str, Any]:
        """Return this file as the ``file`` object of a raw JSON-RPC ``FilePart``.

        Note the **camelCase** ``mimeType``: A2A models use snake_case in Python
        but serialise to camelCase on the wire.

        Returns:
            A plain dict, for use with :func:`build_send_message_payload`.
        """
        if self.content is not None:
            return {
                "bytes": base64.b64encode(self.content).decode("ascii"),
                "mimeType": self.mime_type,
                "name": self.name,
            }
        return {"uri": self.uri, "mimeType": self.mime_type, "name": self.name}


def outbound_file_from_uri(uri: str) -> OutboundFile:
    """Build an :class:`OutboundFile` referencing ``uri``, inferring name and type.

    Both are derived from the URI's **path**, never the full URI. Pre-signed URLs
    are the documented way to send a file and they carry a query string, so
    inferring from the whole URI yields no MIME match (everything becomes
    ``application/octet-stream``) and leaves the query blob in the filename.

    Args:
        uri: The location to reference. A query string is ignored for naming.

    Returns
    -------
        An ``OutboundFile`` with ``uri`` set and nothing inlined.

    Examples
    --------
        >>> f = outbound_file_from_uri("https://s3/bucket/q4.csv?X-Amz-Signature=abc")
        >>> f.name, f.mime_type
        ('q4.csv', 'text/csv')
    """
    path = urlparse(uri).path
    name = path.rstrip("/").rsplit("/", 1)[-1] or "attachment"
    return OutboundFile(
        name=name,
        mime_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
        uri=uri,
    )


def build_client_message(
    text: str | None = None,
    files: list[OutboundFile] | None = None,
    data: dict[str, Any] | None = None,
    *,
    task_id: str | None = None,
    context_id: str | None = None,
) -> Message:
    """Build a typed ``Message`` carrying any mix of part kinds.

    This is what NAT's ``send_message(message_text: str)`` cannot express: it
    always constructs a single ``TextPart``.

    Args:
        text: Optional text body, becomes a ``TextPart``.
        files: Optional attachments, each becoming a ``FilePart``.
        data: Optional structured payload, becomes a ``DataPart``.
        task_id: Set to continue an existing task (multi-turn).
        context_id: Set to stay within an existing conversation context.

    Returns:
        A validated ``Message``, ready for the a2a-sdk client.

    Raises:
        ValueError: If no part of any kind was supplied.
    """
    parts: list[Part] = []
    if text:
        parts.append(Part(root=TextPart(text=text)))
    for outbound_file in files or []:
        parts.append(outbound_file.to_part())
    # Compare against None so an intentionally empty dict still produces a part.
    if data is not None:
        parts.append(Part(root=DataPart(data=data)))

    if not parts:
        raise ValueError(
            "An A2A message needs at least one part: pass text, files or data."
        )

    return Message(
        role=Role.user,
        parts=parts,
        message_id=uuid.uuid4().hex,
        task_id=task_id,
        context_id=context_id,
    )


def build_send_message_payload(
    text: str | None = None,
    files: list[OutboundFile] | None = None,
    data: dict[str, Any] | None = None,
    *,
    task_id: str | None = None,
    context_id: str | None = None,
    request_id: int | str = 1,
) -> dict[str, Any]:
    """Build a complete ``message/send`` JSON-RPC request as a plain dict.

    The transparent counterpart to :func:`build_client_message`: the returned
    dict is exactly what goes on the wire, which makes it useful for ``curl``
    reproductions, documentation and tests. Prefer
    :func:`build_client_message` in application code, where pydantic validation
    is worth having.

    Args:
        text: Optional text body.
        files: Optional attachments.
        data: Optional structured payload.
        task_id: Set to continue an existing task.
        context_id: Set to stay within an existing conversation context.
        request_id: JSON-RPC envelope id.

    Returns:
        The JSON-RPC request as a dict, ready for ``json=`` in an HTTP POST.

    Raises:
        ValueError: If no part of any kind was supplied.
    """
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"kind": "text", "text": text})
    for outbound_file in files or []:
        parts.append({"kind": "file", "file": outbound_file.to_wire()})
    if data is not None:
        parts.append({"kind": "data", "data": data})

    if not parts:
        raise ValueError(
            "An A2A message needs at least one part: pass text, files or data."
        )

    message: dict[str, Any] = {
        "role": "user",
        "parts": parts,
        "messageId": uuid.uuid4().hex,
        "kind": "message",
    }
    if task_id:
        message["taskId"] = task_id
    if context_id:
        message["contextId"] = context_id

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "message/send",
        "params": {"message": message},
    }


# ---------------------------------------------------------------------------
# Inbound: recovering the parts NAT's text extraction discards
# ---------------------------------------------------------------------------


def _iter_tasks(events: list[Any]) -> list[Any]:
    """Yield the ``Task`` carried by each event, tolerating both event shapes.

    The a2a-sdk client yields either a bare ``Message`` or a ``ClientEvent``
    tuple of ``(Task, UpdateEvent | None)``.

    Args:
        events: Events as returned by the client.

    Returns:
        Every object that looks like a ``Task``, in arrival order.
    """
    tasks: list[Any] = []
    for event in events:
        candidate = event[0] if isinstance(event, tuple) and event else event
        # Duck-typed rather than isinstance: the SDK's Task is re-exported from
        # several modules, and a strict check makes this brittle across versions.
        if getattr(candidate, "status", None) is not None or hasattr(
            candidate, "artifacts"
        ):
            tasks.append(candidate)
    return tasks


def iter_artifacts(events: list[Any]) -> list[Any]:
    """Collect every ``Artifact`` from a list of A2A response events.

    Handles both response shapes: a bare ``Message`` (which carries no artifacts)
    and ``ClientEvent`` tuples carrying a ``Task``.

    Artifacts are keyed by ``artifact_id``, because a task and its update events
    can both reference the same artifact. A2A permits an artifact to arrive in
    **chunks** (``TaskArtifactUpdateEvent.append``), so repeat sightings of an id
    have their parts *merged* rather than discarded -- taking only the first
    sighting would silently truncate a chunked artifact. Parts already present are
    not duplicated, so a peer that re-sends a complete artifact is handled too.

    Args:
        events: Events returned by a parts-aware send.

    Returns:
        Artifacts in first-seen order, each carrying every part observed for it.
        Empty if the peer replied with a ``Message``, or with a ``Task`` carrying
        none.
    """
    artifacts: list[Any] = []
    by_key: dict[str, Any] = {}
    anonymous = 0

    for task in _iter_tasks(events):
        for artifact in getattr(task, "artifacts", None) or []:
            artifact_id = getattr(artifact, "artifact_id", None)
            if artifact_id:
                key = str(artifact_id)
            else:
                # No id to correlate on, so it cannot be a chunk of anything.
                key = f"_anon_{anonymous}"
                anonymous += 1

            existing = by_key.get(key)
            if existing is None:
                by_key[key] = artifact
                artifacts.append(artifact)
                continue

            # Same artifact seen again: merge any parts we haven't already got.
            known = getattr(existing, "parts", None)
            if known is None:
                continue
            for part in getattr(artifact, "parts", None) or []:
                if part not in known:
                    known.append(part)

    return artifacts


def _decoded_size(b64: str) -> int:
    """Return the decoded byte length of a base64 string, or 0 if unusable.

    Args:
        b64: Base64 text, as carried by ``FileWithBytes.bytes``.

    Returns:
        Decoded length in bytes; 0 when the value is absent or malformed.
    """
    if not b64:
        return 0
    try:
        return len(base64.b64decode(b64, validate=False))
    except (binascii.Error, ValueError):
        return 0


def summarize_task(events: list[Any]) -> str:
    """Render a human-readable report of a Task, including non-text parts.

    Use this instead of NAT's ``extract_text_from_events()`` when files, images
    or structured data matter -- that helper returns text only.

    The output is deliberately compact and text-only, because a NAT function's
    return value becomes an LLM tool result: text previews are truncated and
    file parts are reported by name, type and size rather than inlined. For
    programmatic access use :func:`iter_artifacts`; to persist attachments use
    :func:`save_task_files`.

    Args:
        events: Events returned by a parts-aware send.

    Returns:
        A multi-line report of task state and every artifact part.
    """
    if not events:
        return "No response events."

    lines: list[str] = []

    tasks = _iter_tasks(events)
    if tasks:
        task = tasks[0]
        status = getattr(task, "status", None)
        state = getattr(status, "state", None)
        lines.append(
            f"Task {getattr(task, 'id', '?')} state={getattr(state, 'value', state)}"
        )
    else:
        lines.append("Response was a bare Message (no Task was created).")

    artifacts = iter_artifacts(events)
    if not artifacts:
        lines.append("No artifacts returned.")
        return "\n".join(lines)

    lines.append(f"{len(artifacts)} artifact(s):")
    for artifact in artifacts:
        lines.append(f"  - {getattr(artifact, 'name', None) or '<unnamed>'}")
        for raw_part in getattr(artifact, "parts", None) or []:
            part = raw_part.root if hasattr(raw_part, "root") else raw_part
            kind = getattr(part, "kind", None)

            if kind == "text":
                preview = (getattr(part, "text", "") or "").replace("\n", " ")
                lines.append(f"      text: {preview[:_TEXT_PREVIEW_CHARS]}")
            elif kind == "data":
                try:
                    rendered = json.dumps(getattr(part, "data", None))
                except (TypeError, ValueError):
                    rendered = repr(getattr(part, "data", None))
                lines.append(f"      data: {rendered[:_TEXT_PREVIEW_CHARS]}")
            elif kind == "file":
                file_obj = getattr(part, "file", None)
                name = getattr(file_obj, "name", None) or "<unnamed>"
                mime = getattr(file_obj, "mime_type", None) or "unknown"
                uri = getattr(file_obj, "uri", None)
                if uri:
                    lines.append(f"      file: {name} ({mime}) uri={uri}")
                else:
                    size = _decoded_size(getattr(file_obj, "bytes", "") or "")
                    lines.append(f"      file: {name} ({mime}) {size:,} bytes inline")
            else:  # pragma: no cover - forward compatibility with new part kinds
                lines.append(f"      {kind}: <unhandled part kind>")

    return "\n".join(lines)


def save_task_files(events: list[Any], output_dir: str | Path) -> list[Path]:
    """Write every inline file artifact to disk.

    Only ``FileWithBytes`` parts are written. ``FileWithUri`` parts are skipped,
    because fetching them is the caller's problem: an A2A URI carries none of the
    A2A call's authentication.

    Filenames are flattened to their basename so an artifact cannot write outside
    ``output_dir``.

    Args:
        events: Events returned by a parts-aware send.
        output_dir: Directory to write into. Created if absent.

    Returns:
        Paths written, in artifact order.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for artifact in iter_artifacts(events):
        for raw_part in getattr(artifact, "parts", None) or []:
            part = raw_part.root if hasattr(raw_part, "root") else raw_part
            if getattr(part, "kind", None) != "file":
                continue
            file_obj = getattr(part, "file", None)
            payload = getattr(file_obj, "bytes", None)
            if not payload:
                continue

            # Basename only: a malicious or careless peer must not be able to
            # traverse out of output_dir with a name like "../../etc/passwd".
            raw_name = getattr(file_obj, "name", None) or "artifact.bin"
            safe_name = Path(str(raw_name)).name or "artifact.bin"

            try:
                decoded = base64.b64decode(payload, validate=False)
            except (binascii.Error, ValueError):
                logger.warning(
                    "Skipping artifact file %s: payload is not valid base64", safe_name
                )
                continue

            destination = directory / safe_name
            destination.write_bytes(decoded)
            written.append(destination)

    logger.info("Wrote %d artifact file(s) to %s", len(written), directory)
    return written
