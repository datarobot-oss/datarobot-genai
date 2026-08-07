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

"""A2A Tasks, Artifacts, Files and Images for DRAgent front ends.

NAT's :class:`~nat.plugins.a2a.server.agent_executor_adapter.NATWorkflowAgentExecutor`
is explicitly *Phase 1 / message-only*: ``get_user_input()`` reads text and drops
inbound files, ``to_type=str`` flattens the result, and it returns a bare
``Message``.  Because A2A ``Artifact`` objects only exist on a ``Task``, artifacts
are structurally impossible through that path -- not merely unsupported.

This module supplies the missing half:

* :func:`extract_request_inputs` parses **every** inbound part kind (text, file,
  data), rather than text alone.
* :class:`TaskArtifactAgentExecutor` owns the task lifecycle and publishes
  artifacts, while inheriting the per-user identity and header forwarding that
  Okta cross-application access depends on.
* An application supplies only an :class:`ArtifactBuilder` -- one method, no base
  class -- and the module-level ``*_artifact`` helpers to construct its outputs.

Usage
-----
Implement a builder::

    from datarobot_genai.dragent.a2a_artifacts import data_artifact, file_artifact

    class FinanceArtifacts:
        async def build_artifacts(self, inputs, response_text):
            return [
                data_artifact("analysis", {"symbol": "AAPL"}),
                file_artifact("chart.png", png_bytes, "image/png"),
            ]

Then reference it from ``workflow.yaml``; no custom front-end plugin and no
subclassing are required::

    general:
      front_end:
        _type: dragent_fastapi
        a2a:
          server:
            name: My Agent
          artifact_builder: "myapp.finance.FinanceArtifacts"
          task_mode: auto

Response shape
--------------
Both ``Task`` and ``Message`` are valid results for ``message/send``
(``SendMessageSuccessResponse.result: Task | Message``), so the response shape is
selectable -- see :class:`TaskMode`.

.. note::
   This module bridges an upstream gap.  If NAT gains native task and artifact
   support, prefer that; the surface here is intentionally small so migration
   stays cheap.
"""

from __future__ import annotations

import base64
import binascii
import logging
import typing
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart
from a2a.types import FilePart
from a2a.types import FileWithBytes
from a2a.types import FileWithUri
from a2a.types import InternalError
from a2a.types import InvalidParamsError
from a2a.types import Part
from a2a.types import TaskNotFoundError
from a2a.types import TextPart
from a2a.utils import new_agent_text_message
from a2a.utils import new_task
from a2a.utils.errors import ServerError

logger = logging.getLogger(__name__)

TaskMode = typing.Literal["auto", "always", "never"]
"""Response strategy for :class:`TaskArtifactAgentExecutor`.

``"auto"`` (default)
    Run the workflow, then decide: no artifacts -> return a ``Message``; one or
    more artifacts -> return a ``Task`` carrying them.  Because the decision needs
    the outcome, no ``submitted``/``working`` progress events are emitted.
    Non-breaking for callers that expect a message.
``"always"``
    Create the ``Task`` up front and emit ``submitted -> working -> completed``.
    Use when callers need progress on long-running work, or when the agent always
    produces artifacts.
``"never"``
    Never open a task; always return a ``Message`` built from the workflow text.
"""

__all__ = [
    "A2ARequestInputs",
    "ArtifactBuilder",
    "InboundFile",
    "OutboundArtifact",
    "TaskArtifactAgentExecutor",
    "TaskMode",
    "artifact",
    "data_artifact",
    "extract_request_inputs",
    "file_artifact",
    "file_uri_artifact",
    "mixed_artifact",
    "text_artifact",
]


# ---------------------------------------------------------------------------
# Inbound request parsing (the half NAT's get_user_input() discards)
# ---------------------------------------------------------------------------


@dataclass
class InboundFile:
    """A ``FilePart`` received from the calling agent.

    Exactly one of :attr:`content` / :attr:`uri` is populated, mirroring the A2A
    spec's ``FileWithBytes`` vs ``FileWithUri`` union.

    Attributes
    ----------
        name: Client-supplied filename, if any.
        mime_type: Declared MIME type, e.g. ``image/png``.
        content: Decoded bytes when the sender inlined the file (``FileWithBytes``).
        uri: Remote location when the sender passed a reference (``FileWithUri``).
    """

    name: str | None
    mime_type: str | None
    content: bytes | None = None
    uri: str | None = None

    @property
    def is_inline(self) -> bool:
        """True when the payload arrived as inline bytes rather than a URI."""
        return self.content is not None

    @property
    def size(self) -> int:
        """Decoded size in bytes; ``0`` for URI references."""
        return len(self.content) if self.content else 0

    def describe(self) -> str:
        """Return a short human-readable summary for logs and text artifacts."""
        where = f"{self.size} bytes inline" if self.is_inline else f"uri={self.uri}"
        return f"{self.name or '<unnamed>'} ({self.mime_type or 'unknown'}, {where})"


@dataclass
class A2ARequestInputs:
    """Everything the caller actually sent, not just the text.

    An :class:`ArtifactBuilder` decides *what to return*, so it needs to see what
    was asked. This carries the request itself plus the conversational context
    around it, so a builder can vary its output per request instead of returning a
    fixed set.

    Attributes
    ----------
        text: Concatenated text of every ``TextPart`` in the request.
        files: Every ``FilePart``, decoded into :class:`InboundFile`.
        data: The ``data`` payload of every ``DataPart``.
        message_id: The inbound message's id.
        task_id: The task this message continues, or ``None`` on a first turn.
        context_id: Conversation grouping id. Stable across turns, unlike
            ``task_id`` — key on this to correlate a multi-turn exchange.
        metadata: The inbound message's ``metadata`` object, if any. Callers use
            this to pass hints out of band, e.g. requested output formats.
        history: Earlier ``Message`` objects on the continued task, oldest first.
            Empty on a first turn, and empty if the client did not request
            history. Raw A2A models, not flattened — use :meth:`history_text`
            for the common case.
        context: The raw ``RequestContext``. An escape hatch for anything not
            surfaced above; prefer the named fields, which are stable.
    """

    text: str = ""
    files: list[InboundFile] = field(default_factory=list)
    data: list[dict[str, Any]] = field(default_factory=list)
    message_id: str | None = None
    task_id: str | None = None
    context_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[Any] = field(default_factory=list)
    context: Any = None

    @property
    def has_attachments(self) -> bool:
        """True when the request carried any file or structured-data part."""
        return bool(self.files or self.data)

    @property
    def is_follow_up(self) -> bool:
        """True when this message continues an existing task."""
        return bool(self.task_id)

    def history_text(self, limit: int | None = None) -> list[str]:
        """Return prior turns as plain strings, newest last.

        Flattens each historical ``Message`` to the text of its ``TextPart``s,
        which is what a builder usually wants when deciding whether something was
        already sent. Non-text parts are ignored here; read :attr:`history`
        directly if they matter.

        Args:
            limit: Return at most this many of the most recent turns. ``None``
                returns all of them.

        Returns
        -------
            One string per historical message that had any text, oldest first.
        """
        out: list[str] = []
        for message in self.history:
            texts = [
                getattr(_unwrap(p), "text", "")
                for p in getattr(message, "parts", None) or []
                if getattr(_unwrap(p), "kind", None) == "text"
            ]
            joined = "\n".join(t for t in texts if t)
            if joined:
                out.append(joined)
        return out[-limit:] if limit else out

    def asked_for(self, *keywords: str) -> bool:
        """True when the request text mentions any of ``keywords``.

        A convenience for the most common branch in a builder -- deciding whether
        to build an expensive artifact at all. Case-insensitive substring match;
        deliberately naive, since anything smarter belongs in the workflow's LLM
        rather than in artifact assembly.

        Args:
            *keywords: Terms to look for.

        Returns
        -------
            True if any keyword appears in the request text.
        """
        haystack = self.text.lower()
        return any(k.lower() in haystack for k in keywords)


def _unwrap(part: Any) -> Any:
    """Return the concrete part model, unwrapping the ``Part`` RootModel if needed.

    ``a2a.types.Part`` is a pydantic ``RootModel`` union of
    ``TextPart | FilePart | DataPart`` (distinguished in JSON by the ``kind``
    literal), so the useful object lives at ``.root``.  Some call sites hand us
    the inner model directly, so tolerate both shapes.
    """
    return part.root if hasattr(part, "root") else part


def _b64decode_tolerant(payload: str, *, context: str) -> bytes | None:
    """Base64-decode, retrying leniently before giving up.

    Strict decoding rejects payloads containing newlines or using the URL-safe
    alphabet, both of which appear in the wild.  Try strict first (fast, catches
    genuine corruption), then a lenient pass, then base64url.

    Args:
        payload: The base64 text to decode.
        context: Identifier used in the warning log when decoding fails.

    Returns
    -------
        Decoded bytes, or ``None`` if every strategy failed.
    """
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        pass

    # Lenient: ignores stray whitespace/newlines from pretty-printed JSON.
    try:
        return base64.b64decode(payload)
    except (binascii.Error, ValueError):
        pass

    # base64url alphabet ('-' and '_' instead of '+' and '/').
    try:
        padded = payload + "=" * (-len(payload) % 4)
        return base64.urlsafe_b64decode(padded)
    except (binascii.Error, ValueError) as exc:
        logger.warning("Could not base64-decode %s: %s", context, exc)
        return None


def _decode_file_part(part: FilePart) -> InboundFile:
    """Convert a ``FilePart`` into an :class:`InboundFile`, decoding base64 bytes.

    Invalid base64 is downgraded to a URI-less, content-less entry with a warning
    so a single bad attachment does not abort the task.
    """
    file_obj = part.file
    name = getattr(file_obj, "name", None)
    mime = getattr(file_obj, "mime_type", None)

    uri = getattr(file_obj, "uri", None)
    if uri:
        return InboundFile(name=name, mime_type=mime, uri=uri)

    raw = getattr(file_obj, "bytes", None)
    if not raw:
        return InboundFile(name=name, mime_type=mime)

    # The A2A wire format carries FileWithBytes.bytes as a base64 string.
    decoded = _b64decode_tolerant(raw, context=f"FilePart {name!r}")
    return InboundFile(name=name, mime_type=mime, content=decoded)


def extract_request_inputs(context: RequestContext) -> A2ARequestInputs:
    """Parse **all** part kinds out of an inbound A2A request.

    This is the deliberate replacement for NAT's ``context.get_user_input()``,
    which returns text only and therefore silently drops files, images and
    structured data.

    Args:
        context: The A2A request context handed to ``execute()``.

    Returns
    -------
        An :class:`A2ARequestInputs` with text, files and data separated.  Parts
        that cannot be decoded are logged and skipped rather than raising, so one
        malformed attachment cannot fail the whole request.
    """
    inputs = A2ARequestInputs(context=context)

    # Conversational context, so a builder can vary output per request rather than
    # returning a fixed set. Read defensively: RequestContext is populated by the
    # a2a-sdk request handler and not every field is set on every call path.
    task = getattr(context, "current_task", None)
    inputs.task_id = getattr(context, "task_id", None) or getattr(task, "id", None)
    inputs.context_id = getattr(context, "context_id", None) or getattr(
        task, "context_id", None
    )
    if task is not None:
        # A2A Tasks carry their prior Messages. Present only when the client asked
        # for history, so treat an empty list as "unknown", not "first turn".
        inputs.history = list(getattr(task, "history", None) or [])

    message = getattr(context, "message", None)
    if message is None or not getattr(message, "parts", None):
        return inputs

    inputs.message_id = getattr(message, "message_id", None)
    inputs.task_id = inputs.task_id or getattr(message, "task_id", None)
    inputs.context_id = inputs.context_id or getattr(message, "context_id", None)
    message_metadata = getattr(message, "metadata", None)
    if isinstance(message_metadata, dict):
        inputs.metadata = dict(message_metadata)

    texts: list[str] = []
    for raw in message.parts:
        part = _unwrap(raw)
        kind = getattr(part, "kind", None)

        if kind == "text" or isinstance(part, TextPart):
            texts.append(part.text)

        elif kind == "file" or isinstance(part, FilePart):
            inputs.files.append(_decode_file_part(part))

        elif kind == "data" or isinstance(part, DataPart):
            payload = getattr(part, "data", None)
            if isinstance(payload, dict):
                inputs.data.append(payload)
            else:  # pragma: no cover - defensive, spec says data is an object
                logger.warning("Ignoring DataPart with non-object payload: %r", type(payload))

    inputs.text = "\n".join(t for t in texts if t)
    if inputs.has_attachments:
        logger.info(
            "Inbound A2A request carried %d file part(s) and %d data part(s)",
            len(inputs.files),
            len(inputs.data),
        )
    logger.debug(
        "A2A request context: context_id=%s task_id=%s follow_up=%s "
        "history_turns=%d metadata_keys=%s",
        inputs.context_id,
        inputs.task_id,
        inputs.is_follow_up,
        len(inputs.history),
        sorted(inputs.metadata) or "-",
    )
    return inputs


# ---------------------------------------------------------------------------
# Outbound artifacts
# ---------------------------------------------------------------------------


@dataclass
class OutboundArtifact:
    """One artifact to publish on the task.

    An artifact is a *named container* of :class:`~a2a.types.Part` objects -- it is
    not itself binary data.  Binary payloads travel inside a ``FilePart``.  A
    single artifact may hold several part kinds at once; see :func:`mixed_artifact`.

    Prefer the module-level helpers (:func:`text_artifact`, :func:`data_artifact`,
    :func:`file_artifact`, :func:`file_uri_artifact`, :func:`mixed_artifact`) over
    constructing this directly.

    Attributes
    ----------
        name: Artifact name shown to the caller.
        parts: The wrapped parts carried by this artifact.
        metadata: Optional metadata published alongside the artifact.
    """

    name: str
    parts: list[Part]
    metadata: dict[str, Any] | None = None


def artifact(
    name: str, parts: list[Part], metadata: dict[str, Any] | None = None
) -> OutboundArtifact:
    """Bundle pre-built parts into one artifact (mixed part kinds allowed).

    Args:
        name: Artifact name shown to the caller.
        parts: Already-wrapped ``Part`` objects.
        metadata: Optional metadata dict published with the artifact.
    """
    return OutboundArtifact(name=name, parts=parts, metadata=metadata)


def text_artifact(name: str, text: str, metadata: dict[str, Any] | None = None) -> OutboundArtifact:
    """Artifact containing a single ``TextPart``."""
    return artifact(name, [Part(root=TextPart(text=text))], metadata)


def data_artifact(
    name: str, data: dict[str, Any], metadata: dict[str, Any] | None = None
) -> OutboundArtifact:
    """Artifact containing a single ``DataPart`` (machine-readable JSON)."""
    return artifact(name, [Part(root=DataPart(data=data))], metadata)


def file_artifact(
    name: str,
    content: bytes,
    mime_type: str,
    *,
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> OutboundArtifact:
    """Artifact containing an **inline** file (``FileWithBytes``).

    Bytes are base64-encoded here because that is what the A2A wire format
    requires -- which inflates the payload by roughly a third.  Best for small
    payloads; prefer :func:`file_uri_artifact` for large files.

    Args:
        name: Artifact name.
        content: Raw file bytes.
        mime_type: MIME type, e.g. ``image/png`` or ``text/csv``.
        filename: Name carried inside the file part (defaults to ``name``).
        metadata: Optional artifact metadata.
    """
    encoded = base64.b64encode(content).decode("ascii")
    part = FilePart(file=FileWithBytes(bytes=encoded, mime_type=mime_type, name=filename or name))
    return artifact(name, [Part(root=part)], metadata)


def file_uri_artifact(
    name: str,
    uri: str,
    mime_type: str,
    *,
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> OutboundArtifact:
    """Artifact referencing a file by URI (``FileWithUri``).

    Nothing is inlined, so this scales to large artifacts -- but the caller must be
    able to reach *and* be authorised for the URI.  A URI inherits no A2A
    credentials, so use a pre-signed URL or a separately authorised endpoint.

    Args:
        name: Artifact name.
        uri: Location of the file.
        mime_type: MIME type of the referenced file.
        filename: Name carried inside the file part (defaults to ``name``).
        metadata: Optional artifact metadata.
    """
    part = FilePart(file=FileWithUri(uri=uri, mime_type=mime_type, name=filename or name))
    return artifact(name, [Part(root=part)], metadata)


def mixed_artifact(
    name: str,
    *,
    text: str | None = None,
    data: dict[str, Any] | None = None,
    files: list[tuple[str, bytes, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> OutboundArtifact:
    """One artifact holding several part kinds at once.

    This is the shape most real deliverables take: a human-readable summary, the
    machine-readable equivalent, and the generated file(s) together.

    Args:
        name: Artifact name.
        text: Optional summary text.
        data: Optional structured payload.
        files: Optional list of ``(filename, content, mime_type)`` tuples, inlined
            as ``FileWithBytes``.
        metadata: Optional artifact metadata.
    """
    parts: list[Part] = []
    if text is not None:
        parts.append(Part(root=TextPart(text=text)))
    if data is not None:
        parts.append(Part(root=DataPart(data=data)))
    for filename, content, mime_type in files or []:
        parts.append(
            Part(
                root=FilePart(
                    file=FileWithBytes(
                        bytes=base64.b64encode(content).decode("ascii"),
                        mime_type=mime_type,
                        name=filename,
                    )
                )
            )
        )
    return artifact(name, parts, metadata)


# ---------------------------------------------------------------------------
# The application's contribution
# ---------------------------------------------------------------------------


@runtime_checkable
class ArtifactBuilder(Protocol):
    """What an application implements to publish artifacts.

    One method, no base class, no imports from private modules::

        class FinanceArtifacts:
            async def build_artifacts(self, inputs, response_text):
                return [data_artifact("analysis", {"symbol": "AAPL"})]

    Returning an empty list means "nothing task-shaped to report", which under
    ``task_mode="auto"`` yields a plain ``Message``.
    """

    async def build_artifacts(
        self, inputs: A2ARequestInputs, response_text: str
    ) -> list[OutboundArtifact]:
        """Describe the artifacts this agent returns.

        Args:
            inputs: Everything the caller sent (text, files, structured data).
            response_text: The workflow's textual result.

        Returns
        -------
            The artifacts to publish, in order.  Empty means none.
        """
        ...


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def _import_class(dotted_path: str, *, what: str) -> type:
    """Import ``pkg.module.Class`` and return the class.

    Raises
    ------
        ValueError: If the path is malformed, unimportable, or not a class.
    """
    module_name, _, class_name = dotted_path.rpartition(".")
    if not module_name:
        raise ValueError(f"{what} must be a dotted path to a class, got {dotted_path!r}")
    try:
        module = __import__(module_name, fromlist=[class_name])
        obj = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Could not import {what} {dotted_path!r}: {exc}") from exc
    if not isinstance(obj, type):
        raise ValueError(f"{what} {dotted_path!r} is not a class")
    return obj


def load_artifact_builder(dotted_path: str) -> ArtifactBuilder:
    """Import and instantiate an :class:`ArtifactBuilder` from a dotted path.

    Args:
        dotted_path: e.g. ``"myapp.finance.FinanceArtifacts"``.

    Returns
    -------
        A builder instance.

    Raises
    ------
        ValueError: If the path cannot be imported, is not a class, or the class
            does not implement ``build_artifacts``.
    """
    builder_cls = _import_class(dotted_path, what="a2a.artifact_builder")
    if not hasattr(builder_cls, "build_artifacts"):
        raise ValueError(
            f"a2a.artifact_builder {dotted_path!r} does not implement "
            "build_artifacts(inputs, response_text)"
        )
    return typing.cast(ArtifactBuilder, builder_cls())


class TaskArtifactAgentExecutor:
    """A2A executor that publishes Tasks and Artifacts.

    Owns the whole lifecycle -- task creation, state transitions, artifact
    publication, failure handling and cancellation -- so an application supplies
    only an :class:`ArtifactBuilder`.

    This class is **concrete and public**.  Subclass it directly if you need to
    change how the workflow is invoked (override :meth:`run_workflow`); there is
    no base-ordering requirement to get wrong.

    .. note::
       The front end composes this over the per-user executor, which supplies the
       identity resolution and inbound-header forwarding that Okta
       cross-application access depends on.  See
       :func:`datarobot_genai.dragent.frontends.fastapi.DRAgentFastApiFrontEndPlugin`.

    Args:
        session_manager: NAT session manager used to run the workflow.
        builder: Supplies the artifacts. When ``None``, no artifacts are produced,
            which under ``task_mode="auto"`` reproduces message-only behaviour.
        task_mode: Response strategy; see :data:`TaskMode`.
        default_artifact_name: Name for the fallback text artifact used in
            ``task_mode="always"`` when the builder returns nothing.
    """

    def __init__(
        self,
        session_manager: Any,
        *,
        builder: ArtifactBuilder | None = None,
        task_mode: TaskMode = "auto",
        default_artifact_name: str = "response",
    ) -> None:
        self.session_manager = session_manager
        self.builder = builder
        self.task_mode: TaskMode = task_mode
        self.default_artifact_name = default_artifact_name

    # -- lifecycle ---------------------------------------------------------

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run the workflow and publish either a ``Task`` or a ``Message``.

        Args:
            context: Inbound A2A request context.
            event_queue: Queue the A2A server drains to stream events to the caller.

        Raises
        ------
            ServerError: ``InvalidParamsError`` for a malformed request,
                ``InternalError`` if the workflow raises.
        """
        if not getattr(context, "message", None):
            logger.error("A2A request context has no message")
            raise ServerError(error=InvalidParamsError(message="Request contained no message."))

        if self.task_mode == "always":
            await self._execute_as_task(context, event_queue)
        else:
            await self._execute_and_choose(context, event_queue)

    async def _run(
        self, context: RequestContext
    ) -> tuple[A2ARequestInputs, str, list[OutboundArtifact]]:
        """Parse the request, run the workflow, and collect artifacts."""
        inputs = extract_request_inputs(context)
        response_text = await self.run_workflow(inputs, context)
        artifacts = await self.build_artifacts(inputs, response_text)
        return inputs, response_text, artifacts

    async def _execute_and_choose(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run first, then return a ``Message`` or a ``Task`` based on the outcome.

        Trade-off: because the choice depends on whether artifacts exist, the task
        cannot be *published* before the workflow runs, so no ``working`` event is
        emitted.  Use ``task_mode="always"`` when progress matters more.

        The task is still *built* up front, though never published unless needed:
        ``new_task()`` validates the message, and discovering a malformed request
        only after the workflow had run would turn a successful execution into an
        ``InvalidParamsError`` and discard the artifacts with it.
        """
        # Validate now, publish later (or never). Skipped when continuing an
        # existing task, since that message was validated on its own turn.
        prepared_task = None if context.current_task is not None else self._build_task(context)

        try:
            _, response_text, artifacts = await self._run(context)
        except Exception as exc:
            logger.error("A2A request failed: %s", exc, exc_info=True)
            raise ServerError(error=InternalError()) from exc

        if self.task_mode == "never" or not artifacts:
            # Nothing task-shaped to return: a plain Message is cheaper and is what
            # callers of a message-only agent already expect.
            if artifacts:
                # task_mode="never" is explicit, so honour it -- but this is a
                # silent data-loss path if it was set by mistake, and A2A gives us
                # nowhere to put artifacts on a Message.
                logger.warning(
                    "task_mode='never' is discarding %d artifact(s) built for this "
                    "request; artifacts can only be carried by a Task. Use "
                    "task_mode='auto' to return them when they exist.",
                    len(artifacts),
                )
            await event_queue.enqueue_event(
                new_agent_text_message(response_text, context_id=context.context_id, task_id=None)
            )
            logger.info("A2A request answered with a Message (%d artifacts)", len(artifacts))
            return

        task = await self._ensure_task(context, event_queue, prepared_task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            await self._publish_artifacts(updater, artifacts)
            await updater.complete()
            logger.info("A2A task %s completed with %d artifact(s)", task.id, len(artifacts))
        except Exception as exc:
            logger.error("A2A task %s failed: %s", task.id, exc, exc_info=True)
            await self._publish_failure(updater, task.id, exc)
            raise ServerError(error=InternalError()) from exc

    async def _execute_as_task(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Open a ``Task`` up front and emit ``submitted -> working -> completed``."""
        task = await self._ensure_task(context, event_queue)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            # submitted -> working. Clients render this as "in progress"; it is also
            # what makes a long-running task observable at all.
            await updater.start_work()

            _, response_text, artifacts = await self._run(context)
            if not artifacts:
                artifacts = [text_artifact(self.default_artifact_name, response_text)]

            await self._publish_artifacts(updater, artifacts)

            # working -> completed. Terminal; TaskUpdater guards double-completion.
            await updater.complete()
            logger.info("A2A task %s completed with %d artifact(s)", task.id, len(artifacts))

        except Exception as exc:
            logger.error("A2A task %s failed: %s", task.id, exc, exc_info=True)
            await self._publish_failure(updater, task.id, exc)
            raise ServerError(error=InternalError()) from exc

    def _build_task(self, context: RequestContext) -> Any:
        """Create -- but do not publish -- the Task for this request.

        ``new_task()`` validates the message as a side effect: it rejects an empty
        ``TextPart``, for instance. Calling it *before* the workflow runs means a
        malformed request is refused up front, rather than after the work is done
        and the artifacts have to be thrown away.

        Args:
            context: Inbound A2A request context.

        Returns
        -------
            A ``Task`` in ``submitted`` state, not yet enqueued.

        Raises
        ------
            ServerError: ``InvalidParamsError`` if the message is malformed,
                translated from ``new_task()``'s ``ValueError`` so callers get the
                error the protocol expects.
        """
        try:
            return new_task(context.message)  # state = submitted
        except ValueError as exc:
            logger.error("Rejecting malformed A2A message: %s", exc)
            raise ServerError(error=InvalidParamsError(message=str(exc))) from exc

    async def _ensure_task(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        prepared: Any = None,
    ) -> Any:
        """Return the task for this request, publishing one if needed.

        A Task must exist before any status or artifact event can reference it.

        In practice ``current_task`` is always ``None`` here: this executor drives
        the task to a terminal state, and the A2A request handler rejects a
        follow-up referencing a terminal task before ``execute()`` is reached.
        Re-use ``contextId`` (not ``taskId``) to continue a conversation.

        Args:
            context: Inbound A2A request context.
            event_queue: Queue the A2A server drains to stream events.
            prepared: A Task already built by :meth:`_build_task`, reused so the
                message is not validated twice and no id is generated twice.

        Returns
        -------
            The Task every subsequent event will reference.
        """
        task = context.current_task
        if task is not None:
            return task
        task = prepared if prepared is not None else self._build_task(context)
        await event_queue.enqueue_event(task)
        return task

    @staticmethod
    async def _publish_artifacts(updater: TaskUpdater, artifacts: list[OutboundArtifact]) -> None:
        """Emit each artifact as its own ``TaskArtifactUpdateEvent``."""
        for item in artifacts:
            await updater.add_artifact(
                parts=item.parts,
                name=item.name,
                metadata=item.metadata,
                last_chunk=True,
            )

    @staticmethod
    async def _publish_failure(updater: TaskUpdater, task_id: str, exc: Exception) -> None:
        """Surface a terminal ``failed`` state so the caller stops waiting."""
        try:
            await updater.failed(
                message=updater.new_agent_message([Part(root=TextPart(text=f"Task failed: {exc}"))])
            )
        except Exception:  # pragma: no cover - queue already closed
            logger.warning("Could not publish failed state for task %s", task_id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Publish a ``canceled`` state for the referenced task.

        NAT's implementation raises ``UnsupportedOperationError`` unconditionally,
        which made sense when no ``Task`` existed.  Now that tasks are real -- and
        NAT's own client registers a ``cancel_task`` tool for the LLM -- always
        failing would leave callers polling a task that never resolves.

        .. warning::
           This is **best-effort bookkeeping, not preemption.**  It moves the task
           to a terminal ``canceled`` state so the caller stops waiting; it does
           **not** interrupt an in-flight workflow, because NAT exposes no
           cancellation hook into a running session.  Any compute already started
           runs to completion and its result is discarded.  If you need true
           cancellation, thread a ``CancellationToken``/``asyncio.Event`` through
           :meth:`run_workflow` and check it between steps.

        Args:
            context: The A2A request context identifying the task to cancel.
            event_queue: Queue used to publish the terminal status update.

        Raises
        ------
            ServerError: ``TaskNotFoundError`` if no task is associated with the
                request, since there is then nothing to cancel.
        """
        task = context.current_task
        if task is None:
            logger.warning("Cancellation requested but no task is associated with the request")
            raise ServerError(error=TaskNotFoundError())

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.cancel(
            message=updater.new_agent_message(
                [
                    Part(
                        root=TextPart(
                            text=(
                                "Task marked canceled. Note: work already in "
                                "progress is not interrupted."
                            )
                        )
                    )
                ]
            )
        )
        logger.info("A2A task %s marked canceled", task.id)

    # -- overridable seams -------------------------------------------------

    async def run_workflow(self, inputs: A2ARequestInputs, context: RequestContext) -> str:
        """Execute the underlying NAT workflow and return its text result.

        Mirrors NAT's own invocation (``session_manager.session()`` ->
        ``session.run(query)`` -> ``runner.result(to_type=str)``) so concurrency
        limits and per-user isolation behave exactly as upstream.

        .. note::
           NAT rejects an empty query with ``InvalidParamsError``.  This
           deliberately allows it, because a **file-only** request ("here is an
           image, describe it") is legitimate under the A2A spec and is precisely
           what this module exists to enable.  Override to reinstate a stricter
           check if your agent always requires text.

        Override this if your workflow returns something richer than text and you
        want structured values to reach the builder.

        Args:
            inputs: Parsed request parts.
            context: Inbound A2A request context (for ids and fallbacks).

        Returns
        -------
            The workflow's response as a string.
        """
        query = inputs.text or context.get_user_input() or ""
        async with self.session_manager.session() as session:
            async with session.run(query) as runner:
                # ``result()`` is typed as returning Any upstream; narrow it here
                # so callers (and mypy --strict) get a concrete str.
                result: Any = await runner.result(to_type=str)
                return str(result)

    async def build_artifacts(
        self, inputs: A2ARequestInputs, response_text: str
    ) -> list[OutboundArtifact]:
        """Delegate to the configured :class:`ArtifactBuilder`.

        Returns an empty list when no builder is configured, which under
        ``task_mode="auto"`` reproduces message-only behaviour.
        """
        if self.builder is None:
            return []
        return await self.builder.build_artifacts(inputs, response_text)
