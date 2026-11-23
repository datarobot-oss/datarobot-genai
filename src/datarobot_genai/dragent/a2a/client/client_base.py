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
# Portions derived from NVIDIA NeMo Agent Toolkit (nvidia-nat-a2a 1.9.0).
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by DataRobot: import paths, formatting, docstrings and type annotations; ported to
# a2a-sdk 1.x.

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from collections.abc import Iterable
from datetime import timedelta
from types import TracebackType
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx
from a2a.client import Client
from a2a.client import ClientCallInterceptor
from a2a.client import ClientConfig
from a2a.client import ClientFactory
from a2a.server.tasks.task_manager import append_artifact_to_task
from a2a.types import AgentCard
from a2a.types import Message
from a2a.types import Part
from a2a.types import Role
from a2a.types import SendMessageRequest
from a2a.types import StreamResponse
from a2a.types import Task

from datarobot_genai.dragent.a2a.cards import fetch_agent_card

if TYPE_CHECKING:
    from nat.authentication.interfaces import AuthProviderBase

logger = logging.getLogger(__name__)


class A2ABaseClient:
    """
    Minimal A2A client for connecting to an A2A agent.

    Args:
        base_url: The base URL of the A2A agent
        agent_card_path: Path to agent card (default: /.well-known/agent-card.json)
        task_timeout: Timeout for task operations (default: 300 seconds)
        streaming: Enable streaming responses (default: True)
        auth_provider: Optional NAT authentication provider for securing requests
    """

    def __init__(
        self,
        base_url: str,
        agent_card_path: str = "/.well-known/agent-card.json",
        task_timeout: timedelta = timedelta(seconds=300),
        streaming: bool = True,
        auth_provider: AuthProviderBase | None = None,
    ):
        self._base_url = base_url
        self._agent_card_path = agent_card_path
        self._task_timeout = task_timeout
        self._streaming = streaming
        self._auth_provider = auth_provider

        self._httpx_client: httpx.AsyncClient | None = None
        self._client: Client | None = None
        self._agent_card: AgentCard | None = None

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def agent_card(self) -> AgentCard | None:
        return self._agent_card

    async def __aenter__(self) -> A2ABaseClient:
        if self._httpx_client is not None or self._client is not None:
            raise RuntimeError("A2ABaseClient already initialized")

        # 1) Create httpx client explicitly
        self._httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._task_timeout.total_seconds())
        )

        # 2) Resolve agent card
        await self._resolve_agent_card()
        if not self._agent_card:
            raise RuntimeError("Agent card not resolved")

        # 3) Setup authentication interceptors if auth is configured
        interceptors: list[ClientCallInterceptor] = []
        if self._auth_provider:
            try:
                from a2a.client import AuthInterceptor

                from datarobot_genai.dragent.a2a.auth.credential_service import A2ACredentialService

                credential_service = A2ACredentialService(
                    auth_provider=self._auth_provider,
                    agent_card=self._agent_card,
                )
                interceptors.append(AuthInterceptor(credential_service))
                logger.info("Authentication configured for A2A client")
            except ImportError as e:
                logger.error("Failed to setup authentication: %s", e)
                raise RuntimeError(
                    "Authentication requires a2a-sdk with AuthInterceptor support"
                ) from e

        # 4) Create A2A client with interceptors
        client_config = ClientConfig(
            httpx_client=self._httpx_client,
            streaming=self._streaming,
        )
        factory = ClientFactory(client_config)
        self._client = factory.create(self._agent_card, interceptors=interceptors)

        logger.info("Connected to A2A agent at %s", self._base_url)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Close A2A client first (if it exposes aclose)
        if self._client is not None:
            aclose = getattr(self._client, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    logger.warning("Error while closing A2A client", exc_info=True)

        # Then close httpx client
        if self._httpx_client is not None:
            try:
                await self._httpx_client.aclose()
            except Exception:
                logger.warning("Error while closing HTTPX client", exc_info=True)

        self._httpx_client = None
        self._client = None
        self._agent_card = None

    async def _resolve_agent_card(self) -> None:
        """Fetch the agent card from the A2A agent."""
        if not self._httpx_client:
            raise RuntimeError("httpx_client is not initialized")

        try:
            logger.info("Fetching agent card from: %s%s", self._base_url, self._agent_card_path)
            self._agent_card = await fetch_agent_card(
                self._httpx_client, self._base_url, self._agent_card_path
            )
            logger.info("Successfully fetched public agent card")
            # TODO: add support for authenticated extended agent card
        except Exception as e:
            logger.error("Failed to fetch agent card: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to fetch agent card from {self._base_url}") from e

    async def send_message(
        self, message_text: str, task_id: str | None = None, context_id: str | None = None
    ) -> AsyncGenerator[StreamResponse]:
        """
        Send a message to the agent and stream response events.

        This is the low-level A2A protocol method that yields events as they arrive.
        For simpler usage, prefer the high-level agent function registered by this client.

        Args:
            message_text: The message text to send
            task_id: Optional task ID to continue an existing conversation
            context_id: Optional context ID for the conversation

        Yields
        ------
            StreamResponse: The agent's response events as they arrive, each holding a
                message, a task, a status update or an artifact update
        """
        if not self._client:
            raise RuntimeError("A2ABaseClient not initialized")

        message = Message(
            role=Role.ROLE_USER,
            parts=[Part(text=message_text)],
            message_id=uuid4().hex,
            task_id=task_id,
            context_id=context_id,
        )

        async for response in self._client.send_message(SendMessageRequest(message=message)):
            yield response

    async def get_task(self, task_id: str, history_length: int | None = None) -> Task:
        """
        Get the status and details of a specific task.

        This is an A2A protocol operation for retrieving task information.

        Args:
            task_id: The unique identifier of the task
            history_length: Optional limit on the number of history messages to retrieve

        Returns
        -------
            Task: The task object with current status and history
        """
        if not self._client:
            raise RuntimeError("A2ABaseClient not initialized")

        from a2a.types import GetTaskRequest

        params = GetTaskRequest(id=task_id, history_length=history_length)
        return await self._client.get_task(params)

    async def cancel_task(self, task_id: str) -> Task:
        """
        Cancel a running task.

        This is an A2A protocol operation for canceling tasks.

        Args:
            task_id: The unique identifier of the task to cancel

        Returns
        -------
            Task: The task object with updated status
        """
        if not self._client:
            raise RuntimeError("A2ABaseClient not initialized")

        from a2a.types import CancelTaskRequest

        params = CancelTaskRequest(id=task_id)
        return await self._client.cancel_task(params)

    async def send_message_streaming(
        self, message_text: str, task_id: str | None = None, context_id: str | None = None
    ) -> AsyncGenerator[StreamResponse]:
        """
        Send a message to the agent and stream response events (alias for send_message).

        This method provides an explicit streaming interface that mirrors the A2A SDK pattern.
        It is functionally identical to send_message(), which already streams events.

        Args:
            message_text: The message text to send
            task_id: Optional task ID to continue an existing conversation
            context_id: Optional context ID for the conversation

        Yields
        ------
            StreamResponse: The agent's response events as they arrive.
        """
        async for event in self.send_message(message_text, task_id=task_id, context_id=context_id):
            yield event

    def extract_text_from_parts(self, parts: Iterable[Part]) -> list[str]:
        """
        Extract text content from A2A message parts.

        Args:
            parts: A2A Part objects

        Returns
        -------
            List of the non-empty text strings in the parts
        """
        return [part.text for part in parts if part.text]

    def extract_text_from_task(self, task: Task) -> str:
        """
        Extract text response from an A2A Task object.

        This method understands the A2A protocol structure and extracts the final
        text response from a completed task, prioritizing artifacts over history.

        Args:
            task: A2A Task object

        Returns
        -------
            Extracted text response or status message

        Priority order:
            1. Check task status (return error/progress if not completed)
            2. Extract from task.artifacts (structured output)
            3. Fallback to last agent message in task.history
        """
        from a2a.types import TaskState

        # Check task status. A protobuf submessage is never falsy, so ask for its presence.
        if task.HasField("status") and task.status.state != TaskState.TASK_STATE_COMPLETED:
            # Task not completed - return status message or indicate in progress
            if task.status.state == TaskState.TASK_STATE_FAILED:
                error = " ".join(self.extract_text_from_parts(task.status.message.parts))
                return f"Task failed: {error or 'Unknown error'}"
            return f"Task in progress (state: {TaskState.Name(task.status.state)})"

        # Priority 1: Extract from artifacts (structured output)
        if task.artifacts:
            # Get text from all artifacts
            all_text = []
            for artifact in task.artifacts:
                if artifact.parts:
                    text_parts = self.extract_text_from_parts(artifact.parts)
                    if text_parts:
                        all_text.extend(text_parts)
            if all_text:
                return " ".join(all_text)

        # Priority 2: Fallback to history (conversational messages)
        if task.history:
            # Get the last agent message from history
            for msg in reversed(task.history):
                if msg.role == Role.ROLE_AGENT:  # Get last agent message
                    text_parts = self.extract_text_from_parts(msg.parts)
                    if text_parts:
                        return " ".join(text_parts)

        return "No response"

    def extract_text_from_events(self, events: list[StreamResponse]) -> str:
        """
        Extract text response from a list of A2A events.

        This is a convenience method that handles both message and task responses.

        Args:
            events: List of A2A events, as yielded by send_message()

        Returns
        -------
            Extracted text response
        """
        if not events:
            return "No response"

        # Get the last event
        last_event = events[-1]

        # If it's a Message, extract text from parts
        if last_event.HasField("message"):
            text_parts = self.extract_text_from_parts(last_event.message.parts)
            return " ".join(text_parts) if text_parts else str(last_event.message)

        # Otherwise the events describe a task
        task = assemble_task(events)
        if task is not None:
            return self.extract_text_from_task(task)

        return str(last_event)


def apply_event(task: Task | None, event: StreamResponse) -> Task | None:
    """Fold one event into the task accumulated so far, as a2a-sdk 0.3 clients did.

    a2a-sdk 1.x clients yield each event as it arrives instead. A message event leaves the
    task as it is. ``task`` may be changed in place.
    """
    kind = event.WhichOneof("payload")
    if kind == "task":
        task = Task()
        task.CopyFrom(event.task)
    elif kind in ("status_update", "artifact_update"):
        update = getattr(event, kind)
        if task is None:
            task = Task(id=update.task_id, context_id=update.context_id)
        if kind == "artifact_update":
            append_artifact_to_task(task, update)
        else:
            if update.status.HasField("message"):
                task.history.append(update.status.message)
            if update.HasField("metadata"):
                task.metadata.update(update.metadata)
            task.status.CopyFrom(update.status)
    return task


def assemble_task(events: list[StreamResponse]) -> Task | None:
    """Apply task events in order and return the resulting task, if there is one."""
    task: Task | None = None
    for event in events:
        task = apply_event(task, event)
    return task
