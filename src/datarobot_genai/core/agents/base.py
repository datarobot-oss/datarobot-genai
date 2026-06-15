# Copyright 2025 DataRobot, Inc. and its affiliates.
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

from __future__ import annotations

import abc
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import Optional
from typing import TypeAlias
from typing import TypedDict
from typing import TypeVar

from ag_ui.core import Event
from ag_ui.core import RunAgentInput
from ag_ui.core import UserMessage

from datarobot_genai.core.agents.history import NormalizedHistoryMessage
from datarobot_genai.core.agents.history import build_history_summary_from_messages
from datarobot_genai.core.agents.history import drop_unpaired_boundary_tool_turns
from datarobot_genai.core.agents.history import extract_history_messages
from datarobot_genai.core.config import get_max_history_messages_default
from datarobot_genai.core.utils.auth import prepare_identity_header
from datarobot_genai.core.utils.urls import get_api_base

if TYPE_CHECKING:
    from ragas import MultiTurnSample

TTool = TypeVar("TTool")

logger = logging.getLogger(__name__)


class BaseAgent(Generic[TTool], abc.ABC):
    """BaseAgent centralizes common initialization for agent templates.

    Fields:
      - api_key: DataRobot API token
      - api_base: Endpoint for DataRobot, normalized for LLM Gateway usage
      - llm: Framework-specific LLM client (constructed outside the agent and passed in)
      - model: Optional model identifier string (e.g. LiteLLM / deployment model name)
      - timeout: Request timeout
      - verbose: Verbosity flag
      - forwarded_headers: Forwarded headers for the agent
      - max_history_messages: Maximum number of prior messages to include in chat history
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        llm: Any | None = None,
        tools: list[TTool] | None = None,
        verbose: bool = True,
        timeout: int = 90,
        forwarded_headers: dict[str, str] | None = None,
        max_history_messages: int | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("DATAROBOT_API_TOKEN")
        self.api_base = (
            api_base or os.environ.get("DATAROBOT_ENDPOINT") or "https://app.datarobot.com"
        )
        self.set_llm(llm)
        self.set_tools(tools or [])
        self.set_timeout(timeout)
        self.set_verbose(verbose)
        self.set_model(model)
        self._forwarded_headers: dict[str, str] = forwarded_headers or {}
        self._identity_header: dict[str, str] = prepare_identity_header(self._forwarded_headers)
        self._max_history_messages = max_history_messages

    def set_llm(self, llm: Any | None) -> None:
        self._llm = llm

    @property
    def llm(self) -> Any | None:
        return self._llm

    def set_model(self, model: str | None) -> None:
        self._model = model

    @property
    def model(self) -> str | None:
        return self._model

    def set_timeout(self, timeout: int) -> None:
        self._timeout = timeout

    @property
    def timeout(self) -> int:
        return self._timeout

    def set_verbose(self, verbose: bool) -> None:
        self._verbose = verbose

    @property
    def verbose(self) -> bool:
        return self._verbose

    def set_tools(self, tools: list[TTool]) -> None:
        self._tools = tools

    @property
    def tools(self) -> list[TTool]:
        """Return the list of tools available to this agent.

        Subclasses can use this to wire tools into the agent.
        """
        return self._tools

    @property
    def max_history_messages(self) -> int:
        """Maximum number of prior messages to include in chat history.

        Defaults to ``DATAROBOT_GENAI_MAX_HISTORY_MESSAGES`` env var (read at
        call time). Subclasses can override via the constructor parameter or
        by overriding this property.
        """
        if self._max_history_messages is not None:
            return self._max_history_messages
        return get_max_history_messages_default()

    @property
    def forwarded_headers(self) -> dict[str, str]:
        """Return the forwarded headers for this agent."""
        return self._forwarded_headers

    def litellm_api_base(self, deployment_id: str | None) -> str:
        return get_api_base(self.api_base, deployment_id)

    @abc.abstractmethod
    def invoke(self, run_agent_input: RunAgentInput) -> InvokeReturn:
        raise NotImplementedError("Not implemented")

    async def invoke_single_message(self, user_message: str) -> InvokeReturn:
        """
        Invoke the agent without chat history with a single user message.

        Parameters
        ----------
        user_message: str
            The user message to invoke the agent with.

        Returns
        -------
        InvokeReturn
            Same async stream as :meth:`invoke` for a fresh run with a single user turn.
        """
        # Generate a new thread and run ID
        run_input = RunAgentInput(
            thread_id=str(uuid.uuid4()),
            run_id=str(uuid.uuid4()),
            messages=[UserMessage(id=str(uuid.uuid4()), role="user", content=user_message)],
            state=[],
            tools=[],
            context=[],
            forwardedProps=None,
        )
        async for output in self.invoke(run_input):
            yield output

    def build_history_summary(
        self,
        run_agent_input: RunAgentInput,
    ) -> str:
        """Instance helper to summarize prior turns as plain-text transcript.

        Subclasses can override ``max_history_messages`` to control how many
        prior messages are included. This is primarily intended for exposing a
        ``chat_history`` variable in prompts across different agent types.
        """
        return build_history_summary_from_messages(run_agent_input, self.max_history_messages)

    def history_messages(
        self,
        run_agent_input: RunAgentInput,
    ) -> list[NormalizedHistoryMessage]:
        """Prior turns as structured dicts (role/content + tool_calls/tool_call_id).

        Same history selection and ``max_history_messages`` truncation as
        :meth:`build_history_summary`, but preserves structured tool-call metadata so
        framework adapters can reconstruct native message history instead of
        flattening to text. See ``datarobot_genai.langgraph.history`` and
        ``datarobot_genai.llama_index.history``.

        Tool-call/result turns left unpaired by ``max_history`` truncation are dropped
        so the reconstructed message sequence stays valid for the model.
        """
        history = extract_history_messages(run_agent_input, self.max_history_messages)
        return drop_unpaired_boundary_tool_turns(history)

    @classmethod
    def create_pipeline_interactions_from_events(
        cls,
        events: list[Any] | None,
    ) -> MultiTurnSample | None:
        """Create a simple MultiTurnSample from a list of generic events/messages."""
        if not events:
            return None
        # Lazy import to reduce memory overhead when ragas is not used
        from ragas import MultiTurnSample

        return MultiTurnSample(user_input=events)


def _message_content_as_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return "".join(getattr(part, "text", "") for part in content if part is not None)


def apply_system_context_to_run_input(run_agent_input: RunAgentInput) -> RunAgentInput:
    """Fold AG-UI system messages into the latest user turn.

    ``streaming_memory_agent`` injects retrieved memories as a system message
    immediately before the last user message. Agents that only read the last
    user turn (for example ``LlamaIndexAgent``) must merge those system
    messages here so retrieved memory reaches the model.
    """
    messages = list(run_agent_input.messages)
    if not messages:
        return run_agent_input

    last_user_idx: int | None = None
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if getattr(message, "role", None) == "user" and _message_content_as_str(
            getattr(message, "content", None)
        ):
            last_user_idx = idx
            break

    if last_user_idx is None:
        return run_agent_input

    system_texts = [
        text
        for message in messages[:last_user_idx]
        if getattr(message, "role", None) == "system"
        for text in [_message_content_as_str(getattr(message, "content", None))]
        if text
    ]
    if not system_texts:
        return run_agent_input

    user_message = messages[last_user_idx]
    user_content = _message_content_as_str(getattr(user_message, "content", None))
    merged_content = "\n\n".join(system_texts) + "\n\n" + user_content

    updated_messages = [
        message
        for message in messages
        if message is not user_message and getattr(message, "role", None) != "system"
    ]
    updated_messages.append(user_message.model_copy(update={"content": merged_content}))

    return run_agent_input.model_copy(update={"messages": updated_messages})


def extract_user_prompt_content(run_agent_input: RunAgentInput) -> Any:
    """Extract the last user message content from input."""
    user_messages = [msg for msg in run_agent_input.messages if msg.role == "user"]
    # Get the last user message
    content: str = user_messages[-1].content if user_messages else ""
    # Try converting prompt from json to a dict
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            pass

    return content


def make_system_prompt(suffix: str = "", *, prefix: str | None = None) -> str:
    """Build a system prompt with optional prefix and suffix.

    Parameters
    ----------
    suffix : str, default ""
        Text appended after the prefix. If non-empty, it is placed on a new line.
    prefix : str | None, keyword-only, default None
        Custom prefix text. When ``None``, a default collaborative assistant
        instruction is used.

    Returns
    -------
    str
        The composed system prompt string.
    """
    default_prefix = (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
    )
    head = prefix if prefix is not None else default_prefix
    if suffix:
        return head + "\n" + suffix
    return head


# Structured type for token usage metrics in responses
class UsageMetrics(TypedDict):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


# Canonical return type for all agent invoke implementations
InvokeReturn: TypeAlias = AsyncGenerator[
    tuple[Event, Optional["MultiTurnSample"], UsageMetrics], None
]


def default_usage_metrics() -> UsageMetrics:
    """Return a metrics dict with required keys for OpenAI-compatible responses."""
    return {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
    }
