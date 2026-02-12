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
import os
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

from datarobot_genai.core.utils.auth import prepare_identity_header
from datarobot_genai.core.utils.urls import get_api_base

if TYPE_CHECKING:
    from ragas import MultiTurnSample

TTool = TypeVar("TTool")


class BaseAgent(Generic[TTool], abc.ABC):
    """BaseAgent centralizes common initialization for agent templates.

    Fields:
      - api_key: DataRobot API token
      - api_base: Endpoint for DataRobot, normalized for LLM Gateway usage
      - model: Preferred model name
      - timeout: Request timeout
      - verbose: Verbosity flag
      - authorization_context: Authorization context for downstream agents/tools
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        verbose: bool | str | None = True,
        timeout: int | None = 90,
        authorization_context: dict[str, Any] | None = None,
        forwarded_headers: dict[str, str] | None = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get("DATAROBOT_API_TOKEN")
        self.api_base = (
            api_base or os.environ.get("DATAROBOT_ENDPOINT") or "https://app.datarobot.com"
        )
        self.model = model
        self.timeout = timeout if timeout is not None else 90
        if isinstance(verbose, str):
            self.verbose = verbose.lower() == "true"
        elif verbose is None:
            self.verbose = True
        else:
            self.verbose = bool(verbose)
        self._mcp_tools: list[TTool] = []
        self._authorization_context = authorization_context or {}
        self._forwarded_headers: dict[str, str] = forwarded_headers or {}
        self._identity_header: dict[str, str] = prepare_identity_header(self._forwarded_headers)

    def set_mcp_tools(self, tools: list[TTool]) -> None:
        self._mcp_tools = tools

    @property
    def mcp_tools(self) -> list[TTool]:
        """Return the list of MCP tools available to this agent.

        Subclasses can use this to wire tools into CrewAI agents/tasks during
        workflow construction inside ``crew``.
        """
        return self._mcp_tools

    @property
    def authorization_context(self) -> dict[str, Any]:
        """Return the authorization context for this agent."""
        return self._authorization_context

    @property
    def forwarded_headers(self) -> dict[str, str]:
        """Return the forwarded headers for this agent."""
        return self._forwarded_headers

    def litellm_api_base(self, deployment_id: str | None) -> str:
        return get_api_base(self.api_base, deployment_id)

    @abc.abstractmethod
    def invoke(self, run_agent_input: RunAgentInput) -> InvokeReturn:
        raise NotImplementedError("Not implemented")

    def build_history_summary(
        self,
        completion_create_params: CompletionCreateParams | Mapping[str, Any],
    ) -> str:
        """Instance helper to summarize prior turns as plain-text transcript.

        Subclasses can override ``MAX_HISTORY_MESSAGES`` to control how many
        prior messages are included. This is primarily intended for exposing a
        ``chat_history`` variable in prompts across different agent types.
        """
        max_history = getattr(self, "MAX_HISTORY_MESSAGES", 20)
        return build_history_summary(completion_create_params, max_history)

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


class _NormalizedHistoryMessageRequired(TypedDict):
    """Required fields for a normalized prior chat message."""

    role: str
    content: str


class NormalizedHistoryMessage(_NormalizedHistoryMessageRequired, total=False):
    """Normalized representation of a single prior chat message.

    This structure is intentionally minimal but preserves enough optional
    metadata for tool-heavy agents to reconstruct richer history when needed.

    Required fields:
    - role: str
    - content: str

    Optional fields (best-effort, may be absent):
    - tool_call_id: str | None
    - name: str | None
    - tool_calls: Any | None  # e.g. OpenAI-style tool_calls payload
    """

    tool_call_id: str | None
    name: str | None
    tool_calls: Any | None


def _get_message_field(message: Any, key: str, default: Any = None) -> Any:
    """Best-effort field access for pydantic-like objects or dict messages."""
    if isinstance(message, Mapping):
        return message.get(key, default)
    return getattr(message, key, default)


def _summarize_tool_calls(tool_calls: Any) -> str:
    """Render a minimal, stable summary for tool-call-only assistant messages."""
    if not tool_calls:
        return "[tool_calls]"

    names: list[str] = []
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            name: Any = None
            if isinstance(tc, Mapping):
                fn = tc.get("function")
                if isinstance(fn, Mapping):
                    name = fn.get("name")
                name = name or tc.get("name") or tc.get("tool_name")
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn is not None else None
                name = name or getattr(tc, "name", None) or getattr(tc, "tool_name", None)
            if name:
                names.append(str(name))

    if names:
        return "[tool_calls] " + ", ".join(names)
    return "[tool_calls]"


def extract_history_messages(
    completion_create_params: CompletionCreateParams | Mapping[str, Any],
    max_history: int,
) -> list[NormalizedHistoryMessage]:
    r"""Return normalized prior messages to use as chat history.

    Behaviour:
    - Considers ``completion_create_params['messages']`` in order.
    - If the *final* message is a ``"user"`` message, treats everything *before*
      it as history (so the latest user turn can be handled separately).
    - Otherwise treats all provided messages as history.
    - Converts messages into ``{role, content}`` dicts with string content.
    - Truncates to the most recent ``max_history`` entries.

    Special cases:
    - When there are no messages, returns an empty list.
    - When ``max_history <= 0``, history is disabled and an empty list is returned.
      This matches the documented semantics where 0 means "no history".
    """
    params = cast(Mapping[str, Any], completion_create_params)
    raw_messages = list(params.get("messages", []))
    if not raw_messages or max_history <= 0:
        return []

    # Only exclude the final user message when it is actually the last message
    # in the provided list. This avoids dropping trailing assistant/tool messages
    # that some runtimes include after the last user message.
    last_role = _get_message_field(raw_messages[-1], "role")
    history_slice = raw_messages[:-1] if last_role == "user" else raw_messages

    # Keep only the most recent N messages in history to avoid unbounded growth
    # when callers provide long transcripts.
    if max_history and len(history_slice) > max_history:
        history_slice = history_slice[-max_history:]

    history: list[NormalizedHistoryMessage] = []
    for message in history_slice:
        role = _get_message_field(message, "role")
        content = _get_message_field(message, "content")
        tool_call_id = _get_message_field(message, "tool_call_id")
        name = _get_message_field(message, "name")
        tool_calls = _get_message_field(message, "tool_calls")

        text = str(content) if content is not None else ""
        if not text and tool_calls is not None:
            # Preserve assistant tool-call messages even when content is empty/None.
            text = _summarize_tool_calls(tool_calls)
        if not text and str(role or "") == "tool":
            # Tool outputs should generally have content, but if they don't,
            # keep a minimal placeholder so downstream adapters don't silently
            # drop tool steps.
            label = str(name) if name is not None else ""
            if not label and tool_call_id is not None:
                label = str(tool_call_id)
            text = f"[tool] {label}".strip() if label else "[tool]"
        if not text:
            continue

        entry: NormalizedHistoryMessage = {
            "role": str(role or "user"),
            "content": text,
        }

        # Preserve optional tool metadata when present so downstream agents can
        # reconstruct richer tool histories if desired.
        if tool_call_id is not None:
            entry["tool_call_id"] = str(tool_call_id)
        if name is not None:
            entry["name"] = str(name)
        if tool_calls is not None:
            entry["tool_calls"] = tool_calls

        history.append(entry)

    return history


def build_history_summary(
    completion_create_params: CompletionCreateParams | Mapping[str, Any],
    max_history: int,
) -> str:
    """Build a plain-text summary of prior turns for prompts.

    This is a convenience helper around ``extract_history_messages`` that:
    - Uses the same history selection semantics as ``extract_history_messages``
    - Truncates to the most recent ``max_history`` entries
    - Normalizes them into ``{role, content}`` dicts
    and then renders a newline-separated transcript of the form
    ``role: content``.
    """
    history = extract_history_messages(completion_create_params, max_history)
    if not history:
        return ""

    lines = [f"{msg['role']}: {msg['content']}" for msg in history]
    return "\n".join(lines)


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
    tuple[str | Event, Optional["MultiTurnSample"], UsageMetrics], None
]


def default_usage_metrics() -> UsageMetrics:
    """Return a metrics dict with required keys for OpenAI-compatible responses."""
    return {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
    }
