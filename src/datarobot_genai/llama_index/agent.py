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
import inspect
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Event
from llama_index.llms.litellm import LiteLLM
from openai.types.chat import CompletionCreateParams

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import default_usage_metrics
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.agents.base import is_streaming

from .mcp import load_mcp_tools

if TYPE_CHECKING:
    from ragas import MultiTurnSample


class DataRobotLiteLLM(LiteLLM):
    """LiteLLM wrapper providing chat/function capability metadata for LlamaIndex."""

    @property
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata."""
        return LLMMetadata(
            context_window=128000,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
        )


def create_pipeline_interactions_from_events(
    events: list[Event] | None,
) -> MultiTurnSample | None:
    if not events:
        return None
    # Lazy import to reduce memory overhead when ragas is not used
    from ragas import MultiTurnSample
    from ragas.integrations.llama_index import convert_to_ragas_messages
    from ragas.messages import AIMessage
    from ragas.messages import HumanMessage
    from ragas.messages import ToolMessage

    # convert_to_ragas_messages expects a list[Event]
    ragas_trace = convert_to_ragas_messages(list(events))
    ragas_messages = cast(list[HumanMessage | AIMessage | ToolMessage], ragas_trace)
    return MultiTurnSample(user_input=ragas_messages)


class LlamaIndexAgent(BaseAgent[BaseTool], abc.ABC):
    """Abstract base agent for LlamaIndex workflows."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mcp_tools: list[Any] = []

    def set_mcp_tools(self, tools: list[Any]) -> None:
        """Set MCP tools for this agent."""
        self._mcp_tools = tools

    @property
    def mcp_tools(self) -> list[Any]:
        """Return the list of MCP tools available to this agent.

        Subclasses can use this to wire tools into LlamaIndex agents during
        workflow construction inside ``build_workflow``.
        """
        return self._mcp_tools

    @abc.abstractmethod
    def build_workflow(self) -> Any:
        """Return an AgentWorkflow instance ready to run."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract_response_text(self, result_state: Any, events: list[Any]) -> str:
        """Extract final response text from workflow state and/or events."""
        raise NotImplementedError

    def make_input_message(self, completion_create_params: CompletionCreateParams) -> str:
        """Create an input string for the workflow from the user prompt."""
        user_prompt_content = extract_user_prompt_content(completion_create_params)
        return str(user_prompt_content)

    async def invoke(self, completion_create_params: CompletionCreateParams) -> InvokeReturn:
        """Run the LlamaIndex workflow with the provided completion parameters."""
        input_message = self.make_input_message(completion_create_params)

        # Load MCP tools (if configured) asynchronously before building workflow
        mcp_tools = await load_mcp_tools(
            authorization_context=self._authorization_context,
            forwarded_headers=self.forwarded_headers,
        )
        self.set_mcp_tools(mcp_tools)

        # Preserve prior template startup print for CLI parity
        try:
            print(
                "Running agent with user prompt:",
                extract_user_prompt_content(completion_create_params),
                flush=True,
            )
        except Exception:
            # Printing is best-effort; proceed regardless
            pass

        workflow = self.build_workflow()
        handler = workflow.run(user_msg=input_message)

        usage_metrics: UsageMetrics = default_usage_metrics()

        # Streaming parity with LangGraph: yield incremental deltas during event processing
        if is_streaming(completion_create_params):

            async def _gen() -> AsyncGenerator[tuple[str, MultiTurnSample | None, UsageMetrics]]:
                events: list[Any] = []
                current_agent_name: str | None = None
                async for event in handler.stream_events():
                    events.append(event)
                    # Best-effort extraction of incremental text from LlamaIndex events
                    delta: str | None = None
                    # Agent switch banner if available on event
                    try:
                        if hasattr(event, "current_agent_name"):
                            new_agent = getattr(event, "current_agent_name")
                            if (
                                isinstance(new_agent, str)
                                and new_agent
                                and new_agent != current_agent_name
                            ):
                                current_agent_name = new_agent
                                # Print banner for agent switch (do not emit as streamed content)
                                print("\n" + "=" * 50, flush=True)
                                print(f"🤖 Agent: {current_agent_name}", flush=True)
                                print("=" * 50 + "\n", flush=True)
                    except Exception:
                        pass

                    try:
                        if hasattr(event, "delta") and isinstance(getattr(event, "delta"), str):
                            delta = getattr(event, "delta")
                        # Some event types may carry incremental text under "text" or similar
                        elif hasattr(event, "text") and isinstance(getattr(event, "text"), str):
                            delta = getattr(event, "text")
                    except Exception:
                        # Ignore malformed events and continue
                        delta = None

                    if delta:
                        # Yield token/content delta with current (accumulated) usage metrics
                        yield delta, None, usage_metrics

                    # Best-effort debug/event messages printed to CLI (do not stream as content)
                    try:
                        event_type = type(event).__name__
                        if event_type == "AgentInput" and hasattr(event, "input"):
                            print("📥 Input:", getattr(event, "input"), flush=True)
                        elif event_type == "AgentOutput":
                            # Output content
                            resp = getattr(event, "response", None)
                            if (
                                resp is not None
                                and hasattr(resp, "content")
                                and getattr(resp, "content")
                            ):
                                print("📤 Output:", getattr(resp, "content"), flush=True)
                            # Planned tool calls
                            tcalls = getattr(event, "tool_calls", None)
                            if isinstance(tcalls, list) and tcalls:
                                names = []
                                for c in tcalls:
                                    try:
                                        nm = getattr(c, "tool_name", None) or (
                                            c.get("tool_name") if isinstance(c, dict) else None
                                        )
                                        if nm:
                                            names.append(str(nm))
                                    except Exception:
                                        pass
                                if names:
                                    print("🛠️  Planning to use tools:", names, flush=True)
                        elif event_type == "ToolCallResult":
                            tname = getattr(event, "tool_name", None)
                            tkwargs = getattr(event, "tool_kwargs", None)
                            tout = getattr(event, "tool_output", None)
                            print(f"🔧 Tool Result ({tname}):", flush=True)
                            print(f"  Arguments: {tkwargs}", flush=True)
                            print(f"  Output: {tout}", flush=True)
                        elif event_type == "ToolCall":
                            tname = getattr(event, "tool_name", None)
                            tkwargs = getattr(event, "tool_kwargs", None)
                            print(f"🔨 Calling Tool: {tname}", flush=True)
                            print(f"  With arguments: {tkwargs}", flush=True)
                    except Exception:
                        # Ignore best-effort debug rendering errors
                        pass

                # After streaming completes, build final interactions and finish chunk
                # Extract state from workflow context (supports sync/async get or attribute)
                state = None
                ctx = getattr(handler, "ctx", None)
                try:
                    if ctx is not None:
                        get = getattr(ctx, "get", None)
                        if callable(get):
                            result = get("state")
                            state = await result if inspect.isawaitable(result) else result
                        elif hasattr(ctx, "state"):
                            state = getattr(ctx, "state")
                except (AttributeError, TypeError):
                    state = None

                # Run subclass-defined response extraction (not streamed) for completeness
                _ = self.extract_response_text(state, events)

                pipeline_interactions = create_pipeline_interactions_from_events(events)
                # Final empty chunk indicates end of stream, carrying interactions and usage
                yield "", pipeline_interactions, usage_metrics

            return _gen()

        # Non-streaming path: run to completion, emit debug prints, then return final response
        events: list[Any] = []
        current_agent_name: str | None = None
        async for event in handler.stream_events():
            events.append(event)

            # Replicate prior template CLI prints for non-streaming mode
            try:
                if hasattr(event, "current_agent_name"):
                    new_agent = getattr(event, "current_agent_name")
                    if isinstance(new_agent, str) and new_agent and new_agent != current_agent_name:
                        current_agent_name = new_agent
                        print(f"\n{'=' * 50}", flush=True)
                        print(f"🤖 Agent: {current_agent_name}", flush=True)
                        print(f"{'=' * 50}\n", flush=True)
            except Exception:
                pass

            try:
                if hasattr(event, "delta") and isinstance(getattr(event, "delta"), str):
                    print(getattr(event, "delta"), end="", flush=True)
                elif hasattr(event, "text") and isinstance(getattr(event, "text"), str):
                    print(getattr(event, "text"), end="", flush=True)
                else:
                    event_type = type(event).__name__
                    if event_type == "AgentInput" and hasattr(event, "input"):
                        print("📥 Input:", getattr(event, "input"), flush=True)
                    elif event_type == "AgentOutput":
                        resp = getattr(event, "response", None)
                        if (
                            resp is not None
                            and hasattr(resp, "content")
                            and getattr(resp, "content")
                        ):
                            print("📤 Output:", getattr(resp, "content"), flush=True)
                        tcalls = getattr(event, "tool_calls", None)
                        if isinstance(tcalls, list) and tcalls:
                            names: list[str] = []
                            for c in tcalls:
                                try:
                                    nm = getattr(c, "tool_name", None) or (
                                        c.get("tool_name") if isinstance(c, dict) else None
                                    )
                                    if nm:
                                        names.append(str(nm))
                                except Exception:
                                    pass
                            if names:
                                print("🛠️  Planning to use tools:", names, flush=True)
                    elif event_type == "ToolCallResult":
                        tname = getattr(event, "tool_name", None)
                        tkwargs = getattr(event, "tool_kwargs", None)
                        tout = getattr(event, "tool_output", None)
                        print(f"🔧 Tool Result ({tname}):", flush=True)
                        print(f"  Arguments: {tkwargs}", flush=True)
                        print(f"  Output: {tout}", flush=True)
                    elif event_type == "ToolCall":
                        tname = getattr(event, "tool_name", None)
                        tkwargs = getattr(event, "tool_kwargs", None)
                        print(f"🔨 Calling Tool: {tname}", flush=True)
                        print(f"  With arguments: {tkwargs}", flush=True)
            except Exception:
                # Best-effort debug printing; continue on errors
                pass

        # Extract state from workflow context (supports sync/async get or attribute)
        state = None
        ctx = getattr(handler, "ctx", None)
        try:
            if ctx is not None:
                get = getattr(ctx, "get", None)
                if callable(get):
                    result = get("state")
                    state = await result if inspect.isawaitable(result) else result
                elif hasattr(ctx, "state"):
                    state = getattr(ctx, "state")
        except (AttributeError, TypeError):
            state = None
        response_text = self.extract_response_text(state, events)

        pipeline_interactions = create_pipeline_interactions_from_events(events)

        return response_text, pipeline_interactions, usage_metrics
