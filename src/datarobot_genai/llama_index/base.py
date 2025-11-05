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

"""
Base class for LlamaIndex-based agents.

Provides a standard ``invoke`` that runs an AgentWorkflow, collects events,
and converts them into pipeline interactions. Subclasses provide the workflow
and response extraction logic.
"""

from __future__ import annotations

import abc
import inspect
from collections.abc import AsyncGenerator
from typing import Any

from openai.types.chat import CompletionCreateParams
from ragas import MultiTurnSample

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import default_usage_metrics
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.agents.base import is_streaming

from .agent import create_pipeline_interactions_from_events


class LlamaIndexAgent(BaseAgent, abc.ABC):
    """Abstract base agent for LlamaIndex workflows."""

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
                                banner = (
                                    f"\n{'=' * 50}\n🤖 Agent: {current_agent_name}\n{'=' * 50}\n"
                                )
                                yield banner, None, usage_metrics
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

                    # Best-effort debug/event messages similar to prior template prints
                    try:
                        event_type = type(event).__name__
                        if event_type == "AgentInput" and hasattr(event, "input"):
                            yield f"📥 Input: {getattr(event, 'input')}", None, usage_metrics
                        elif event_type == "AgentOutput":
                            # Output content
                            resp = getattr(event, "response", None)
                            if (
                                resp is not None
                                and hasattr(resp, "content")
                                and getattr(resp, "content")
                            ):
                                yield f"📤 Output: {getattr(resp, 'content')}", None, usage_metrics
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
                                    yield f"🛠️  Planning to use tools: {names}", None, usage_metrics
                        elif event_type == "ToolCallResult":
                            tname = getattr(event, "tool_name", None)
                            tkwargs = getattr(event, "tool_kwargs", None)
                            tout = getattr(event, "tool_output", None)
                            lines = []
                            if tname:
                                lines.append(f"🔧 Tool Result ({tname}):")
                            if tkwargs is not None:
                                lines.append(f"  Arguments: {tkwargs}")
                            if tout is not None:
                                lines.append(f"  Output: {tout}")
                            if lines:
                                yield "\n".join(lines), None, usage_metrics
                        elif event_type == "ToolCall":
                            tname = getattr(event, "tool_name", None)
                            tkwargs = getattr(event, "tool_kwargs", None)
                            if tname:
                                msg = f"🔨 Calling Tool: {tname}"
                                if tkwargs is not None:
                                    msg += f"\n  With arguments: {tkwargs}"
                                yield msg, None, usage_metrics
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

        # Non-streaming path: run to completion, then return final response
        events: list[Any] = []
        async for event in handler.stream_events():
            events.append(event)

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
