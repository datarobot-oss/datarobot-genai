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
import json
import uuid
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import StepFinishedEvent
from ag_ui.core import StepStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallResultEvent
from ag_ui.core import ToolCallStartEvent
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Event
from llama_index.llms.litellm import LiteLLM

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.core.agents import extract_user_prompt_content
from datarobot_genai.core.agents import to_llama_index_messages
from datarobot_genai.core.agents import truncate_messages
from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics

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


class LlamaIndexAgent(BaseAgent[BaseTool], abc.ABC):
    """Abstract base agent for LlamaIndex workflows."""

    @abc.abstractmethod
    async def build_workflow(self) -> Any:
        """Return an AgentWorkflow instance ready to run."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract_response_text(self, result_state: Any, events: list[Any]) -> str:
        """Extract final response text from workflow state and/or events."""
        raise NotImplementedError

    def make_input_message(self, run_agent_input: RunAgentInput) -> str:
        """Create an input string for the workflow from the user prompt."""
        user_prompt_content = extract_user_prompt_content(run_agent_input)
        return str(user_prompt_content)

    async def invoke(self, run_agent_input: RunAgentInput) -> InvokeReturn:
        """Run the LlamaIndex workflow with the provided completion parameters."""
        messages = list(run_agent_input.messages)

        # Multi-turn: split into user_msg (last user text) and chat_history
        # (everything before, converted to LlamaIndex types).
        if len(messages) > 1 and self.max_history_messages > 0:
            user_msg = str(extract_user_prompt_content(run_agent_input))
            history = truncate_messages(messages, self.max_history_messages, exclude_current=True)
            chat_history = to_llama_index_messages(history) if history else None
        else:
            # Single-turn: use make_input_message (existing behavior)
            user_msg = self.make_input_message(run_agent_input)
            chat_history = None

        # Preserve prior template startup print for CLI parity
        try:
            print("Running agent with user prompt:", user_msg, flush=True)
        except Exception:
            pass

        thread_id = run_agent_input.thread_id
        run_id = run_agent_input.run_id
        usage_metrics: UsageMetrics = default_usage_metrics()

        yield (
            RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id),
            None,
            usage_metrics,
        )

        built: Any = self.build_workflow()
        workflow = await built if inspect.isawaitable(built) else built

        run_kwargs: dict[str, Any] = {"user_msg": user_msg}
        if chat_history:
            run_kwargs["chat_history"] = chat_history
        handler = workflow.run(**run_kwargs)

        events: list[Any] = []
        current_agent_name: str | None = None
        message_id = str(uuid.uuid4())
        text_started = False
        agent: str | None = None

        async for event in handler.stream_events():
            events.append(event)
            # Best-effort extraction of incremental text from LlamaIndex events
            delta: str | None = None

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
                if not text_started:
                    yield (
                        TextMessageStartEvent(
                            type=EventType.TEXT_MESSAGE_START, message_id=message_id
                        ),
                        None,
                        usage_metrics,
                    )
                    text_started = True
                yield (
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT, message_id=message_id, delta=delta
                    ),
                    None,
                    usage_metrics,
                )

                # Agent switch banner if available on event
            if hasattr(event, "current_agent_name"):
                agent = getattr(event, "current_agent_name", None)
                if agent is not None and agent != current_agent_name:
                    if current_agent_name is not None:
                        yield (
                            StepFinishedEvent(step_name=current_agent_name),
                            None,
                            usage_metrics,
                        )

                    yield (
                        StepStartedEvent(step_name=agent),
                        None,
                        usage_metrics,
                    )
                    current_agent_name = agent
                    agent = None
                    # Print banner for agent switch (do not emit as streamed content)
                    print("\n" + "=" * 50, flush=True)
                    print(f"🤖 Agent: {current_agent_name}", flush=True)
                    print("=" * 50 + "\n", flush=True)

            event_type = type(event).__name__
            if event_type == "AgentInput" and hasattr(event, "input"):
                print("📥 Input:", getattr(event, "input"), flush=True)
            elif event_type == "AgentOutput":
                # Output content
                resp = getattr(event, "response", None)
                if resp is not None and hasattr(resp, "content") and getattr(resp, "content"):
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
                tid = getattr(event, "tool_id", None)
                tkwargs = getattr(event, "tool_kwargs", None)
                tout = getattr(event, "tool_output", None)
                print(f"🔧 Tool Result ({tname}):", flush=True)
                print(f"  Arguments: {tkwargs}", flush=True)
                print(f"  Output: {tout}", flush=True)
                yield (
                    ToolCallEndEvent(
                        type=EventType.TOOL_CALL_END,
                        tool_call_id=tid,
                    ),
                    None,
                    usage_metrics,
                )
                yield (
                    ToolCallResultEvent(
                        type=EventType.TOOL_CALL_RESULT,
                        message_id=message_id,
                        tool_call_id=tid,
                        content=json.dumps(tout, default=str),
                        role="tool",
                    ),
                    None,
                    usage_metrics,
                )
            elif event_type == "ToolCall":
                tname = getattr(event, "tool_name", None)
                tkwargs = getattr(event, "tool_kwargs", None)
                tid = getattr(event, "tool_id", None)
                print(f"🔨 Calling Tool: {tname}", flush=True)
                print(f"  With arguments: {tkwargs}", flush=True)
                yield (
                    ToolCallStartEvent(
                        type=EventType.TOOL_CALL_START,
                        tool_call_id=tid,
                        tool_call_name=tname,
                    ),
                    None,
                    usage_metrics,
                )
                yield (
                    ToolCallArgsEvent(
                        type=EventType.TOOL_CALL_ARGS,
                        tool_call_id=tid,
                        delta=json.dumps(tkwargs, default=str),
                    ),
                    None,
                    usage_metrics,
                )

        if agent is not None:
            yield (
                StepFinishedEvent(step_name=agent),
                None,
                usage_metrics,
            )
            agent = None

        if text_started:
            yield (
                TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                ),
                None,
                usage_metrics,
            )

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

        pipeline_interactions = self.create_pipeline_interactions_from_events(events)
        # TODO: find a way to count usage (LlamaIndex does not report it)
        yield (
            RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id),
            pipeline_interactions,
            usage_metrics,
        )

    @classmethod
    def create_pipeline_interactions_from_events(
        cls,
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
