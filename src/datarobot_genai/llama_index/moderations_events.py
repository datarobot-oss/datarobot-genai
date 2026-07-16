from typing import Any


def convert_to_moderations_messages(
    events: list[Event],
) -> list[HumanMessage | AIMessage | ToolMessage]:
    """Convert LlamaIndex agent events into pipeline-interaction messages.

    Ports the old ``ragas.integrations.llama_index.convert_to_ragas_messages``: it walks
    ``AgentInput`` / ``AgentOutput`` / ``ToolCallResult`` events and emits the matching
    Human / AI / Tool messages, de-duplicating tool calls by their tool id.
    """
    # Lazy import so the moderations-backed primitives load only when a run
    # actually records pipeline interactions.
    from datarobot_dome.guards.agent_goal_accuracy import AIMessage
    from datarobot_dome.guards.agent_goal_accuracy import HumanMessage
    from datarobot_dome.guards.agent_goal_accuracy import ToolCall
    from datarobot_dome.guards.agent_goal_accuracy import ToolMessage
    from llama_index.core.agent.workflow import AgentInput
    from llama_index.core.agent.workflow import AgentOutput
    from llama_index.core.agent.workflow import ToolCallResult
    from llama_index.core.base.llms.types import MessageRole
    from llama_index.core.base.llms.types import TextBlock

    messages: list[HumanMessage | AIMessage | ToolMessage] = []
    tool_call_ids: set[Any] = set()

    for event in events:
        if isinstance(event, AgentInput):
            last_chat_message = event.input[-1]
            content = ""
            if last_chat_message.blocks:
                content = "\n".join(
                    str(block.text)
                    for block in last_chat_message.blocks
                    if isinstance(block, TextBlock)
                )
            if last_chat_message.role == MessageRole.USER:
                # A user turn that only echoes a preceding tool result is noise; skip it.
                if messages and isinstance(messages[-1], ToolMessage):
                    continue
                messages.append(HumanMessage(content=content))
        elif isinstance(event, AgentOutput):
            content = "\n".join(
                str(block.text) for block in event.response.blocks if isinstance(block, TextBlock)
            )
            tool_calls: list[ToolCall] | None = None
            if hasattr(event, "tool_calls"):
                tool_calls = []
                for tc in event.tool_calls:
                    if tc.tool_id not in tool_call_ids:
                        tool_call_ids.add(tc.tool_id)
                        tool_calls.append(ToolCall(name=tc.tool_name, args=tc.tool_kwargs))
            messages.append(AIMessage(content=content, tool_calls=tool_calls or None))
        elif isinstance(event, ToolCallResult):
            if event.return_direct:
                messages.append(AIMessage(content=event.tool_output.content))
            else:
                messages.append(ToolMessage(content=event.tool_output.content))

    return messages