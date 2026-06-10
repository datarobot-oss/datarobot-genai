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
from ag_ui.core import ReasoningMessageChunkEvent
from ag_ui.core import RunAgentInput
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallResultEvent
from ag_ui.core import ToolCallStartEvent
from ag_ui.core import UserMessage

from datarobot_genai.core.agents.events import events_to_messages
from datarobot_genai.core.agents.history import drop_unpaired_boundary_tool_turns
from datarobot_genai.core.agents.history import extract_history_messages


def test_text_only() -> None:
    # GIVEN only assistant text deltas
    messages = events_to_messages(
        [
            TextMessageContentEvent(message_id="m1", delta="Hello "),
            TextMessageContentEvent(message_id="m1", delta="world"),
        ]
    )

    # THEN a single assistant message carries the concatenated text
    assert [m.role for m in messages] == ["assistant"]
    assert messages[0].content == "Hello world"
    assert messages[0].tool_calls is None


def test_tool_call_with_result() -> None:
    # GIVEN a tool call (args streamed), its result, then a final answer
    messages = events_to_messages(
        [
            ToolCallStartEvent(tool_call_id="c1", tool_call_name="generate_objectid"),
            ToolCallArgsEvent(tool_call_id="c1", delta='{"type":'),
            ToolCallArgsEvent(tool_call_id="c1", delta=' "deployment"}'),
            ToolCallEndEvent(tool_call_id="c1"),
            ToolCallResultEvent(
                message_id="r1", tool_call_id="c1", content="69cbb73789723b6936c6c9e1"
            ),
            TextMessageContentEvent(message_id="m2", delta="69cbb73789723b6936c6c9e1"),
        ]
    )

    # THEN assistant(tool_calls) -> paired tool result -> final assistant text
    assert [m.role for m in messages] == ["assistant", "tool", "assistant"]
    assistant_tool_call, tool_result, final = messages
    assert assistant_tool_call.content is None
    assert assistant_tool_call.tool_calls is not None
    call = assistant_tool_call.tool_calls[0]
    assert call.id == "c1"
    assert call.function.name == "generate_objectid"
    # Argument deltas are concatenated into the OpenAI-wire JSON string.
    assert call.function.arguments == '{"type": "deployment"}'
    assert tool_result.tool_call_id == "c1"
    assert tool_result.content == "69cbb73789723b6936c6c9e1"
    assert final.content == "69cbb73789723b6936c6c9e1"


def test_keeps_text_with_its_tool_call() -> None:
    # GIVEN an assistant step that BOTH says something AND calls a tool, then a
    # final answer after the result (the case that used to drop the call).
    messages = events_to_messages(
        [
            TextMessageContentEvent(message_id="m1", delta="Let me check the weather."),
            ToolCallStartEvent(tool_call_id="c1", tool_call_name="get_weather"),
            ToolCallArgsEvent(tool_call_id="c1", delta='{"city": "Paris"}'),
            ToolCallEndEvent(tool_call_id="c1"),
            ToolCallResultEvent(message_id="r1", tool_call_id="c1", content="18C, sunny"),
            TextMessageContentEvent(message_id="m2", delta="It's 18C and sunny in Paris."),
        ]
    )

    # THEN text and the tool call ride on ONE assistant message (no drop, no reorder);
    # the post-result answer is its own assistant message.
    assert [m.role for m in messages] == ["assistant", "tool", "assistant"]
    step, tool_result, answer = messages
    assert step.content == "Let me check the weather."
    assert step.tool_calls is not None and step.tool_calls[0].function.name == "get_weather"
    assert step.tool_calls[0].function.arguments == '{"city": "Paris"}'
    assert tool_result.tool_call_id == "c1"
    assert tool_result.content == "18C, sunny"
    assert answer.content == "It's 18C and sunny in Paris."


def test_idless_turns_do_not_merge_across_tool_result() -> None:
    # GIVEN two id-less assistant turns (empty message_id -- e.g. a provider whose
    # streamed chunks carry no id) separated by a tool call and its result.
    messages = events_to_messages(
        [
            TextMessageContentEvent(message_id="", delta="Let me check."),
            ToolCallStartEvent(tool_call_id="c1", tool_call_name="get_weather"),
            ToolCallArgsEvent(tool_call_id="c1", delta='{"city": "Paris"}'),
            ToolCallResultEvent(message_id="r1", tool_call_id="c1", content="18C"),
            TextMessageContentEvent(message_id="", delta="It is 18C."),
        ]
    )

    # THEN the pre-tool text+call and the post-result answer stay SEPARATE assistant
    # messages: the id-less second turn must not fold back into the closed first step.
    assert [m.role for m in messages] == ["assistant", "tool", "assistant"]
    step, tool_result, answer = messages
    assert step.content == "Let me check."
    assert step.tool_calls is not None and [c.id for c in step.tool_calls] == ["c1"]
    assert tool_result.tool_call_id == "c1"
    assert answer.content == "It is 18C."
    assert answer.tool_calls is None


def test_parent_message_id_attaches_tool_call_to_named_message() -> None:
    # GIVEN a tool call whose parent_message_id names an EARLIER assistant message,
    # not the most-recently-open one.
    messages = events_to_messages(
        [
            TextMessageStartEvent(message_id="m1"),
            TextMessageContentEvent(message_id="m1", delta="first bubble"),
            TextMessageEndEvent(message_id="m1"),
            TextMessageStartEvent(message_id="m2"),
            TextMessageContentEvent(message_id="m2", delta="second bubble"),
            TextMessageEndEvent(message_id="m2"),
            ToolCallStartEvent(tool_call_id="c1", tool_call_name="search", parent_message_id="m1"),
            ToolCallArgsEvent(tool_call_id="c1", delta="{}"),
        ]
    )

    # THEN the call attaches to m1 (its parent), not the more recent m2.
    first, second = messages[0], messages[1]
    assert first.content == "first bubble"
    assert first.tool_calls is not None and first.tool_calls[0].function.name == "search"
    assert second.content == "second bubble"
    assert second.tool_calls is None


def test_prepends_reasoning() -> None:
    # GIVEN reasoning chunks emitted before the answer
    messages = events_to_messages(
        [
            ReasoningMessageChunkEvent(message_id="thought", delta="The sky "),
            ReasoningMessageChunkEvent(message_id="thought", delta="scatters blue light."),
            TextMessageContentEvent(message_id="m1", delta="The sky is blue."),
        ]
    )

    # THEN a leading reasoning message carries the concatenated reasoning
    assert [m.role for m in messages] == ["reasoning", "assistant"]
    assert messages[0].content == "The sky scatters blue light."
    assert messages[1].content == "The sky is blue."


def test_parallel_tool_calls() -> None:
    # GIVEN two tool calls in one turn, each with its own result
    messages = events_to_messages(
        [
            ToolCallStartEvent(tool_call_id="c1", tool_call_name="get_weather"),
            ToolCallArgsEvent(tool_call_id="c1", delta='{"city": "Paris"}'),
            ToolCallStartEvent(tool_call_id="c2", tool_call_name="log_event"),
            ToolCallArgsEvent(tool_call_id="c2", delta='{"e": "x"}'),
            ToolCallResultEvent(message_id="r1", tool_call_id="c1", content="18C"),
            ToolCallResultEvent(message_id="r2", tool_call_id="c2", content="ok"),
        ]
    )

    # THEN one assistant turn holds both calls, followed by both paired results in order
    assert [m.role for m in messages] == ["assistant", "tool", "tool"]
    assert [c.id for c in messages[0].tool_calls] == ["c1", "c2"]
    assert [c.function.name for c in messages[0].tool_calls] == ["get_weather", "log_event"]
    assert [(m.tool_call_id, m.content) for m in messages[1:]] == [("c1", "18C"), ("c2", "ok")]


def test_tool_result_inserted_after_owner_when_answer_streams_first() -> None:
    # GIVEN a chat -> tool -> chat loop where the follow-up answer text streams
    # BEFORE the tool result is recorded.
    messages = events_to_messages(
        [
            ToolCallStartEvent(tool_call_id="c1", tool_call_name="search"),
            ToolCallArgsEvent(tool_call_id="c1", delta="{}"),
            TextMessageContentEvent(message_id="m2", delta="here is the answer"),
            ToolCallResultEvent(message_id="r1", tool_call_id="c1", content="result"),
        ]
    )

    # THEN the result is inserted right after the assistant that issued the call,
    # not appended at the end -- so the transcript stays valid
    # (assistant(tool_calls) -> tool -> answer), not assistant -> answer -> tool.
    assert [m.role for m in messages] == ["assistant", "tool", "assistant"]
    assert messages[0].tool_calls[0].id == "c1"
    assert messages[1].tool_call_id == "c1"
    assert messages[1].content == "result"
    assert messages[2].content == "here is the answer"


def test_trailing_tool_call_without_result() -> None:
    # GIVEN a turn that ended on a tool call with no result (e.g. interrupted)
    messages = events_to_messages(
        [
            ToolCallStartEvent(tool_call_id="c1", tool_call_name="search"),
            ToolCallArgsEvent(tool_call_id="c1", delta="{}"),
        ]
    )

    # THEN the tool call is committed as the assistant message (no fabricated tool
    # result); the unpaired trailing call is trimmed by
    # drop_unpaired_boundary_tool_turns when this is replayed as history.
    assert [m.role for m in messages] == ["assistant"]
    assert messages[0].tool_calls[0].id == "c1"


def test_empty() -> None:
    assert events_to_messages([]) == []


def test_is_replayable_as_history() -> None:
    """The folded messages survive history extraction as a valid, paired transcript.

    This is the multi-turn contract: turn 1's events, folded and replayed inside a
    later request's ``messages``, must keep the assistant tool call adjacent to its
    tool result (no boundary drop, correct ordering).
    """
    # GIVEN a folded tool-using turn 1
    history = events_to_messages(
        [
            ToolCallStartEvent(tool_call_id="c1", tool_call_name="generate_objectid"),
            ToolCallArgsEvent(tool_call_id="c1", delta='{"type": "deployment"}'),
            ToolCallResultEvent(
                message_id="r1", tool_call_id="c1", content="69cbb73789723b6936c6c9e1"
            ),
            TextMessageContentEvent(message_id="m2", delta="69cbb73789723b6936c6c9e1"),
        ]
    )

    # WHEN that turn is replayed as history before a new user turn
    run_input = RunAgentInput(
        thread_id="t",
        run_id="run-2",
        state={},
        messages=[
            UserMessage(id="u1", content="generate an object id for a deployment"),
            *history,
            UserMessage(id="u2", content="what id did the tool return?"),
        ],
        tools=[],
        context=[],
        forwarded_props={},
    )
    extracted = extract_history_messages(run_input, max_history=20)
    paired = drop_unpaired_boundary_tool_turns(extracted)

    # THEN the assistant tool call is kept immediately before its tool result, and the
    # tool result content survives for the model to recall.
    roles = [m["role"] for m in paired]
    assistant_idx = roles.index("assistant")
    assert paired[assistant_idx].get("tool_calls"), "tool call dropped from replayed history"
    assert paired[assistant_idx + 1]["role"] == "tool"
    assert "69cbb73789723b6936c6c9e1" in paired[assistant_idx + 1]["content"]
