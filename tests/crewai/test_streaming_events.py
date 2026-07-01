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

"""Tests for CrewAIStreamingEventListener tool-call capture."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

from crewai.events import crewai_event_bus
from crewai.events.event_types import AgentExecutionStartedEvent
from crewai.events.event_types import ToolUsageErrorEvent
from crewai.events.event_types import ToolUsageFinishedEvent
from crewai.events.event_types import ToolUsageStartedEvent

from datarobot_genai.crewai.streaming_events import CrewAIStreamingEventListener
from datarobot_genai.crewai.streaming_events import ToolCallRecord

_T = datetime(2026, 1, 1)
_SRC = object()


def _emit(event: Any) -> None:
    # The bus dispatches handlers on a background executor and returns a Future; wait on it so
    # the listener's queue is populated before we drain (in production the loop drains later).
    fut = crewai_event_bus.emit(_SRC, event=event)
    if fut is not None:
        fut.result(timeout=5)


def _drain(listener: CrewAIStreamingEventListener) -> list[ToolCallRecord]:
    out = []
    while not listener.tool_call_events.empty():
        out.append(listener.tool_call_events.get_nowait())
    return out


def _finished(name: str, output: str, args: Any = None) -> ToolUsageFinishedEvent:
    # Real CrewAI emits the call's own tool_args on Finished (same dict as Started); pairing keys
    # on it, so tests must pass matching args.
    return ToolUsageFinishedEvent(
        tool_name=name, tool_args=args or {}, started_at=_T, finished_at=_T, output=output
    )


def test_started_then_finished_pairs_call_and_result_records() -> None:
    listener = CrewAIStreamingEventListener()
    with crewai_event_bus.scoped_handlers():
        listener.setup_listeners(crewai_event_bus)
        _emit(ToolUsageStartedEvent(tool_name="word_counter", tool_args={"text": "a b"}))
        _emit(_finished("word_counter", "2", {"text": "a b"}))

    records = _drain(listener)
    assert [r.kind for r in records] == ["call", "result"]
    call, result = records
    assert call.name == "word_counter"
    assert call.args == '{"text": "a b"}'  # dict args serialized to JSON
    assert result.content == "2"
    assert call.tool_call_id == result.tool_call_id  # paired by (tool_name, tool_args)


def test_concurrent_results_pair_by_content_not_arrival_order() -> None:
    # Three calls with distinct args whose results arrive OUT of start order (real native multi-tool
    # behaviour). Pairing must follow content, not FIFO -- else a result lands on the wrong call.
    listener = CrewAIStreamingEventListener()
    with crewai_event_bus.scoped_handlers():
        listener.setup_listeners(crewai_event_bus)
        _emit(ToolUsageStartedEvent(tool_name="wc", tool_args={"text": "alpha"}))
        _emit(ToolUsageStartedEvent(tool_name="wc", tool_args={"text": "beta gamma"}))
        _emit(ToolUsageStartedEvent(tool_name="wc", tool_args={"text": "delta epsilon zeta"}))
        _emit(_finished("wc", "1", {"text": "alpha"}))
        _emit(_finished("wc", "3", {"text": "delta epsilon zeta"}))  # finishes before "beta gamma"
        _emit(_finished("wc", "2", {"text": "beta gamma"}))

    records = _drain(listener)
    id_by_args = {r.args: r.tool_call_id for r in records if r.kind == "call"}
    content_by_id = {r.tool_call_id: r.content for r in records if r.kind == "result"}
    # each call's result matches ITS OWN args, regardless of completion order (FIFO would swap them)
    assert content_by_id[id_by_args['{"text": "alpha"}']] == "1"
    assert content_by_id[id_by_args['{"text": "beta gamma"}']] == "2"
    assert content_by_id[id_by_args['{"text": "delta epsilon zeta"}']] == "3"


def test_result_before_its_start_is_buffered_and_paired() -> None:
    # Handler reorder (thread pool doesn't preserve emit order): a Finished arriving before its
    # Started must be buffered and emitted call-before-result, not dropped or orphaned.
    listener = CrewAIStreamingEventListener()
    with crewai_event_bus.scoped_handlers():
        listener.setup_listeners(crewai_event_bus)
        _emit(_finished("wc", "2", {"text": "a b"}))  # Finished first, no Started yet
        _emit(ToolUsageStartedEvent(tool_name="wc", tool_args={"text": "a b"}))

    records = _drain(listener)
    assert [r.kind for r in records] == ["call", "result"]  # call emitted before result
    assert records[0].tool_call_id == records[1].tool_call_id
    assert records[1].content == "2"


def test_error_emits_result_record_paired_with_its_call() -> None:
    listener = CrewAIStreamingEventListener()
    with crewai_event_bus.scoped_handlers():
        listener.setup_listeners(crewai_event_bus)
        _emit(ToolUsageStartedEvent(tool_name="flaky", tool_args={}))
        _emit(ToolUsageErrorEvent(tool_name="flaky", tool_args={}, error="boom"))

    records = _drain(listener)
    assert [r.kind for r in records] == ["call", "result"]
    assert records[1].content == "Error: boom"
    assert records[0].tool_call_id == records[1].tool_call_id


def test_extra_result_for_one_start_is_dropped() -> None:
    # One Started followed by two ending events (a stray/duplicate result) must yield exactly one
    # result -- the extra, unpaired one is dropped rather than given a phantom id.
    listener = CrewAIStreamingEventListener()
    with crewai_event_bus.scoped_handlers():
        listener.setup_listeners(crewai_event_bus)
        _emit(ToolUsageStartedEvent(tool_name="flaky", tool_args={}))
        _emit(ToolUsageErrorEvent(tool_name="flaky", tool_args={}, error="boom"))
        _emit(_finished("flaky", "boom"))  # extra ending event, no matching Started left

    records = _drain(listener)
    assert [r.kind for r in records] == ["call", "result"]  # exactly one result, no orphan
    assert records[1].content == "Error: boom"
    assert records[0].tool_call_id == records[1].tool_call_id


def test_unpaired_finished_emits_no_result() -> None:
    # A Finished with no matching Started must not synthesize a lone result.
    listener = CrewAIStreamingEventListener()
    with crewai_event_bus.scoped_handlers():
        listener.setup_listeners(crewai_event_bus)
        _emit(_finished("ghost", "x"))

    assert _drain(listener) == []


def _agent_started(role: str) -> AgentExecutionStartedEvent:
    # model_construct: the real event requires a BaseAgent (needs provider/keys to build);
    # the handler only reads event.agent.role, so a stub agent is enough.
    return AgentExecutionStartedEvent.model_construct(
        agent=SimpleNamespace(role=role),
        task=None,
        tools=None,
        task_prompt="p",
        type="agent_execution_started",
    )


def test_agent_execution_started_tracks_active_role() -> None:
    listener = CrewAIStreamingEventListener()
    with crewai_event_bus.scoped_handlers():
        listener.setup_listeners(crewai_event_bus)
        _emit(_agent_started("Planner"))
        assert listener.active_agent_role == "Planner"
        _emit(_agent_started("Writer"))
        assert listener.active_agent_role == "Writer"


def test_paired_buffers_do_not_retain_empty_entries() -> None:
    # After a call+result reconcile, the per-key deque is emptied; leaving the key behind grows the
    # buffers by one dead entry per distinct tool for the run. Reconciled keys must be dropped.
    listener = CrewAIStreamingEventListener()
    with crewai_event_bus.scoped_handlers():
        listener.setup_listeners(crewai_event_bus)
        _emit(ToolUsageStartedEvent(tool_name="wc", tool_args={"text": "a"}))
        _emit(_finished("wc", "1", {"text": "a"}))

    assert not listener._calls_awaiting_result
    assert not listener._results_awaiting_call


def test_tool_call_role_comes_from_event_not_shared_active_role() -> None:
    # The tool's own agent rides on the event; the shared active_agent_role can already point at the
    # NEXT agent (its AgentExecutionStarted handler ran first), so attributing by the field would
    # open the wrong agent's step. Prefer the event's role.
    listener = CrewAIStreamingEventListener()
    with crewai_event_bus.scoped_handlers():
        listener.setup_listeners(crewai_event_bus)
        _emit(_agent_started("Writer"))  # active role has advanced to the next agent
        _emit(
            ToolUsageStartedEvent(
                tool_name="wc",
                tool_args={"text": "a"},
                from_agent=SimpleNamespace(id="p1", role="Planner"),
            )
        )

    records = _drain(listener)
    assert records[0].kind == "call"
    assert records[0].agent_role == "Planner"  # from the event, not the "Writer" active role


def test_tool_call_role_falls_back_to_active_when_event_lacks_role() -> None:
    # Paths that don't set from_agent leave the event's agent_role empty; fall back to the
    # bus-tracked active role so single-agent streams still get their step.
    listener = CrewAIStreamingEventListener()
    with crewai_event_bus.scoped_handlers():
        listener.setup_listeners(crewai_event_bus)
        _emit(_agent_started("Planner"))
        _emit(ToolUsageStartedEvent(tool_name="wc", tool_args={}))

    records = _drain(listener)
    assert records[0].agent_role == "Planner"


def test_two_sequential_calls_keep_distinct_paired_ids() -> None:
    listener = CrewAIStreamingEventListener()
    with crewai_event_bus.scoped_handlers():
        listener.setup_listeners(crewai_event_bus)
        _emit(ToolUsageStartedEvent(tool_name="a", tool_args={}))
        _emit(_finished("a", "r1"))
        _emit(ToolUsageStartedEvent(tool_name="b", tool_args={}))
        _emit(_finished("b", "r2"))

    records = _drain(listener)
    assert [r.kind for r in records] == ["call", "result", "call", "result"]
    assert records[0].tool_call_id == records[1].tool_call_id  # pair A
    assert records[2].tool_call_id == records[3].tool_call_id  # pair B
    assert records[0].tool_call_id != records[2].tool_call_id  # distinct calls
