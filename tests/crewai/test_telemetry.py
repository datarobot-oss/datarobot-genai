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

import os
from collections.abc import Iterator
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from opentelemetry import baggage
from opentelemetry import trace

from datarobot_genai.crewai import telemetry


@pytest.fixture(autouse=True)
def reset_state() -> Iterator[None]:
    """Reset the module-level instrumentation flag around each test."""
    telemetry._INSTRUMENTED["crewai"] = False
    yield
    telemetry._INSTRUMENTED["crewai"] = False


def test_instrument_enables_instrumentor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CREWAI_TESTING", raising=False)
    with patch.object(telemetry, "DataRobotCrewAIInstrumentor") as instrumentor:
        telemetry.instrument()

    instrumentor.return_value.instrument.assert_called_once()
    assert telemetry._INSTRUMENTED["crewai"] is True
    assert os.environ["CREWAI_TESTING"] == "true"


def test_instrument_is_idempotent() -> None:
    with patch.object(telemetry, "DataRobotCrewAIInstrumentor") as instrumentor:
        telemetry.instrument()
        telemetry.instrument()

    instrumentor.return_value.instrument.assert_called_once()


def test_instrument_swallows_errors() -> None:
    with patch.object(telemetry, "DataRobotCrewAIInstrumentor") as instrumentor:
        instrumentor.return_value.instrument.side_effect = RuntimeError("boom")
        # Should not raise
        telemetry.instrument()

    assert telemetry._INSTRUMENTED["crewai"] is False


def test_subclass_wraps_async_methods() -> None:
    """The subclass delegates the sync path to the base and adds the async wrappers."""
    instrumentor = telemetry.DataRobotCrewAIInstrumentor()
    with (
        patch.object(telemetry.CrewAIInstrumentor, "_instrument") as super_instrument,
        patch.object(telemetry, "wrap_function_wrapper") as wrap,
    ):
        instrumentor._instrument()

    super_instrument.assert_called_once()
    wrapped_targets = {(call.args[0], call.args[1]) for call in wrap.call_args_list}
    assert wrapped_targets == {
        ("crewai.crew", "Crew.akickoff"),
        ("crewai.agent", "Agent.aexecute_task"),
        ("crewai.task", "Task.aexecute_sync"),
        ("crewai.llm", "LLM.acall"),
    }


def test_subclass_skips_missing_async_methods() -> None:
    """Wrapping is guarded so missing async methods don't break instrumentation."""
    instrumentor = telemetry.DataRobotCrewAIInstrumentor()
    with (
        patch.object(telemetry.CrewAIInstrumentor, "_instrument"),
        patch.object(telemetry, "wrap_function_wrapper", side_effect=AttributeError),
    ):
        # Should not raise even though every wrap attempt fails.
        instrumentor._instrument()


def test_subclass_unwraps_async_methods() -> None:
    instrumentor = telemetry.DataRobotCrewAIInstrumentor()
    with (
        patch.object(telemetry.CrewAIInstrumentor, "_uninstrument") as super_uninstrument,
        patch.object(telemetry, "unwrap") as unwrap,
    ):
        instrumentor._uninstrument()

    super_uninstrument.assert_called_once()
    unwrapped_targets = {(call.args[0], call.args[1]) for call in unwrap.call_args_list}
    assert unwrapped_targets == {
        ("crewai.crew.Crew", "akickoff"),
        ("crewai.agent.Agent", "aexecute_task"),
        ("crewai.task.Task", "aexecute_sync"),
        ("crewai.llm.LLM", "acall"),
    }


class TestWrapAexecuteTaskPropagatesAgentNameAsBaggage:
    """gen_ai.agent.name must be visible as Baggage for the duration of the
    wrapped call, so a tool call made from inside it can read which agent
    triggered it.
    """

    async def test_agent_role_is_visible_as_baggage_during_the_call(self) -> None:
        instance = MagicMock()
        instance.role = "researcher"
        seen_agent_name = None

        async def wrapped(*args: object, **kwargs: object) -> str:
            nonlocal seen_agent_name
            seen_agent_name = baggage.get_baggage("gen_ai.agent.name")
            return "ok"

        wrapper = telemetry.wrap_aexecute_task(trace.get_tracer(__name__), None, None)
        result = await wrapper(wrapped, instance, (), {})

        assert result == "ok"
        assert seen_agent_name == "researcher"
        assert baggage.get_baggage("gen_ai.agent.name") is None

    async def test_missing_role_leaves_baggage_untouched(self) -> None:
        instance = MagicMock()
        del instance.role
        seen_agent_name = "sentinel"

        async def wrapped(*args: object, **kwargs: object) -> str:
            nonlocal seen_agent_name
            seen_agent_name = baggage.get_baggage("gen_ai.agent.name")
            return "ok"

        wrapper = telemetry.wrap_aexecute_task(trace.get_tracer(__name__), None, None)
        await wrapper(wrapped, instance, (), {})

        assert seen_agent_name is None
