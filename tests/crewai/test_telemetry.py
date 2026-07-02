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
from unittest.mock import patch

import pytest

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
        # Isolate the "which targets get wrapped" logic from the double-wrap
        # guard, whose result depends on whether real instrumentation already
        # wrapped these modules earlier in the test session.
        patch.object(telemetry, "_is_already_wrapped", return_value=False),
    ):
        instrumentor._instrument()

    super_instrument.assert_called_once()
    wrapped_targets = {(call.args[0], call.args[1]) for call in wrap.call_args_list}
    assert wrapped_targets == {
        ("crewai.crew", "Crew.akickoff"),
        ("crewai.agent", "Agent.aexecute_task"),
        ("crewai.task", "Task.aexecute_sync"),
        ("crewai.llm", "LLM.acall"),
        # DataRobot-specific LLM choke-point wrappers (not upstream). Wrapped at
        # the source module and in every executor module that imports them by
        # name (see _DATAROBOT_WRAP_TARGETS).
        ("crewai.utilities.agent_utils", "aget_llm_response"),
        ("crewai.utilities.agent_utils", "get_llm_response"),
        ("crewai.agents.crew_agent_executor", "aget_llm_response"),
        ("crewai.agents.crew_agent_executor", "get_llm_response"),
        ("crewai.lite_agent", "get_llm_response"),
        ("crewai.experimental.agent_executor", "get_llm_response"),
    }


def test_instrument_wraps_executor_llm_choke_points() -> None:
    """The LLM choke points are wrapped where the executors call them.

    CrewAI's executors import ``get_llm_response`` / ``aget_llm_response`` by
    name, so wrapping only ``crewai.utilities.agent_utils`` leaves the
    executor's own reference unwrapped and no ``{model}.llm`` span is emitted on
    the agent paths. This guards that the executor namespaces are wrapped.
    """
    import crewai.agents.crew_agent_executor as executor

    telemetry.instrument()

    assert hasattr(executor.aget_llm_response, "__wrapped__"), (
        "async LLM choke point in crew_agent_executor must be wrapped so the "
        "async agent path emits an LLM span"
    )
    assert hasattr(executor.get_llm_response, "__wrapped__"), (
        "sync LLM choke point in crew_agent_executor must be wrapped so the "
        "sync agent path emits an LLM span"
    )


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
        # DataRobot-specific LLM choke-point wrappers (not upstream).
        ("crewai.utilities.agent_utils", "aget_llm_response"),
        ("crewai.utilities.agent_utils", "get_llm_response"),
        ("crewai.agents.crew_agent_executor", "aget_llm_response"),
        ("crewai.agents.crew_agent_executor", "get_llm_response"),
        ("crewai.lite_agent", "get_llm_response"),
        ("crewai.experimental.agent_executor", "get_llm_response"),
    }
