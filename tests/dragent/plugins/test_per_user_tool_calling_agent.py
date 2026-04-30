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

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallResultEvent
from ag_ui.core import ToolCallStartEvent
from nat.builder.function_info import FunctionInfo
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig

from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.plugins.per_user_tool_calling_agent import (
    PerUserToolCallAgentWorkflowConfig,
)
from datarobot_genai.dragent.plugins.per_user_tool_calling_agent import _per_user_tool_calling_agent


class TestPerUserToolCallAgentWorkflowConfig:
    def test_is_subclass_of_tool_call_agent_workflow_config(self):
        assert issubclass(PerUserToolCallAgentWorkflowConfig, ToolCallAgentWorkflowConfig)

    def test_registered_name(self):
        assert PerUserToolCallAgentWorkflowConfig.static_type() == "per_user_tool_calling_agent"

    def test_inherits_all_fields_from_parent(self):
        parent_fields = set(ToolCallAgentWorkflowConfig.model_fields) - {"type"}
        child_fields = set(PerUserToolCallAgentWorkflowConfig.model_fields) - {"type"}
        assert parent_fields == child_fields

    def test_default_instantiation(self):
        config = PerUserToolCallAgentWorkflowConfig(llm_name="gpt-4o")
        assert config is not None

    def test_registered_as_per_user_function(self):
        """Importing the module registers a per-user function; verify no exception is raised."""
        import datarobot_genai.dragent.plugins.per_user_tool_calling_agent  # noqa: F401


def _make_fn_info(stream_fn=MagicMock()):
    return FunctionInfo.create(
        single_fn=AsyncMock(),
        stream_fn=stream_fn,
        description="test",
    )


class TestPerUserToolCallingAgentWrapper:
    @pytest.mark.asyncio
    async def test_wraps_stream_fn_when_present(self):
        """When stream_fn is provided, the wrapper yields a FunctionInfo with wrapped stream."""
        original_fn_info = _make_fn_info(stream_fn=AsyncMock())

        async def fake_gen(config, builder):
            yield original_fn_info

        with patch(
            "datarobot_genai.dragent.plugins.per_user_tool_calling_agent"
            ".tool_calling_agent_workflow"
        ) as mock_workflow:
            mock_workflow.__wrapped__ = fake_gen
            config = MagicMock()
            builder = MagicMock()

            gen = _per_user_tool_calling_agent(config, builder)
            result = await gen.__anext__()

            assert isinstance(result, FunctionInfo)
            # The stream_fn should be wrapped, not the original
            assert result.stream_fn is not original_fn_info.stream_fn
            assert result.single_fn is original_fn_info.single_fn
            assert result.description == original_fn_info.description

            await gen.aclose()

    @pytest.mark.asyncio
    async def test_yields_fn_info_unchanged_when_stream_fn_is_none(self):
        """When stream_fn is None, the wrapper yields fn_info as-is."""
        original_fn_info = MagicMock(spec=FunctionInfo)
        original_fn_info.stream_fn = None

        async def fake_gen(config, builder):
            yield original_fn_info

        with patch(
            "datarobot_genai.dragent.plugins.per_user_tool_calling_agent"
            ".tool_calling_agent_workflow"
        ) as mock_workflow:
            mock_workflow.__wrapped__ = fake_gen
            config = MagicMock()
            builder = MagicMock()

            gen = _per_user_tool_calling_agent(config, builder)
            result = await gen.__anext__()

            assert result is original_fn_info
            await gen.aclose()

    @pytest.mark.asyncio
    async def test_original_generator_is_closed_on_exit(self):
        """The original generator must be closed via aclose() in the finally block."""
        original_fn_info = _make_fn_info(stream_fn=AsyncMock())
        closed = False

        async def fake_gen(config, builder):
            nonlocal closed
            try:
                yield original_fn_info
            finally:
                closed = True

        with patch(
            "datarobot_genai.dragent.plugins.per_user_tool_calling_agent"
            ".tool_calling_agent_workflow"
        ) as mock_workflow:
            mock_workflow.__wrapped__ = fake_gen
            config = MagicMock()
            builder = MagicMock()

            gen = _per_user_tool_calling_agent(config, builder)
            await gen.__anext__()
            await gen.aclose()

            assert closed


def _make_step(event_type, name, uuid_, output=None):
    """Build a minimal IntermediateStep-shaped object for the bridge's _on_step."""
    metadata = SimpleNamespace(tool_outputs=output)
    payload = SimpleNamespace(
        event_type=event_type,
        UUID=uuid_,
        name=name,
        metadata=metadata,
        data=None,
    )
    return SimpleNamespace(payload=payload)


async def _drive_wrapped_stream(converter_events):
    """Drive `_per_user_tool_calling_agent`'s wrapped_stream and return (subscriber, run).

    `converter_events` is a list of either DRAgentEventResponse (yielded by the
    fake converter) or callables (invoked between yields, given the captured
    on_step subscriber). Returns (subscription_mock, output_responses).
    """
    captured: dict[str, object] = {}

    async def fake_converter(_chunks):
        on_step = captured["on_step"]
        for item in converter_events:
            if callable(item):
                item(on_step)
            else:
                yield item

    subscription = MagicMock()
    step_mgr = MagicMock()
    step_mgr.subscribe.side_effect = lambda on_next, **_: (
        (captured.setdefault("on_step", on_next) and subscription) or subscription
    )
    context = MagicMock()
    context.intermediate_step_manager = step_mgr

    original_fn_info = _make_fn_info(stream_fn=AsyncMock())

    async def fake_gen(config, builder):
        yield original_fn_info

    with (
        patch(
            "datarobot_genai.dragent.plugins.per_user_tool_calling_agent"
            ".tool_calling_agent_workflow"
        ) as mock_workflow,
        patch(
            "datarobot_genai.dragent.plugins.per_user_tool_calling_agent"
            ".convert_chunks_to_agui_events",
            new=fake_converter,
        ),
        patch(
            "datarobot_genai.dragent.plugins.per_user_tool_calling_agent.Context.get",
            return_value=context,
        ),
    ):
        mock_workflow.__wrapped__ = fake_gen
        gen = _per_user_tool_calling_agent(MagicMock(), MagicMock())
        fn_info = await gen.__anext__()
        outputs: list[DRAgentEventResponse] = []
        async for response in fn_info.stream_fn(MagicMock()):
            outputs.append(response)
        await gen.aclose()
        return subscription, outputs


def _flatten(outputs):
    return [e for response in outputs for e in response.events]


class TestSubscriberLifecycle:
    @pytest.mark.asyncio
    async def test_subscription_unsubscribed_on_stream_completion(self):
        """The intermediate-step subscription is disposed when wrapped_stream exits."""
        subscription, _ = await _drive_wrapped_stream([])
        subscription.unsubscribe.assert_called_once()


class TestToolCallResultBridge:
    async def test_preserves_parallel_same_name_results_when_function_starts_arrive_first(self):
        """Parallel same-name tools keep invocation order after late starts."""
        zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # GIVEN two same-name tool invocations whose NAT starts arrive before streamed tool starts
        events = [
            lambda on_step: on_step(_make_step(IntermediateStepType.FUNCTION_START, "foo", "u1")),
            lambda on_step: on_step(_make_step(IntermediateStepType.FUNCTION_START, "foo", "u2")),
            DRAgentEventResponse(
                events=[
                    ToolCallStartEvent(tool_call_id="tc1", tool_call_name="foo"),
                    ToolCallStartEvent(tool_call_id="tc2", tool_call_name="foo"),
                ],
                usage_metrics=zero,
            ),
            lambda on_step: on_step(
                _make_step(IntermediateStepType.FUNCTION_END, "foo", "u2", output="B")
            ),
            lambda on_step: on_step(
                _make_step(IntermediateStepType.FUNCTION_END, "foo", "u1", output="A")
            ),
        ]

        # WHEN the wrapped stream is drained
        _, outputs = await _drive_wrapped_stream(events)
        results = [event for event in _flatten(outputs) if isinstance(event, ToolCallResultEvent)]

        # THEN each completion stays attached to the correct tool call id
        assert [result.tool_call_id for result in results] == ["tc2", "tc1"]
        assert [result.content for result in results] == ["B", "A"]

    @pytest.mark.asyncio
    async def test_synthesizes_end_before_queued_result_for_open_tool(self):
        """If a tool completes before the converter emits END, the bridge synthesizes one."""
        zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Sequence: TOOL_CALL_START(tc1) → fire FUNCTION_END for tc1 → next tool starts.
        # The converter never emits a TOOL_CALL_END for tc1, so the bridge must synthesize one.
        events = [
            DRAgentEventResponse(
                events=[ToolCallStartEvent(tool_call_id="tc1", tool_call_name="foo")],
                usage_metrics=zero,
            ),
            lambda on_step: on_step(_make_step(IntermediateStepType.FUNCTION_START, "foo", "u1")),
            lambda on_step: on_step(
                _make_step(IntermediateStepType.FUNCTION_END, "foo", "u1", output="A")
            ),
            DRAgentEventResponse(
                events=[ToolCallStartEvent(tool_call_id="tc2", tool_call_name="foo")],
                usage_metrics=zero,
            ),
        ]
        _, outputs = await _drive_wrapped_stream(events)
        flat = _flatten(outputs)

        types = [type(e).__name__ for e in flat]
        assert types == [
            "ToolCallStartEvent",
            "ToolCallEndEvent",
            "ToolCallResultEvent",
            "ToolCallStartEvent",
        ]
        synthetic_end = flat[1]
        result = flat[2]
        assert isinstance(synthetic_end, ToolCallEndEvent)
        assert synthetic_end.tool_call_id == "tc1"
        assert isinstance(result, ToolCallResultEvent)
        assert result.tool_call_id == "tc1"

    @pytest.mark.asyncio
    async def test_correlates_parallel_same_name_tools_via_invocation_uuid(self):
        """Same-name parallel tools match results by step UUID, not FIFO completion order."""
        zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Two starts for the same tool name, dispatch order tc1 then tc2.
        # FUNCTION_START fires in dispatch order → invocation u1↔tc1, u2↔tc2.
        # Tool tc2 completes first (u2), then tc1 (u1). Results must use the
        # correct tc_ids despite the swapped completion order.
        events = [
            DRAgentEventResponse(
                events=[
                    ToolCallStartEvent(tool_call_id="tc1", tool_call_name="foo"),
                    ToolCallStartEvent(tool_call_id="tc2", tool_call_name="foo"),
                ],
                usage_metrics=zero,
            ),
            lambda on_step: on_step(_make_step(IntermediateStepType.FUNCTION_START, "foo", "u1")),
            lambda on_step: on_step(_make_step(IntermediateStepType.FUNCTION_START, "foo", "u2")),
            lambda on_step: on_step(
                _make_step(IntermediateStepType.FUNCTION_END, "foo", "u2", output="B")
            ),
            lambda on_step: on_step(
                _make_step(IntermediateStepType.FUNCTION_END, "foo", "u1", output="A")
            ),
            DRAgentEventResponse(
                events=[
                    ToolCallEndEvent(tool_call_id="tc1"),
                    ToolCallEndEvent(tool_call_id="tc2"),
                ],
                usage_metrics=zero,
            ),
        ]
        _, outputs = await _drive_wrapped_stream(events)
        results = [e for e in _flatten(outputs) if isinstance(e, ToolCallResultEvent)]

        assert [r.tool_call_id for r in results] == ["tc2", "tc1"]
        assert [r.content for r in results] == ["B", "A"]
