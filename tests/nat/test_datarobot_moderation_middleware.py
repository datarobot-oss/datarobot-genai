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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("datarobot_dome")

from ag_ui.core import TextMessageContentEvent
from ag_ui.core import ToolCallStartEvent
from ag_ui.core import UserMessage
from datarobot_dome.constants import NONE_CUSTOM_PY_RESPONSE
from datarobot_dome.constants import GuardStage
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.dragent.frontends.request import DRAgentRunAgentInput
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.nat.datarobot_moderation_middleware import DataRobotModerationConfig
from datarobot_genai.nat.datarobot_moderation_middleware import DataRobotModerationMiddleware

PROMPT_COL = "prompt_col"
RESPONSE_COL = "response_col"


def _make_run_input(content: str = "hello") -> DRAgentRunAgentInput:
    return DRAgentRunAgentInput(
        thread_id="t1",
        run_id="r1",
        messages=[UserMessage(id="u1", content=content)],
        tools=[],
        context=[],
        forwarded_props={},
        state={},
    )


def _fn_context() -> FunctionMiddlewareContext:
    return FunctionMiddlewareContext(
        name="test_fn",
        config={},
        description=None,
        input_schema=None,
        single_output_schema=DRAgentEventResponse,
        stream_output_schema=DRAgentEventResponse,
    )


def _invocation(
    run_input: DRAgentRunAgentInput,
    output: DRAgentEventResponse | None = None,
) -> InvocationContext:
    return InvocationContext(
        function_context=_fn_context(),
        original_args=(run_input,),
        original_kwargs={},
        modified_args=(run_input,),
        modified_kwargs={},
        output=output,
    )


def _text_response(text: str) -> DRAgentEventResponse:
    mid = "msg-1"
    return DRAgentEventResponse(
        events=[TextMessageContentEvent(message_id=mid, delta=text)],
        usage_metrics=default_usage_metrics(),
    )


def _pipeline_mock() -> MagicMock:
    pipeline = MagicMock()
    pipeline.get_input_column.side_effect = lambda stage: (
        PROMPT_COL if stage == GuardStage.PROMPT else RESPONSE_COL
    )
    pipeline.get_association_id_column_name.return_value = ""
    pipeline.get_new_metrics_payload.return_value = None
    pipeline.extra_model_output_for_chat_enabled = False
    return pipeline


def _moderation_mock(pipeline: MagicMock) -> MagicMock:
    mod = MagicMock()
    mod._pipeline = pipeline
    return mod


@pytest.fixture
def builder_mock() -> MagicMock:
    return MagicMock()


def test_enabled_false_when_pipeline_not_loaded(builder_mock: MagicMock) -> None:
    # GIVEN load_llm_moderation_pipeline returns None
    # WHEN middleware is constructed
    # THEN enabled is False
    with patch(
        "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
        return_value=None,
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
    assert mw.enabled is False


async def test_pre_invoke_blocks_and_sets_output(builder_mock: MagicMock) -> None:
    # GIVEN prescore ran and evaluate_prompt reports a blocked prompt
    # WHEN pre_invoke runs
    # THEN context.output is set to a DRAgentEventResponse and call_next must not run
    pipeline = _pipeline_mock()
    moderation = _moderation_mock(pipeline)
    blocked = SimpleNamespace(
        blocked=True,
        blocked_message="blocked-by-test",
        replaced=False,
        replacement=None,
    )
    moderation.evaluate_prompt.return_value = (blocked, None)

    prescore_df = pd.DataFrame({PROMPT_COL: ["bad"]})

    run_input = _make_run_input("bad")
    ctx = _invocation(run_input)

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.run_prescore_guards",
            return_value=(prescore_df, prescore_df, 0.0),
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.report_otel_evaluation_set_metric",
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.set_moderation_attribute_to_completion",
            side_effect=lambda _p, completion, _df, association_id=None: completion,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        out = await mw.pre_invoke(ctx)

    assert out is not None
    assert out.output is not None
    assert isinstance(out.output, DRAgentEventResponse)
    assert any(
        isinstance(ev, TextMessageContentEvent) and "blocked-by-test" in ev.delta
        for ev in out.output.events
    )


async def test_pre_invoke_replaces_last_user_message(builder_mock: MagicMock) -> None:
    # GIVEN evaluate_prompt requests a replacement string
    # WHEN pre_invoke runs
    # THEN the last UserMessage content is updated and context.output stays unset
    pipeline = _pipeline_mock()
    moderation = _moderation_mock(pipeline)
    replaced = SimpleNamespace(
        blocked=False,
        blocked_message=None,
        replaced=True,
        replacement="[redacted]",
    )
    moderation.evaluate_prompt.return_value = (replaced, None)

    prescore_df = pd.DataFrame({PROMPT_COL: ["secret"]})

    run_input = _make_run_input("secret")
    ctx = _invocation(run_input)

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.run_prescore_guards",
            return_value=(prescore_df, prescore_df, 0.0),
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        out = await mw.pre_invoke(ctx)

    assert out is not None
    assert out.output is None
    assert run_input.messages[-1].content == "[redacted]"
    assert mw.data is not None
    assert mw.data.loc[0, PROMPT_COL] == "[redacted]"


async def test_post_invoke_skips_non_text_first_event(builder_mock: MagicMock) -> None:
    # GIVEN the first AG-UI event is not a text delta
    # WHEN post_invoke runs
    # THEN output is unchanged (None return)
    pipeline = _pipeline_mock()
    moderation = _moderation_mock(pipeline)
    tool_first = DRAgentEventResponse(
        events=[
            ToolCallStartEvent(
                tool_call_id="tc1",
                tool_call_name="noop",
                parent_message_id=None,
            )
        ],
        usage_metrics=default_usage_metrics(),
    )
    ctx = _invocation(_make_run_input(), output=tool_first)

    with patch(
        "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
        return_value=moderation,
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)

    out = await mw.post_invoke(ctx)
    assert out is None
    assert ctx.output is tool_first


async def test_post_invoke_rewrites_completion_when_postscore_succeeds(
    builder_mock: MagicMock,
) -> None:
    # GIVEN a normal assistant text response and postscore produces a final message
    # WHEN post_invoke runs
    # THEN context.output is replaced with a new DRAgentEventResponse containing that message
    pipeline = _pipeline_mock()
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_response.return_value = (
        SimpleNamespace(blocked=False, replaced=False, replacement=None, blocked_message=None),
        None,
    )

    predictions = pd.DataFrame({PROMPT_COL: ["p"], RESPONSE_COL: ["model-out"]})
    postscore = pd.DataFrame({RESPONSE_COL: ["final-out"]})

    run_input = _make_run_input()
    ctx = _invocation(run_input, output=_text_response("model-out"))

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.build_predictions_df_from_completion",
            return_value=(predictions, {}),
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.run_postscore_guards",
            return_value=(postscore, 0.0),
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.format_result_df",
            return_value=pd.DataFrame({"dummy": [1]}),
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.report_otel_evaluation_set_metric",
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.set_moderation_attribute_to_completion",
            side_effect=lambda _p, completion, _df, association_id=None: completion,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        mw.data = pd.DataFrame({PROMPT_COL: ["p"]})
        mw.prescore_df = mw.data.copy()

        out = await mw.post_invoke(ctx)

    assert out is not None
    assert ctx.output is not None
    text = "".join(ev.delta for ev in ctx.output.events if isinstance(ev, TextMessageContentEvent))
    assert text == "final-out"


async def test_post_invoke_uses_none_custom_response_when_postscore_empty(
    builder_mock: MagicMock,
) -> None:
    # GIVEN the model response column is None so postscore is skipped
    # WHEN post_invoke selects the message for an empty postscore frame
    # THEN NONE_CUSTOM_PY_RESPONSE is used
    pipeline = _pipeline_mock()
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_response.return_value = (
        SimpleNamespace(blocked=False, replaced=False, replacement=None, blocked_message=None),
        None,
    )

    predictions = pd.DataFrame({PROMPT_COL: ["p"], RESPONSE_COL: [None]})

    ctx = _invocation(_make_run_input(), output=_text_response("ignored"))

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.build_predictions_df_from_completion",
            return_value=(predictions, {}),
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.run_postscore_guards",
            return_value=(pd.DataFrame(), 0.0),
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.format_result_df",
            return_value=pd.DataFrame(),
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.report_otel_evaluation_set_metric",
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.set_moderation_attribute_to_completion",
            side_effect=lambda _p, completion, _df, association_id=None: completion,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        mw.data = pd.DataFrame({PROMPT_COL: ["p"]})
        mw.prescore_df = mw.data.copy()

        await mw.post_invoke(ctx)

    text = "".join(ev.delta for ev in ctx.output.events if isinstance(ev, TextMessageContentEvent))
    assert text == NONE_CUSTOM_PY_RESPONSE


async def test_function_middleware_invoke_blocked_short_circuits(builder_mock: MagicMock) -> None:
    # GIVEN pre_invoke sets context.output (blocked)
    # WHEN function_middleware_invoke runs
    # THEN call_next is never awaited
    pipeline = _pipeline_mock()
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_prompt.return_value = (
        SimpleNamespace(
            blocked=True,
            blocked_message="stop",
            replaced=False,
            replacement=None,
        ),
        None,
    )
    prescore_df = pd.DataFrame({PROMPT_COL: ["x"]})

    call_next = AsyncMock()

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.run_prescore_guards",
            return_value=(prescore_df, prescore_df, 0.0),
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.report_otel_evaluation_set_metric",
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.set_moderation_attribute_to_completion",
            side_effect=lambda _p, completion, _df, association_id=None: completion,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        result = await mw.function_middleware_invoke(
            _make_run_input(),
            call_next=call_next,
            context=_fn_context(),
        )

    call_next.assert_not_awaited()
    assert isinstance(result, DRAgentEventResponse)


async def test_function_middleware_stream_yields_blocked_pre_invoke(
    builder_mock: MagicMock,
) -> None:
    # GIVEN streaming entrypoint and prescore blocks
    # WHEN function_middleware_stream runs
    # THEN a single blocked response is yielded and the worker is never started
    pipeline = _pipeline_mock()
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_prompt.return_value = (
        SimpleNamespace(
            blocked=True,
            blocked_message="no-stream",
            replaced=False,
            replacement=None,
        ),
        None,
    )
    prescore_df = pd.DataFrame({PROMPT_COL: ["x"]})
    stream_next = MagicMock()

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.run_prescore_guards",
            return_value=(prescore_df, prescore_df, 0.0),
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.report_otel_evaluation_set_metric",
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.set_moderation_attribute_to_completion",
            side_effect=lambda _p, completion, _df, association_id=None: completion,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        chunks = [
            item
            async for item in mw.function_middleware_stream(
                _make_run_input(),
                call_next=stream_next,
                context=_fn_context(),
            )
        ]

    stream_next.assert_not_called()
    assert len(chunks) == 1
    assert isinstance(chunks[0], DRAgentEventResponse)


async def test_function_middleware_stream_echoes_single_text_chunk(builder_mock: MagicMock) -> None:
    # GIVEN one streamed text chunk from upstream and ModerationIterator echoes chunks
    # WHEN function_middleware_stream runs
    # THEN one moderated DRAgentEventResponse is yielded
    pipeline = _pipeline_mock()
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_prompt.return_value = (
        SimpleNamespace(blocked=False, blocked_message=None, replaced=False, replacement=None),
        None,
    )
    prescore_df = pd.DataFrame({PROMPT_COL: ["hi"]})

    async def upstream():
        yield _text_response("delta-one")

    stream_next = MagicMock(return_value=upstream())

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.run_prescore_guards",
            return_value=(prescore_df, prescore_df, 0.0),
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.ModerationIterator",
            side_effect=lambda _sc, src: src,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        chunks = [
            item
            async for item in mw.function_middleware_stream(
                _make_run_input("hi"),
                call_next=stream_next,
                context=_fn_context(),
            )
        ]

    assert len(chunks) == 1
    assert isinstance(chunks[0], DRAgentEventResponse)
    deltas = "".join(ev.delta for ev in chunks[0].events if isinstance(ev, TextMessageContentEvent))
    assert deltas == "delta-one"


async def test_function_middleware_stream_passthrough_when_no_run_agent_input(
    builder_mock: MagicMock,
) -> None:
    # GIVEN pre_invoke returns before prescore (no first positional arg / no run input)
    # WHEN function_middleware_stream runs
    # THEN upstream chunks are yielded and ModerationIterator is never used
    pipeline = _pipeline_mock()
    moderation = _moderation_mock(pipeline)

    async def upstream():
        yield _text_response("passthrough")

    stream_next = MagicMock(return_value=upstream())

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.ModerationIterator",
            side_effect=AssertionError("ModerationIterator should not run without prescore"),
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        chunks = [
            item
            async for item in mw.function_middleware_stream(
                call_next=stream_next,
                context=_fn_context(),
            )
        ]

    stream_next.assert_called_once()
    assert len(chunks) == 1
    assert chunks[0].events == _text_response("passthrough").events
