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

import asyncio
from datetime import UTC
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("datarobot_dome")

from ag_ui.core import EventType
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallStartEvent
from ag_ui.core import UserMessage
from datarobot_dome.api import EvaluationResult
from datarobot_dome.constants import DATAROBOT_MODERATIONS_ATTR
from datarobot_dome.constants import NONE_CUSTOM_PY_RESPONSE
from datarobot_dome.constants import GuardStage
from datarobot_moderation_interface.drum_integration import get_chat_prompt
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChoice
from nat.data_models.api_server import ChoiceMessage
from nat.data_models.api_server import Message as NATAPIMessage
from nat.data_models.api_server import Usage as NATChatUsage
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as OpenAIChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta as OpenAIChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall as OpenAIChoiceDeltaToolCall
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCallFunction as OpenAIChoiceDeltaToolCallFunction,
)

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.core.agents.verify import validate_sequence
from datarobot_genai.dragent.frontends.request import DRAgentRunAgentInput
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.nat.datarobot_moderation_middleware import DataRobotModerationConfig
from datarobot_genai.nat.datarobot_moderation_middleware import DataRobotModerationMiddleware
from datarobot_genai.nat.datarobot_moderation_middleware import (
    _clear_moderation_invoke_state_if_set,
)
from datarobot_genai.nat.datarobot_moderation_middleware import _moderation_invoke_state_ctx
from datarobot_genai.nat.datarobot_moderation_middleware import _set_moderation_invoke_state
from datarobot_genai.nat.datarobot_moderation_middleware import (
    chat_completion_to_dragent_event_response,
)
from datarobot_genai.nat.datarobot_moderation_middleware import (
    moderation_prompt_from_workflow_input,
)
from datarobot_genai.nat.datarobot_moderation_middleware import workflow_input_to_completion_dict

PROMPT_COL = "prompt_col"
RESPONSE_COL = "response_col"


def _nat_chat_response_assistant_text(content: str) -> ChatResponse:
    return ChatResponse(
        id="cr-id",
        choices=[
            ChatResponseChoice(
                index=0,
                finish_reason="stop",
                message=ChoiceMessage(role="assistant", content=content),
            )
        ],
        created=datetime.now(UTC),
        model="test-model",
        usage=NATChatUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )


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
    mod._executor = MagicMock()
    return mod


def _prescore_df_blocked(prompt: str, blocked_message: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            PROMPT_COL: [prompt],
            f"blocked_{PROMPT_COL}": [True],
            f"blocked_message_{PROMPT_COL}": [blocked_message],
            f"replaced_{PROMPT_COL}": [False],
        }
    )


def _prescore_df_ok(prompt: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            PROMPT_COL: [prompt],
            f"blocked_{PROMPT_COL}": [False],
            f"replaced_{PROMPT_COL}": [False],
        }
    )


def _prescore_df_replaced(prompt: str, replacement: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            PROMPT_COL: [prompt],
            f"blocked_{PROMPT_COL}": [False],
            f"replaced_{PROMPT_COL}": [True],
            f"replaced_message_{PROMPT_COL}": [replacement],
        }
    )


def test_workflow_input_to_completion_dict_chat_request_or_message() -> None:
    """NAT LLM Gateway passes ChatRequestOrMessage (no ``forwarded_props``);
    prescore must accept it.
    """
    crm = ChatRequestOrMessage(messages=[NATAPIMessage(role="user", content="hello gateway")])
    params = workflow_input_to_completion_dict(crm)
    assert params["tools"] == []
    assert get_chat_prompt(params) == "hello gateway"
    assert moderation_prompt_from_workflow_input(crm) == get_chat_prompt(params)


def test_moderation_prompt_from_workflow_input_parity_with_completion_dict() -> None:
    run_input = _make_run_input("plan the thing")
    direct = moderation_prompt_from_workflow_input(run_input)
    via_ccp = get_chat_prompt(workflow_input_to_completion_dict(run_input))
    assert direct == via_ccp


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
    # GIVEN prescore ``run_guards`` marks the prompt blocked (single execution path)
    # WHEN pre_invoke runs
    # THEN context.output is set to a DRAgentEventResponse and call_next must not run
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_blocked("bad", "blocked-by-test")
    moderation._executor.run_guards.return_value = (prescore_df, 0.0)

    run_input = _make_run_input("bad")
    ctx = _invocation(run_input)

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        try:
            out = await mw.pre_invoke(ctx)
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is not None
    assert out.output is not None
    assert isinstance(out.output, DRAgentEventResponse)
    assert any(
        isinstance(ev, TextMessageContentEvent) and "blocked-by-test" in ev.delta
        for ev in out.output.events
    )
    assert out.output.datarobot_moderations is None


async def test_pre_invoke_blocked_includes_datarobot_moderations_from_prescore_metrics(
    builder_mock: MagicMock,
) -> None:
    # GIVEN prescore row carries guard metric columns (e.g. token counts) alongside blocked flags
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_blocked("bad", "blocked-by-test")
    prescore_df["token_count_prompt"] = [42]
    moderation._executor.run_guards.return_value = (prescore_df, 0.0)

    run_input = _make_run_input("bad")
    ctx = _invocation(run_input)

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        try:
            out = await mw.pre_invoke(ctx)
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is not None and out.output is not None
    assert out.output.datarobot_moderations == {"token_count_prompt": 42}


async def test_pre_invoke_replaces_last_user_message(builder_mock: MagicMock) -> None:
    # GIVEN prescore ``run_guards`` requests a replacement string
    # WHEN pre_invoke runs
    # THEN the last UserMessage content is updated and context.output stays unset
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_replaced("secret", "[redacted]")
    moderation._executor.run_guards.return_value = (prescore_df, 0.0)

    run_input = _make_run_input("secret")
    ctx = _invocation(run_input)

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        try:
            out = await mw.pre_invoke(ctx)
            st = _moderation_invoke_state_ctx.get()
            assert st is not None
            assert st.data.loc[0, PROMPT_COL] == "[redacted]"
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is not None
    assert out.output is None
    assert run_input.messages[-1].content == "[redacted]"


async def test_pre_invoke_replaces_chat_request_or_message_input_message(
    builder_mock: MagicMock,
) -> None:
    # GIVEN LLM Gateway ``ChatRequestOrMessage`` with only ``input_message`` and replacement
    # WHEN pre_invoke runs
    # THEN input_message is updated so downstream ``call_next`` sees the moderated string
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_replaced("secret", "[redacted]")
    moderation._executor.run_guards.return_value = (prescore_df, 0.0)

    crm = ChatRequestOrMessage(input_message="secret")
    ctx = InvocationContext(
        function_context=_fn_context(),
        original_args=(crm,),
        original_kwargs={},
        modified_args=(crm,),
        modified_kwargs={},
        output=None,
    )

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        try:
            out = await mw.pre_invoke(ctx)
            st = _moderation_invoke_state_ctx.get()
            assert st is not None
            assert st.data.loc[0, PROMPT_COL] == "[redacted]"
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is not None
    assert out.output is None
    assert crm.input_message == "[redacted]"


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


async def test_post_invoke_preserves_aggregate_ag_ui_when_response_text_unchanged(
    builder_mock: MagicMock,
) -> None:
    """Non-streaming /generate aggregates lifecycle events; post_invoke must not drop them."""
    pipeline = _pipeline_mock()
    pipeline.get_postscore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_response.return_value = (EvaluationResult(blocked=False), 0.0)

    predictions = pd.DataFrame({PROMPT_COL: ["p"], RESPONSE_COL: ["hi"]})

    mid = "msg-1"
    aggregate = DRAgentEventResponse(
        events=[
            RunStartedEvent(type=EventType.RUN_STARTED, thread_id="t", run_id="r"),
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=mid,
                role="assistant",
            ),
            TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT, message_id=mid, delta="hi"
            ),
            TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=mid),
            RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id="t", run_id="r"),
        ],
        usage_metrics=default_usage_metrics(),
    )
    run_input = _make_run_input("p")
    ctx = _invocation(run_input, output=aggregate)

    moderated_sidecar = DRAgentEventResponse(
        events=[TextMessageContentEvent(message_id="x", delta="hi")],
        datarobot_moderations={"token_count": 2},
        usage_metrics=default_usage_metrics(),
    )

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
            "datarobot_genai.nat.datarobot_moderation_middleware.chat_completion_to_dragent_event_response",
            return_value=moderated_sidecar,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        prescore = pd.DataFrame({PROMPT_COL: ["p"]})
        _set_moderation_invoke_state(
            data=prescore,
            prescore_df=prescore.copy(),
            latency_so_far=0.0,
        )
        try:
            out = await mw.post_invoke(ctx)
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is not None
    assert ctx.output.events[0].type == EventType.RUN_STARTED
    assert ctx.output.datarobot_moderations == {"token_count": 2}
    validate_sequence(ctx.output.events)


async def test_post_invoke_rewrites_completion_when_postscore_succeeds(
    builder_mock: MagicMock,
) -> None:
    # GIVEN a normal assistant text response and postscore produces a final message
    # WHEN post_invoke runs
    # THEN context.output is replaced with a new DRAgentEventResponse containing that message
    pipeline = _pipeline_mock()
    pipeline.get_postscore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_response.return_value = (
        EvaluationResult(blocked=False, replaced=True, replacement="final-out"),
        0.0,
    )

    predictions = pd.DataFrame({PROMPT_COL: ["p"], RESPONSE_COL: ["model-out"]})

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
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        prescore = pd.DataFrame({PROMPT_COL: ["p"]})
        _set_moderation_invoke_state(
            data=prescore,
            prescore_df=prescore.copy(),
            latency_so_far=0.0,
        )
        try:
            out = await mw.post_invoke(ctx)
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is not None
    assert ctx.output is not None
    text = "".join(ev.delta for ev in ctx.output.events if isinstance(ev, TextMessageContentEvent))
    assert text == "final-out"


async def test_post_invoke_rewrites_nat_chat_response_when_postscore_succeeds(
    builder_mock: MagicMock,
) -> None:
    """NAT ``single_fn`` returns ``ChatResponse``, not ``DRAgentEventResponse``;
    postscore still runs.
    """
    pipeline = _pipeline_mock()
    pipeline.get_postscore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_response.return_value = (
        EvaluationResult(blocked=False, replaced=True, replacement="final-out"),
        0.0,
    )

    predictions = pd.DataFrame({PROMPT_COL: ["p"], RESPONSE_COL: ["model-out"]})

    nat_out = _nat_chat_response_assistant_text("model-out")
    ctx = _invocation(_make_run_input(), output=nat_out)

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.build_predictions_df_from_completion",
            return_value=(predictions, {}),
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        prescore = pd.DataFrame({PROMPT_COL: ["p"]})
        _set_moderation_invoke_state(
            data=prescore,
            prescore_df=prescore.copy(),
            latency_so_far=0.0,
        )
        try:
            out = await mw.post_invoke(ctx)
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is not None
    assert isinstance(ctx.output, ChatResponse)
    assert ctx.output.choices[0].message.content == "final-out"


async def test_post_invoke_rewrites_plain_str_when_postscore_succeeds(
    builder_mock: MagicMock,
) -> None:
    pipeline = _pipeline_mock()
    pipeline.get_postscore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_response.return_value = (
        EvaluationResult(blocked=False, replaced=True, replacement="final-out"),
        0.0,
    )

    predictions = pd.DataFrame({PROMPT_COL: ["p"], RESPONSE_COL: ["model-out"]})

    ctx = _invocation(_make_run_input(), output="model-out")

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.build_predictions_df_from_completion",
            return_value=(predictions, {}),
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        prescore = pd.DataFrame({PROMPT_COL: ["p"]})
        _set_moderation_invoke_state(
            data=prescore,
            prescore_df=prescore.copy(),
            latency_so_far=0.0,
        )
        try:
            out = await mw.post_invoke(ctx)
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is not None
    assert ctx.output == "final-out"


@pytest.mark.parametrize("missing_cell", [None, np.nan, pd.NA])
async def test_post_invoke_uses_none_custom_response_when_postscore_empty(
    builder_mock: MagicMock,
    missing_cell: Any,
) -> None:
    # GIVEN the model response column is null-like (None / NaN / NA) so postscore merge is skipped
    # WHEN post_invoke selects the message for an empty postscore frame
    # THEN NONE_CUSTOM_PY_RESPONSE is used
    pipeline = _pipeline_mock()
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_response.return_value = (
        SimpleNamespace(
            blocked=False, replaced=False, replacement=None, blocked_message=None, metrics={}
        ),
        None,
    )

    predictions = pd.DataFrame({PROMPT_COL: ["p"], RESPONSE_COL: [missing_cell]})

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
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        prescore = pd.DataFrame({PROMPT_COL: ["p"]})
        _set_moderation_invoke_state(
            data=prescore,
            prescore_df=prescore.copy(),
            latency_so_far=0.0,
        )
        try:
            await mw.post_invoke(ctx)
        finally:
            _clear_moderation_invoke_state_if_set()

    text = "".join(ev.delta for ev in ctx.output.events if isinstance(ev, TextMessageContentEvent))
    assert text == NONE_CUSTOM_PY_RESPONSE


@pytest.mark.parametrize("missing_cell", [None, np.nan, pd.NA])
async def test_post_invoke_blocked_empty_postscore_coerces_none_blocked_message_to_empty_str(
    builder_mock: MagicMock,
    missing_cell: Any,
) -> None:
    # GIVEN null-like response cell (empty postscore_df path) and postscore marks blocked with
    # no replacement message
    # WHEN post_invoke builds the completion
    # THEN assistant content is "" (not None) and finish_reason is content_filter
    pipeline = _pipeline_mock()
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_response.return_value = (
        EvaluationResult(blocked=True, blocked_message=None),
        0.0,
    )

    predictions = pd.DataFrame({PROMPT_COL: ["p"], RESPONSE_COL: [missing_cell]})

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
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)
        prescore = pd.DataFrame({PROMPT_COL: ["p"]})
        _set_moderation_invoke_state(
            data=prescore,
            prescore_df=prescore.copy(),
            latency_so_far=0.0,
        )
        try:
            out = await mw.post_invoke(ctx)
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is not None
    assert isinstance(ctx.output, DRAgentEventResponse)
    text = "".join(ev.delta for ev in ctx.output.events if isinstance(ev, TextMessageContentEvent))
    assert text == ""
    assert ctx.output.original_chunk is not None
    assert ctx.output.original_chunk.choices[0].finish_reason == "content_filter"


async def test_function_middleware_invoke_blocked_short_circuits(builder_mock: MagicMock) -> None:
    # GIVEN pre_invoke sets context.output (blocked)
    # WHEN function_middleware_invoke runs
    # THEN call_next is never awaited
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_blocked("x", "stop")
    moderation._executor.run_guards.return_value = (prescore_df, 0.0)

    call_next = AsyncMock()

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
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


async def test_function_middleware_invoke_preserves_prescore_data_across_concurrent_tasks(
    builder_mock: MagicMock,
) -> None:
    """Each asyncio task must keep its own prescore frame across ``await call_next``."""
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)

    def run_guards_side_effect(
        data: pd.DataFrame, guards: Any, stage: Any
    ) -> tuple[pd.DataFrame, float]:
        prompt = str(data.loc[0, PROMPT_COL])
        return _prescore_df_ok(prompt), 0.0

    moderation._executor.run_guards.side_effect = run_guards_side_effect
    seen_in_build: dict[asyncio.Task[Any], str] = {}

    def build_df_side_effect(
        data: pd.DataFrame, pipeline: Any, chat_completion: Any
    ) -> tuple[pd.DataFrame, dict[Any, Any]]:
        task = asyncio.current_task()
        assert task is not None
        seen_in_build[task] = str(data.loc[0, PROMPT_COL])
        prompt = str(data.loc[0, PROMPT_COL])
        predictions = pd.DataFrame({PROMPT_COL: [prompt], RESPONSE_COL: ["x"]})
        return predictions, {}

    async def slow_call_next(*_a: Any, **_k: Any) -> DRAgentEventResponse:
        await asyncio.sleep(0.05)
        return _text_response("x")

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.build_predictions_df_from_completion",
            side_effect=build_df_side_effect,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(model_dir=None), builder_mock)

        async def run_one(prompt: str) -> None:
            await mw.function_middleware_invoke(
                _make_run_input(prompt),
                call_next=slow_call_next,
                context=_fn_context(),
            )

        t_aaa = asyncio.create_task(run_one("aaa-only"))
        await asyncio.sleep(0)
        t_bbb = asyncio.create_task(run_one("bbb-only"))
        await asyncio.gather(t_aaa, t_bbb)

    assert seen_in_build[t_aaa] == "aaa-only"
    assert seen_in_build[t_bbb] == "bbb-only"


async def test_function_middleware_stream_yields_blocked_pre_invoke(
    builder_mock: MagicMock,
) -> None:
    # GIVEN streaming entrypoint and prescore blocks
    # WHEN function_middleware_stream runs
    # THEN a single blocked response is yielded and the worker is never started
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_blocked("x", "no-stream")
    moderation._executor.run_guards.return_value = (prescore_df, 0.0)
    stream_next = MagicMock()

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
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
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_ok("hi")
    moderation._executor.run_guards.return_value = (prescore_df, 0.0)

    async def upstream():
        yield _text_response("delta-one")

    stream_next = MagicMock(return_value=upstream())

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
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


async def test_function_middleware_stream_defers_text_message_end_before_run_finished(
    builder_mock: MagicMock,
) -> None:
    """Deferred TEXT_MESSAGE_END must be emitted before RUN_FINISHED (AG-UI ordering)."""
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_ok("hi")
    moderation._executor.run_guards.return_value = (prescore_df, 0.0)
    mid = "msg-1"
    zero = default_usage_metrics()

    async def upstream():
        yield DRAgentEventResponse(
            events=[RunStartedEvent(thread_id="t1", run_id="r1")],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[TextMessageStartEvent(message_id=mid, role="assistant")],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[TextMessageContentEvent(message_id=mid, delta="hi")],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[TextMessageEndEvent(message_id=mid)],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[RunFinishedEvent(thread_id="t1", run_id="r1")],
            usage_metrics=zero,
        )

    stream_next = MagicMock(return_value=upstream())

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
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

    flat = [ev for resp in chunks for ev in resp.events]
    validate_sequence(flat)
    end_idx = next(i for i, e in enumerate(flat) if e.type == EventType.TEXT_MESSAGE_END)
    finished_idx = next(i for i, e in enumerate(flat) if e.type == EventType.RUN_FINISHED)
    assert end_idx < finished_idx


def test_chat_completion_to_dragent_event_response_keeps_tool_calls_with_text() -> None:
    """Moderation rehydration must not drop tool AG-UI events bundled with a text delta."""
    mid = "msg-1"
    source = [
        TextMessageContentEvent(message_id=mid, delta="call"),
        TextMessageEndEvent(message_id=mid),
        ToolCallStartEvent(
            tool_call_id="call_1",
            tool_call_name="generate_objectid",
            parent_message_id=mid,
        ),
        ToolCallArgsEvent(tool_call_id="call_1", delta='{"x": 1}'),
    ]
    moderated_chunk = ChatCompletionChunk(
        id="chunk-1",
        choices=[
            OpenAIChunkChoice(
                index=0,
                delta=OpenAIChoiceDelta(
                    content="call",
                    tool_calls=[
                        OpenAIChoiceDeltaToolCall(
                            index=0,
                            id="call_1",
                            type="function",
                            function=OpenAIChoiceDeltaToolCallFunction(
                                name="generate_objectid",
                                arguments='{"x": 1}',
                            ),
                        )
                    ],
                ),
                finish_reason=None,
            )
        ],
        created=1700000000,
        model="test-model",
        object="chat.completion.chunk",
    )
    out = chat_completion_to_dragent_event_response(
        moderated_chunk,
        source_ag_ui_events=source,
        stream_tool_index_map={},
    )
    types = [e.type for e in out.events]
    assert EventType.TEXT_MESSAGE_CONTENT in types
    assert EventType.TEXT_MESSAGE_END in types
    assert EventType.TOOL_CALL_START in types
    assert EventType.TOOL_CALL_ARGS in types
    starts = [e for e in out.events if e.type == EventType.TOOL_CALL_START]
    assert starts[0].tool_call_name == "generate_objectid"


def test_chat_completion_to_dragent_event_response_serializes_numpy_moderations() -> None:
    """datarobot_dome may attach numpy scalars; SSE must still serialize via model_dump_json."""
    chunk = ChatCompletionChunk(
        id="chunk-1",
        choices=[
            OpenAIChunkChoice(
                index=0,
                delta=OpenAIChoiceDelta(content="hi"),
                finish_reason=None,
            )
        ],
        created=1700000000,
        model="test-model",
        object="chat.completion.chunk",
    )
    setattr(
        chunk,
        DATAROBOT_MODERATIONS_ATTR,
        {
            "count": np.int64(42),
            "nested": {"x": np.float64(1.5)},
            "ts": pd.Timestamp("2026-01-01T00:00:00Z"),
        },
    )
    out = chat_completion_to_dragent_event_response(chunk)
    assert out.datarobot_moderations is not None
    assert out.datarobot_moderations["count"] == 42
    assert out.datarobot_moderations["nested"]["x"] == 1.5
    assert "2026-01-01" in str(out.datarobot_moderations["ts"])
    out.model_dump_json()
