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
import os
from collections.abc import Iterator
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("datarobot_dome")

from ag_ui.core import EventType
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import StepFinishedEvent
from ag_ui.core import StepStartedEvent
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallStartEvent
from ag_ui.core import UserMessage
from datarobot_dome.api import EvaluationResult
from datarobot_dome.api import _from_dataframe
from datarobot_dome.async_http_client import AsyncHTTPClient
from datarobot_dome.constants import DATAROBOT_MODERATIONS_ATTR
from datarobot_dome.constants import DEFAULT_RESPONSE_COLUMN_NAME
from datarobot_dome.constants import MODERATION_MODEL_NAME
from datarobot_dome.constants import GuardStage
from datarobot_dome.schema.moderation_config import ModerationConfig
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
from datarobot_genai.nat.datarobot_moderation_middleware import dome_chunk_to_dragent_event_response
from datarobot_genai.nat.datarobot_moderation_middleware import load_llm_moderation_pipeline
from datarobot_genai.nat.datarobot_moderation_middleware import (
    moderation_prompt_from_workflow_input,
)
from datarobot_genai.nat.datarobot_moderation_middleware import workflow_input_to_completion_dict


@pytest.fixture(autouse=True)
def _noop_datarobot_moderation_credential_check() -> Iterator[None]:
    """11.2.28+ validates DR credentials when loading pipelines; unit tests use stub endpoints."""
    with patch("datarobot_dome.api._verify_datarobot_credentials"):
        yield


PROMPT_COL = "prompt_col"
RESPONSE_COL = "response_col"
INTEGRATION_MODERATION_MODEL_DIR = Path(__file__).parent / "fixtures" / "moderation_integration"
INTEGRATION_MODERATION_PROMPT_LENGTH_BLOCK_DIR = (
    Path(__file__).parent / "fixtures" / "moderation_prompt_length_block"
)
INTEGRATION_MODERATION_RESPONSE_LENGTH_BLOCK_DIR = (
    Path(__file__).parent / "fixtures" / "moderation_response_token_length_block"
)
INTEGRATION_MODERATION_MODEL_PROMPT_REPLACE_DIR = (
    Path(__file__).parent / "fixtures" / "moderation_model_prompt_replace"
)
INTEGRATION_MODERATION_MODEL_RESPONSE_REPLACE_DIR = (
    Path(__file__).parent / "fixtures" / "moderation_model_response_replace"
)


def _moderation_config_from_fixture_dir(model_dir: Path) -> ModerationConfig:
    path = model_dir / "moderation_config.yaml"
    return ModerationConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# ``moderation_prompt_length_block/moderation_config.yaml`` blocks when prompt token count
# is greater than 32 (81 tokens for this string).
_LONG_PROMPT_FOR_TOKEN_BLOCK_GUARD = " ".join(["moderation"] * 80)
_PROMPT_TOKEN_CAP_BLOCK_MESSAGE = "Prompt exceeds the configured maximum length (token count)."

# ``moderation_response_token_length_block/moderation_config.yaml`` blocks when response
# token count is greater than 32 (80 tokens for this string).
_SHORT_USER_PROMPT_FOR_RESPONSE_BLOCK = "hi"
_LONG_RESPONSE_FOR_TOKEN_BLOCK_GUARD = " ".join(["reply"] * 80)
_RESPONSE_TOKEN_CAP_BLOCK_MESSAGE = "Response exceeds the configured maximum length (token count)."

# Model-guard replace fixtures use deployment_id below; tests stub ``Deployment.get`` and
# ``AsyncHTTPClient.predict`` so ``GuardFactory`` / ``run_model_guard`` / ``run_guards`` run
# without contacting a real prediction server.
_MODEL_GUARD_DEPLOYMENT_ID = "507f191e810c19729de860ea"
_MODEL_PROMPT_REPLACEMENT = "[model-moderated-prompt]"
_MODEL_RESPONSE_REPLACEMENT = "[model-moderated-response]"


def _assistant_text_joined_from_dragent_response(response: DRAgentEventResponse) -> str:
    return "".join(ev.delta for ev in response.events if isinstance(ev, TextMessageContentEvent))


def _model_guard_deployment_stub() -> MagicMock:
    dep = MagicMock()
    dep.id = _MODEL_GUARD_DEPLOYMENT_ID
    dep.default_prediction_server = {
        "url": "https://test-moderation.invalid",
        "datarobot-key": "test-key",
    }
    return dep


def _model_guard_predict_stub(
    replacement_text: str,
    *,
    predict_inputs: list[pd.DataFrame],
) -> Any:
    """Return an ``AsyncHTTPClient.predict`` replacement that records inputs and fires replace."""

    async def fake_predict(_self: Any, deployment: Any, input_df: pd.DataFrame) -> pd.DataFrame:
        predict_inputs.append(input_df.copy())
        return pd.DataFrame(
            {"trigger_pred": [1.0], "replacement_col": [replacement_text]},
            index=input_df.index,
        )

    return fake_predict


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
    mod.evaluate_prompt_async = AsyncMock()
    mod.evaluate_response_async = AsyncMock()
    mod.stream_response_async = _passthrough_stream_response_async
    return mod


def _set_evaluate_prompt_async_return(
    moderation: MagicMock,
    prescore_df: pd.DataFrame,
    *,
    latency: float = 0.0,
) -> None:
    prompt_eval = _from_dataframe(prescore_df, PROMPT_COL)
    moderation.evaluate_prompt_async.return_value = (prompt_eval, latency, prescore_df)


async def _passthrough_stream_response_async(
    completion: Any,
    **kwargs: Any,
) -> Any:
    """Mirror ``ModerationPipeline.stream_response_async`` peek-ahead chunk ordering."""
    aiter = completion.__aiter__()
    try:
        current = await aiter.__anext__()
    except StopAsyncIteration:
        return
    while True:
        try:
            peek = await aiter.__anext__()
        except StopAsyncIteration:
            yield current
            return
        yield current
        current = peek


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
    assert moderation_prompt_from_workflow_input(crm) == "hello gateway"


def test_moderation_prompt_from_workflow_input_run_agent_input() -> None:
    run_input = _make_run_input("plan the thing")
    assert moderation_prompt_from_workflow_input(run_input) == "plan the thing"


def test_moderation_prompt_from_workflow_input_input_message_only() -> None:
    crm = ChatRequestOrMessage(input_message="gateway string only")
    assert moderation_prompt_from_workflow_input(crm) == "gateway string only"


def test_load_llm_moderation_pipeline_from_config_moderation_field() -> None:
    moderation = ModerationConfig.model_validate(
        {
            "guards": [
                {
                    "name": "Prompt Tokens",
                    "description": (
                        "Track the number of tokens associated with the input to the LLM, and/or "
                        "retrieved text from the vector database."
                    ),
                    "ootb_type": "token_count",
                    "stage": "prompt",
                    "type": "ootb",
                }
            ],
            "timeout_sec": 60,
        }
    )
    cfg = DataRobotModerationConfig(moderation=moderation)
    with patch.dict(
        os.environ,
        {
            "DATAROBOT_API_TOKEN": "test-token",
            "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
        },
    ):
        pipeline = load_llm_moderation_pipeline(cfg)
    assert pipeline is not None
    assert pipeline._pipeline.get_prescore_guards()


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
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
    assert mw.enabled is False


async def test_pre_invoke_no_prescore_guards_prescore_df_has_blocked_and_replaced_columns(
    builder_mock: MagicMock,
) -> None:
    # GIVEN the pipeline has no prescore guards
    # WHEN pre_invoke runs
    # THEN prescore state is stored for downstream streaming/postscore
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = []
    moderation = _moderation_mock(pipeline)
    prescore_df = pd.DataFrame({PROMPT_COL: ["hello"]})
    moderation.evaluate_prompt_async.return_value = (
        EvaluationResult(blocked=False),
        0.0,
        prescore_df,
    )

    run_input = _make_run_input("hello")
    ctx = _invocation(run_input)

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        try:
            await mw.pre_invoke(ctx)
            st = _moderation_invoke_state_ctx.get()
            assert st is not None
            assert st.prescore_df.equals(prescore_df)
        finally:
            _clear_moderation_invoke_state_if_set()

    moderation.evaluate_prompt_async.assert_awaited_once_with("hello")


async def test_pre_invoke_blocks_and_sets_output(builder_mock: MagicMock) -> None:
    # GIVEN prescore marks the prompt blocked
    # WHEN pre_invoke runs
    # THEN context.output is set to a DRAgentEventResponse and call_next must not run
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_blocked("bad", "blocked-by-test")
    _set_evaluate_prompt_async_return(moderation, prescore_df)

    run_input = _make_run_input("bad")
    ctx = _invocation(run_input)

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
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
    _set_evaluate_prompt_async_return(moderation, prescore_df)

    run_input = _make_run_input("bad")
    ctx = _invocation(run_input)

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        try:
            out = await mw.pre_invoke(ctx)
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is not None and out.output is not None
    assert out.output.datarobot_moderations == {"token_count_prompt": 42}


async def test_pre_invoke_replaces_last_user_message(builder_mock: MagicMock) -> None:
    # GIVEN prescore requests a replacement string
    # WHEN pre_invoke runs
    # THEN the last UserMessage content is updated and context.output stays unset
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_replaced("secret", "[redacted]")
    _set_evaluate_prompt_async_return(moderation, prescore_df)

    run_input = _make_run_input("secret")
    ctx = _invocation(run_input)

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        try:
            out = await mw.pre_invoke(ctx)
            st = _moderation_invoke_state_ctx.get()
            assert st is not None
            assert st.input_df.loc[0, PROMPT_COL] == "[redacted]"
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
    _set_evaluate_prompt_async_return(moderation, prescore_df)

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
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        try:
            out = await mw.pre_invoke(ctx)
            st = _moderation_invoke_state_ctx.get()
            assert st is not None
            assert st.input_df.loc[0, PROMPT_COL] == "[redacted]"
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is not None
    assert out.output is None
    assert crm.input_message == "[redacted]"


async def test_pre_invoke_replacement_apply_failure_clears_prescore_replaced_flags(
    builder_mock: MagicMock,
) -> None:
    # GIVEN prescore requests a replacement but it cannot be applied to the workflow object
    # WHEN pre_invoke runs
    # THEN prescore_df flags are cleared so metadata matches the original prompt sent to the LLM,
    # and pre_invoke returns None (no InvocationContext rewrite).
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_replaced("secret", "[redacted]")
    _set_evaluate_prompt_async_return(moderation, prescore_df)

    run_input = _make_run_input("secret")
    ctx = _invocation(run_input)

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware._apply_moderated_prompt_text_to_workflow_input",
            return_value=False,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        try:
            out = await mw.pre_invoke(ctx)
            st = _moderation_invoke_state_ctx.get()
            assert st is not None
            assert st.input_df.loc[0, PROMPT_COL] == "secret"
            assert bool(st.prescore_df.loc[0, f"replaced_{PROMPT_COL}"]) is False
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is None
    assert run_input.messages[-1].content == "secret"


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
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)

    out = await mw.post_invoke(ctx)
    assert out is None
    assert ctx.output is tool_first


@pytest.mark.parametrize(
    "events",
    [
        (
            TextMessageChunkEvent(
                type=EventType.TEXT_MESSAGE_CHUNK,
                message_id="msg-1",
                delta="",
            ),
            TextMessageChunkEvent(
                type=EventType.TEXT_MESSAGE_CHUNK,
                message_id="msg-1",
                delta="",
            ),
        ),
        (
            TextMessageChunkEvent(
                type=EventType.TEXT_MESSAGE_CHUNK,
                message_id="msg-1",
                delta="   ",
            ),
        ),
        (
            TextMessageChunkEvent(
                type=EventType.TEXT_MESSAGE_CHUNK,
                message_id="msg-1",
                delta="\n\t",
            ),
        ),
    ],
)
async def test_post_invoke_skips_dr_agent_when_joined_text_blank(
    builder_mock: MagicMock,
    events: tuple[Any, ...],
) -> None:
    """AG-UI text events whose deltas join to empty/whitespace must not run postscore."""
    pipeline = _pipeline_mock()
    pipeline.get_postscore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    response = DRAgentEventResponse(
        events=list(events),
        usage_metrics=default_usage_metrics(),
    )
    ctx = _invocation(_make_run_input(), output=response)

    with patch(
        "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
        return_value=moderation,
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        prescore = pd.DataFrame({PROMPT_COL: ["p"]})
        _set_moderation_invoke_state(
            input_df=prescore,
            prescore_df=prescore.copy(),
            latency_so_far=0.0,
        )
        try:
            out = await mw.post_invoke(ctx)
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is None
    assert ctx.output is response
    moderation.evaluate_response_async.assert_not_awaited()


async def test_post_invoke_preserves_aggregate_ag_ui_when_response_text_unchanged(
    builder_mock: MagicMock,
) -> None:
    """Non-streaming /generate aggregates lifecycle events; post_invoke must not drop them."""
    pipeline = _pipeline_mock()
    pipeline.get_postscore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_response_async.return_value = (
        EvaluationResult(blocked=False, metrics={"token_count": 2}),
        0.0,
        pd.DataFrame(),
    )

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

    with patch(
        "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
        return_value=moderation,
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        prescore = pd.DataFrame({PROMPT_COL: ["p"]})
        _set_moderation_invoke_state(
            input_df=prescore,
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
    moderation.evaluate_response_async.return_value = (
        EvaluationResult(blocked=False, replaced=True, replacement="final-out"),
        0.0,
        pd.DataFrame(),
    )

    run_input = _make_run_input()
    ctx = _invocation(run_input, output=_text_response("model-out"))

    with patch(
        "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
        return_value=moderation,
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        prescore = pd.DataFrame({PROMPT_COL: ["p"]})
        _set_moderation_invoke_state(
            input_df=prescore,
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
    moderation.evaluate_response_async.return_value = (
        EvaluationResult(blocked=False, replaced=True, replacement="final-out"),
        0.0,
        pd.DataFrame(),
    )

    nat_out = _nat_chat_response_assistant_text("model-out")
    ctx = _invocation(_make_run_input(), output=nat_out)

    with patch(
        "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
        return_value=moderation,
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        prescore = pd.DataFrame({PROMPT_COL: ["p"]})
        _set_moderation_invoke_state(
            input_df=prescore,
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
    moderation.evaluate_response_async.return_value = (
        EvaluationResult(blocked=False, replaced=True, replacement="final-out"),
        0.0,
        pd.DataFrame(),
    )

    ctx = _invocation(_make_run_input(), output="model-out")

    with patch(
        "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
        return_value=moderation,
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        prescore = pd.DataFrame({PROMPT_COL: ["p"]})
        _set_moderation_invoke_state(
            input_df=prescore,
            prescore_df=prescore.copy(),
            latency_so_far=0.0,
        )
        try:
            out = await mw.post_invoke(ctx)
        finally:
            _clear_moderation_invoke_state_if_set()

    assert out is not None
    assert ctx.output == "final-out"


async def test_post_invoke_blocked_empty_postscore_coerces_none_blocked_message_to_empty_str(
    builder_mock: MagicMock,
) -> None:
    # GIVEN postscore marks blocked with no replacement message
    # WHEN post_invoke builds the completion
    # THEN assistant content is "" (not None) and finish_reason is content_filter
    pipeline = _pipeline_mock()
    moderation = _moderation_mock(pipeline)
    moderation.evaluate_response_async.return_value = (
        EvaluationResult(blocked=True, blocked_message=None),
        0.0,
        pd.DataFrame(),
    )

    ctx = _invocation(_make_run_input(), output=_text_response("ignored"))

    with patch(
        "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
        return_value=moderation,
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        prescore = pd.DataFrame({PROMPT_COL: ["p"]})
        _set_moderation_invoke_state(
            input_df=prescore,
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
    _set_evaluate_prompt_async_return(moderation, prescore_df)

    call_next = AsyncMock()

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
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
    """Each asyncio task must keep its own prescore frame across ``await call_next``.

    Postscore reads task-local state via ``evaluate_response_async``; the prompt passed there
    must match the prescore row for that task, not a sibling task.
    """
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    seen_in_evaluate_response: dict[asyncio.Task[Any], str] = {}

    async def evaluate_prompt_async_side_effect(prompt: str) -> tuple[Any, float, pd.DataFrame]:
        prescore_df = _prescore_df_ok(prompt)
        return _from_dataframe(prescore_df, PROMPT_COL), 0.0, prescore_df

    async def evaluate_response_async_side_effect(
        _response_text: str, *, prompt: str | None = None, **_: Any
    ) -> tuple[EvaluationResult, float, pd.DataFrame]:
        task = asyncio.current_task()
        assert task is not None
        seen_in_evaluate_response[task] = prompt or ""
        return EvaluationResult(blocked=False), 0.0, pd.DataFrame()

    moderation.evaluate_prompt_async.side_effect = evaluate_prompt_async_side_effect
    moderation.evaluate_response_async.side_effect = evaluate_response_async_side_effect

    async def slow_call_next(*_a: Any, **_k: Any) -> DRAgentEventResponse:
        await asyncio.sleep(0.05)
        return _text_response("x")

    with patch(
        "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
        return_value=moderation,
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)

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

    assert seen_in_evaluate_response[t_aaa] == "aaa-only"
    assert seen_in_evaluate_response[t_bbb] == "bbb-only"


async def test_function_middleware_invoke_integration_executes_real_moderations(
    builder_mock: MagicMock,
) -> None:
    # GIVEN a real moderation config with token-count, cost, and ROUGE-1 guards (ROUGE-1 needs
    # citation columns; this path only asserts metrics that are always emitted here)
    model_dir = INTEGRATION_MODERATION_MODEL_DIR
    result: DRAgentEventResponse
    with patch.dict(
        os.environ,
        {
            "DATAROBOT_API_TOKEN": "test-token",
            "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
        },
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True

        async def call_next(*_a: Any, **_k: Any) -> DRAgentEventResponse:
            return _text_response("This is a test response.")

        # WHEN middleware invoke runs end-to-end without patched moderation internals
        result = await mw.function_middleware_invoke(
            _make_run_input("Count moderation tokens for this prompt."),
            call_next=call_next,
            context=_fn_context(),
        )

    # THEN real moderation metadata includes prompt/response token counts and priced cost
    # (fixture: $1/1k prompt tokens, $2/1k completion tokens → 7/1000 + 12/1000)
    assert isinstance(result, DRAgentEventResponse)
    assert result.datarobot_moderations is not None
    mods = result.datarobot_moderations
    assert mods["Prompts_token_count"] == 7
    assert mods["Responses_token_count"] == 6
    assert mods["cost"] == pytest.approx(0.019)
    assert mods["prompt_token_count_from_usage"] == mods["Prompts_token_count"]
    assert mods["response_token_count_from_usage"] == mods["Responses_token_count"]


async def test_function_middleware_invoke_integration_nat_chat_input_chat_response_real_moderations(
    builder_mock: MagicMock,
) -> None:
    # GIVEN NAT ``ChatRequestOrMessage`` (not AG-UI ``RunAgentInput``) and ``call_next`` returns
    # ``ChatResponse`` (single_fn / LLM Gateway style), with real moderation loaded from disk
    model_dir = INTEGRATION_MODERATION_MODEL_DIR
    crm = ChatRequestOrMessage(
        messages=[
            NATAPIMessage(role="user", content="Count moderation tokens for this prompt."),
        ],
    )
    result: ChatResponse
    with patch.dict(
        os.environ,
        {
            "DATAROBOT_API_TOKEN": "test-token",
            "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
        },
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True

        async def call_next(*_a: Any, **_k: Any) -> ChatResponse:
            return _nat_chat_response_assistant_text("This is a test response.")

        result = await mw.function_middleware_invoke(
            crm,
            call_next=call_next,
            context=_fn_context(),
        )

    assert isinstance(result, ChatResponse)
    assert result.choices[0].message.content == "This is a test response."
    assert result.usage == NATChatUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    assert result.datarobot_moderations is not None
    mods = result.datarobot_moderations
    assert mods["Prompts_token_count"] == 7
    assert mods["Responses_token_count"] == 6
    assert mods["cost"] == pytest.approx(0.019)
    assert mods["prompt_token_count_from_usage"] == mods["Prompts_token_count"]
    assert mods["response_token_count_from_usage"] == mods["Responses_token_count"]


async def test_function_middleware_stream_integration_executes_real_moderations(
    builder_mock: MagicMock,
) -> None:
    # GIVEN a real moderation config with token-count, cost, and ROUGE-1 guards for streaming
    model_dir = INTEGRATION_MODERATION_MODEL_DIR

    async def upstream() -> Any:
        yield _text_response("This is a test response.")

    stream_next = MagicMock(return_value=upstream())
    chunks: list[DRAgentEventResponse]
    with patch.dict(
        os.environ,
        {
            "DATAROBOT_API_TOKEN": "test-token",
            "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
        },
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True

        # WHEN middleware stream runs end-to-end without patched moderation internals
        chunks = [
            item
            async for item in mw.function_middleware_stream(
                _make_run_input("Count moderation tokens for this prompt."),
                call_next=stream_next,
                context=_fn_context(),
            )
        ]

    # THEN streamed output includes real moderation metadata with prompt/response tokens and cost
    assert chunks
    moderation_payloads = [c.datarobot_moderations for c in chunks if c.datarobot_moderations]
    assert moderation_payloads
    all_keys = {k for payload in moderation_payloads for k in payload}
    assert "Prompts_token_count" in all_keys
    assert "Responses_token_count" in all_keys
    assert "cost" in all_keys
    final_mods = moderation_payloads[-1]
    assert final_mods["Prompts_token_count"] == 7
    assert final_mods["Responses_token_count"] == 6
    assert final_mods["cost"] == pytest.approx(0.019)


async def test_function_middleware_invoke_prompt_token_limit_blocks(
    builder_mock: MagicMock,
) -> None:
    model_dir = INTEGRATION_MODERATION_PROMPT_LENGTH_BLOCK_DIR
    call_next = AsyncMock()
    with patch.dict(
        os.environ,
        {
            "DATAROBOT_API_TOKEN": "test-token",
            "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
        },
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True
        result = await mw.function_middleware_invoke(
            _make_run_input(_LONG_PROMPT_FOR_TOKEN_BLOCK_GUARD),
            call_next=call_next,
            context=_fn_context(),
        )

    call_next.assert_not_awaited()
    assert isinstance(result, DRAgentEventResponse)
    assert result.model == MODERATION_MODEL_NAME
    assert _assistant_text_joined_from_dragent_response(result) == _PROMPT_TOKEN_CAP_BLOCK_MESSAGE
    assert result.datarobot_moderations is not None
    mods = result.datarobot_moderations
    assert mods["Prompts_token_count"] == 81
    assert any(k.endswith("_blocked_promptText") and mods[k] is True for k in mods), (
        f"expected a blocked_promptText flag in {mods!r}"
    )


async def test_function_middleware_invoke_nat_chat_input_chat_response_prompt_token_limit_blocks(
    builder_mock: MagicMock,
) -> None:
    model_dir = INTEGRATION_MODERATION_PROMPT_LENGTH_BLOCK_DIR
    crm = ChatRequestOrMessage(
        messages=[
            NATAPIMessage(role="user", content=_LONG_PROMPT_FOR_TOKEN_BLOCK_GUARD),
        ],
    )
    call_next = AsyncMock()
    with patch.dict(
        os.environ,
        {
            "DATAROBOT_API_TOKEN": "test-token",
            "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
        },
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True
        result = await mw.function_middleware_invoke(
            crm,
            call_next=call_next,
            context=_fn_context(),
        )

    call_next.assert_not_awaited()
    assert isinstance(result, DRAgentEventResponse)
    assert _assistant_text_joined_from_dragent_response(result) == _PROMPT_TOKEN_CAP_BLOCK_MESSAGE
    assert result.datarobot_moderations is not None
    mods = result.datarobot_moderations
    assert mods["Prompts_token_count"] == 81
    assert any(k.endswith("_blocked_promptText") and mods[k] is True for k in mods), (
        f"expected a blocked_promptText flag in {mods!r}"
    )


async def test_function_middleware_stream_prompt_token_limit_blocks(
    builder_mock: MagicMock,
) -> None:
    model_dir = INTEGRATION_MODERATION_PROMPT_LENGTH_BLOCK_DIR

    async def upstream() -> Any:
        yield _text_response("This path must not run.")

    stream_next = MagicMock(return_value=upstream())
    with patch.dict(
        os.environ,
        {
            "DATAROBOT_API_TOKEN": "test-token",
            "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
        },
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True
        chunks = [
            item
            async for item in mw.function_middleware_stream(
                _make_run_input(_LONG_PROMPT_FOR_TOKEN_BLOCK_GUARD),
                call_next=stream_next,
                context=_fn_context(),
            )
        ]

    stream_next.assert_not_called()
    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, DRAgentEventResponse)
    assert _assistant_text_joined_from_dragent_response(chunk) == _PROMPT_TOKEN_CAP_BLOCK_MESSAGE
    assert chunk.datarobot_moderations is not None
    mods = chunk.datarobot_moderations
    assert mods["Prompts_token_count"] == 81
    assert any(k.endswith("_blocked_promptText") and mods[k] is True for k in mods), (
        f"expected a blocked_promptText flag in {mods!r}"
    )


async def test_function_middleware_invoke_response_token_limit_blocks(
    builder_mock: MagicMock,
) -> None:
    model_dir = INTEGRATION_MODERATION_RESPONSE_LENGTH_BLOCK_DIR
    call_next = AsyncMock(return_value=_text_response(_LONG_RESPONSE_FOR_TOKEN_BLOCK_GUARD))
    with patch.dict(
        os.environ,
        {
            "DATAROBOT_API_TOKEN": "test-token",
            "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
        },
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True
        result = await mw.function_middleware_invoke(
            _make_run_input(_SHORT_USER_PROMPT_FOR_RESPONSE_BLOCK),
            call_next=call_next,
            context=_fn_context(),
        )

    call_next.assert_awaited_once()
    assert isinstance(result, DRAgentEventResponse)
    assert result.model == MODERATION_MODEL_NAME
    assert _assistant_text_joined_from_dragent_response(result) == _RESPONSE_TOKEN_CAP_BLOCK_MESSAGE
    assert result.original_chunk is not None
    assert result.original_chunk.choices[0].finish_reason == "content_filter"
    assert result.datarobot_moderations is not None
    mods = result.datarobot_moderations
    assert mods["Responses_token_count"] == 80
    blocked_suffix = f"_blocked_{DEFAULT_RESPONSE_COLUMN_NAME}"
    assert any(k.endswith(blocked_suffix) and mods[k] is True for k in mods), (
        f"expected a {blocked_suffix!r} flag in {mods!r}"
    )


async def test_function_middleware_invoke_nat_chat_input_chat_response_response_token_limit_blocks(
    builder_mock: MagicMock,
) -> None:
    model_dir = INTEGRATION_MODERATION_RESPONSE_LENGTH_BLOCK_DIR
    crm = ChatRequestOrMessage(
        messages=[
            NATAPIMessage(role="user", content=_SHORT_USER_PROMPT_FOR_RESPONSE_BLOCK),
        ],
    )
    call_next = AsyncMock(
        return_value=_nat_chat_response_assistant_text(_LONG_RESPONSE_FOR_TOKEN_BLOCK_GUARD)
    )
    with patch.dict(
        os.environ,
        {
            "DATAROBOT_API_TOKEN": "test-token",
            "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
        },
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True
        result = await mw.function_middleware_invoke(
            crm,
            call_next=call_next,
            context=_fn_context(),
        )

    call_next.assert_awaited_once()
    assert isinstance(result, ChatResponse)
    assert result.choices[0].message.content == _RESPONSE_TOKEN_CAP_BLOCK_MESSAGE
    assert result.choices[0].finish_reason == "content_filter"
    assert result.model == MODERATION_MODEL_NAME
    assert result.datarobot_moderations is not None
    mods = result.datarobot_moderations
    assert mods["Responses_token_count"] == 80
    blocked_suffix = f"_blocked_{DEFAULT_RESPONSE_COLUMN_NAME}"
    assert any(k.endswith(blocked_suffix) and mods[k] is True for k in mods), (
        f"expected a {blocked_suffix!r} flag in {mods!r}"
    )


async def test_function_middleware_stream_response_token_limit_blocks(
    builder_mock: MagicMock,
) -> None:
    model_dir = INTEGRATION_MODERATION_RESPONSE_LENGTH_BLOCK_DIR

    async def upstream() -> Any:
        yield _text_response(_LONG_RESPONSE_FOR_TOKEN_BLOCK_GUARD)

    stream_next = MagicMock(return_value=upstream())
    with patch.dict(
        os.environ,
        {
            "DATAROBOT_API_TOKEN": "test-token",
            "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
        },
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True
        chunks = [
            item
            async for item in mw.function_middleware_stream(
                _make_run_input(_SHORT_USER_PROMPT_FOR_RESPONSE_BLOCK),
                call_next=stream_next,
                context=_fn_context(),
            )
        ]

    stream_next.assert_called_once()
    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, DRAgentEventResponse)
    assert _assistant_text_joined_from_dragent_response(chunk) == _RESPONSE_TOKEN_CAP_BLOCK_MESSAGE
    assert chunk.datarobot_moderations is not None
    mods = chunk.datarobot_moderations
    assert mods["Responses_token_count"] == 80
    blocked_suffix = f"_blocked_{DEFAULT_RESPONSE_COLUMN_NAME}"
    assert any(k.endswith(blocked_suffix) and mods[k] is True for k in mods), (
        f"expected a {blocked_suffix!r} flag in {mods!r}"
    )


async def test_function_middleware_invoke_prompt_replace(
    builder_mock: MagicMock,
) -> None:
    pytest.importorskip("datarobot")
    model_dir = INTEGRATION_MODERATION_MODEL_PROMPT_REPLACE_DIR
    run_input = _make_run_input("user-secret-before-moderation")
    call_next = AsyncMock(return_value=_text_response("model-out"))
    predict_inputs: list[pd.DataFrame] = []
    fake_predict = _model_guard_predict_stub(
        _MODEL_PROMPT_REPLACEMENT,
        predict_inputs=predict_inputs,
    )

    with (
        patch.dict(
            os.environ,
            {
                "DATAROBOT_API_TOKEN": "test-token",
                "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
            },
        ),
        patch("datarobot.Deployment.get", return_value=_model_guard_deployment_stub()),
        patch.object(AsyncHTTPClient, "predict", fake_predict),
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True
        result = await mw.function_middleware_invoke(
            run_input,
            call_next=call_next,
            context=_fn_context(),
        )

    call_next.assert_awaited_once()
    assert predict_inputs, "model guard should call AsyncHTTPClient.predict"
    assert run_input.messages[-1].content == _MODEL_PROMPT_REPLACEMENT
    assert isinstance(result, DRAgentEventResponse)
    assert _assistant_text_joined_from_dragent_response(result) == "model-out"


async def test_function_middleware_invoke_nat_chat_input_chat_response_prompt_replace(
    builder_mock: MagicMock,
) -> None:
    pytest.importorskip("datarobot")
    model_dir = INTEGRATION_MODERATION_MODEL_PROMPT_REPLACE_DIR
    crm = ChatRequestOrMessage(
        messages=[NATAPIMessage(role="user", content="user-secret-before-moderation")],
    )
    call_next = AsyncMock(return_value=_nat_chat_response_assistant_text("model-out"))
    predict_inputs: list[pd.DataFrame] = []
    fake_predict = _model_guard_predict_stub(
        _MODEL_PROMPT_REPLACEMENT,
        predict_inputs=predict_inputs,
    )

    with (
        patch.dict(
            os.environ,
            {
                "DATAROBOT_API_TOKEN": "test-token",
                "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
            },
        ),
        patch("datarobot.Deployment.get", return_value=_model_guard_deployment_stub()),
        patch.object(AsyncHTTPClient, "predict", fake_predict),
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True
        result = await mw.function_middleware_invoke(
            crm,
            call_next=call_next,
            context=_fn_context(),
        )

    call_next.assert_awaited_once()
    assert predict_inputs
    assert crm.messages[-1].content == _MODEL_PROMPT_REPLACEMENT
    assert isinstance(result, ChatResponse)
    assert result.choices[0].message.content == "model-out"


async def test_function_middleware_stream_prompt_replace(
    builder_mock: MagicMock,
) -> None:
    pytest.importorskip("datarobot")
    model_dir = INTEGRATION_MODERATION_MODEL_PROMPT_REPLACE_DIR
    run_input = _make_run_input("user-secret-before-moderation")

    async def upstream() -> Any:
        yield _text_response("streamed-model-out")

    stream_next = MagicMock(return_value=upstream())
    predict_inputs: list[pd.DataFrame] = []
    fake_predict = _model_guard_predict_stub(
        _MODEL_PROMPT_REPLACEMENT,
        predict_inputs=predict_inputs,
    )

    with (
        patch.dict(
            os.environ,
            {
                "DATAROBOT_API_TOKEN": "test-token",
                "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
            },
        ),
        patch("datarobot.Deployment.get", return_value=_model_guard_deployment_stub()),
        patch.object(AsyncHTTPClient, "predict", fake_predict),
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True
        chunks = [
            item
            async for item in mw.function_middleware_stream(
                run_input,
                call_next=stream_next,
                context=_fn_context(),
            )
        ]

    stream_next.assert_called_once()
    assert predict_inputs
    assert run_input.messages[-1].content == _MODEL_PROMPT_REPLACEMENT
    assert len(chunks) == 1
    assert _assistant_text_joined_from_dragent_response(chunks[0]) == "streamed-model-out"


async def test_function_middleware_invoke_response_replace(
    builder_mock: MagicMock,
) -> None:
    pytest.importorskip("datarobot")
    model_dir = INTEGRATION_MODERATION_MODEL_RESPONSE_REPLACE_DIR
    call_next = AsyncMock(return_value=_text_response("upstream-assistant-text"))
    predict_inputs: list[pd.DataFrame] = []
    fake_predict = _model_guard_predict_stub(
        _MODEL_RESPONSE_REPLACEMENT,
        predict_inputs=predict_inputs,
    )

    with (
        patch.dict(
            os.environ,
            {
                "DATAROBOT_API_TOKEN": "test-token",
                "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
            },
        ),
        patch("datarobot.Deployment.get", return_value=_model_guard_deployment_stub()),
        patch.object(AsyncHTTPClient, "predict", fake_predict),
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True
        result = await mw.function_middleware_invoke(
            _make_run_input("hi"),
            call_next=call_next,
            context=_fn_context(),
        )

    call_next.assert_awaited_once()
    assert predict_inputs
    assert isinstance(result, DRAgentEventResponse)
    assert _assistant_text_joined_from_dragent_response(result) == _MODEL_RESPONSE_REPLACEMENT
    assert result.model == MODERATION_MODEL_NAME
    assert result.original_chunk is not None
    assert result.original_chunk.choices[0].finish_reason == "content_filter"


async def test_function_middleware_invoke_nat_chat_input_chat_response_response_replace(
    builder_mock: MagicMock,
) -> None:
    pytest.importorskip("datarobot")
    model_dir = INTEGRATION_MODERATION_MODEL_RESPONSE_REPLACE_DIR
    crm = ChatRequestOrMessage(messages=[NATAPIMessage(role="user", content="hi")])
    call_next = AsyncMock(return_value=_nat_chat_response_assistant_text("upstream-assistant"))
    predict_inputs: list[pd.DataFrame] = []
    fake_predict = _model_guard_predict_stub(
        _MODEL_RESPONSE_REPLACEMENT,
        predict_inputs=predict_inputs,
    )

    with (
        patch.dict(
            os.environ,
            {
                "DATAROBOT_API_TOKEN": "test-token",
                "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
            },
        ),
        patch("datarobot.Deployment.get", return_value=_model_guard_deployment_stub()),
        patch.object(AsyncHTTPClient, "predict", fake_predict),
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True
        result = await mw.function_middleware_invoke(
            crm,
            call_next=call_next,
            context=_fn_context(),
        )

    call_next.assert_awaited_once()
    assert predict_inputs
    assert isinstance(result, ChatResponse)
    assert result.choices[0].message.content == _MODEL_RESPONSE_REPLACEMENT
    assert result.choices[0].finish_reason == "content_filter"
    assert result.model == MODERATION_MODEL_NAME


async def test_function_middleware_stream_response_replace(
    builder_mock: MagicMock,
) -> None:
    pytest.importorskip("datarobot")
    model_dir = INTEGRATION_MODERATION_MODEL_RESPONSE_REPLACE_DIR

    async def upstream() -> Any:
        yield _text_response("upstream-stream-chunk")

    stream_next = MagicMock(return_value=upstream())
    predict_inputs: list[pd.DataFrame] = []
    fake_predict = _model_guard_predict_stub(
        _MODEL_RESPONSE_REPLACEMENT,
        predict_inputs=predict_inputs,
    )

    with (
        patch.dict(
            os.environ,
            {
                "DATAROBOT_API_TOKEN": "test-token",
                "DATAROBOT_ENDPOINT": "https://example.test/api/v2",
            },
        ),
        patch("datarobot.Deployment.get", return_value=_model_guard_deployment_stub()),
        patch.object(AsyncHTTPClient, "predict", fake_predict),
    ):
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
        assert mw.enabled is True
        chunks = [
            item
            async for item in mw.function_middleware_stream(
                _make_run_input("hi"),
                call_next=stream_next,
                context=_fn_context(),
            )
        ]

    stream_next.assert_called_once()
    assert predict_inputs
    assert len(chunks) == 1
    assert _assistant_text_joined_from_dragent_response(chunks[0]) == _MODEL_RESPONSE_REPLACEMENT


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
    _set_evaluate_prompt_async_return(moderation, prescore_df)
    stream_next = MagicMock()

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
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
    # GIVEN one streamed text chunk from upstream and stream_response_async echoes chunks
    # WHEN function_middleware_stream runs
    # THEN one moderated DRAgentEventResponse is yielded
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_ok("hi")
    _set_evaluate_prompt_async_return(moderation, prescore_df)

    async def upstream():
        yield _text_response("delta-one")

    stream_next = MagicMock(return_value=upstream())

    with (
        patch(
            "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
            return_value=moderation,
        ),
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
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
    # THEN upstream chunks are yielded and stream_response_async is never used
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
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
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


async def test_function_middleware_stream_preserves_message_id_per_text_chunk(
    builder_mock: MagicMock,
) -> None:
    """Each moderated chunk must use the source response for that chunk, not the peeked next one."""
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_ok("hi")
    _set_evaluate_prompt_async_return(moderation, prescore_df)
    mid_a = "msg-chunk-a"
    mid_b = "msg-chunk-b"
    zero = default_usage_metrics()

    async def upstream():
        yield DRAgentEventResponse(
            events=[TextMessageContentEvent(message_id=mid_a, delta="a")],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[TextMessageContentEvent(message_id=mid_b, delta="b")],
            usage_metrics=zero,
        )

    stream_next = MagicMock(return_value=upstream())

    with patch(
        "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
        return_value=moderation,
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        chunks = [
            item
            async for item in mw.function_middleware_stream(
                _make_run_input("hi"),
                call_next=stream_next,
                context=_fn_context(),
            )
        ]

    content_events = [
        ev for resp in chunks for ev in resp.events if isinstance(ev, TextMessageContentEvent)
    ]
    assert len(content_events) == 2
    assert content_events[0].message_id == mid_a
    assert content_events[0].delta == "a"
    assert content_events[1].message_id == mid_b
    assert content_events[1].delta == "b"


async def test_function_middleware_stream_defers_text_message_end_before_run_finished(
    builder_mock: MagicMock,
) -> None:
    """Deferred TEXT_MESSAGE_END must be emitted before RUN_FINISHED (AG-UI ordering)."""
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_ok("hi")
    _set_evaluate_prompt_async_return(moderation, prescore_df)
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
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
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


async def test_function_middleware_stream_preserves_step_order_at_agent_transition(
    builder_mock: MagicMock,
) -> None:
    """CrewAI emits END, STEP_FINISHED, STEP_STARTED, START between agent text segments."""
    pipeline = _pipeline_mock()
    pipeline.get_prescore_guards.return_value = [MagicMock()]
    moderation = _moderation_mock(pipeline)
    prescore_df = _prescore_df_ok("topic")
    _set_evaluate_prompt_async_return(moderation, prescore_df)
    planner_mid = "msg-planner"
    writer_mid = "msg-writer"
    zero = default_usage_metrics()

    async def upstream():
        yield DRAgentEventResponse(
            events=[RunStartedEvent(thread_id="t1", run_id="r1")],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[StepStartedEvent(step_name="Content Planner")],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[
                TextMessageStartEvent(message_id=planner_mid, role="assistant"),
            ],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[TextMessageContentEvent(message_id=planner_mid, delta="plan")],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[TextMessageEndEvent(message_id=planner_mid)],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[StepFinishedEvent(step_name="Content Planner")],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[StepStartedEvent(step_name="Content Writer")],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[
                TextMessageStartEvent(message_id=writer_mid, role="assistant"),
            ],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[TextMessageContentEvent(message_id=writer_mid, delta="write")],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[TextMessageEndEvent(message_id=writer_mid)],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[StepFinishedEvent(step_name="Content Writer")],
            usage_metrics=zero,
        )
        yield DRAgentEventResponse(
            events=[RunFinishedEvent(thread_id="t1", run_id="r1")],
            usage_metrics=zero,
        )

    stream_next = MagicMock(return_value=upstream())

    with patch(
        "datarobot_genai.nat.datarobot_moderation_middleware.load_llm_moderation_pipeline",
        return_value=moderation,
    ):
        mw = DataRobotModerationMiddleware(DataRobotModerationConfig(), builder_mock)
        chunks = [
            item
            async for item in mw.function_middleware_stream(
                _make_run_input("topic"),
                call_next=stream_next,
                context=_fn_context(),
            )
        ]

    flat = [ev for resp in chunks for ev in resp.events]
    validate_sequence(flat)
    writer_start_idx = next(
        i
        for i, e in enumerate(flat)
        if e.type == EventType.STEP_STARTED and e.step_name == "Content Writer"
    )
    writer_finish_idx = next(
        i
        for i, e in enumerate(flat)
        if e.type == EventType.STEP_FINISHED and e.step_name == "Content Writer"
    )
    assert writer_start_idx < writer_finish_idx


def test_dome_chunk_to_dragent_event_response_keeps_tool_calls_with_text() -> None:
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
    out = dome_chunk_to_dragent_event_response(
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


def test_dome_chunk_to_dragent_event_response_serializes_numpy_moderations() -> None:
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
    out = dome_chunk_to_dragent_event_response(chunk)
    assert out.datarobot_moderations is not None
    assert out.datarobot_moderations["count"] == 42
    assert out.datarobot_moderations["nested"]["x"] == 1.5
    assert "2026-01-01" in str(out.datarobot_moderations["ts"])
    out.model_dump_json()
