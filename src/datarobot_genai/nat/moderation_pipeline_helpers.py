# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chat moderation helpers used by NAT middleware (datarobot_dome only; no drum_integration)."""

from __future__ import annotations

import copy
import logging
import os
import time
import traceback
import uuid
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from datarobot_dome.api import ModerationPipeline
from datarobot_dome.chat_helper import add_citations_to_df
from datarobot_dome.chat_helper import add_token_count_columns_to_df
from datarobot_dome.chat_helper import build_moderations_attribute_for_completion
from datarobot_dome.chat_helper import calculate_token_counts_and_confidence_score
from datarobot_dome.chat_helper import remove_unnecessary_columns
from datarobot_dome.constants import AGENTIC_PIPELINE_INTERACTIONS_ATTR
from datarobot_dome.constants import CHAT_COMPLETION_OBJECT
from datarobot_dome.constants import CITATIONS_ATTR
from datarobot_dome.constants import DATAROBOT_ASSOCIATION_ID_FIELD_NAME
from datarobot_dome.constants import DATAROBOT_METRICS_DICT_FIELD_NAME
from datarobot_dome.constants import DATAROBOT_MODERATIONS_ATTR
from datarobot_dome.constants import DISABLE_MODERATION_RUNTIME_PARAM_NAME
from datarobot_dome.constants import LLM_BLUEPRINT_ID_ATTR
from datarobot_dome.constants import LLM_CONTEXT_COLUMN_NAME
from datarobot_dome.constants import LLM_PROVIDER_GUARDS_ATTR
from datarobot_dome.constants import MODERATION_CONFIG_FILE_NAME
from datarobot_dome.constants import MODERATION_MODEL_NAME
from datarobot_dome.constants import NONE_CUSTOM_PY_RESPONSE
from datarobot_dome.constants import PROMPT_VECTOR_ATTR
from datarobot_dome.constants import USAGE_ATTR
from datarobot_dome.constants import GuardStage
from datarobot_dome.guard_executor import AsyncGuardExecutor
from datarobot_dome.runtime import get_runtime_parameter_value_bool
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage

_logger = logging.getLogger(__name__)

datarobot_metadata_columns = [
    "datarobot_token_count",
    "datarobot_latency",
    "datarobot_confidence_score",
]


def load_llm_moderation_pipeline(model_dir: str | None) -> ModerationPipeline | None:
    """Load YAML-configured LLM moderation, matching DRUM ``init()`` behavior without DRUM."""
    if get_runtime_parameter_value_bool(DISABLE_MODERATION_RUNTIME_PARAM_NAME, default_value=False):
        _logger.warning("Moderation is disabled via runtime parameter on the model")
        return None

    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

    base = model_dir if model_dir is not None else os.getcwd()
    guard_config_file = os.path.join(base, MODERATION_CONFIG_FILE_NAME)
    if not os.path.exists(guard_config_file):
        _logger.warning(
            "Guard config file: %s not found; moderations will not be enforced",
            guard_config_file,
        )
        return None

    pipeline = ModerationPipeline.from_yaml(guard_config_file)
    os.environ["PROMPT_COLUMN_NAME"] = pipeline._pipeline.get_input_column(GuardStage.PROMPT)
    os.environ["RESPONSE_COLUMN_NAME"] = pipeline._pipeline.get_input_column(GuardStage.RESPONSE)
    return pipeline


def block_citations_if_prompt_blocked(pipeline: Any, result_df: pd.DataFrame) -> None:
    if LLM_CONTEXT_COLUMN_NAME not in result_df.columns:
        return

    prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
    blocked_prompt_column_name = f"blocked_{prompt_column_name}"
    for index, row in result_df.iterrows():
        if row[blocked_prompt_column_name]:
            result_df.loc[index, LLM_CONTEXT_COLUMN_NAME] = ""


def handle_result_df_error_cases(
    prompt_column_name: str, df: pd.DataFrame, latency: float
) -> pd.DataFrame:
    replaced_message_prompt_column_name = f"replaced_message_{prompt_column_name}"
    moderated_prompt_column_name = f"moderated_{prompt_column_name}"
    replaced_prompt_column_name = f"replaced_{prompt_column_name}"
    for index, row in df.iterrows():
        if row.get(replaced_prompt_column_name):
            df.loc[index, moderated_prompt_column_name] = row[replaced_message_prompt_column_name]
        else:
            df.loc[index, moderated_prompt_column_name] = row[prompt_column_name]
    df["datarobot_latency"] = latency / df.shape[0]
    df["datarobot_token_count"] = 0
    df["datarobot_confidence_score"] = 0.0
    if prompt_column_name in df.columns:
        df.drop(prompt_column_name, axis=1, inplace=True)
    return df


def run_prescore_guards(
    pipeline: Any, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
    blocked_prompt_column_name = f"blocked_{prompt_column_name}"
    replaced_prompt_column_name = f"replaced_{prompt_column_name}"
    replaced_message_prompt_column_name = f"replaced_message_{prompt_column_name}"

    input_df = data.copy(deep=True)
    if len(pipeline.get_prescore_guards()) == 0:
        input_df[blocked_prompt_column_name] = False
        return input_df, input_df, 0

    start_time = time.time()

    try:
        prescore_df, prescore_latency = AsyncGuardExecutor(pipeline).run_guards(
            input_df, pipeline.get_prescore_guards(), GuardStage.PROMPT
        )
    except Exception as e:
        end_time = time.time()
        _logger.error("Failed to run prescore guards: %s", e)
        _logger.error(traceback.format_exc())
        prescore_df = input_df
        prescore_df[blocked_prompt_column_name] = False
        prescore_latency = end_time - start_time

    _logger.debug("%s", prescore_df)
    if blocked_prompt_column_name in prescore_df.columns:
        filtered_df = prescore_df[~prescore_df[blocked_prompt_column_name]]
    else:
        filtered_df = prescore_df

    for index, row in filtered_df.iterrows():
        if row.get(replaced_prompt_column_name):
            filtered_df.loc[index, prompt_column_name] = row[replaced_message_prompt_column_name]

    return prescore_df, filtered_df[data.columns], prescore_latency


def format_result_df(
    pipeline: Any,
    prescore_df: pd.DataFrame,
    postscore_df: pd.DataFrame,
    data: pd.DataFrame,
    none_predictions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
    blocked_prompt_column_name = f"blocked_{prompt_column_name}"
    blocked_message_prompt_column_name = f"blocked_message_{prompt_column_name}"
    response_column_name = pipeline.get_input_column(GuardStage.RESPONSE)
    blocked_completion_column_name = f"blocked_{response_column_name}"
    unmoderated_response_column_name = f"unmoderated_{response_column_name}"
    moderated_prompt_column_name = f"moderated_{prompt_column_name}"
    replaced_prompt_column_name = f"replaced_{prompt_column_name}"
    replaced_message_prompt_column_name = f"replaced_message_{prompt_column_name}"

    result_columns = (
        set(postscore_df.columns)
        .union(set(prescore_df.columns))
        .union(set(datarobot_metadata_columns))
        .union({unmoderated_response_column_name, moderated_prompt_column_name})
    )
    result_df = pd.DataFrame(index=data.index, columns=list(result_columns))

    for index, row in prescore_df.iterrows():
        if row.get(blocked_prompt_column_name):
            result_df.loc[index, response_column_name] = row[blocked_message_prompt_column_name]
            result_df.loc[index, unmoderated_response_column_name] = np.nan
        elif row.get(replaced_prompt_column_name):
            result_df.loc[index, moderated_prompt_column_name] = row[
                replaced_message_prompt_column_name
            ]
        else:
            result_df.loc[index, moderated_prompt_column_name] = row[prompt_column_name]
        for column in prescore_df.columns:
            result_df.loc[index, column] = row[column]

    if none_predictions_df is not None and not none_predictions_df.empty:
        for index, row in none_predictions_df.iterrows():
            result_df.loc[index, response_column_name] = NONE_CUSTOM_PY_RESPONSE
            result_df.loc[index, unmoderated_response_column_name] = NONE_CUSTOM_PY_RESPONSE
            result_df.loc[index, blocked_completion_column_name] = False
            for column in none_predictions_df.columns:
                if column != response_column_name:
                    result_df.loc[index, column] = row[column]

    blocked_message_completion_column_name = f"blocked_message_{response_column_name}"
    replaced_response_column_name = f"replaced_{response_column_name}"
    replaced_message_response_column_name = f"replaced_message_{response_column_name}"
    for index, row in postscore_df.iterrows():
        if row.get(blocked_completion_column_name):
            result_df.loc[index, response_column_name] = row[blocked_message_completion_column_name]
        elif row.get(replaced_response_column_name):
            result_df.loc[index, response_column_name] = row[replaced_message_response_column_name]
        else:
            result_df.loc[index, response_column_name] = row[response_column_name]
        result_df.loc[index, unmoderated_response_column_name] = row[response_column_name]
        for column in postscore_df.columns:
            if column != response_column_name:
                result_df.loc[index, column] = row[column]

    block_citations_if_prompt_blocked(pipeline, result_df)
    calculate_token_counts_and_confidence_score(pipeline, result_df)

    result_df = remove_unnecessary_columns(pipeline, result_df)

    pipeline.report_custom_metrics(result_df)

    for column in data.columns:
        if column in result_df.columns:
            result_df.drop(column, axis=1, inplace=True)

    _logger.debug("Return df")
    _logger.debug("%s", result_df)

    return result_df


def filter_extra_body(
    completion_create_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    datarobot_extra_body_params: dict[str, Any] = {}
    name = DATAROBOT_METRICS_DICT_FIELD_NAME
    if name in completion_create_params:
        value = completion_create_params[name]
        _logger.debug("found DataRobot metrics in extra_body: %s=%s", name, value)
        if isinstance(value, dict):
            datarobot_extra_body_params = copy.deepcopy(value)
        else:
            _logger.warning("DataRobot metrics in extra_body is not a dict: %s=%s", name, value)
        completion_create_params.pop(name, None)
    return completion_create_params, datarobot_extra_body_params


def filter_association_id(
    completion_create_params: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    name = DATAROBOT_ASSOCIATION_ID_FIELD_NAME
    if name in completion_create_params:
        value = completion_create_params[name]
        _logger.debug("found association ID in extra_body: %s=%s", name, value)
        completion_create_params.pop(name, None)
        return completion_create_params, value
    return completion_create_params, None


def build_predictions_df_from_completion(
    data: pd.DataFrame, pipeline: Any, chat_completion: Any
) -> tuple[pd.DataFrame, dict[str, Any]]:
    response_column_name = pipeline.get_input_column(GuardStage.RESPONSE)
    predictions_df = data.copy(deep=True)
    source_object: Any
    if isinstance(chat_completion, ChatCompletion):
        if len(chat_completion.choices) == 0:
            raise ValueError("Invalid response from custom.py, len(choices) = 0")
        predictions_df[response_column_name] = chat_completion.choices[0].message.content
        citations = getattr(chat_completion, CITATIONS_ATTR, None)
        if citations:
            predictions_df = add_citations_to_df(citations, predictions_df)
        if getattr(chat_completion, USAGE_ATTR, None):
            predictions_df = add_token_count_columns_to_df(
                pipeline, predictions_df, usage=chat_completion.usage
            )
        pipeline_interactions = getattr(chat_completion, AGENTIC_PIPELINE_INTERACTIONS_ATTR, None)
        if pipeline_interactions:
            predictions_df[AGENTIC_PIPELINE_INTERACTIONS_ATTR] = pipeline_interactions
        else:
            predictions_df[AGENTIC_PIPELINE_INTERACTIONS_ATTR] = [None] * len(predictions_df)

        source_object = chat_completion
    elif isinstance(chat_completion, Iterable):
        messages = []
        last_chunk = None
        for index, chunk in enumerate(chat_completion):
            if not isinstance(chunk, ChatCompletionChunk):
                raise ValueError(
                    f"Chunk at index {index} is not of type 'ChatCompletionChunk',"
                    f" but is of type '{type(chunk)}'"
                )
            last_chunk = chunk
            if len(chunk.choices) == 0:
                _logger.warning("No chunk delta at index %s, skipping it..", index)
                continue
            if chunk.choices[0].delta.content:
                messages.append(chunk.choices[0].delta.content)
        predictions_df[response_column_name] = "".join(messages)
        chunk_citations = (
            getattr(last_chunk, CITATIONS_ATTR, None) if last_chunk is not None else None
        )
        if chunk_citations:
            predictions_df = add_citations_to_df(chunk_citations, predictions_df)
        source_object = last_chunk
    else:
        raise ValueError(
            "Object returned by custom.py is not of type 'ChatCompletion' or an "
            f"'Iterable[ChatCompletionChunk], but is of type '{type(chat_completion)}'"
        )

    extra_attributes = {
        attr: getattr(source_object, attr, None)
        for attr in [
            LLM_BLUEPRINT_ID_ATTR,
            LLM_PROVIDER_GUARDS_ATTR,
            PROMPT_VECTOR_ATTR,
            CITATIONS_ATTR,
            USAGE_ATTR,
        ]
    }
    extra_attributes[AGENTIC_PIPELINE_INTERACTIONS_ATTR] = getattr(
        source_object, AGENTIC_PIPELINE_INTERACTIONS_ATTR, None
    )
    return predictions_df, extra_attributes


def build_non_streaming_chat_completion(
    message: str | None, reason: str, extra_attributes: dict[str, Any] | None = None
) -> ChatCompletion:
    msg = ChatCompletionMessage(content=message, role="assistant")
    choice = Choice(finish_reason=reason, index=0, message=msg)
    completion = ChatCompletion(
        id=str(uuid.uuid4()),
        choices=[choice],
        created=int(time.time()),
        model=MODERATION_MODEL_NAME,
        object=CHAT_COMPLETION_OBJECT,
    )
    if extra_attributes:
        for attr, attr_value in extra_attributes.items():
            setattr(completion, attr, attr_value)
    return completion


def set_moderation_attribute_to_completion(
    pipeline: Any,
    chat_completion: Any,
    df: pd.DataFrame,
    association_id: str | None = None,
) -> Any:
    if not pipeline.extra_model_output_for_chat_enabled:
        return chat_completion

    moderations = build_moderations_attribute_for_completion(pipeline, df)
    if moderations is None:
        return chat_completion

    if association_id:
        moderations["association_id"] = association_id
    if isinstance(chat_completion, ChatCompletion):
        setattr(chat_completion, DATAROBOT_MODERATIONS_ATTR, moderations)
    else:
        setattr(chat_completion[-1], DATAROBOT_MODERATIONS_ATTR, moderations)

    return chat_completion


def get_chat_prompt(completion_create_params: dict[str, Any]) -> str:
    if (
        "messages" not in completion_create_params
        or completion_create_params["messages"] is None
        or len(completion_create_params["messages"]) == 0
        or not isinstance(completion_create_params["messages"][-1], dict)
        or "content" not in completion_create_params["messages"][-1]
    ):
        raise ValueError(
            f"Chat input for moderation does not contain a message: {completion_create_params}"
        )

    last_user_message = None
    tool_calls = []
    for message in completion_create_params["messages"]:
        if message["role"] == "user":
            last_user_message = message
        if message["role"] == "tool":
            tool_calls.append(f"{message.get('name', '')}_{message['content']}")
    if last_user_message is None:
        raise ValueError("No message with 'user' role found in input")

    prompt_content = last_user_message["content"]
    tool_names = []
    if "tools" in completion_create_params:
        for tool in completion_create_params["tools"]:
            if "function" in tool and "name" in tool["function"]:
                tool_names.append(tool["function"]["name"])
    if isinstance(prompt_content, str):
        chat_prompt = prompt_content
    elif isinstance(prompt_content, list):
        concatenated_prompt = []
        for content in prompt_content:
            if content["type"] == "text":
                message = content["text"]
            elif content["type"] == "image_url":
                message = f"Image URL: {content['image_url']['url']}"
            elif content["type"] == "input_audio":
                message = f"Audio Input, Format: {content['input_audio']['format']}"
            else:
                message = f"Unhandled content type: {content['type']}"
            concatenated_prompt.append(message)
        chat_prompt = "\n".join(concatenated_prompt)
    else:
        raise ValueError(f"Unhandled prompt type: {type(prompt_content)}")

    if len(tool_calls) > 0:
        return "\n".join([chat_prompt, "Tool Calls:", "\n".join(tool_calls)])

    if len(tool_names) > 0:
        return "\n".join([chat_prompt, "Tool Names:", "\n".join(tool_names)])

    return chat_prompt
