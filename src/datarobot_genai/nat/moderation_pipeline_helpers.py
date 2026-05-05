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
"""NAT moderation helpers: DRUM integration utilities plus NAT-only pipeline loading."""

from __future__ import annotations

import logging
import os

from datarobot_dome.api import ModerationPipeline
from datarobot_dome.constants import DISABLE_MODERATION_RUNTIME_PARAM_NAME
from datarobot_dome.constants import MODERATION_CONFIG_FILE_NAME
from datarobot_dome.constants import GuardStage
from datarobot_dome.runtime import get_runtime_parameter_value_bool
from datarobot_moderation_interface.drum_integration import (
    _handle_result_df_error_cases as handle_result_df_error_cases,
)
from datarobot_moderation_interface.drum_integration import (
    _set_moderation_attribute_to_completion as set_moderation_attribute_to_completion,
)
from datarobot_moderation_interface.drum_integration import build_non_streaming_chat_completion
from datarobot_moderation_interface.drum_integration import build_predictions_df_from_completion
from datarobot_moderation_interface.drum_integration import filter_association_id
from datarobot_moderation_interface.drum_integration import filter_extra_body
from datarobot_moderation_interface.drum_integration import format_result_df
from datarobot_moderation_interface.drum_integration import get_chat_prompt
from datarobot_moderation_interface.drum_integration import run_prescore_guards

_logger = logging.getLogger(__name__)


def load_llm_moderation_pipeline(model_dir: str | None) -> ModerationPipeline | None:
    """Load YAML LLM moderation for NAT via ``ModerationPipeline.from_yaml`` (not DRUM ``init``)."""
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


__all__ = [
    "build_non_streaming_chat_completion",
    "build_predictions_df_from_completion",
    "filter_association_id",
    "filter_extra_body",
    "format_result_df",
    "get_chat_prompt",
    "handle_result_df_error_cases",
    "load_llm_moderation_pipeline",
    "run_prescore_guards",
    "set_moderation_attribute_to_completion",
]
