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

"""Compatibility patches for nvidia-nat-crewai streaming with crewai >= 1.1.0.

When ``stream=True``, LiteLLM returns a stream object (e.g. ``CustomStreamWrapper``)
without ``.choices``. NAT's ``wrapped_llm_call`` calls ``output.choices[0].model_dump()``
which crashes for streaming responses, so streaming calls must bypass that wrapper
and call ``litellm.completion`` directly.

TODO(BUZZOK-29844): Remove once nvidia-nat-crewai fixes streaming handling upstream.
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def patch_crewai_callback_handler() -> None:
    """Patch CrewAIProfilerHandler._llm_call_monkey_patch to bypass NAT's wrapper for streaming."""
    try:
        from nat.plugins.crewai.crewai_callback_handler import CrewAIProfilerHandler
    except ImportError:
        return

    if getattr(CrewAIProfilerHandler._llm_call_monkey_patch, "_dr_patched", False):
        return

    _original_method = CrewAIProfilerHandler._llm_call_monkey_patch

    def _patched_llm_call_monkey_patch(self: Any) -> Callable[..., Any]:
        """Wrap the original monkey patch to bypass NAT's wrapper for streaming calls."""
        original_litellm = self._original_llm_call
        patched_wrapped = _original_method(self)

        def dispatch(*args: Any, **kwargs: Any) -> Any:
            # NAT's wrapped_llm_call calls output.choices[0].model_dump()
            # which crashes for streaming responses (CustomStreamWrapper).
            if kwargs.get("stream"):
                return original_litellm(*args, **kwargs)
            return patched_wrapped(*args, **kwargs)

        return dispatch

    CrewAIProfilerHandler._llm_call_monkey_patch = _patched_llm_call_monkey_patch
    CrewAIProfilerHandler._llm_call_monkey_patch._dr_patched = True
    logger.debug("Patched CrewAIProfilerHandler for crewai >= 1.1.0 compatibility")
