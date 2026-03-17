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

"""Compatibility patches for nvidia-nat-crewai 1.4.1 with crewai >= 1.1.0.

nvidia-nat-crewai 1.4.1 expects crewai < 1.0.0 response format where
``choice.model_extra["message"]`` holds the LLM message. In crewai >= 1.1.0,
``message`` is a proper attribute on the choice object. This module patches
the callback handler to support both formats.

TODO(BUZZOK-29844): Remove once nvidia-nat-crewai ships a fix upstream.
Upstream issue: https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues/1802
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def patch_crewai_callback_handler() -> None:
    """Patch CrewAIProfilerHandler._llm_call_monkey_patch for crewai >= 1.1.0 compatibility."""
    try:
        from nat.plugins.crewai.crewai_callback_handler import CrewAIProfilerHandler
    except ImportError:
        return

    if getattr(CrewAIProfilerHandler._llm_call_monkey_patch, "_dr_patched", False):
        return

    _original_method = CrewAIProfilerHandler._llm_call_monkey_patch

    def _patched_llm_call_monkey_patch(self: Any) -> Callable[..., Any]:
        """Wrap the original monkey patch to inject message into model_extra before it runs."""
        original_func = self._original_llm_call

        def fixed_wrapped(*args: Any, **kwargs: Any) -> Any:
            def compat_completion(*a: Any, **kw: Any) -> Any:
                output = original_func(*a, **kw)
                for choice in output.choices:
                    if (
                        choice.model_extra is not None
                        and "message" not in choice.model_extra
                        and hasattr(choice, "message")
                        and choice.message is not None
                    ):
                        choice.model_extra["message"] = choice.message.model_dump()
                return output

            self._original_llm_call = compat_completion
            try:
                patched_wrapped = _original_method(self)
                return patched_wrapped(*args, **kwargs)
            finally:
                self._original_llm_call = original_func

        return fixed_wrapped

    CrewAIProfilerHandler._llm_call_monkey_patch = _patched_llm_call_monkey_patch
    CrewAIProfilerHandler._llm_call_monkey_patch._dr_patched = True
    logger.debug("Patched CrewAIProfilerHandler for crewai >= 1.1.0 compatibility")
