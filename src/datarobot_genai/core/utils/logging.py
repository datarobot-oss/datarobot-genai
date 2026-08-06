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

import logging

logger = logging.getLogger(__name__)

_LITELLM_LOGGER_NAMES = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")


def _configure_litellm_loggers() -> None:
    """Keep LiteLLM's native handler/formatter and suppress duplicate root propagation.

    LiteLLM attaches its own ``StreamHandler`` (``HH:MM:SS - LiteLLM:LEVEL: file:line``)
    when ``litellm._logging`` is imported. With ``propagate=True`` (the prior default here),
    the same record also reaches the application root logger and prints twice in different
    formats. We keep LiteLLM's handler and set ``propagate=False`` so only the native
    LiteLLM line is emitted.
    """
    for name in _LITELLM_LOGGER_NAMES:
        lg = logging.getLogger(name)
        lg.propagate = False


def setup_logging() -> None:
    """Setup uniform logging for the application."""  # noqa: D401
    current_log_level = logging.getLogger().getEffectiveLevel()
    logger.info(f"Setting up logging, log level: {logging._levelToName[current_log_level]}")

    # Import litellm so its module-level handler registration runs before we configure it.
    try:
        import litellm  # noqa: F401, PLC0415
    except ImportError:
        pass

    _configure_litellm_loggers()
