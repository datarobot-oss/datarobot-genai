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
    """Route LiteLLM log records through the application root logger only.

    LiteLLM attaches its own ``StreamHandler`` (with a distinct ANSI formatter) when
    ``litellm._logging`` is imported. If that happens after ``setup_logging()`` has
    already run, or if handlers are left attached while ``propagate`` is true, each
    log line is emitted twice (LiteLLM formatter + app formatter).
    """
    for name in _LITELLM_LOGGER_NAMES:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def setup_logging() -> None:
    """Setup uniform logging for the application."""  # noqa: D401
    current_log_level = logging.getLogger().getEffectiveLevel()
    logger.info(f"Setting up logging, log level: {logging._levelToName[current_log_level]}")

    # Import litellm so its module-level handler registration runs before we strip it.
    try:
        import litellm  # noqa: F401, PLC0415
    except ImportError:
        pass

    _configure_litellm_loggers()
