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


def setup_logging() -> None:
    """Setup uniform logging for the application."""  # noqa: D401
    current_log_level = logging.getLogger().getEffectiveLevel()
    logger.info(f"Setting up logging, log level: {logging._levelToName[current_log_level]}")

    # Import LiteLLM first: it attaches its own stderr handler at import time while
    # leaving propagate=True, so each line is logged twice. Importing here ensures
    # that handler exists before we strip it, even when callers import litellm later.
    try:
        import litellm  # noqa: F401  (imported for its logging side effect)
    except ImportError:
        pass

    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
        lg = logging.getLogger(name)
        if lg:
            logger.debug(f"Resetting logger {name}")
            lg.handlers.clear()
            lg.propagate = True
