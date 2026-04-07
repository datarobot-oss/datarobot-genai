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

import os

DEFAULT_MAX_HISTORY_MESSAGES = 20


def get_max_history_messages_default() -> int:
    """Return the default maximum number of history messages.

    This can be overridden globally via the
    ``DATAROBOT_GENAI_MAX_HISTORY_MESSAGES`` environment variable.
    Invalid values fall back to the built-in default. Negative values are
    treated as 0 (disable history).
    """
    raw = os.getenv("DATAROBOT_GENAI_MAX_HISTORY_MESSAGES")
    if not raw:
        return DEFAULT_MAX_HISTORY_MESSAGES

    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_HISTORY_MESSAGES

    # 0 means "no history". Clamp negatives to 0 to allow "disable history"
    # semantics via env var while preventing unbounded/undefined behavior.
    return max(0, value)
