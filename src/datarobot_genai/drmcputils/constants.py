# Copyright 2026 DataRobot, Inc.
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

DEFAULT_DATAROBOT_ENDPOINT = "https://app.datarobot.com/api/v2"
RUNTIME_PARAM_ENV_VAR_NAME_PREFIX = "MLOPS_RUNTIME_PARAM_"

MAX_INLINE_SIZE = 1024 * 1024  # 1MB

AUTH_CTX_KEY = "authorization_context"

LOG_LEVELS = ("debug", "info", "warn", "error")

IMPORTANCE_VALUES = ("low", "moderate", "high", "critical")

ARTIFACT_STATUSES: tuple[str, ...] = ("draft", "locked")

ARTIFACT_TYPES: tuple[str, ...] = ("service", "nim")

WORKLOAD_TERMINAL_FAILURE_STATUS = "errored"

REPLACEMENT_STRATEGIES: tuple[str, ...] = ("rolling",)

# Header names to check for authorization tokens (in order of preference)
HEADER_TOKEN_CANDIDATE_NAMES = [
    "x-datarobot-authorization",
    "x-datarobot-api-key",
    "x-datarobot-api-token",
    "authorization",
]
