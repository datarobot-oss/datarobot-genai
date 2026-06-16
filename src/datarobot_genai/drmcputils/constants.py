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

CREDENTIAL_TYPE_KEYS: dict[str, list[str]] = {
    "basic": ["login", "password"],
    "s3": ["awsAccessKeyId", "awsSecretAccessKey", "awsSessionToken"],
    "azure": ["azureConnectionString"],
    "gcp": ["gcpKey"],
    "oauth": ["oauthRefreshToken", "oauthClientId", "oauthClientSecret"],
    "api_token": ["apiToken"],
    "snowflake_key_pair_user_account": ["privateKeyStr", "passphrase"],
    "databricks_access_token": ["token"],
    "databricks_service_principal": ["clientId", "clientSecret"],
    "azure_service_principal": ["clientId", "clientSecret", "tenantId"],
}

KNOWN_CREDENTIAL_TYPES: tuple[str, ...] = tuple(sorted(CREDENTIAL_TYPE_KEYS))

OPENAPI_BUNDLED_SPEC_CANDIDATES: tuple[str, ...] = (
    "/app/openapi.yaml",
    "/app/openapi.json",
    "/workload-api/charts/openapi.yaml",
)

OPENAPI_DEFAULT_REMOTE_PATH = "openapi.json"

# Header names to check for authorization tokens (in order of preference)
HEADER_TOKEN_CANDIDATE_NAMES = [
    "x-datarobot-authorization",
    "x-datarobot-api-key",
    "x-datarobot-api-token",
    "authorization",
]
