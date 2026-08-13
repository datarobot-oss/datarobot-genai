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

import pytest

from datarobot_genai.drmcputils.log_redaction import redact_secrets

_MESSAGES_CONTAINING_SECRET = [
    "sk-a1b1c1111111111A1B1C1111119111191111111181111117",  # OpenAI key
    "sk-proj-a1b1c111_111111A1B1C111111911119111-11_181111117991122aaa",  # newer OpenAI key
    "AKIAA1B11C11111D1111",  # AWS key
    "dummy.user@datarobot.com",  # Email
    "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.c2lnbmF0dXJl",  # JWT
    "Bearer abcDEF123456789xyz",  # Authorization header value
    "basic dXNlcjpwYXNzd29yZA==",  # Basic auth value (case-insensitive)
    "password=Sup3rS3cret!",  # key=value assignment
    "api_key: 12345secretvalue",  # key: value assignment
    "client_secret=abc-def_123",  # key=value with - and _
    "DATAROBOT_API_TOKEN=NjM4someToken-value_here",  # env-style token assignment
    "password=hunter2",  # short but digit-bearing credential
    "client_secret=correcthorsebatterystaple",  # long all-lowercase credential
]

# Regression guard for the removed catch-all ``([a-zA-Z0-9]{20,})``: operational
# identifiers must survive redaction so logs stay debuggable.
_MESSAGES_WITHOUT_SECRETS = [
    "UserMCPProvider: API client not yet initialised",  # class name
    "request_id=abc123def456ghi789jkl012 completed",  # long request id
    "deployment 6513f86cd799439011abcdef failed",  # 24-hex ObjectId
    "trace_id=4bf92f3577b34da6a3ce929d0e0e4736",  # trace id
    "tokens=1500 completion_tokens=300",  # LLM usage counters (plural 'tokens')
    "Registered tool datarobot_docs_search_documentation",  # long tool name
    "completion_token=300 prompt_token=12",  # singular *_token usage counters
    "Basic authentication disabled",  # prose after 'Basic', not a credential
    "Bearer integration tests passed",  # prose after 'Bearer'
    "basic connectivity check passed",  # lowercase prose
    "api_key: missing",  # diagnostic word, not a credential value
    "authorization: denied",  # diagnostic word
]


@pytest.mark.parametrize("msg", _MESSAGES_CONTAINING_SECRET)
def test_redact_secrets_redacts_secrets(msg: str) -> None:
    # GIVEN a message containing a secret
    # WHEN it is redacted / THEN the secret is fully replaced
    assert redact_secrets(msg) == "[REDACTED]"


@pytest.mark.parametrize("msg", _MESSAGES_WITHOUT_SECRETS)
def test_redact_secrets_preserves_operational_identifiers(msg: str) -> None:
    # GIVEN a message with ids/class names but no secret
    # WHEN it is redacted / THEN the message is untouched
    assert redact_secrets(msg) == msg


def test_redact_secrets_redacts_secret_within_context() -> None:
    # GIVEN a realistic line mixing a secret with operational context
    msg = "Auth failed for deployment 6513f86cd799439011abcdef: Authorization: Bearer eyJa.eyJb.sig"

    redacted = redact_secrets(msg)

    # THEN the token is gone but the surrounding context survives
    assert "eyJa.eyJb.sig" not in redacted
    assert "[REDACTED]" in redacted
    assert "6513f86cd799439011abcdef" in redacted
