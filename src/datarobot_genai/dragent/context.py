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

from typing import Any

from nat.runtime.runner import Context

from datarobot_genai.core.chat.auth import get_authorization_context_from_headers


def extract_headers_from_context(headers_to_forward: list[str]) -> dict[str, str]:
    context = Context.get()
    headers = context.metadata.headers
    extracted_headers: dict[str, str] = {}
    if not headers:
        return extracted_headers

    for header in headers_to_forward:
        if header in headers:
            extracted_headers[header] = headers[header]

    return extracted_headers


def extract_datarobot_headers_from_context() -> dict[str, str]:
    context = Context.get()
    headers = context.metadata.headers
    extracted_headers: dict[str, str] = {}
    if not headers:
        return extracted_headers

    for header in headers:
        # Already lowercase from NAT
        if header.startswith("x-datarobot-") or header.startswith("x-untrusted-"):
            extracted_headers[header] = headers[header]

    return extracted_headers


def extract_authorization_from_context(secret_key: str | None = None) -> dict[str, Any] | None:
    context = Context.get()
    headers = context.metadata.headers
    if not headers:
        return None

    return get_authorization_context_from_headers(headers, secret_key=secret_key)
