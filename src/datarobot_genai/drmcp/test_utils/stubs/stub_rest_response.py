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


class StubRestResponse:
    """Stub HTTP response for client.get()/client.post()/patch()/delete() REST calls."""

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        text: str | None = None,
        content: bytes | None = None,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ):
        self._data = data or {}
        self._text = text
        self._content = content
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._data

    @property
    def text(self) -> str:
        if self._text is not None:
            return self._text
        return ""

    @property
    def content(self) -> bytes:
        if self._content is not None:
            return self._content
        if self._data:
            import json

            return json.dumps(self._data).encode()
        return b""
