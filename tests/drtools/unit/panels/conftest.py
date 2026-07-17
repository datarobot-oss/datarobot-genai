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

"""Shared fixtures for panel unit tests."""

import pytest

from datarobot_genai.drmcputils.files.store import BlobRef

FAKE_CONTAINER_ID = "fakecontainer1"


class FakeBlobStore:
    """In-memory path-based :class:`BlobStore` for hermetic panel tests.

    Mirrors the shared-container backend's semantics: blobs live at
    ``/``-separated paths inside one container (``container``); a bare token
    without ``/`` addresses a *legacy* standalone blob by Files id
    (``legacy``); ``delete`` silently ignores missing paths; ``list`` filters
    by path prefix server-side.
    """

    def __init__(self) -> None:
        self.container: dict[str, bytes] = {}
        self.legacy: dict[str, bytes] = {}
        self.container_id = FAKE_CONTAINER_ID

    async def put(
        self,
        data: bytes,
        *,
        path: str,
        content_type: str | None = None,
        timeout: int = 600,
    ) -> BlobRef:
        # content_type is advisory and not persisted (mirrors the real backend).
        self.container[path] = data
        return BlobRef(path=path, container_id=self.container_id, size=len(data))

    async def get(self, path: str) -> bytes:
        blobs = self.container if "/" in path else self.legacy
        if path not in blobs:
            raise KeyError(path)
        return blobs[path]

    async def delete(self, paths: str | list[str]) -> None:
        for path in [paths] if isinstance(paths, str) else paths:
            (self.container if "/" in path else self.legacy).pop(path, None)

    async def move(self, from_path: str, to_path: str) -> None:
        if from_path not in self.container:
            raise KeyError(from_path)
        self.container[to_path] = self.container.pop(from_path)

    async def list(
        self,
        *,
        prefix: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BlobRef]:
        paths = sorted(p for p in self.container if prefix is None or p.startswith(prefix))
        page = paths[offset : offset + limit] if limit else paths[offset:]
        return [
            BlobRef(path=p, container_id=self.container_id, size=len(self.container[p]))
            for p in page
        ]


@pytest.fixture
def fake_blob_store() -> FakeBlobStore:
    return FakeBlobStore()
