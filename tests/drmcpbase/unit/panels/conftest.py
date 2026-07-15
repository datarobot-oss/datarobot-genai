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

"""Shared fixtures for panel resource unit tests."""

import pytest

from datarobot_genai.drmcputils.files.store import BlobRef


class FakeBlobStore:
    """In-memory :class:`BlobStore` implementation for hermetic panel tests.

    Tags are stored alongside each blob and ``list`` returns blobs whose tags
    are a superset of the requested tags — matching the AND semantics the
    Files-API backend uses.
    """

    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, BlobRef, set[str]]] = {}
        self._counter = 0

    async def put(
        self,
        data: bytes,
        *,
        name: str,
        content_type: str | None = None,
        tags: list[str] | None = None,
    ) -> BlobRef:
        # content_type is advisory and not persisted (mirrors the real backend).
        self._counter += 1
        files_id = f"blob{self._counter}"
        ref = BlobRef(files_id=files_id, name=name, tags=tuple(tags or []))
        self.blobs[files_id] = (data, ref, set(tags or []))
        return ref

    async def get(self, ref: BlobRef | str) -> bytes:
        files_id = ref.files_id if isinstance(ref, BlobRef) else ref
        if files_id not in self.blobs:
            raise KeyError(files_id)
        return self.blobs[files_id][0]

    async def delete(self, ref: BlobRef | str) -> None:
        files_id = ref.files_id if isinstance(ref, BlobRef) else ref
        self.blobs.pop(files_id, None)

    async def set_tags(self, ref: BlobRef | str, tags: list[str]) -> None:
        files_id = ref.files_id if isinstance(ref, BlobRef) else ref
        if files_id not in self.blobs:
            raise KeyError(files_id)
        data, old_ref, _old_tags = self.blobs[files_id]
        new_ref = BlobRef(files_id=old_ref.files_id, name=old_ref.name, tags=tuple(tags))
        self.blobs[files_id] = (data, new_ref, set(tags))

    async def list(
        self,
        *,
        search: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BlobRef]:
        wanted = set(tags or [])
        matches = [
            ref for _data, ref, blob_tags in self.blobs.values() if wanted.issubset(blob_tags)
        ]
        return matches[offset : offset + limit] if limit else matches


@pytest.fixture
def fake_blob_store() -> FakeBlobStore:
    return FakeBlobStore()
