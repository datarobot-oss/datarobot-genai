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

r"""Live, opt-in integration test for ``DataRobotFilesBlobStore``.

Unlike the mocked unit tests, this exercises the real DataRobot Files API end to
end (put → get → list → delete) and so verifies the SDK call signatures *and*
behavior against a live cluster.

It is **opt-in** to avoid any accidental writes to a real account: it runs only
when ``DR_FILES_LIVE_INTEGRATION`` is truthy and DataRobot credentials are
present. It is also excluded from the default CI test run (the
``tests/drtools/integration`` directory is ignored). Run it against staging::

    DR_FILES_LIVE_INTEGRATION=1 \
    DATAROBOT_ENDPOINT=https://staging.datarobot.com/api/v2 \
    DATAROBOT_API_TOKEN=*** \
    uv run pytest tests/drtools/integration/files -m integration -vv -s
"""

from __future__ import annotations

import os
import uuid

import pytest

from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.files.store import BlobRef
from datarobot_genai.drtools.files.store import DataRobotFilesBlobStore

pytestmark = pytest.mark.integration

_OPT_IN = os.getenv("DR_FILES_LIVE_INTEGRATION", "").lower() in {"1", "true", "yes"}
_HAS_CREDS = bool(os.getenv("DATAROBOT_API_TOKEN")) and bool(os.getenv("DATAROBOT_ENDPOINT"))

requires_live_dr = pytest.mark.skipif(
    not (_OPT_IN and _HAS_CREDS),
    reason=(
        "Live Files API integration is opt-in: set DR_FILES_LIVE_INTEGRATION=1 plus "
        "DATAROBOT_ENDPOINT and DATAROBOT_API_TOKEN to run."
    ),
)


@requires_live_dr
@pytest.mark.asyncio
async def test_blob_store_roundtrip_live() -> None:
    # headers_auth_only=False: no request headers in a standalone test, so fall
    # back to the ambient DATAROBOT_API_TOKEN.
    store = DataRobotFilesBlobStore(headers_auth_only=False)
    payload = f"hello {uuid.uuid4()}".encode()
    name = f"drtools-blobstore-it-{uuid.uuid4().hex}.txt"
    tag = f"drtools-it-{uuid.uuid4().hex[:8]}"

    ref = await store.put(payload, name=name, content_type="text/plain", tags=[tag])
    assert isinstance(ref, BlobRef)
    assert ref.files_id
    assert tag in ref.tags

    try:
        # get() round-trips the exact bytes.
        assert await store.get(ref) == payload

        # list() by our unique tag finds the blob, and its ref equals put()'s ref.
        listed = await store.list(tags=[tag])
        match = [r for r in listed if r.files_id == ref.files_id]
        assert match, f"stored blob {ref.files_id} not found via list(tags={tag!r})"
        assert match[0] == ref
    finally:
        await store.delete(ref)

    # After deletion the blob is gone.
    with pytest.raises(ToolError):
        await store.get(ref)
