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
end (put → get → list → move → delete) against the shared container, and so
verifies the SDK call signatures *and* server behavior — in particular that
``/``-separated paths uploaded via ``prefix`` survive intact (a path embedded in
an uploaded *filename* would be basename-sanitized server-side).

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

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.files.store import BlobRef
from datarobot_genai.drmcputils.files.store import DataRobotFilesBlobStore

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
    # A unique folder under a unique source segment keeps this run isolated and
    # trivially cleanable even if an assertion fails mid-way.
    run = f"it_{uuid.uuid4().hex[:12]}"
    path = f"{run}/conv_a/blob.txt"
    moved_path = f"{run}/conv_a/blob-moved.txt"

    ref = await store.put(payload, path=path, content_type="text/plain")
    assert isinstance(ref, BlobRef)
    assert ref.path == path
    assert ref.container_id

    try:
        # get() by path round-trips the exact bytes — proof the '/'-separated
        # path survived the upload (no server-side basename sanitization).
        assert await store.get(path) == payload

        # list() by prefix finds the blob at its full path.
        listed = await store.list(prefix=f"{run}/", limit=0)
        assert [r.path for r in listed] == [path]
        assert listed[0].container_id == ref.container_id

        # move() renames in place; the old path is gone, the new one reads back.
        await store.move(path, moved_path)
        assert await store.get(moved_path) == payload
        assert [r.path for r in await store.list(prefix=f"{run}/", limit=0)] == [moved_path]
    finally:
        await store.delete([path, moved_path])

    # After deletion the folder is empty and the blob is gone.
    assert await store.list(prefix=f"{run}/", limit=0) == []
    with pytest.raises(ToolError):
        await store.get(moved_path)
