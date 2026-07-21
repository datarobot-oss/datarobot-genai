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

r"""Live, opt-in integration test for conversation-scoped ``PanelStore``.

The unit suite runs against an in-memory fake, which is a *model* of the Files
API — and 0.26.1 shipped a bug the model missed: scoped id resolution probed
existence with an exact file path as a listing ``prefix``, which the live API
rejects with ``400: Prefix must end with a forward slash "/"`` (every scoped
``get``/``delete``/``move`` failed in production while all unit tests passed).
This test exercises the real API through the exact code paths that broke:
scoped create → get (hinted + unhinted directory probes) → cross-conversation
isolation → id-preserving move → delete.

It is **opt-in** like the files-store live test (same flag): panels are written
under ``staging``/``main`` in the account's shared ``panels`` container, but
inside a unique per-run conversation directory, so runs are isolated and
cleaned up even on mid-test failure. Run it against staging::

    DR_FILES_LIVE_INTEGRATION=1 \
    DATAROBOT_ENDPOINT=https://staging.datarobot.com/api/v2 \
    DATAROBOT_API_TOKEN=*** \
    uv run pytest tests/drtools/integration/panels -m integration -vv -s
"""

from __future__ import annotations

import os
import uuid

import pytest

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.files.store import DataRobotFilesBlobStore
from datarobot_genai.drmcputils.panels.models import Text
from datarobot_genai.drmcputils.panels.store import PanelStore

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
async def test_scoped_panel_store_roundtrip_live() -> None:
    # headers_auth_only=False: no request headers in a standalone test, so fall
    # back to the ambient DATAROBOT_API_TOKEN.
    run = uuid.uuid4().hex[:12]
    conversation = f"it_conv_{run}"
    store = PanelStore(
        DataRobotFilesBlobStore(headers_auth_only=False), conversation_id=conversation
    )

    created = await store.create(Text(title="live", text=run), source="staging")
    assert created.id is not None

    try:
        # Scoped get with a source hint — the exact probe that 0.26.1 sent as
        # a file-path prefix and the live API 400'd on.
        hinted = await store.get(created.id, source="staging")
        assert hinted.id == created.id
        assert isinstance(hinted, Text)
        assert hinted.text == run

        # Unhinted get walks the known-source directory probes.
        unhinted = await store.get(created.id)
        assert unhinted.id == created.id

        # The conversation's listing sees exactly this panel.
        listed = await store.list(source="staging", limit=0)
        assert [p.id for p in listed] == [created.id]

        # Another conversation cannot resolve the id (scope enforcement
        # against the real backend; resolution falls through to the legacy
        # id lookup, which the API answers with not-found).
        intruder = PanelStore(
            DataRobotFilesBlobStore(headers_auth_only=False),
            conversation_id=f"it_other_{run}",
        )
        with pytest.raises(ToolError):
            await intruder.get(created.id)

        # Promote staging → main: id preserved, both directory probes agree.
        moved = await store.move(created.id, to_source="main")
        assert moved.id == created.id
        assert (await store.get(created.id, source="main")).id == created.id
        assert await store.list(source="staging", limit=0) == []
    finally:
        await store.delete(created.id)

    # The conversation directories are empty again in both sources.
    assert await store.list(source="staging", limit=0) == []
    assert await store.list(source="main", limit=0) == []
