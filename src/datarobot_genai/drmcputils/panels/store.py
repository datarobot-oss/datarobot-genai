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

"""Server-side panel store.

Persists panels through a :class:`~datarobot_genai.drmcputils.files.store.BlobStore`:
each panel is a small JSON *manifest* blob (the panel metadata) tagged for
discovery, plus an optional separate *payload* blob for bulky content (a
Dataset's Parquet, a Chart's spec). The manifest blob's id is the panel id, so
no separate id allocation is needed and ids are globally unique.

The store depends only on the ``BlobStore`` Protocol, so it is backed by the
DataRobot Files API in production and by an in-memory fake in tests.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging

from datarobot_genai.drmcputils.files.store import BlobStore
from datarobot_genai.drmcputils.panels.models import BasePanel
from datarobot_genai.drmcputils.panels.models import Panel
from datarobot_genai.drmcputils.panels.models import panel_from_manifest

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = "main"
DEFAULT_LIST_LIMIT = 100

_MANIFEST_TAG = "dr-panel"
_PAYLOAD_TAG = "dr-panel-payload"


def _source_tag(source: str) -> str:
    return f"dr-panel-source:{source}"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


class PanelStore:
    """CRUD + listing for panels over a :class:`BlobStore`."""

    def __init__(self, blob_store: BlobStore) -> None:
        self._blobs = blob_store

    async def create(
        self,
        panel: BasePanel,
        *,
        source: str = DEFAULT_SOURCE,
        payload: bytes | None = None,
        payload_name: str | None = None,
        content_type: str | None = None,
    ) -> Panel:
        """Persist ``panel`` (and an optional ``payload`` blob); returns it with ``id`` set."""
        panel.updated_at = _now_iso()
        if payload is not None:
            payload_ref = await self._blobs.put(
                payload,
                name=payload_name or f"{panel.type.value}-payload",
                content_type=content_type,
                tags=[_PAYLOAD_TAG, _source_tag(source)],
            )
            panel.payload_files_id = payload_ref.files_id
            panel.payload_name = payload_ref.name

        manifest = json.dumps(panel.model_dump(mode="json", exclude={"id"})).encode("utf-8")
        try:
            manifest_ref = await self._blobs.put(
                manifest,
                name=f"panel-{panel.type.value}.json",
                content_type="application/json",
                tags=[_MANIFEST_TAG, _source_tag(source), f"dr-panel-type:{panel.type.value}"],
            )
        except Exception:
            # The panel was not created; don't leave the payload blob orphaned.
            if panel.payload_files_id:
                try:
                    await self._blobs.delete(panel.payload_files_id)
                except Exception:  # noqa: BLE001 - cleanup is best-effort; surface the original error
                    logger.warning(
                        "Failed to clean up payload blob %s after manifest write failed",
                        panel.payload_files_id,
                    )
                panel.payload_files_id = None
                panel.payload_name = None
            raise
        panel.id = manifest_ref.files_id
        return panel  # type: ignore[return-value]

    async def get(self, panel_id: str) -> Panel:
        """Load a panel by id (its manifest blob id). Payload is not hydrated here."""
        raw = await self._blobs.get(panel_id)
        manifest = json.loads(raw.decode("utf-8"))
        panel = panel_from_manifest(manifest)
        panel.id = panel_id
        return panel

    async def list(
        self,
        *,
        source: str = DEFAULT_SOURCE,
        limit: int = DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> list[Panel]:
        """List panels in ``source`` (metadata only); page with ``limit``/``offset``."""
        refs = await self._blobs.list(
            tags=[_MANIFEST_TAG, _source_tag(source)], limit=limit, offset=offset
        )
        panels: list[Panel] = []
        for ref in refs:
            raw = await self._blobs.get(ref.files_id)
            try:
                panel = panel_from_manifest(json.loads(raw.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError) as exc:
                # One corrupt manifest must not break listing the healthy ones.
                logger.warning("Skipping unreadable panel manifest %s: %s", ref.files_id, exc)
                continue
            panel.id = ref.files_id
            panels.append(panel)
        return panels

    async def delete(self, panel_id: str) -> None:
        """Delete a panel's manifest and its payload blob (if any).

        Transient fetch errors propagate (so the caller can retry without
        orphaning the payload); only an unreadable/corrupt manifest falls back
        to deleting the manifest alone, since the payload id is unknowable.
        """
        payload_files_id: str | None = None
        try:
            payload_files_id = (await self.get(panel_id)).payload_files_id
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError) as exc:
            logger.warning(
                "Panel %s manifest is unreadable (%s); deleting the manifest only", panel_id, exc
            )
        if payload_files_id:
            await self._blobs.delete(payload_files_id)
        await self._blobs.delete(panel_id)
