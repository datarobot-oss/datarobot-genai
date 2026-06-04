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

"""Panel data models.

Panels are typed, persisted analytical artifacts with lineage. Metadata is a
small Pydantic record; bulky payloads (a Dataset's Parquet, a Chart's spec) are
stored separately via a :class:`~datarobot_genai.drtools.files.store.BlobStore`
and referenced here by ``payload_files_id`` (the Files container id), keeping
the manifest cheap to list and serialize.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class PanelType(StrEnum):
    """Discriminator for the concrete panel types."""

    DATASET = "dataset"
    CHART = "chart"
    TEXT = "text"
    JSON = "json"


class BasePanel(BaseModel):
    """Common metadata shared by every panel type.

    ``id`` is assigned by the store on create (it is the manifest's Files
    container id) and is ``None`` until then. Inline-content types (Text, Json)
    carry their payload on the model; bulky types (Dataset, Chart) reference an
    external blob via ``payload_files_id``.
    """

    id: str | None = None
    type: PanelType
    title: str
    description: str | None = None
    parents: list[str] = Field(default_factory=list)
    execution_context: dict[str, Any] | None = None
    updated_by: str | None = None
    updated_at: str | None = None
    payload_files_id: str | None = None
    payload_name: str | None = None


class Dataset(BasePanel):
    type: Literal[PanelType.DATASET] = PanelType.DATASET
    row_count: int | None = None
    columns: list[str] | None = None


class Chart(BasePanel):
    type: Literal[PanelType.CHART] = PanelType.CHART
    chart_library: str | None = None


class Text(BasePanel):
    type: Literal[PanelType.TEXT] = PanelType.TEXT
    text: str = ""


class Json(BasePanel):
    type: Literal[PanelType.JSON] = PanelType.JSON
    data: dict[str, Any] = Field(default_factory=dict)


Panel = Dataset | Chart | Text | Json

PANEL_TYPE_TO_MODEL: dict[PanelType, type[BasePanel]] = {
    PanelType.DATASET: Dataset,
    PanelType.CHART: Chart,
    PanelType.TEXT: Text,
    PanelType.JSON: Json,
}


def panel_from_manifest(raw: dict[str, Any]) -> Panel:
    """Reconstruct the concrete panel type from a stored manifest dict."""
    panel_type = PanelType(raw["type"])
    model = PANEL_TYPE_TO_MODEL[panel_type]
    return model.model_validate(raw)  # type: ignore[return-value]
