# Copyright 2025 DataRobot, Inc.
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
import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Literal

from .base import MetadataBase

# Path to the schemas directory
SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


class DrumTargetType(str, Enum):
    BINARY = "binary"
    REGRESSION = "regression"
    ANOMALY = "anomaly"
    UNSTRUCTURED = "unstructured"
    MULTICLASS = "multiclass"
    TEXT_GENERATION = "textgeneration"
    GEO_POINT = "geopoint"
    VECTOR_DATABASE = "vectordatabase"
    AGENTIC_WORKFLOW = "agenticworkflow"

    @classmethod
    def prediction_types(cls) -> set["DrumTargetType"]:
        """Get the set of DRUM target types that correspond to structured predictions."""
        return {
            DrumTargetType.BINARY,
            DrumTargetType.REGRESSION,
            DrumTargetType.ANOMALY,
            DrumTargetType.MULTICLASS,
            DrumTargetType.TEXT_GENERATION,
            DrumTargetType.GEO_POINT,
            DrumTargetType.VECTOR_DATABASE,
        }


@lru_cache(maxsize=1)
def _get_prediction_fallback_schema() -> dict[str, Any]:
    """Get the default prediction input schema for DRUM deployments."""
    schema_path = SCHEMAS_DIR / "drum_prediction_fallback_schema.json"
    with open(schema_path) as f:
        schema: dict[str, Any] = json.load(f)
        return schema


@lru_cache(maxsize=1)
def _get_agentic_fallback_schema() -> dict[str, Any]:
    """Get the default agentic workflow input schema for DRUM deployments."""
    schema_path = SCHEMAS_DIR / "drum_agentic_fallback_schema.json"
    with open(schema_path) as f:
        schema: dict[str, Any] = json.load(f)
        return schema


def get_default_schema(target_type: str) -> dict[str, Any]:
    """Get the default input schema for a given DRUM target type, when
    the deployment does not provide one. This fallback mechanism is here
    to lower the friction of using DRUM deployments with MCP, for more
    advanced use cases it is recommended to provide a custom input and
    expose it via model-metadata.yaml inputSchema parameter.

    Args:
        target_type: The target type of the DRUM deployment.

    Returns
    -------
        A dictionary representing the default input schema wrapped in HTTP request structure.
    """
    if target_type == DrumTargetType.AGENTIC_WORKFLOW:
        return _get_agentic_fallback_schema()

    if target_type in DrumTargetType.prediction_types():
        return _get_prediction_fallback_schema()

    return {}


def is_drum(metadata: dict[str, Any]) -> bool:
    """Check if the deployment is a DRUM deployment.

    DRUM deployments are identified by the presence of both drum_server
    and drum_version fields in the metadata response.

    Args:
        metadata: The response retrieved from the custom model /info/ route.

    Returns
    -------
        True if this is a DRUM deployment, False otherwise.
    """
    drum_server = metadata.get("drum_server")
    drum_version = metadata.get("drum_version")
    return bool(drum_server or drum_version)


class DrumMetadataAdapter(MetadataBase):
    """Adapter for DRUM deployment metadata."""

    def __init__(self, metadata: dict[str, Any]):
        """Initialize adapter with validated metadata.

        Args:
            metadata: Dictionary containing at minimum a 'target_type' key.

        Note:
            Use class methods `from_deployment_metadata()` or `from_target_type()`
            for construction instead of calling this directly.
        """
        self.metadata = metadata
        self._validate_tool_support()

    def _validate_tool_support(self) -> None:
        """Validate that DRUM deployments are supported in the current environment.

        Raises
        ------
            ValueError: If DRUM deployments are not supported.
        """
        if self.target_type not in list(DrumTargetType):
            raise ValueError(
                f"The deployment target_type: {self.target_type} "
                f"is not supported, to be registered as MCP Tool."
            )

    @classmethod
    def from_deployment_metadata(cls, metadata: dict[str, Any]) -> "DrumMetadataAdapter":
        """Create adapter from full deployment metadata.

        Args:
            metadata: The response retrieved from the custom model /info/ route.

        Returns
        -------
            DrumMetadataAdapter instance.

        Raises
        ------
            ValueError: If metadata is not from a DRUM deployment.
        """
        if not is_drum(metadata):
            raise ValueError("Provided metadata is not from a DRUM deployment.")
        return cls(metadata)

    @classmethod
    def from_target_type(cls, target_type: str) -> "DrumMetadataAdapter":
        """Create adapter from target type only.

        Used for testing/minimal setup when broader set of information
        from metadata built from model-metadata.yaml information is
        not available i.e. datarobot predictive models.

        Args:
            target_type: The DRUM target type (e.g., 'binary', 'regression').

        Returns
        -------
            DrumMetadataAdapter instance with minimal metadata.
        """
        return cls({"target_type": target_type.lower()})

    @property
    def target_type(self) -> str:
        return str(self.metadata["target_type"])

    @property
    def name(self) -> str:
        return str(self.model_metadata.get("name", ""))

    @property
    def description(self) -> str:
        return str(self.model_metadata.get("description", ""))

    @property
    def endpoint(self) -> str:
        """Return the appropriate endpoint for the DRUM target type."""
        predictions_endpoint = "/predictions"

        endpoint_map: dict[str, str] = {
            DrumTargetType.BINARY: predictions_endpoint,
            DrumTargetType.REGRESSION: predictions_endpoint,
            DrumTargetType.ANOMALY: predictions_endpoint,
            DrumTargetType.MULTICLASS: predictions_endpoint,
            DrumTargetType.TEXT_GENERATION: predictions_endpoint,
            DrumTargetType.GEO_POINT: predictions_endpoint,
            DrumTargetType.UNSTRUCTURED: "/predictionsUnstructured",
            DrumTargetType.VECTOR_DATABASE: predictions_endpoint,
            DrumTargetType.AGENTIC_WORKFLOW: "/chat/completions",
        }

        return endpoint_map[self.target_type]

    @property
    def model_metadata(self) -> dict[str, Any]:
        result = self.metadata.get("model_metadata", {})
        return dict(result)

    @property
    def input_schema(self) -> dict[str, Any]:
        input_schema = self.model_metadata.get("input_schema", get_default_schema(self.target_type))

        if not input_schema or not isinstance(input_schema, dict):
            raise ValueError(
                "DRUM deployment is missing a valid input schema. Please make "
                "sure the model-metadata.yaml file includes `inputSchema` "
                "definition and that custom model is using datarobot-drum in "
                "version v1.17.2 or later."
            )
        return dict(input_schema)

    @property
    def method(self) -> Literal["GET", "POST", "PATCH", "PUT", "DELETE"]:
        return "POST"

    @property
    def headers(self) -> dict[str, str]:
        """Return HTTP headers required for this deployment type."""
        if self.target_type in DrumTargetType.prediction_types():
            # structured predictions send data as CSV bytes, which
            # requires an explicit Content-Type header since aiohttp
            # won't set it automatically
            return {"Content-Type": "text/csv"}

        return {}
