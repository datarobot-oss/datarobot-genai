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
from unittest.mock import MagicMock


class StubModel:
    """Stub DataRobot model object."""

    def __init__(self, model_id: str, model_type: str, metrics: dict):
        self.id = model_id
        self.model_type = model_type
        self.metrics = metrics

    def score(self, dataset_url: str) -> MagicMock:
        """Stub scoring method."""
        return MagicMock(id=f"job_{self.id}_{hash(dataset_url) % 1000}")


class StubDatetimePartitioning:
    """Stub datetime partitioning for get_deployment_info time_series_config."""

    datetime_partition_column = "date"
    forecast_window_start = 1
    forecast_window_end = 14
    multiseries_id_columns: list[str] = []


class StubProject:
    """Stub DataRobot project object."""

    def __init__(self, project_id: str, models: list[StubModel] | None = None):
        self.id = project_id
        self._models = models or []
        self.target = "sentiment"
        self.target_type = "Binary"
        self.datetime_partitioning = StubDatetimePartitioning()

    def get_models(self) -> list:
        """Stub get_models (matches real API: no arguments)."""
        return self._models


class StubDeployment:
    """Stub DataRobot deployment object."""

    def __init__(
        self, deployment_id: str, project_id: str = "test_project_123", model_id: str = "model_1"
    ):
        self.id = deployment_id
        self.model = {"project_id": project_id, "id": model_id}

    def get_features(self) -> list:
        """
        Stub get_features;
        returns: list of feature dicts
        (must include feature_type for generate_prediction_data_template).
        """
        return [
            {"name": "text_review", "importance": 1, "feature_type": "text"},
            {"name": "product_category", "importance": 0, "feature_type": "categorical"},
        ]

    def get_capabilities(self) -> MagicMock:
        """Stub get_capabilities."""
        return MagicMock()


class StubDRClient:
    """Stub DataRobot client for tests (canned responses; use with dr_client_stubs)."""

    def __init__(self, projects: list[StubProject] | None = None):
        self.Project = MagicMock()
        self.Model = MagicMock()
        self.Deployment = MagicMock()
        self.client = MagicMock()


def test_create_dr_client() -> StubDRClient:
    """Create a stub DataRobot client with test project and models."""
    client = StubDRClient()
    # Create test project with stub models
    project = StubProject(
        "test_project_123",
        models=[
            StubModel(
                "model_1",
                "Keras Text Convolutional Neural Network Classifier",
                {"AUC": 0.95, "LogLoss": 0.12},
            ),
            StubModel("model_2", "Random Forest", {"AUC": 0.92, "LogLoss": 0.15}),
            StubModel("model_3", "LightGBM", {"AUC": 0.94, "LogLoss": 0.13}),
        ],
    )
    # Create standalone model
    standalone_model = StubModel("standalone_model", "Neural Network", {"AUC": 0.88})

    def get_project(project_id: str) -> StubProject | None:
        """Stub Project.get that returns appropriate project or raises exception."""
        if project_id == "test_project_123":
            return project
        elif project_id == "nonexistent_project":
            return None
        elif project_id == "test_project":
            raise Exception("DataRobot API error")
        else:
            return None

    def get_model(
        project: StubProject | None = None, model_id: str | None = None, **kwargs: Any
    ) -> StubModel | None:
        """Stub Model.get; signature matches SDK (project, model_id)."""
        mid = model_id or kwargs.get("model_id")
        proj = project or kwargs.get("project")
        if mid == "standalone_model":
            return standalone_model
        if mid in ("model_1", "model_2", "model_3"):
            return (
                next((m for m in proj._models if m.id == mid), None)
                if proj and hasattr(proj, "_models")
                else standalone_model
            )
        return standalone_model

    def get_deployment(deployment_id: str | None = None, **kwargs: Any) -> StubDeployment:
        """Stub Deployment.get; accepts deployment_id as positional or keyword."""
        did = deployment_id or kwargs.get("deployment_id")
        return StubDeployment(
            did or "stub_deployment_id", project_id="test_project_123", model_id="model_1"
        )

    # Configure the stub methods
    client.Project.get = get_project
    client.Model.get = get_model
    client.Deployment.get = get_deployment
    return client
