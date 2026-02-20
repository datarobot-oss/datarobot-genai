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


class MockModel:
    """Mock DataRobot model object."""

    def __init__(self, model_id: str, model_type: str, metrics: dict):
        self.id = model_id
        self.model_type = model_type
        self.metrics = metrics

    def score(self, dataset_url: str) -> MagicMock:
        """Mock scoring method."""
        return MagicMock(id=f"job_{self.id}_{hash(dataset_url) % 1000}")


class MockDatetimePartitioning:
    """Mock datetime partitioning for get_deployment_info time_series_config."""

    datetime_partition_column = "date"
    forecast_window_start = 1
    forecast_window_end = 14
    multiseries_id_columns: list[str] = []


class MockProject:
    """Mock DataRobot project object."""

    def __init__(self, project_id: str, models: list[MockModel] | None = None):
        self.id = project_id
        self._models = models or []
        self.target = "sentiment"
        self.target_type = "Binary"
        self.datetime_partitioning = MockDatetimePartitioning()

    def get_models(self, model_id: str) -> list:
        """Mock getting model method."""
        return self._models


class MockDeployment:
    """Mock DataRobot deployment object."""

    def __init__(
        self, deployment_id: str, project_id: str = "test_project_123", model_id: str = "model_1"
    ):
        self.id = deployment_id
        self.model = {"project_id": project_id, "id": model_id}

    def get_features(self) -> list:
        """
        Mock get_features;
        returns: list of feature dicts
        (must include feature_type for generate_prediction_data_template).
        """
        return [
            {"name": "text_review", "importance": 1, "feature_type": "text"},
            {"name": "product_category", "importance": 0, "feature_type": "categorical"},
        ]

    def get_capabilities(self) -> MagicMock:
        """Mock get_capabilities."""
        return MagicMock()


class MockDRClient:
    """Mock DataRobot client object."""

    def __init__(self, projects: list[MockProject] | None = None):
        self.Project = MagicMock()
        self.Model = MagicMock()
        self.Deployment = MagicMock()
        self.client = MagicMock()


def test_create_dr_client() -> MockDRClient:
    """Create a mock DataRobot client with test project and models."""
    client = MockDRClient()
    # Create test project with mock models
    project = MockProject(
        "test_project_123",
        models=[
            MockModel(
                "model_1",
                "Keras Text Convolutional Neural Network Classifier",
                {"AUC": 0.95, "LogLoss": 0.12},
            ),
            MockModel("model_2", "Random Forest", {"AUC": 0.92, "LogLoss": 0.15}),
            MockModel("model_3", "LightGBM", {"AUC": 0.94, "LogLoss": 0.13}),
        ],
    )
    # Create standalone model
    standalone_model = MockModel("standalone_model", "Neural Network", {"AUC": 0.88})

    def get_project(project_id: str) -> MockProject | None:
        """Mock Project.get that returns appropriate project or raises exception."""
        if project_id == "test_project_123":
            return project
        elif project_id == "nonexistent_project":
            return None
        elif project_id == "test_project":
            raise Exception("DataRobot API error")
        else:
            return None

    def get_model(
        model_id: str | None = None, project: MockProject | None = None, **kwargs: Any
    ) -> MockModel | None:
        """Mock Model.get; accepts model_id or project=, model_id= (kwargs)."""
        mid = model_id or (kwargs.get("model_id") if kwargs else None)
        if mid == "standalone_model":
            return standalone_model
        if mid in ("model_1", "model_2", "model_3"):
            return (
                next((m for m in project._models if m.id == mid), None)
                if project and hasattr(project, "_models")
                else standalone_model
            )
        return standalone_model

    def get_deployment(deployment_id: str | None = None, **kwargs: Any) -> MockDeployment:
        """Mock Deployment.get; accepts deployment_id as positional or keyword."""
        did = deployment_id or kwargs.get("deployment_id")
        return MockDeployment(
            did or "stub_deployment_id", project_id="test_project_123", model_id="model_1"
        )

    # Configure the mock methods
    client.Project.get = get_project
    client.Model.get = get_model
    client.Deployment.get = get_deployment
    return client
