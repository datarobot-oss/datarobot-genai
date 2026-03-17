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

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

# Project id used by test_create_dr_client(); use for integration tests with stubs.
STUB_PROJECT_ID = "test_project_123"


class StubModel:
    """Stub DataRobot model object."""

    def __init__(self, model_id: str, model_type: str, metrics: dict):
        self.id = model_id
        self.model_type = model_type
        self.metrics = metrics

    def score(self, dataset_url: str) -> MagicMock:
        """Stub scoring method. Raises for fake URLs so integration tests can assert errors."""
        if "example.com" in (dataset_url or ""):
            raise Exception("404 client error: {'message': 'Not Found'}")
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
        self.datetime_partitioning = None

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
        self.status = "active"

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


class StubDataset:
    """Stub DataRobot dataset object."""

    def __init__(
        self,
        dataset_id: str,
        name: str = "stub_dataset",
        row_count: int = 100,
    ):
        self.id = dataset_id
        self.name = name
        self.created_at = "2025-01-01T00:00:00Z"
        self.row_count = row_count

    def get_as_dataframe(self) -> Any:
        """Return a small stub DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_2": ["a", "b", "c", "d", "e"],
                "target": [0, 1, 0, 1, 0],
            }
        )

    def get_raw_sample_data(self) -> Any:
        """Return a small stub sample DataFrame (subset, avoids full download)."""
        return self.get_as_dataframe()


class StubDataStore:
    """Stub DataRobot datastore object."""

    def __init__(self, datastore_id: str, canonical_name: str = "stub_datastore"):
        self.id = datastore_id
        self.canonical_name = canonical_name
        self.creator_id = "stub_creator"
        self.params = {"type": "jdbc", "driver": "postgresql"}


class StubRestResponse:
    """Stub HTTP response for client.get()/client.post() REST calls."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def json(self) -> dict[str, Any]:
        return self._data


class StubDRClient:
    """Stub DataRobot client for tests (canned responses; use with dr_client_stubs)."""

    def __init__(self, projects: list[StubProject] | None = None):
        self.Project = MagicMock()
        self.Model = MagicMock()
        self.Deployment = MagicMock()
        self.Dataset = MagicMock()
        self.DataStore = MagicMock()
        self.client = MagicMock()
        self.stub_rest_get: Any = None
        self.stub_rest_post: Any = None


def test_create_dr_client() -> StubDRClient:
    """Create a stub DataRobot client with test project and models."""
    client = StubDRClient()

    # Metrics shape: get_best_model expects metrics[metric].get("validation")
    def _metrics(auc: float, logloss: float) -> dict:
        return {
            "AUC": {"validation": auc},
            "LogLoss": {"validation": logloss},
        }

    # Create test project with stub models (model_1 best by AUC for integration test_model)
    project = StubProject(
        "test_project_123",
        models=[
            StubModel(
                "model_1",
                "Keras Text Convolutional Neural Network Classifier",
                _metrics(0.95, 0.12),
            ),
            StubModel("model_2", "Random Forest", _metrics(0.92, 0.15)),
            StubModel("model_3", "LightGBM", _metrics(0.94, 0.13)),
        ],
    )
    # Create standalone model
    standalone_model = StubModel("standalone_model", "Neural Network", _metrics(0.88, 0.2))

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

    # --- Dataset stubs ---
    stub_dataset = StubDataset("stub_dataset_id", name="stub_dataset.csv", row_count=100)

    def get_dataset(dataset_id: str) -> StubDataset:
        if dataset_id == "stub_dataset_id":
            return stub_dataset
        raise Exception(f"404 client error: {{'message': 'Dataset {dataset_id} not found'}}")

    def list_datasets() -> list[StubDataset]:
        return [stub_dataset]

    # --- DataStore stubs ---
    stub_datastore = StubDataStore("stub_datastore_id", canonical_name="Test PostgreSQL")

    def list_datastores() -> list[StubDataStore]:
        return [stub_datastore]

    # --- REST method stubs for client.get() / client.post() ---
    def stub_get(url: str, params: dict | None = None, **kwargs: Any) -> StubRestResponse:
        """Stub for client.get() REST calls."""
        if "externalDataDrivers" in url and "tables" in url:
            return StubRestResponse({"data": [{"name": "public.users"}, {"name": "public.orders"}]})
        return StubRestResponse({"data": [], "next": None})

    def stub_post(url: str, json: dict | None = None, **kwargs: Any) -> StubRestResponse:
        """Stub for client.post() REST calls."""
        if "externalDataDrivers" in url and "execute" in url:
            return StubRestResponse(
                {
                    "data": [{"id": 1, "name": "test"}],
                    "columns": ["id", "name"],
                }
            )
        return StubRestResponse({"data": []})

    # Configure the stub methods
    client.Project.get = get_project
    client.Model.get = get_model
    client.Deployment.get = get_deployment
    client.Dataset.get = get_dataset
    client.Dataset.list = list_datasets
    client.DataStore.list = list_datastores
    # Store REST stubs on the client so integration_mcp_server can wire them
    # onto mock_rest after replacing client.client.
    client.stub_rest_get = stub_get
    client.stub_rest_post = stub_post
    return client


def get_stub_classification_project() -> dict[str, Any]:
    """Return a stub project dict for integration tests (id matches test_create_dr_client)."""
    return {"project": SimpleNamespace(id=STUB_PROJECT_ID)}
