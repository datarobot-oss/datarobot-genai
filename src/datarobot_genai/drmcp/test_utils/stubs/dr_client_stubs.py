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

# Dataset id used by test_create_dr_client(); use for integration tests with stubs.
STUB_DATASET_ID = "stub_dataset_id"


class StubRocCurve:
    """Stub DataRobot ROC curve object."""

    def __init__(self) -> None:
        self.roc_points = [
            {"fpr": 0.0, "tpr": 0.0, "threshold": 1.0},
            {"fpr": 0.1, "tpr": 0.7, "threshold": 0.5},
            {"fpr": 1.0, "tpr": 1.0, "threshold": 0.0},
        ]


class StubModel:
    """Stub DataRobot model object."""

    def __init__(self, model_id: str, model_type: str, metrics: dict):
        self.id = model_id
        self.model_type = model_type
        self.metrics = metrics
        self.featurelist_name = "Informative Features"
        self.sample_pct = 64.0

    def score(self, dataset_url: str) -> MagicMock:
        """Stub scoring method. Raises for fake URLs so integration tests can assert errors."""
        if "example.com" in (dataset_url or ""):
            raise Exception("404 client error: {'message': 'Not Found'}")
        return MagicMock(id=f"job_{self.id}_{hash(dataset_url) % 1000}")

    def request_feature_impact(self) -> None:
        """Stub request_feature_impact; no-op in tests."""

    def get_or_request_feature_impact(self) -> list[dict[str, Any]]:
        """Stub get_or_request_feature_impact; returns canned feature impact list."""
        return [
            {"featureName": "text_review", "impactNormalized": 1.0, "impactUnnormalized": 0.45},
            {
                "featureName": "product_category",
                "impactNormalized": 0.4,
                "impactUnnormalized": 0.18,
            },
        ]

    def get_roc_curve(self, source: str = "validation") -> "StubRocCurve":
        """Stub get_roc_curve; returns canned ROC curve object."""
        return StubRocCurve()


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
        self.metric = "AUC"
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

    def __init__(self, dataset_id: str, name: str = "stub_dataset", row_count: int = 200):
        self.id = dataset_id
        self.name = name
        self.created_at = "2025-01-01T00:00:00Z"
        self.row_count = row_count

    def get_as_dataframe(self) -> Any:
        """Return a stub DataFrame suitable for time-series eligibility checks."""
        import numpy as np
        import pandas as pd

        n = self.row_count
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        return pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "sales": np.random.default_rng(42).uniform(100, 1000, n),
                "store_id": [f"store_{i % 5}" for i in range(n)],
            }
        )


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
        self.client = MagicMock()


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
    stub_dataset = StubDataset(STUB_DATASET_ID, name="stub_dataset.csv", row_count=200)

    def get_dataset(dataset_id: str) -> StubDataset:
        if dataset_id == STUB_DATASET_ID:
            return stub_dataset
        raise Exception(f"404 client error: {{'message': 'Dataset {dataset_id} not found'}}")

    def list_datasets() -> list[StubDataset]:
        return [stub_dataset]

    # --- REST method stubs for client.get() / client.post() ---
    def stub_get(url: str, params: dict | None = None, **kwargs: Any) -> StubRestResponse:
        """Stub for client.get() REST calls."""
        if "predictionResults" in url:
            # Stub for get_prediction_history; return canned prediction rows.
            limit = (params or {}).get("limit", 100)
            rows = [
                {
                    "rowId": i,
                    "predictionValue": round(0.7 + i * 0.01, 2),
                    "timestamp": f"2024-01-{i + 1:02d}T00:00:00Z",
                }
                for i in range(min(limit, 5))
            ]
            return StubRestResponse({"data": rows, "next": None})
        if url.rstrip("/") == "deployments":
            # Stub for list_vector_databases; return mixed deployments.
            return StubRestResponse(
                {
                    "data": [
                        {
                            "id": "vdb_deployment_1",
                            "label": "Product VDB",
                            "status": "active",
                            "model": {"targetType": "VectorDatabase"},
                            "capabilities": {"supportsVectorDatabaseQuerying": True},
                        },
                        {
                            "id": "regular_deployment_1",
                            "label": "Regular Model",
                            "status": "active",
                            "model": {"targetType": "Binary"},
                            "capabilities": {},
                        },
                    ],
                    "next": None,
                }
            )
        if "useCases" in url:
            return StubRestResponse(
                {
                    "data": [
                        {"id": "uc_1", "name": "Test Use Case", "description": "Stub use case"},
                    ],
                    "count": 1,
                    "next": None,
                }
            )
        if "externalDataDrivers" in url and "tables" in url:
            return StubRestResponse(
                {"data": [{"name": "public.users"}, {"name": "public.orders"}]}
            )
        return StubRestResponse({"data": [], "next": None})

    def stub_post(url: str, json: dict | None = None, **kwargs: Any) -> StubRestResponse:
        """Stub for client.post() REST calls."""
        payload = json or {}
        # cuOpt predictions: data contains objects with "mode" key
        if "predictions" in url and isinstance(payload.get("data"), list):
            items = payload["data"]
            if items and isinstance(items[0], dict) and "mode" in items[0]:
                mode = items[0].get("mode", "solve")
                if mode == "validate":
                    return StubRestResponse({"valid": True, "errors": []})
                return StubRestResponse(
                    {
                        "data": [
                            {
                                "status": "optimal",
                                "objective_value": 42.0,
                                "solution": {"x": 1.0, "y": 0.0},
                                "solver_info": {"solver": "cuopt", "iterations": 10},
                            }
                        ]
                    }
                )
        # VDB predictions: payload has "query" key
        if "predictions" in url and "query" in payload:
            return StubRestResponse(
                {
                    "data": [
                        {"page_content": "Result 1", "metadata": {"source": "doc1"}, "score": 0.9},
                        {"page_content": "Result 2", "metadata": {"source": "doc2"}, "score": 0.8},
                    ]
                }
            )
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
    client.get = stub_get
    client.post = stub_post
    return client


def get_stub_classification_project() -> dict[str, Any]:
    """Return a stub project dict for integration tests (id matches test_create_dr_client)."""
    return {"project": SimpleNamespace(id=STUB_PROJECT_ID)}
