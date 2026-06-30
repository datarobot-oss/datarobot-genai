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

import polars as pl

from datarobot_genai.drmcp.test_utils.stubs.stub_rest_response import StubRestResponse
from datarobot_genai.drmcp.test_utils.stubs.workload_stubs import workload_stub_delete
from datarobot_genai.drmcp.test_utils.stubs.workload_stubs import workload_stub_get
from datarobot_genai.drmcp.test_utils.stubs.workload_stubs import workload_stub_patch
from datarobot_genai.drmcp.test_utils.stubs.workload_stubs import workload_stub_post

# Project id used by test_create_dr_client(); use for integration tests with stubs.
STUB_PROJECT_ID = "test_project_123"

# Dataset id used by test_create_dr_client(); use for integration tests with stubs.
STUB_DATASET_ID = "stub_dataset_id"

# Catalog dataset id from ``classification_predict_dataset`` when using stub token (conftest).
STUB_PREDICT_CATALOG_DATASET_ID = "stub_predict_dataset_id"

# Batch prediction job id returned by stub ``BatchPredictionJob.score`` / ``get``.
STUB_BATCH_PREDICTION_JOB_ID = "stub_batch_prediction_job_id"

# Use case id used by test_create_dr_client(); use for integration tests with stubs.
STUB_USE_CASE_ID = "stub_use_case_id"

# VDB deployment id used by test_create_dr_client(); use for integration tests with stubs.
STUB_VDB_DEPLOYMENT_ID = "stub_vdb_deployment_id"


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

    def request_predictions(
        self, dataset: Any = None, dataset_id: Any = None, **kwargs: Any
    ) -> MagicMock:
        """Stub ``Model.request_predictions`` (used by modeling_score_dataset)."""
        return MagicMock(id=f"pred_job_{self.id}")

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
    """Stub datetime partitioning for deployment_get_info time_series_config."""

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
        self.project_name = f"Project {project_id}"

    def get_models(self) -> list:
        """Stub get_models (matches real API: no arguments)."""
        return self._models

    def get_model_records(self, limit: int = 100, offset: int = 0, **kwargs: Any) -> list:
        """Stub paginated model list (matches ``Project.get_model_records`` slice semantics)."""
        return self._models[offset : offset + limit]

    def upload_dataset_from_catalog(self, dataset_id: str, **kwargs: Any) -> Any:
        """Stub ``Project.upload_dataset_from_catalog`` (used by modeling_score_dataset)."""
        return SimpleNamespace(id=f"prediction_dataset_for_{dataset_id}")


class StubDeployment:
    """Stub DataRobot deployment object."""

    def __init__(
        self, deployment_id: str, project_id: str = "test_project_123", model_id: str = "model_1"
    ):
        self.id = deployment_id
        self.label = f"Deployment {deployment_id}"
        self.model = {"project_id": project_id, "id": model_id}
        self.status = "active"

    def get_features(self) -> list:
        """
        Stub get_features;
        returns: list of feature dicts
        (must include feature_type for deployment_generate_prediction_sample).
        """
        return [
            {"name": "text_review", "importance": 1, "feature_type": "text"},
            {"name": "product_category", "importance": 0, "feature_type": "categorical"},
        ]

    def get_capabilities(self) -> MagicMock:
        """Stub get_capabilities."""
        return MagicMock()


class StubCatalogDatasetFeature:
    """Stand-in for SDK ``DatasetFeature`` (catalog / allFeaturesDetails) in stub tests."""

    def __init__(self, name: str, feature_type: str = "Categorical", **kwargs: Any) -> None:
        self.name = name
        self.feature_type = feature_type
        self.unique_count = kwargs.get("unique_count", 5)
        self.na_count = kwargs.get("na_count", 0)
        self.date_format = kwargs.get("date_format")
        self.min = kwargs.get("min")
        self.max = kwargs.get("max")
        self.mean = kwargs.get("mean")
        self.median = kwargs.get("median")
        self.std_dev = kwargs.get("std_dev")
        self.low_information = kwargs.get("low_information", False)
        self.time_series_eligible = kwargs.get("time_series_eligible", False)
        self.time_series_eligibility_reason = kwargs.get("time_series_eligibility_reason", "")
        self.time_step = kwargs.get("time_step")
        self.time_unit = kwargs.get("time_unit")
        self.target_leakage = kwargs.get("target_leakage", "FALSE")
        self.target_leakage_reason = kwargs.get("target_leakage_reason")

    def get_histogram(self, bin_limit: int | None = None, key_name: str | None = None) -> Any:
        return SimpleNamespace(
            plot=[
                {
                    "label": "stub-bin",
                    "count": max(1, int(self.unique_count or 1)),
                    "target": None,
                }
            ]
        )


class StubDataset:
    """Stub DataRobot dataset object."""

    def __init__(
        self,
        dataset_id: str,
        name: str = "stub_dataset",
        row_count: int = 200,
    ):
        self.id = dataset_id
        self.name = name
        self.created_at = "2025-01-01T00:00:00Z"
        self.row_count = row_count

    def get_details(self) -> Any:
        """Stub ``Dataset.get_details`` (catalog extended profile flag)."""
        return SimpleNamespace(data_persisted=True)

    def _stub_catalog_features(self) -> list[StubCatalogDatasetFeature]:
        """Feature names align with ``get_as_dataframe`` (date, sales, store_id)."""
        rc = self.row_count
        return [
            StubCatalogDatasetFeature(
                "date",
                "Categorical",
                unique_count=rc,
                na_count=0,
                time_series_eligible=True,
                time_series_eligibility_reason="suitable",
            ),
            StubCatalogDatasetFeature(
                "sales",
                "Float",
                unique_count=min(100, max(1, rc)),
                na_count=0,
                min=100.0,
                max=1000.0,
                mean=550.0,
                median=520.0,
                std_dev=100.0,
            ),
            StubCatalogDatasetFeature(
                "store_id",
                "Categorical",
                unique_count=min(5, max(1, rc)),
                na_count=0,
            ),
        ]

    def iterate_all_features(
        self,
        offset: int | None = None,
        limit: int | None = None,
        order_by: str | None = None,
    ) -> Any:
        """Stub ``Dataset.iterate_all_features`` for catalog API-backed EDA in tools."""
        features = self._stub_catalog_features()
        if offset:
            features = features[offset:]
        if limit is not None:
            features = features[:limit]
        return iter(features)

    def get_as_dataframe(self) -> Any:
        """Return a stub pandas DataFrame suitable for time-series eligibility checks.

        Returns pandas because the real DataRobot SDK's get_as_dataframe() returns pandas.
        Production code converts to polars internally via pl.from_pandas().
        """
        import random

        n = self.row_count
        rng = random.Random(42)
        dates = pl.date_range(
            pl.date(2023, 1, 1), pl.date(2023, 1, 1) + pl.duration(days=n - 1), eager=True
        )
        df = pl.DataFrame(
            {
                "date": dates.cast(pl.Utf8),
                "sales": [rng.uniform(100, 1000) for _ in range(n)],
                "store_id": [f"store_{i % 5}" for i in range(n)],
            }
        )
        return df.to_pandas()

    def get_raw_sample_data(self) -> Any:
        """Return a small stub sample DataFrame (subset, avoids full download)."""
        return self.get_as_dataframe()


class StubUseCase:
    """Stub DataRobot use case object."""

    def __init__(self, use_case_id: str, name: str = "Stub Use Case"):
        self.id = use_case_id
        self.name = name

    def list_datasets(self) -> list:
        """Return stub datasets associated with this use case."""
        return [StubDataset("uc_dataset_1", name="Use Case Dataset")]

    def list_deployments(self) -> list:
        """Return stub deployments associated with this use case."""
        return [
            StubDeployment("uc_deployment_1", project_id="test_project_123", model_id="model_1")
        ]

    def list_projects(self) -> list:
        """Return stub projects (experiments) associated with this use case."""
        return [StubProject("test_project_123")]


class StubBatchJob:
    """Stub batch prediction job (submit, status poll, download) for MCP integration tests."""

    def __init__(self, job_id: str = STUB_BATCH_PREDICTION_JOB_ID) -> None:
        self.id = job_id

    def get_status(self) -> dict[str, Any]:
        return {
            "status": "COMPLETED",
            "links": {"download": "https://stub.app.example/api/v2/batchPredictions/download"},
            "percentage_completed": 100.0,
            "elapsed_time_sec": 0,
            "status_details": "",
            "job_spec": {"output_settings": {"type": "localFile"}},
        }

    def download(self, buf: Any, timeout: int = 120, read_timeout: int = 660) -> None:
        _ = (timeout, read_timeout)
        buf.write(b"text_review,sentiment_PREDICTION\nstub,positive\n")


class StubBatchPredictionJobAPI:
    """Subset of ``datarobot.BatchPredictionJob`` used by predictive batch tools."""

    @staticmethod
    def score(*args: Any, **kwargs: Any) -> StubBatchJob:
        _ = (args, kwargs)
        return StubBatchJob(STUB_BATCH_PREDICTION_JOB_ID)

    @staticmethod
    def get(job_id: str) -> StubBatchJob:
        return StubBatchJob(job_id.strip())


class StubDataStore:
    """Stub DataRobot datastore object."""

    def __init__(self, datastore_id: str, canonical_name: str = "stub_datastore"):
        self.id = datastore_id
        self.canonical_name = canonical_name
        self.creator_id = "stub_creator"
        self.params = {"type": "jdbc", "driver": "postgresql"}


class StubDRClient:
    """Stub DataRobot client for tests (canned responses; use with dr_client_stubs)."""

    def __init__(self, projects: list[StubProject] | None = None):
        self.Project = MagicMock()
        self.Model = MagicMock()
        self.Deployment = MagicMock()
        self.Dataset = MagicMock()
        self.DataStore = MagicMock()
        self.UseCase = MagicMock()
        self.client = MagicMock()
        self.BatchPredictionJob = StubBatchPredictionJobAPI()
        self.stub_rest_get: Any = None
        self.stub_rest_post: Any = None
        self.stub_rest_patch: Any = None
        self.stub_rest_delete: Any = None


def test_create_dr_client() -> StubDRClient:
    """Create a stub DataRobot client with test project and models."""
    client = StubDRClient()

    # Metrics shape: models_get_bestmodel expects metrics[metric].get("validation")
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
        if dataset_id in (STUB_DATASET_ID, STUB_PREDICT_CATALOG_DATASET_ID):
            return stub_dataset
        raise Exception(f"404 client error: {{'message': 'Dataset {dataset_id} not found'}}")

    def list_datasets() -> list[StubDataset]:
        return [stub_dataset]

    def _catalog_stub_datasets() -> list[StubDataset]:
        """Several catalog rows so ``Dataset.iterate`` can honor offset/limit in tests."""
        return [
            StubDataset(f"stub_cat_{i}", name=f"catalog_item_{i}.csv", row_count=10)
            for i in range(5)
        ]

    def iterate_datasets(offset: int = 0, limit: int | None = None) -> Any:
        """Stub ``Dataset.iterate``; supports offset/limit for ``catalog_list_datasets``."""
        items = _catalog_stub_datasets()[offset:]
        if limit is not None:
            items = items[:limit]
        return iter(items)

    # --- DataStore stubs ---
    stub_datastore = StubDataStore("stub_datastore_id", canonical_name="Test PostgreSQL")

    def list_datastores() -> list[StubDataStore]:
        return [stub_datastore]

    # --- UseCase stubs ---
    def get_use_case(use_case_id: str) -> StubUseCase:
        """Stub UseCase.get; returns a stub use case or raises for unknown IDs."""
        if use_case_id == "nonexistent_use_case":
            raise Exception(f"404 client error: {{'message': 'Use case {use_case_id} not found'}}")
        name = "Stub Use Case" if use_case_id == STUB_USE_CASE_ID else f"Use Case {use_case_id}"
        return StubUseCase(use_case_id, name=name)

    # --- REST method stubs for dr_module.client.get_client() ---
    def _stub_get_non_workload(url: str, params: dict | None) -> StubRestResponse:
        response = StubRestResponse({"data": [], "next": None})
        if "predictionResults" in url:
            limit = (params or {}).get("limit", 100)
            rows = [
                {
                    "rowId": i,
                    "predictionValue": round(0.7 + i * 0.01, 2),
                    "timestamp": f"2024-01-{i + 1:02d}T00:00:00Z",
                }
                for i in range(min(limit, 5))
            ]
            response = StubRestResponse({"data": rows, "next": None})
        elif "externalDataStores" in url:
            response = StubRestResponse(
                {
                    "data": [
                        {
                            "id": stub_datastore.id,
                            "canonicalName": stub_datastore.canonical_name,
                            "creatorId": stub_datastore.creator_id,
                            "params": dict(stub_datastore.params),
                        }
                    ],
                    "next": None,
                }
            )
        elif "externalDataDrivers" in url and "tables" in url:
            response = StubRestResponse(
                {"data": [{"name": "public.users"}, {"name": "public.orders"}]}
            )
        elif url.rstrip("/") == "deployments":
            all_deployments: list[dict[str, Any]] = [
                {
                    "id": STUB_VDB_DEPLOYMENT_ID,
                    "label": "Stub VDB Deployment",
                    "status": "active",
                    "capabilities": {"supportsVectorDatabaseQuerying": True},
                    "model": {"targetType": "VectorDatabase"},
                },
                {
                    "id": "stub_regular_deployment_id",
                    "label": "Regular Deployment",
                    "status": "active",
                    "capabilities": {"supportsVectorDatabaseQuerying": False},
                    "model": {"targetType": "Binary"},
                },
            ]
            model_target_type = (params or {}).get("modelTargetType")
            if model_target_type:
                all_deployments = [
                    d
                    for d in all_deployments
                    if isinstance(d.get("model"), dict)
                    and d["model"].get("targetType") == model_target_type
                ]
            response = StubRestResponse({"data": all_deployments, "next": None})
        elif "useCases" in url:
            data: list[dict] = [
                {"id": STUB_USE_CASE_ID, "name": "Stub Use Case"},
                {"id": "stub_use_case_id_2", "name": "Another Use Case"},
            ]
            search = (params or {}).get("search")
            if search:
                data = [uc for uc in data if search.lower() in uc["name"].lower()]
            response = StubRestResponse({"data": data, "next": None})
        return response

    def stub_get(url: str, params: dict | None = None, **kwargs: Any) -> StubRestResponse:
        """Stub for rest_client.get() REST calls."""
        workload_response = workload_stub_get(url, params, **kwargs)
        if workload_response is not None:
            return workload_response
        return _stub_get_non_workload(url, params)

    def stub_post(url: str, json: dict | None = None, **kwargs: Any) -> StubRestResponse:
        """Stub for rest_client.post() REST calls."""
        workload_response = workload_stub_post(url, json, **kwargs)
        if workload_response is not None:
            return workload_response

        if "deployments" in url and "predictions" in url:
            # vdb_query calls POST deployments/{id}/predictions/
            return StubRestResponse(
                {
                    "data": [
                        {
                            "page_content": "DataRobot is a leading AI platform.",
                            "metadata": {"source": "doc1.pdf", "page": 1},
                            "score": 0.92,
                        },
                        {
                            "page_content": "AutoML makes machine learning accessible.",
                            "metadata": {"source": "doc2.pdf", "page": 3},
                            "score": 0.87,
                        },
                    ]
                }
            )
        if "externalDataStores" in url and "previewQuery" in url:
            return StubRestResponse(
                {
                    "records": [{"id": 1, "name": "test"}],
                    "columns": ["id", "name"],
                }
            )
        return StubRestResponse({"data": []})

    def stub_patch(url: str, json: dict | None = None, **kwargs: Any) -> StubRestResponse:
        """Stub for rest_client.patch() REST calls."""
        workload_response = workload_stub_patch(url, json, **kwargs)
        if workload_response is not None:
            return workload_response
        return StubRestResponse({})

    def stub_delete(url: str, **kwargs: Any) -> StubRestResponse:
        """Stub for rest_client.delete() REST calls."""
        workload_response = workload_stub_delete(url, **kwargs)
        if workload_response is not None:
            return workload_response
        return StubRestResponse({})

    # Configure the stub methods
    client.Project.get = get_project
    client.Model.get = get_model
    client.Deployment.get = get_deployment
    client.Dataset.get = get_dataset
    client.Dataset.list = list_datasets
    client.Dataset.iterate = iterate_datasets
    client.DataStore.list = list_datastores
    client.UseCase.get = get_use_case
    # Store REST stubs on the client so integration_mcp_server can wire them
    # onto mock_rest after replacing client.client.
    client.stub_rest_get = stub_get
    client.stub_rest_post = stub_post
    client.stub_rest_patch = stub_patch
    client.stub_rest_delete = stub_delete
    return client


def get_stub_classification_project() -> dict[str, Any]:
    """Return a stub project dict for integration tests (id matches test_create_dr_client)."""
    return {"project": SimpleNamespace(id=STUB_PROJECT_ID)}
