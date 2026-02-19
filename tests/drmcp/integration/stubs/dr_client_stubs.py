# Copyright 2025 DataRobot, Inc. and its affiliates.
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

from unittest.mock import MagicMock


class MockModel:
    """Mock DataRobot model object."""

    def __init__(self, model_id: str, model_type: str, metrics: dict):
        self.id = model_id
        self.model_type = model_type
        self.metrics = metrics

    def score(self, dataset_url: str):
        """Mock scoring method."""
        return MagicMock(id=f"job_{self.id}_{hash(dataset_url) % 1000}")


class MockProject:
    """Mock DataRobot project object."""

    def __init__(self, project_id: str, models=None):
        self.id = project_id
        self._models = models or []

    def get_models(self, model_id: str):
        """Mock getting model method."""
        return self._models


class MockDRClient:
    """Mock DataRobot client object."""

    def __init__(self, projects=None):
        self.Project = MagicMock()
        self.Model = MagicMock()


def create_test_stub_dr_client() -> MockDRClient:
    """Create a stub DataRobot client with test project and models (no real API calls)."""
    client = MockDRClient()
    # Create test project with mock models
    project = MockProject(
        "test_project_123",
        models=[
            MockModel("model_1", "XGBoost", {"AUC": 0.95, "LogLoss": 0.12}),
            MockModel("model_2", "Random Forest", {"AUC": 0.92, "LogLoss": 0.15}),
            MockModel("model_3", "LightGBM", {"AUC": 0.94, "LogLoss": 0.13}),
        ],
    )
    # Create standalone model
    standalone_model = MockModel("standalone_model", "Neural Network", {"AUC": 0.88})

    def get_project(project_id):
        """Mock Project.get that returns appropriate project or raises exception."""
        if project_id == "test_project_123":
            return project
        elif project_id == "nonexistent_project":
            return None
        elif project_id == "test_project":
            raise Exception("DataRobot API error")
        else:
            return None

    def get_model(model_id):
        """Mock Model.get that returns appropriate model."""
        if model_id == "standalone_model":
            return standalone_model
        else:
            return None

    # Configure the stub methods
    client.Project.get = get_project
    client.Model.get = get_model
    return client


# Alias for backward compatibility
test_create_dr_client = create_test_stub_dr_client
