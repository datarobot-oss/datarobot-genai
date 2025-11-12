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
from collections.abc import Generator
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import datarobot as dr
import pytest

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import DrPrompt
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import DrPromptVersion
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import DrVariable


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to the test data directory."""
    return Path("tests/data")


# Only used for fixtures, the tests use the MCP session directly
@pytest.fixture(scope="session")
def dr_client() -> Any:
    """Get DataRobot client for integration tests."""
    return get_sdk_client()


@pytest.fixture(scope="session")
def timeseries_regression_dataset_name() -> str:
    return "timeseries_regression_train.csv"


@pytest.fixture(scope="session")
def timeseries_regression_project_name() -> str:
    return "MCP Test TS Regression Project"


@pytest.fixture(scope="session")
def timeseries_regression_project(
    dr_client: Any,
    test_data_dir: Path,
    timeseries_regression_dataset_name: str,
    timeseries_regression_project_name: str,
) -> Generator[dict[str, Any], None, None]:
    """Create a time series regression project and return the best model deployment."""
    deployment_label = "MCP Test TS Regression Deployment"

    # First, check if deployment already exists
    try:
        deployments = dr.Deployment.list()
        for deployment in deployments:
            if deployment.label and deployment.label.startswith(deployment_label):
                print(f"Reusing existing deployment: {deployment.id}")
                # Get the model and project info
                if deployment.model:
                    model = dr.Model.get(
                        project=str(deployment.model["project_id"]),
                        model_id=str(deployment.model["id"]),
                    )
                    project = dr.Project.get(str(deployment.model["project_id"]))

                yield {
                    "project": project,
                    "model": model,
                    "deployment": deployment,
                    "deployment_id": deployment.id,
                    "source_dataset_id": project.catalog_id,
                }
                return  # Exit early, don't clean up existing deployment
    except Exception as e:
        print(f"Error checking for existing deployments: {e}")

    # If no existing deployment found, create new one
    train_file = test_data_dir / timeseries_regression_dataset_name
    dataset: Any = dr.Dataset.create_from_file(file_path=train_file)

    # Create project
    project = dr.Project.create_from_dataset(
        dataset.id, project_name=timeseries_regression_project_name
    )

    # Set up time series partitioning
    datetime_spec = dr.DatetimePartitioningSpecification(
        datetime_partition_column="date",
        use_time_series=True,
        forecast_window_start=1,
        forecast_window_end=7,
    )

    # Start modeling with time series configuration
    project.analyze_and_model(  # type: ignore[attr-defined]
        target="sales",
        partitioning_method=datetime_spec,
        mode=dr.enums.AUTOPILOT_MODE.MANUAL,
    )

    # Train just one model instead of full autopilot
    blueprints = project.get_blueprints()  # type: ignore[attr-defined]
    # Get a simple time series blueprint (usually first one is good)
    blueprint = blueprints[0]

    # Train the model
    model_job = project.train_datetime(blueprint.id)  # type: ignore[attr-defined]
    model = model_job.get_result_when_complete()

    # Get available prediction servers and use the first one
    prediction_servers = dr.PredictionServer.list()
    if not prediction_servers:
        raise RuntimeError("No prediction servers available")

    # Create deployment from leaderboard
    deployment = dr.Deployment.create_from_learning_model(
        model_id=model.id,
        label=deployment_label,  # No timestamp
        description="Integration test deployment for time series regression",
        default_prediction_server_id=prediction_servers[0].id,
    )

    yield {
        "project": project,
        "model": model,
        "deployment": deployment,
        "deployment_id": deployment.id,
        "source_dataset_id": dataset.id,
    }

    # Cleanup - commented out to preserve deployment for future test runs
    # try:
    #     deployment.delete()
    # except Exception as e:
    #     print(f"Warning: Could not delete deployment {deployment.id}: {e}")

    # try:
    #     project.delete()
    # except Exception as e:
    #     print(f"Warning: Could not delete project {project.id}: {e}")

    # try:
    #     dataset.delete()
    # except Exception as e:
    #     print(f"Warning: Could not delete dataset {dataset.id}: {e}")


@pytest.fixture(scope="session")
def multiseries_regression_dataset_name() -> str:
    return "multiseries_regression_train.csv"


@pytest.fixture(scope="session")
def multiseries_regression_project_name() -> str:
    return "MCP Test Multiseries TS Regression Project"


@pytest.fixture(scope="session")
def multiseries_regression_project(
    dr_client: Any,
    test_data_dir: Path,
    multiseries_regression_dataset_name: str,
    multiseries_regression_project_name: str,
) -> Generator[dict[str, Any], None, None]:
    """Create a multiseries time series regression project and return the best model deployment."""
    deployment_label = "MCP Test Multiseries TS Regression Deployment"

    # First, check if deployment already exists
    try:
        deployments = dr.Deployment.list()
        for deployment in deployments:
            if deployment.label and deployment.label.startswith(deployment_label):
                print(f"Reusing existing multiseries deployment: {deployment.id}")
                # Get the model and project info
                if deployment.model:
                    model = dr.Model.get(
                        project=str(deployment.model["project_id"]),
                        model_id=str(deployment.model["id"]),
                    )
                    project = dr.Project.get(str(deployment.model["project_id"]))

                yield {
                    "project": project,
                    "model": model,
                    "deployment": deployment,
                    "deployment_id": deployment.id,
                    "source_dataset_id": project.catalog_id,
                }
                return  # Exit early, don't clean up existing deployment
    except Exception as e:
        print(f"Error checking for existing multiseries deployments: {e}")

    # If no existing deployment found, create new one
    train_file = test_data_dir / multiseries_regression_dataset_name

    # Create multiseries project
    dataset: Any = dr.Dataset.create_from_file(file_path=train_file)
    project = dr.Project.create_from_dataset(
        dataset.id, project_name=multiseries_regression_project_name
    )

    # Set up multiseries time series partitioning
    datetime_spec = dr.DatetimePartitioningSpecification(
        datetime_partition_column="date",
        use_time_series=True,
        multiseries_id_columns=["store_id"],
        forecast_window_start=1,
        forecast_window_end=7,
    )

    # Start modeling with multiseries time series configuration
    project.analyze_and_model(  # type: ignore[attr-defined]
        target="sales",
        partitioning_method=datetime_spec,
        mode=dr.enums.AUTOPILOT_MODE.MANUAL,
    )

    # Train just one model instead of full autopilot
    blueprints = project.get_blueprints()  # type: ignore[attr-defined]
    # Get a simple time series blueprint (usually first one is good)
    blueprint = blueprints[0]

    # Train the model
    model_job = project.train_datetime(blueprint.id)  # type: ignore[attr-defined]
    model = model_job.get_result_when_complete()

    # Get available prediction servers and use the first one
    prediction_servers = dr.PredictionServer.list()
    if not prediction_servers:
        raise RuntimeError("No prediction servers available")

    # Create deployment from leaderboard
    deployment = dr.Deployment.create_from_learning_model(
        model_id=model.id,
        label=deployment_label,  # No timestamp
        description="Integration test deployment for multiseries time series regression",
        default_prediction_server_id=prediction_servers[0].id,
    )

    yield {
        "project": project,
        "model": model,
        "deployment": deployment,
        "deployment_id": deployment.id,
        "source_dataset_id": dataset.id,
    }

    # Cleanup - commented out to preserve deployment for future test runs
    # try:
    #     deployment.delete()
    # except Exception as e:
    #     print(f"Warning: Could not delete deployment {deployment.id}: {e}")

    # try:
    #     project.delete()
    # except Exception as e:
    #     print(f"Warning: Could not delete project {project.id}: {e}")

    # try:
    #     dataset.delete()
    # except Exception as e:
    #     print(f"Warning: Could not delete dataset {dataset.id}: {e}")


@pytest.fixture(scope="session")
def classification_dataset_name() -> str:
    return "text_classification_train.csv"


@pytest.fixture(scope="session")
def classification_project_name() -> str:
    return "MCP Test Text Classification Project"


@pytest.fixture(scope="session")
def classification_project(
    dr_client: Any,
    test_data_dir: Path,
    classification_dataset_name: str,
    classification_project_name: str,
) -> Generator[dict[str, Any], None, None]:
    """Create a text classification project and return the best model deployment."""
    deployment_label = "MCP Test Text Classification Deployment"

    # First, check if deployment already exists
    try:
        deployments = dr.Deployment.list()
        for deployment in deployments:
            if deployment.label and deployment.label.startswith(deployment_label):
                print(f"Reusing existing text deployment: {deployment.id}")
                # Get the model and project info
                if deployment.model:
                    model = dr.Model.get(
                        project=str(deployment.model["project_id"]),
                        model_id=str(deployment.model["id"]),
                    )
                    project = dr.Project.get(str(deployment.model["project_id"]))

                yield {
                    "project": project,
                    "model": model,
                    "deployment": deployment,
                    "deployment_id": deployment.id,
                    "source_dataset_id": project.catalog_id,
                }
                return  # Exit early, don't clean up existing deployment
    except Exception as e:
        print(f"Error checking for existing text deployments: {e}")

    # If no existing deployment found, create new one
    train_file = test_data_dir / classification_dataset_name

    # Create a Dataset from the file
    dataset: Any = dr.Dataset.create_from_file(file_path=train_file)

    # Create a Project from the Dataset's ID
    project = dr.Project.create_from_dataset(dataset.id, project_name=classification_project_name)

    # Start modeling for text classification
    project.analyze_and_model(  # type: ignore[attr-defined]
        target="sentiment",
        mode=dr.enums.AUTOPILOT_MODE.MANUAL,
    )

    # Get blueprints and find a text-based one if possible
    blueprints = project.get_blueprints()  # type: ignore[attr-defined]
    # Look for blueprints that support text mining/NLP
    text_blueprint = None
    for bp in blueprints:
        if any(keyword in bp.model_type.lower() for keyword in ["text", "tfidf", "word", "ngram"]):
            text_blueprint = bp
            break

    # If no specific text blueprint found, use the first one
    if text_blueprint is None:
        text_blueprint = blueprints[0]

    # Train the model
    model_job_id = project.train(text_blueprint.id)  # type: ignore[attr-defined]
    model_job = dr.ModelJob.get(project.id, model_job_id)  # type: ignore[attr-defined]
    model = model_job.get_result_when_complete()  # type: ignore[no-untyped-call]

    # Get available prediction servers and use the first one
    prediction_servers = dr.PredictionServer.list()
    if not prediction_servers:
        raise RuntimeError("No prediction servers available")

    # Create deployment from leaderboard
    deployment = dr.Deployment.create_from_learning_model(
        model_id=model.id,
        label=deployment_label,
        description="Integration test deployment for text classification",
        default_prediction_server_id=prediction_servers[0].id,
    )

    yield {
        "project": project,
        "model": model,
        "deployment": deployment,
        "deployment_id": deployment.id,
        "source_dataset_id": dataset.id,
    }
    # Cleanup - commented out to preserve deployment for future test runs
    # try:
    #     deployment.delete()
    # except Exception as e:
    #     print(f"Warning: Could not delete deployment {deployment.id}: {e}")

    # try:
    #     project.delete()
    # except Exception as e:
    #     print(f"Warning: Could not delete project {project.id}: {e}")

    # try:
    #     dataset.delete()
    # except Exception as e:
    #     print(f"Warning: Could not delete dataset {dataset.id}: {e}")


@pytest.fixture(scope="session")
def classification_predict_dataset_name() -> str:
    return "text_classification_predict.csv"


@pytest.fixture(scope="session")
def classification_predict_file_path(
    test_data_dir: Path,
    classification_predict_dataset_name: str,
) -> Path:
    """Return the path to the text classification prediction CSV file."""
    return test_data_dir / classification_predict_dataset_name


@pytest.fixture(scope="session")
def classification_predict_dataset(
    dr_client: Any,
    classification_predict_dataset_name: str,
    classification_predict_file_path: Path,
) -> Generator[dict[str, Any], None, None]:
    """Upload text classification prediction dataset to AI Catalog."""
    # Check if dataset already exists in AI Catalog
    try:
        datasets = dr.Dataset.list()
        for dataset in datasets:
            if dataset.name == classification_predict_dataset_name:
                print(f"Reusing existing prediction dataset: {dataset.id}")
                yield {
                    "dataset": dataset,
                    "dataset_id": dataset.id,
                    "file_path": str(classification_predict_file_path),
                }
                return
    except Exception as e:
        print(f"Error checking for existing prediction datasets: {e}")

    # Upload new dataset to AI Catalog
    try:
        dataset: Any = dr.Dataset.create_from_file(file_path=classification_predict_file_path)
        print(f"Uploaded new prediction dataset: {dataset.id}")

        yield {
            "dataset": dataset,
            "dataset_id": dataset.id,
            "file_path": str(classification_predict_file_path),
        }

        # Cleanup - commented out to preserve dataset for future test runs
        # try:
        #     dataset.delete()
        # except Exception as e:
        #     print(f"Warning: Could not delete prediction dataset {dataset.id}: {e}")

    except Exception as e:
        print(f"Error uploading prediction dataset: {e}")
        raise


@pytest.fixture
def prompt_template_id_ok() -> str:
    return "69086ea4834952718366b2ce"


@pytest.fixture
def prompt_template_version_id_ok() -> str:
    return "69086ea4b65d70489c5b198d"


@pytest.fixture
def get_prompt_template_mock(
    prompt_template_id_ok: str, prompt_template_version_id_ok: str
) -> Iterator[None]:
    """Set up all API endpoint mocks."""
    dr_prompt_version = DrPromptVersion(
        id=prompt_template_version_id_ok,
        version=3,
        prompt_text="Write greeting for {{name}} in max {{sentences}} sentences.",
        variables=[
            DrVariable(name="name", description="Person name"),
            DrVariable(name="sentences", description="Number of sentences"),
        ],
    )
    dr_prompt = DrPrompt(
        id=prompt_template_id_ok,
        name="Dummy prompt name",
        description="Dummy description",
    )
    dr_prompt.get_latest_version = lambda: dr_prompt_version

    with patch(
        "datarobot_genai.drmcp.core.dynamic_prompts.register.get_datarobot_prompt_templates",
        return_value=[dr_prompt],
    ):
        yield
