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

"""DataRobot API client and utilities for tools (e.g. predictive tools)."""

import logging
import os
from typing import Any

import datarobot as dr
from datarobot.context import Context as DRContext
from fastmcp.exceptions import ToolError

from datarobot_genai.drmcp.core.clients import resolve_token_from_headers
from datarobot_genai.drmcp.core.credentials import get_credentials

logger = logging.getLogger(__name__)


async def get_datarobot_access_token() -> str:
    """
    Get DataRobot API token from HTTP headers.

    Uses the same token extraction as core (auth headers and authorization
    context metadata). For use in tools only; core modules use get_sdk_client()
    from drmcp.core.clients.

    Returns
    -------
        API token string

    Raises
    ------
        ToolError: If no API token is found in headers
    """
    token = resolve_token_from_headers()
    if not token:
        logger.warning("DataRobot API token not found in headers")
        raise ToolError(
            "DataRobot API token not found in headers. "
            "Please provide it via 'Authorization' (Bearer), 'x-datarobot-api-token' headers."
        )
    return token


class DataRobotClient:
    """Client for interacting with DataRobot API in tools.

    Wraps the DataRobot Python SDK (datarobot package). Obtain the token
    via get_datarobot_access_token() and pass it to the constructor.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    def get_client(self) -> Any:
        """
        Configure the DataRobot SDK with this client's token and return the dr module.

        The returned value is the global datarobot module (dr) after
        dr.Client(token=..., endpoint=...) has been called. Use it as
        client.Project.list(), client.Deployment.get(...), etc.

        Returns
        -------
            The datarobot module (dr) configured for the current token and endpoint.
        """
        credentials = get_credentials()
        dr.Client(token=self._token, endpoint=credentials.datarobot.endpoint)
        # Avoid use-case context from trafaret affecting tool calls
        DRContext.use_case = None
        return dr


MODEL_EXTENSIONS = (".pkl", ".joblib", ".onnx", ".pth")
REQUIRED_FILES = ("custom.py", "requirements.txt")


def find_model_file_in_folder(model_folder: str) -> str | None:
    for name in os.listdir(model_folder):
        if name.lower().endswith(MODEL_EXTENSIONS):
            return os.path.join(model_folder, name)
    return None


def _select_execution_environment(
    client: Any, execution_environment_id: str | None
) -> tuple[Any, str | None]:
    if execution_environment_id:
        for e in client.ExecutionEnvironment.list():
            if e.id == execution_environment_id:
                vid = e.latest_successful_version.id if e.latest_successful_version else None
                return e, vid
        raise ValueError(f"Execution environment not found: {execution_environment_id}")
    envs = client.ExecutionEnvironment.list()
    scikit = [
        e
        for e in envs
        if "[DataRobot] Python" in e.name
        and "Scikit-Learn" in e.name
        and "3.11" in e.name
        and "Drop-In" in e.name
    ]
    if scikit:
        env = scikit[0]
    else:
        py_envs = [
            e
            for e in envs
            if "[DataRobot] Python" in e.name
            and ("Prediction" in e.name or "Drop-In" in e.name)
            and ("3.11" in e.name or "3.12" in e.name)
        ]
        if py_envs:
            env = py_envs[-1]
        else:
            py_envs = [
                e
                for e in envs
                if "[DataRobot] Python" in e.name
                and ("Prediction" in e.name or "Drop-In" in e.name)
            ]
            if not py_envs:
                raise ValueError("No suitable Python execution environment found")
            env = py_envs[-1]
    vid = env.latest_successful_version.id if env.latest_successful_version else None
    return env, vid


def _target_type_from_string(s: str) -> str:
    m = {
        "binary": dr.TARGET_TYPE.BINARY,
        "regression": dr.TARGET_TYPE.REGRESSION,
        "multiclass": dr.TARGET_TYPE.MULTICLASS,
    }
    return m.get(s.lower(), s)


def deploy_custom_model_impl(
    client: Any,
    model_folder: str,
    model_file_path: str,
    name: str,
    target_type: str,
    target_name: str,
    positive_class_label: str | None = None,
    negative_class_label: str | None = None,
    class_labels: list[str] | None = None,
    deployment_label: str | None = None,
    execution_environment_id: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    for f in REQUIRED_FILES:
        p = os.path.join(model_folder, f)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    if not os.path.isfile(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    model_basename = os.path.basename(model_file_path)
    files = [
        (os.path.join(model_folder, "custom.py"), "custom.py"),
        (os.path.join(model_folder, "requirements.txt"), "requirements.txt"),
        (model_file_path, model_basename),
    ]
    env, env_version_id = _select_execution_environment(client, execution_environment_id)
    create_kw: dict[str, Any] = {
        "name": name,
        "target_type": _target_type_from_string(target_type),
        "target_name": target_name,
        "description": description or "",
        "language": "python",
    }
    if positive_class_label is not None:
        create_kw["positive_class_label"] = positive_class_label
    if negative_class_label is not None:
        create_kw["negative_class_label"] = negative_class_label
    if class_labels is not None:
        create_kw["class_labels"] = class_labels
    custom_model = client.CustomInferenceModel.create(**create_kw)
    version = client.CustomModelVersion.create_clean(
        custom_model_id=custom_model.id,
        base_environment_id=env.id,
        base_environment_version_id=env_version_id,
        files=files,
        is_major_update=True,
    )
    build_info = client.CustomModelVersionDependencyBuild.start_build(
        custom_model_id=custom_model.id,
        custom_model_version_id=version.id,
        max_wait=3600,
    )
    if build_info is None:
        build_info = client.CustomModelVersionDependencyBuild.get_build_info(
            custom_model_id=custom_model.id,
            custom_model_version_id=version.id,
        )
    if build_info.build_status == "failed":
        log = build_info.get_log()
        raise RuntimeError(f"Dependency build failed: {log}")
    registered_model_name = f"{name} ({custom_model.id})"
    rmv = client.RegisteredModelVersion.create_for_custom_model_version(
        custom_model_version_id=version.id,
        registered_model_name=registered_model_name,
        description=description or name,
    )
    label = deployment_label or name
    prediction_servers = client.PredictionServer.list()
    if not prediction_servers:
        raise ValueError("No prediction servers available for deployment.")
    ps_id = prediction_servers[0].id
    deployment = client.Deployment.create_from_registered_model_version(
        model_package_id=rmv.id,
        label=label,
        description=description or "",
        default_prediction_server_id=ps_id,
        max_wait=600,
    )
    return {
        "deployment_id": deployment.id,
        "label": deployment.label,
        "custom_model_id": custom_model.id,
        "custom_model_version_id": version.id,
        "registered_model_version_id": rmv.id,
    }
