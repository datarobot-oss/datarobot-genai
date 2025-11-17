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

"""Client for interacting with Agent Tools deployments for chat and scoring."""

import json
import os
from collections.abc import Iterator
from typing import Any

import datarobot as dr
import openai
import pandas as pd
from datarobot.models.genai.agent.auth import get_authorization_context
from datarobot_predict.deployment import PredictionResult
from datarobot_predict.deployment import UnstructuredPredictionResult
from datarobot_predict.deployment import predict
from datarobot_predict.deployment import predict_unstructured
from openai.types import CompletionCreateParams
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk

from ..utils.urls import get_api_base


class ToolClient:
    """Client for interacting with Agent Tools Deployments.

    This class provides methods to call the custom model tool using various hooks:
    `score`, `score_unstructured`, and `chat`. When the `authorization_context` is set,
    the client automatically propagates it to the agent tool. The `authorization_context`
    is required for retrieving access tokens to connect to external services.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        authorization_context: dict[str, Any] | None = None,
    ):
        """Initialize the ToolClient.

        Args:
            api_key (str | None): API key for authentication. Defaults to the
                environment variable `DATAROBOT_API_TOKEN`.
            base_url (str | None): Base URL for the DataRobot API. Defaults to the
                environment variable `DATAROBOT_ENDPOINT` or 'app.datarobot.com'.
            authorization_context (dict[str, Any] | None): Authorization context to use
                for tool calls. If None, will attempt to get from ContextVar (for backward
                compatibility).
        """
        self.api_key = api_key or os.getenv("DATAROBOT_API_TOKEN")
        base_url = base_url or os.getenv("DATAROBOT_ENDPOINT") or "https://app.datarobot.com"
        base_url = get_api_base(base_url, deployment_id=None)
        self.base_url = base_url
        self._authorization_context = authorization_context

    @property
    def datarobot_api_endpoint(self) -> str:
        return self.base_url + "api/v2"

    def get_deployment(self, deployment_id: str) -> dr.Deployment:
        """Retrieve a deployment by its ID.

        Args:
            deployment_id (str): The ID of the deployment.

        Returns
        -------
            dr.Deployment: The deployment object.
        """
        dr.Client(self.api_key, self.datarobot_api_endpoint)
        return dr.Deployment.get(deployment_id=deployment_id)

    def call(
        self,
        deployment_id: str,
        payload: dict[str, Any],
        authorization_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> UnstructuredPredictionResult:
        """Run the custom model tool using score_unstructured hook.

        Args:
            deployment_id (str): The ID of the deployment.
            payload (dict[str, Any]): The input payload.
            authorization_context (dict[str, Any] | None): Authorization context to use.
                If None, uses the context from initialization or falls back to ContextVar.
            **kwargs: Additional keyword arguments.

        Returns
        -------
            UnstructuredPredictionResult: The response content and headers.
        """
        # Use explicit context, fall back to instance context, then ContextVar
        auth_ctx = authorization_context or self._authorization_context
        if auth_ctx is None:
            try:
                auth_ctx = get_authorization_context()
            except LookupError:
                auth_ctx = {}

        data = {
            "payload": payload,
            "authorization_context": auth_ctx,
        }
        return predict_unstructured(
            deployment=self.get_deployment(deployment_id),
            data=json.dumps(data),
            content_type="application/json",
            **kwargs,
        )

    def score(
        self, deployment_id: str, data_frame: pd.DataFrame, **kwargs: Any
    ) -> PredictionResult:
        """Run the custom model tool using score hook.

        Args:
            deployment_id (str): The ID of the deployment.
            data_frame (pd.DataFrame): The input data frame.
            **kwargs: Additional keyword arguments.

        Returns
        -------
            PredictionResult: The response content and headers.
        """
        return predict(
            deployment=self.get_deployment(deployment_id),
            data_frame=data_frame,
            **kwargs,
        )

    def chat(
        self,
        completion_create_params: CompletionCreateParams,
        model: str,
        authorization_context: dict[str, Any] | None = None,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Run the custom model tool with the chat hook.

        Args:
            completion_create_params (CompletionCreateParams): Parameters for the chat completion.
            model (str): The model to use.
            authorization_context (dict[str, Any] | None): Authorization context to use.
                If None, uses the context from initialization or falls back to ContextVar.

        Returns
        -------
            Union[ChatCompletion, Iterator[ChatCompletionChunk]]: The chat completion response.
        """
        # Use explicit context, fall back to instance context, then ContextVar
        auth_ctx = authorization_context or self._authorization_context
        if auth_ctx is None:
            try:
                auth_ctx = get_authorization_context()
            except LookupError:
                auth_ctx = {}

        extra_body = {
            "authorization_context": auth_ctx,
        }
        return openai.chat.completions.create(
            **completion_create_params,
            model=model,
            extra_body=extra_body,
        )
