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

import logging
import os
from collections.abc import Callable

from ag_ui.core import Event
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi.responses import StreamingResponse
from nat.builder.context import Context
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.api_server import ChatRequest
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.front_ends.fastapi.response_helpers import generate_single_response
from nat.front_ends.fastapi.response_helpers import generate_streaming_response_as_str
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.runtime.session import SessionManager
from pydantic import BaseModel
from pydantic import Field

from datarobot_genai.dragent.request import DRAgentRunAgentInput
from datarobot_genai.dragent.response import DRAgentChatResponse
from datarobot_genai.dragent.response import DRAgentChatResponseChunk
from datarobot_genai.dragent.response import DRAgentEventResponse
from datarobot_genai.dragent.step_adaptor import DRAgentEmptyStepAdaptor

logger = logging.getLogger(__name__)


def prefix_mount_path(endpoint: str) -> str:
    mount_path = os.getenv("URL_PREFIX", "")

    if mount_path == "/":
        return endpoint

    if mount_path.endswith("/"):
        mount_path = mount_path[:-1]

    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    return mount_path + endpoint


class DRAgentFastApiFrontEndPluginWorker(FastApiFrontEndPluginWorker):
    def get_step_adaptor(self) -> StepAdaptor:
        return DRAgentEmptyStepAdaptor(self.front_end_config.step_adaptor)

    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder) -> None:
        # For now, only subset of routes exposed by the default fastapi front end are exposed by the
        # dragent fastapi frontend.
        # We will be adding more routes as we figure out integrations
        await self.add_default_route(app, await self._create_session_manager(builder))
        await self.add_health_route(app)

    async def add_health_route(self, app: FastAPI) -> None:
        """Override to apply prefix_mount_path to the health endpoint."""

        class HealthResponse(BaseModel):
            status: str = Field(description="Health status of the server")

        async def health_check() -> HealthResponse:
            return HealthResponse(status="healthy")

        app.add_api_route(
            path=prefix_mount_path("/health"),
            endpoint=health_check,
            methods=["GET"],
            response_model=HealthResponse,
            description="Health check endpoint for liveness/readiness probes",
            tags=["Health"],
        )
        logger.info("Added health check endpoint at %s", prefix_mount_path("/health"))

    async def add_route(
        self,
        app: FastAPI,
        endpoint: FastApiFrontEndConfig.EndpointBase,
        session_manager: SessionManager,
    ) -> None:
        # Copy of add_route from the base class to limit the routes to our contract: AG-UI

        # TODO: this has to be moved to a base class upstream
        def add_context_headers_to_response(response: Response) -> None:
            """Add context-based headers to response if available."""
            observability_trace_id = Context.get().observability_trace_id
            if observability_trace_id:
                response.headers["Observability-Trace-Id"] = observability_trace_id

        # TODO: this has to be moved to a base class upstream
        response_500 = {
            "description": "Internal Server Error",
            "content": {
                "application/json": {"example": {"detail": "Internal server error occurred"}}
            },
        }

        def post_streaming_endpoint() -> Callable:
            async def post_stream(
                request: Request, payload: DRAgentRunAgentInput
            ) -> StreamingResponse:
                async with session_manager.session(
                    http_connection=request,
                    user_authentication_callback=self._http_flow_handler.authenticate,
                ) as session:
                    return StreamingResponse(
                        headers={"Content-Type": "text/event-stream; charset=utf-8"},
                        content=generate_streaming_response_as_str(
                            payload,
                            session=session,
                            streaming=True,
                            step_adaptor=self.get_step_adaptor(),
                            output_type=DRAgentEventResponse,
                        ),
                    )

            return post_stream

        def post_openai_api_compatible_endpoint() -> Callable:
            """
            OpenAI-compatible endpoint that handles both streaming and non-streaming
            based on the 'stream' parameter in the request.
            """

            async def post_openai_api_compatible(
                response: Response, request: Request, payload: ChatRequest
            ) -> StreamingResponse | DRAgentChatResponse:
                # Check if streaming is requested

                response.headers["Content-Type"] = "application/json"
                stream_requested = payload.stream

                async with session_manager.session(http_connection=request) as session:
                    if stream_requested:
                        # Return streaming response
                        return StreamingResponse(
                            headers={"Content-Type": "text/event-stream; charset=utf-8"},
                            content=generate_streaming_response_as_str(
                                payload,
                                session=session,
                                streaming=True,
                                step_adaptor=self.get_step_adaptor(),
                                result_type=DRAgentChatResponseChunk,
                                output_type=DRAgentChatResponseChunk,
                            ),
                        )

                    result = await generate_single_response(
                        payload, session, result_type=DRAgentChatResponse
                    )
                    add_context_headers_to_response(response)
                    return result

            return post_openai_api_compatible

        primary_route = endpoint.openai_api_path or "/chat"
        app.add_api_route(
            path=prefix_mount_path(f"{primary_route}/stream"),
            endpoint=post_streaming_endpoint(),
            methods=[endpoint.method],
            response_model=Event,
            description=endpoint.description,
            responses={500: response_500},
        )

        if endpoint.openai_api_v1_path:
            # OpenAI v1 Compatible Mode: Create single endpoint that handles both streaming and
            # non-streaming
            app.add_api_route(
                path=prefix_mount_path(endpoint.openai_api_v1_path),
                endpoint=post_openai_api_compatible_endpoint(),
                methods=[endpoint.method],
                response_model=DRAgentChatResponse | DRAgentChatResponseChunk,
                description=f"{endpoint.description} (OpenAI Chat Completions API compatible)",
                responses={500: response_500},
            )


class DRAgentFastApiFrontEndPlugin(FastApiFrontEndPlugin):
    def get_worker_class(self) -> type[FastApiFrontEndPluginWorker]:
        return DRAgentFastApiFrontEndPluginWorker
