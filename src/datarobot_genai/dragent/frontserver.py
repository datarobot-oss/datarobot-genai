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
from collections.abc import AsyncGenerator
from collections.abc import Callable

from ag_ui.core import Event
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi.responses import StreamingResponse
from nat.builder.context import Context
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.front_ends.fastapi.response_helpers import generate_single_response
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.runtime.session import SessionManager

from datarobot_genai.dragent.request import DRAgentChatRequest
from datarobot_genai.dragent.request import DRAgentRunAgentInput
from datarobot_genai.dragent.response import DRAgentChatResponse
from datarobot_genai.dragent.response import DRAgentChatResponseChunk
from datarobot_genai.dragent.response import DRAgentEventResponse
from datarobot_genai.dragent.response_helpers import dragent_generate_streaming_response
from datarobot_genai.dragent.response_helpers import dragent_generate_streaming_response_as_str
from datarobot_genai.dragent.step_adaptor import DRAgentNestedReasoningStepAdaptor

logger = logging.getLogger(__name__)


class DRAgentFastApiFrontEndPluginWorker(FastApiFrontEndPluginWorker):
    def get_step_adaptor(self) -> StepAdaptor:
        return DRAgentNestedReasoningStepAdaptor(self.front_end_config.step_adaptor)

    async def add_default_route(self, app: FastAPI, session_manager: SessionManager) -> None:
        # The default route is rewritten to our contract
        await self.add_agent_route(app, self.front_end_config.workflow, session_manager)

    async def add_agent_route(
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

                    async def gen() -> AsyncGenerator[str]:
                        async for item in dragent_generate_streaming_response(
                            payload,
                            session=session,
                            step_adaptor=self.get_step_adaptor(),
                            output_type=DRAgentEventResponse,
                            streaming=True,
                        ):
                            if isinstance(item, DRAgentEventResponse) and item.events:
                                # Handle RUN events
                                for event in item.events:
                                    if getattr(event, "thread_id", None) is None:
                                        setattr(event, "thread_id", payload.thread_id)
                                    if getattr(event, "run_id", None) is None:
                                        setattr(event, "run_id", payload.run_id)
                                    if getattr(event, "parent_run_id", None) is None:
                                        setattr(event, "parent_run_id", payload.parent_run_id)
                                    if getattr(event, "input", None) is None:
                                        setattr(event, "input", payload)

                            yield item.get_stream_data()

                    return StreamingResponse(
                        headers={"Content-Type": "text/event-stream; charset=utf-8"},
                        content=gen(),
                    )

            return post_stream

        def post_openai_api_compatible_endpoint() -> Callable:
            """
            OpenAI-compatible endpoint that handles both streaming and non-streaming
            based on the 'stream' parameter in the request.
            """

            async def post_openai_api_compatible(
                response: Response, request: Request, payload: DRAgentChatRequest
            ) -> StreamingResponse | DRAgentChatResponse:
                # Check if streaming is requested

                response.headers["Content-Type"] = "application/json"
                stream_requested = payload.stream

                async with session_manager.session(http_connection=request) as session:
                    if stream_requested:
                        # Return streaming response
                        return StreamingResponse(
                            headers={"Content-Type": "text/event-stream; charset=utf-8"},
                            content=dragent_generate_streaming_response_as_str(
                                payload,
                                session=session,
                                streaming=True,
                                step_adaptor=self.get_step_adaptor(),
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
            path=f"{primary_route}/stream",
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
                path=endpoint.openai_api_v1_path,
                endpoint=post_openai_api_compatible_endpoint(),
                methods=[endpoint.method],
                response_model=DRAgentChatResponse | DRAgentChatResponseChunk,
                description=f"{endpoint.description} (OpenAI Chat Completions API compatible)",
                responses={500: response_500},
            )


class DRAgentFastApiFrontEndPlugin(FastApiFrontEndPlugin):
    def get_worker_class(self) -> type[FastApiFrontEndPluginWorker]:
        return DRAgentFastApiFrontEndPluginWorker
