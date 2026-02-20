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

import asyncio
import logging
import typing
from collections.abc import AsyncGenerator

from ag_ui.core import Event
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ResponseBaseModelOutput
from nat.data_models.api_server import ResponsePayloadOutput
from nat.data_models.api_server import ResponseSerializable
from nat.data_models.step_adaptor import StepAdaptorConfig
from nat.front_ends.fastapi.intermediate_steps_subscriber import pull_intermediate
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.runtime.session import Session
from nat.utils.producer_consumer_queue import AsyncIOProducerConsumerQueue
from nat.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)


class DRAgentEventResponse(ResponseBaseModelOutput):
    event: Event | None = None
    delta: str | None = None
    pipeline_interactions: str | None = None
    usage_metrics: dict[str, int] | None = None


class DRAgentChatResponseChunk(ChatResponseChunk):
    pipeline_interactions: str | None = None
    event: Event | None = None


class DRAgentChatResponse(ChatResponse):
    pipeline_interactions: str | None = None


async def dragent_generate_streaming_response_as_str(
    payload: typing.Any,
    *,
    session: Session,
    streaming: bool,
    step_adaptor: StepAdaptor = StepAdaptor(StepAdaptorConfig()),
    result_type: type | None = None,
    output_type: type | None = None,
) -> AsyncGenerator[str]:

    async for item in dragent_generate_streaming_response(
        payload,
        session=session,
        streaming=streaming,
        step_adaptor=step_adaptor,
        result_type=result_type,
        output_type=output_type,
    ):
        if isinstance(item, ResponseSerializable):
            yield item.get_stream_data()
        else:
            raise ValueError(
                "Unexpected item type in stream. Expected ChatResponseSerializable, got: "
                + str(type(item))
            )


async def dragent_generate_streaming_response(
    payload: typing.Any,
    *,
    session: Session,
    streaming: bool,
    step_adaptor: StepAdaptor = StepAdaptor(StepAdaptorConfig()),
    result_type: type | None = None,
    output_type: type | None = None,
) -> AsyncGenerator[ResponseSerializable]:

    async with session.run(payload) as runner:
        q: AsyncIOProducerConsumerQueue[ResponseSerializable] = AsyncIOProducerConsumerQueue()

        # Start the intermediate stream
        intermediate_complete = await pull_intermediate(q, step_adaptor)

        async def pull_result() -> None:
            try:
                if session.workflow.has_streaming_output and streaming:
                    async for chunk in runner.result_stream(to_type=output_type):
                        await q.put(chunk)
                else:
                    result = await runner.result(to_type=result_type)
                    await q.put(runner.convert(result, output_type))
            except Exception:
                raise
            finally:
                # Wait until the intermediate subscription is done before closing q
                # But we have no direct "intermediate_done" reference here
                # because it's encapsulated in pull_intermediate. So we can do:
                #    await some_event.wait()
                # If needed. Alternatively, you can skip that if the intermediate
                # subscriber won't block the main flow.
                #
                # For example, if you *need* to guarantee the subscriber is done before
                # closing the queue, you can structure the code to store or return
                # the 'intermediate_done' event from pull_intermediate.
                #

                await intermediate_complete.wait()

                await q.close()

        try:
            # Start the result stream
            task = asyncio.create_task(pull_result())

            async for item in q:
                if isinstance(item, ResponseSerializable):
                    yield item
                else:
                    yield ResponsePayloadOutput(payload=item)

            if task.exception():
                raise task.exception()  # type: ignore[misc]
        except Exception as e:
            # Return chunk with error message
            error_message = str(e)
            response = GlobalTypeConverter.get().convert(error_message, output_type)
            if isinstance(response, ResponseSerializable):
                yield response
            else:
                yield ResponsePayloadOutput(payload=response)
        finally:
            await q.close()
