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

from ag_ui.core import RunErrorEvent
from nat.builder.context import Context
from nat.builder.context import IntermediateStep
from nat.data_models.api_server import ResponsePayloadOutput
from nat.data_models.api_server import ResponseSerializable
from nat.data_models.step_adaptor import StepAdaptorConfig
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.runtime.session import Session
from nat.utils.producer_consumer_queue import AsyncIOProducerConsumerQueue
from nat.utils.type_converter import GlobalTypeConverter

from datarobot_genai.dragent.response import DRAgentEventResponse

logger = logging.getLogger(__name__)


async def dragent_pull_intermediate(
    _q: AsyncIOProducerConsumerQueue[ResponseSerializable],
    adapter: StepAdaptor,
) -> asyncio.Event:
    """
    Subscribe to the runner's event stream (which is now a simplified Observable)
    using direct callbacks. Processes each event with the adapter and enqueues
    results to `_q`.
    """
    intermediate_done = asyncio.Event()
    context = Context.get()
    loop = asyncio.get_running_loop()
    trace_id_emitted = False

    async def set_intermediate_done() -> None:
        intermediate_done.set()

    def on_next_cb(item: IntermediateStep) -> None:
        """
        Call synchronously called whenever the runner publishes an event.
        We process it, then place it into the async queue (via a small async task).
        If adapter is None, convert the raw IntermediateStep into the complete
        ResponseIntermediateStep and place it into the queue.
        """
        nonlocal trace_id_emitted

        # Check if trace ID is now available and emit it once
        if not trace_id_emitted:
            observability_trace_id = context.observability_trace_id
            if observability_trace_id:
                from nat.data_models.api_server import ResponseObservabilityTrace

                loop.create_task(
                    _q.put(
                        ResponseObservabilityTrace(observability_trace_id=observability_trace_id)
                    )
                )
                trace_id_emitted = True

        adapted = adapter.process(item)

        if adapted is not None:
            loop.create_task(_q.put(adapted))

    def on_error_cb(exc: Exception) -> None:
        """Call if the runner signals an error. We log it and unblock our wait."""
        logger.error("Hit on_error: %s", exc)
        loop.create_task(set_intermediate_done())
        # Handle error by sending an error event
        event = RunErrorEvent(message=str(exc))
        event_chunk = DRAgentEventResponse(event=event)
        loop.create_task(_q.put(event_chunk))

    def on_complete_cb() -> None:
        """Call once the runner signals no more items. We unblock our wait."""
        logger.debug("Completed reading intermediate steps")

        loop.create_task(set_intermediate_done())

    # Subscribe to the runner's "reactive_event_stream" (now a simple Observable)
    _ = context.intermediate_step_manager.subscribe(
        on_next=on_next_cb, on_error=on_error_cb, on_complete=on_complete_cb
    )

    # Wait until on_complete or on_error sets intermediate_done
    return intermediate_done


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
        intermediate_complete = await dragent_pull_intermediate(q, step_adaptor)

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
                item_converted = GlobalTypeConverter.get().convert(item, output_type)
                if isinstance(item_converted, ResponseSerializable):
                    yield item_converted
                else:
                    yield ResponsePayloadOutput(payload=item_converted)

            if task.exception():
                raise task.exception()  # type: ignore[misc]
        except Exception as e:
            # Return chunk with error message
            logger.exception("Error in response stream: %s", e)
            error_message = str(e)
            response = GlobalTypeConverter.get().convert(error_message, output_type)
            if isinstance(response, ResponseSerializable):
                yield response
            else:
                yield ResponsePayloadOutput(payload=response)
        finally:
            await q.close()
