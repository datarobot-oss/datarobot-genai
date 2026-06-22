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

import json
from typing import Any

from crewai import LLM

from datarobot_genai.core.config import DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM
from datarobot_genai.core.config import Config
from datarobot_genai.core.config import LLMConfig
from datarobot_genai.core.config import LLMType
from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.config import default_datarobot_llm_gateway_url
from datarobot_genai.core.config import default_deployment_url
from datarobot_genai.core.config import default_model_name


class LitellmStopWordLLM(LLM):
    """CrewAI LLM subclass that forces LiteLLM usage and enforces client-side stop-word truncation.

    CrewAI's ``LLM.__new__`` may choose a native client instead of LiteLLM for some
    model strings.  The ``__new__`` override forces ``object.__new__`` so that LiteLLM
    is always used.  The ``call()`` override ensures stop words are honoured even when
    the underlying API silently ignores the stop parameter.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "LitellmStopWordLLM":
        return object.__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.is_litellm = True

    def _apply_stop_words(self, content: str) -> str:
        """Apply configured stop words, then truncate inline ReAct hallucinations."""
        truncated = super()._apply_stop_words(content)
        return self._truncate_react_hallucination_after_action_input(truncated)

    @staticmethod
    def _truncate_react_hallucination_after_action_input(content: str) -> str:
        """Truncate hallucinated text appended after ``Action Input:``.

        Models sometimes emit fake tool results or a second ReAct step inline,
        without an ``Observation:`` label. Keep only the action-input value:
        text on the same line as ``Action Input:``, stopping before any inline
        ``Thought:`` or ``Final Answer:`` label.
        """
        marker = "Action Input:"
        marker_start = content.find(marker)
        if marker_start == -1:
            return content

        value_start = marker_start + len(marker)
        value_text = content[value_start:]

        # Earliest index in ``content`` where hallucinated suffix may begin.
        truncation_points = [len(content)]

        newline_in_value = value_text.find("\n")
        if newline_in_value != -1:
            truncation_points.append(value_start + newline_in_value)

        for react_label in ("Thought:", "Final Answer:"):
            label_offset = value_text.find(react_label)
            # Ignore labels glued directly to the marker (offset 0).
            if label_offset > 0:
                truncation_points.append(value_start + label_offset)

        cut_end = min(truncation_points)
        if cut_end == len(content):
            return content
        return content[:cut_end].rstrip()

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Enforce client-side stop-word truncation when API ignores stop parameter."""
        result = super().call(*args, **kwargs)
        if isinstance(result, str):
            return self._apply_stop_words(result)
        return result

    def _format_messages_for_provider(self, messages: list) -> list:
        """Ensure conversation does not end with an assistant message.

        Some models routed through the DataRobot LLM Gateway (e.g. claude-sonnet-4-6)
        reject assistant-message prefill. When the conversation ends with an assistant
        message we append a minimal user message so the API accepts the request while
        preserving the full conversation context.
        """
        formatted = super()._format_messages_for_provider(messages)
        if formatted and formatted[-1].get("role") == "assistant":
            formatted = [*formatted, {"role": "user", "content": "Please continue."}]
        return formatted

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """Async variant of :meth:`call` used by ``Crew.akickoff``."""
        result = await super().acall(*args, **kwargs)
        if isinstance(result, str):
            return self._apply_stop_words(result)
        return result


def _crewai_model_factory(config: dict) -> LLM:
    config["stream_options"] = config.get("stream_options", {"include_usage": True})
    # Strip NAT-internal keys that cause "extra inputs" errors in litellm.
    # Multiple config types (Deployment, Component, Litellm) flow through here.
    config.pop("verify_ssl", None)
    return LitellmStopWordLLM(**config)


def get_datarobot_gateway_llm(model_name: str | None = None, parameters: dict | None = None) -> LLM:
    config = {
        "api_key": default_api_key(),
        "api_base": default_datarobot_llm_gateway_url(),
    }

    if parameters:
        config.update(parameters)

    model_name = model_name or default_model_name()
    if model_name is None:
        raise ValueError("Model name is required")

    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config["model"] = model_name

    return _crewai_model_factory(config)


def get_datarobot_deployment_llm(
    deployment_id: str, model_name: str | None = None, parameters: dict | None = None
) -> LLM:
    config = {
        "api_key": default_api_key(),
        "api_base": default_deployment_url(deployment_id),
    }

    if parameters:
        config.update(parameters)

    model_name = model_name or default_model_name() or DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM
    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config["model"] = model_name
    return _crewai_model_factory(config)


def get_datarobot_nim_llm(
    nim_deployment_id: str, model_name: str | None = None, parameters: dict | None = None
) -> LLM:
    config = {
        "api_key": default_api_key(),
        "api_base": default_deployment_url(nim_deployment_id),
    }

    if parameters:
        config.update(parameters)

    model_name = model_name or default_model_name()
    if model_name is None:
        raise ValueError("Model name is required")

    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config["model"] = model_name
    return _crewai_model_factory(config)


def get_external_llm(model_name: str | None = None, parameters: dict | None = None) -> LLM:
    config = {
        # Everything else is loaded from the environment by LiteLLM
    }

    if parameters:
        config.update(parameters)
    model_name = model_name or default_model_name()
    if model_name is None:
        raise ValueError("Model name is required")

    model_name = model_name.removeprefix("datarobot/")
    config["model"] = model_name

    return _crewai_model_factory(config)


def get_router_llm(
    primary: LLMConfig,
    fallbacks: list[LLMConfig],
    router_settings: dict | None = None,
) -> LLM:
    """Return a CrewAI ``LLM`` whose calls are routed through a ``litellm.Router``.

    Args:
        primary: ``LLMConfig`` for the primary model.
        fallbacks: Ordered list of ``LLMConfig`` fallback configs.
        router_settings: Extra kwargs forwarded to ``litellm.Router``.
    """
    import uuid  # noqa: PLC0415

    from crewai.events import crewai_event_bus  # noqa: PLC0415
    from crewai.events.types.llm_events import LLMStreamChunkEvent  # noqa: PLC0415

    from datarobot_genai.core.router import build_litellm_router  # noqa: PLC0415
    from datarobot_genai.core.router import merge_streaming_tool_calls  # noqa: PLC0415

    router = build_litellm_router(primary, fallbacks, router_settings)

    class RouterLitellmOnlyLLM(LLM):
        def __new__(cls, *args: Any, **kwargs: Any) -> "RouterLitellmOnlyLLM":
            return object.__new__(cls)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.is_litellm = True
            self._llm_router = router

        def call(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            callbacks: list | None = None,
            available_tools: list[dict] | None = None,
            **kwargs: Any,
        ) -> str:
            call_id = str(uuid.uuid4())
            accumulated = []
            tool_calls_seen: list[Any] = []
            for chunk in self._llm_router.completion(
                "primary",
                messages=messages,
                stream=True,
                **({"tools": tools} if tools else {}),
            ):
                delta = chunk.choices[0].delta
                if delta.content:
                    accumulated.append(delta.content)
                    crewai_event_bus.emit(
                        self,
                        event=LLMStreamChunkEvent(chunk=delta.content, call_id=call_id),
                    )
                    if callbacks:
                        for cb in callbacks:
                            if hasattr(cb, "on_llm_new_token"):
                                cb.on_llm_new_token(delta.content)
                if getattr(delta, "tool_calls", None):
                    tool_calls_seen.extend(delta.tool_calls)
            if tool_calls_seen:
                return json.dumps({"tool_calls": merge_streaming_tool_calls(tool_calls_seen)})
            return "".join(accumulated)

        async def acall(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            callbacks: list | None = None,
            available_tools: list[dict] | None = None,
            **kwargs: Any,
        ) -> str:
            call_id = str(uuid.uuid4())
            accumulated = []
            tool_calls_seen: list[Any] = []
            response = await self._llm_router.acompletion(
                "primary",
                messages=messages,
                stream=True,
                **({"tools": tools} if tools else {}),
            )
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    accumulated.append(delta.content)
                    crewai_event_bus.emit(
                        self,
                        event=LLMStreamChunkEvent(chunk=delta.content, call_id=call_id),
                    )
                    if callbacks:
                        for cb in callbacks:
                            if hasattr(cb, "on_llm_new_token"):
                                cb.on_llm_new_token(delta.content)
                if getattr(delta, "tool_calls", None):
                    tool_calls_seen.extend(delta.tool_calls)
            if tool_calls_seen:
                return json.dumps({"tool_calls": merge_streaming_tool_calls(tool_calls_seen)})
            return "".join(accumulated)

    return RouterLitellmOnlyLLM(model="datarobot-router")


def get_llm(model_name: str | None = None, parameters: dict | None = None) -> LLM:
    config = Config()
    llm_type = config.get_llm_type()
    if llm_type == LLMType.GATEWAY:
        return get_datarobot_gateway_llm(model_name, parameters)
    elif llm_type == LLMType.DEPLOYMENT:
        return get_datarobot_deployment_llm(config.llm_deployment_id, model_name, parameters)  # type: ignore[arg-type]
    elif llm_type == LLMType.NIM:
        return get_datarobot_nim_llm(config.nim_deployment_id, model_name, parameters)  # type: ignore[arg-type]
    elif llm_type == LLMType.EXTERNAL:
        return get_external_llm(model_name, parameters)
    else:
        raise ValueError(f"Invalid LLM type inferred from config: {llm_type}, config: {config}")
