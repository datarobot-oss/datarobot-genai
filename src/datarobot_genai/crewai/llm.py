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
import re
import uuid
from typing import Any

from crewai import LLM
from crewai.events import crewai_event_bus
from crewai.events.types.llm_events import LLMStreamChunkEvent

from datarobot_genai.core.config import DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM
from datarobot_genai.core.config import Config
from datarobot_genai.core.config import LLMConfig
from datarobot_genai.core.config import LLMType
from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.config import default_datarobot_llm_gateway_url
from datarobot_genai.core.config import default_deployment_url
from datarobot_genai.core.config import default_model_name
from datarobot_genai.core.model_info import get_model_info


def _model_supports_tool_calling(model: str) -> bool | None:
    """Tool-calling support for *model* via litellm, or ``None`` if unresolved."""
    try:
        info = get_model_info(model)
    except Exception:
        return None
    return bool(info.get("supports_function_calling")) or bool(info.get("supports_tool_choice"))


# Recover tool calls a model emits as text markup (Anthropic <invoke>, MCP
# <use_mcp_tool>, or a bare <use_tool>) — name via `name=` attribute or <tool_name>
# child; args via <parameter name=> children or a <parameters>/<arguments> block
# (JSON or <k>v</k>). The model invents the call tag, so all observed forms are matched.
_CALL_BLOCK_RE = re.compile(r"<(invoke|use_mcp_tool|use_tool)\b([^>]*)>(.*?)</\1>", re.DOTALL)
_NAME_ATTR_RE = re.compile(r'name\s*=\s*"([^"]+)"')
_TOOL_NAME_RE = re.compile(r"<tool_name>\s*(.*?)\s*</tool_name>", re.DOTALL)
_PARAM_ATTR_RE = re.compile(r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>', re.DOTALL)
_ARGS_BLOCK_RE = re.compile(
    r"<(?:parameters|arguments)>(.*?)</(?:parameters|arguments)>", re.DOTALL
)
_CHILD_TAG_RE = re.compile(r"<([^/>\s]+)>(.*?)</\1>", re.DOTALL)


def _parse_tool_args(block: str) -> dict:
    """Extract a tool call's arguments from a recovered call body.

    Handles ``<parameter name="k">v</parameter>`` children (Anthropic), a
    ``<parameters>``/``<arguments>`` block holding JSON or ``<k>v</k>`` child tags, or
    bare ``<k>v</k>`` children directly in the body (the tool-name-as-tag form).
    """
    params = _PARAM_ATTR_RE.findall(block)
    if params:
        return {key: value.strip() for key, value in params}
    block_match = _ARGS_BLOCK_RE.search(block)
    if block_match:
        body = block_match.group(1).strip()
        try:
            parsed = json.loads(body)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {key: value.strip() for key, value in _CHILD_TAG_RE.findall(body)}
    return {key: value.strip() for key, value in _CHILD_TAG_RE.findall(block) if key != "tool_name"}


def _openai_tool_names(tools: Any) -> list[str]:
    """Names from an OpenAI ``tools`` schema list (``[{"function": {"name": ...}}]``)."""
    names = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if name:
            names.append(name)
    return names


# Keywords that are invalid when null/empty under JSON Schema draft 2020-12.
_EMPTY_INVALID_KEYS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems", "enum"})


def _sanitize_tool_schema(node: Any) -> Any:
    """Strip JSON-Schema-invalid placeholder keys before sending tools to the gateway.

    ``mcpadapt`` renders MCP tool params with ``"anyOf": []``, ``"enum": null``,
    ``"items": null`` etc.; bedrock rejects the whole request as not draft-2020-12
    compliant. Drop ``null`` values and empty array-keywords, recursively.
    """
    if isinstance(node, dict):
        cleaned = {}
        for key, value in node.items():
            if value is None or (key in _EMPTY_INVALID_KEYS and value == []):
                continue
            cleaned[key] = _sanitize_tool_schema(value)
        return cleaned
    if isinstance(node, list):
        return [_sanitize_tool_schema(item) for item in node]
    return node


def _recover_text_tool_calls(text: str, tool_names: list[str] | None = None) -> list[dict]:
    """Recover tool calls a model emitted as TEXT into OpenAI tool-call dicts.

    Claude under extended thinking emits a call as text markup instead of a structured
    ``tool_call``, leaking it to the final answer unexecuted. The wrapper it invents varies
    (``<function_calls><invoke>``, ``<use_mcp_tool>``, ``<use_tool>``, ``<budget:function_calls>``)
    and it sometimes drops the wrapper entirely and uses the tool name as the tag
    (``<generate_objectid><object_type>x</object_type></generate_objectid>``). So rather than
    chase wrappers, recover two ways: known wrapper tags, and — anchored on the actual
    ``tool_names`` from the request so prose can't trigger it — the tool-name-as-tag form.
    Duplicates across forms are collapsed. Returns ``[]`` when nothing recoverable is found.
    """
    calls: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(name: str, body: str) -> None:
        name = name.strip()
        if not name:
            return
        arguments = json.dumps(_parse_tool_args(body))
        if (name, arguments) in seen:
            return
        seen.add((name, arguments))
        calls.append(
            {
                "id": f"call_{len(calls)}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )

    # Wrapper forms. Gated on a known call tag (``function_calls`` matched namespace-agnostically)
    # so prose merely quoting a bare ``<invoke>`` is not mistaken for a real call.
    if any(tag in text for tag in ("function_calls", "<use_mcp_tool", "<use_tool")):
        for _tag, attrs, body in _CALL_BLOCK_RE.findall(text):
            match = _NAME_ATTR_RE.search(attrs) or _TOOL_NAME_RE.search(body)
            add(match.group(1) if match else "", body)

    # Tool-name-as-tag form, anchored on real tool names (safe against arbitrary prose markup).
    for name in tool_names or []:
        block_re = re.compile(rf"<{re.escape(name)}\b[^>]*>(.*?)</{re.escape(name)}>", re.DOTALL)
        for body in block_re.findall(text):
            add(name, body)

    return calls


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

    def _collect_delta(
        self,
        delta: Any,
        text: list[str],
        tool_calls: list[Any],
        call_id: str,
        callbacks: list | None,
    ) -> None:
        """Accumulate one stream delta: emit content tokens, gather tool-call parts."""
        if delta.content:
            text.append(delta.content)
            event = LLMStreamChunkEvent(chunk=delta.content, call_id=call_id)
            crewai_event_bus.emit(self, event=event)
            for cb in callbacks or []:
                if hasattr(cb, "on_llm_new_token"):
                    cb.on_llm_new_token(delta.content)
        if getattr(delta, "tool_calls", None):
            tool_calls.extend(delta.tool_calls)

    def _finalize_native(
        self, text: list[str], tool_calls: list[Any], tool_names: list[str] | None = None
    ) -> list[dict] | str:
        """Return the tool calls for CrewAI's native loop (a bare list it runs), else text.

        Prefer structured streamed ``tool_calls``; otherwise recover any the model
        emitted as markup text (``tool_names`` anchors the tool-name-as-tag form);
        otherwise return the stop-word-truncated text.
        """
        if tool_calls:
            from datarobot_genai.core.router import merge_streaming_tool_calls  # noqa: PLC0415

            return merge_streaming_tool_calls(tool_calls)
        joined = "".join(text)
        return _recover_text_tool_calls(joined, tool_names) or self._apply_stop_words(joined)

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

    @staticmethod
    def _wants_native_tool_calls(kwargs: dict) -> bool:
        """CrewAI's native loop calls us with ``tools`` set and ``available_functions=None``."""
        return bool(kwargs.get("tools")) and kwargs.get("available_functions") is None

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Stream and return native tool calls, else stop-word-truncated text.

        CrewAI's streaming response handler discards the assembled ``tool_calls`` and
        returns the (empty) text for a tool-call response, breaking its native loop. So
        when CrewAI asks for native tool calls we stream the completion ourselves and
        return the assembled calls (as :class:`RouterLitellmOnlyLLM` does); the ReAct
        path is unchanged.
        """
        if self._wants_native_tool_calls(kwargs):
            import litellm  # noqa: PLC0415

            tools = _sanitize_tool_schema(kwargs["tools"])
            params = self._prepare_completion_params(args[0], tools)
            params["stream"] = True
            call_id = str(uuid.uuid4())
            text: list[str] = []
            tool_calls: list[Any] = []
            for chunk in litellm.completion(**params):
                self._collect_delta(
                    chunk.choices[0].delta, text, tool_calls, call_id, kwargs.get("callbacks")
                )
            return self._finalize_native(text, tool_calls, _openai_tool_names(tools))
        result = super().call(*args, **kwargs)
        return self._apply_stop_words(result) if isinstance(result, str) else result

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
        if self._wants_native_tool_calls(kwargs):
            import litellm  # noqa: PLC0415

            tools = _sanitize_tool_schema(kwargs["tools"])
            params = self._prepare_completion_params(args[0], tools)
            params["stream"] = True
            call_id = str(uuid.uuid4())
            text: list[str] = []
            tool_calls: list[Any] = []
            async for chunk in await litellm.acompletion(**params):
                self._collect_delta(
                    chunk.choices[0].delta, text, tool_calls, call_id, kwargs.get("callbacks")
                )
            return self._finalize_native(text, tool_calls, _openai_tool_names(tools))
        result = await super().acall(*args, **kwargs)
        return self._apply_stop_words(result) if isinstance(result, str) else result

    def supports_function_calling(self) -> bool:
        supported = _model_supports_tool_calling(self.model)
        return supported if supported is not None else super().supports_function_calling()


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
    primary_model = primary.to_litellm_params().get("model", "")

    class RouterLitellmOnlyLLM(LLM):
        def __new__(cls, *args: Any, **kwargs: Any) -> "RouterLitellmOnlyLLM":
            return object.__new__(cls)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.is_litellm = True
            self._llm_router = router

        def supports_function_calling(self) -> bool:
            # self.model is a sentinel; resolve the routed primary model instead.
            supported = _model_supports_tool_calling(primary_model)
            return supported if supported is not None else super().supports_function_calling()

        def call(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            callbacks: list | None = None,
            available_tools: list[dict] | None = None,
            **kwargs: Any,
        ) -> list[dict] | str:
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
            # Bare list, not a json string: CrewAI's native loop executes a list of tool-call
            # dicts; a string falls through to a final answer so the calls would never run.
            if tool_calls_seen:
                return merge_streaming_tool_calls(tool_calls_seen)
            return "".join(accumulated)

        async def acall(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            callbacks: list | None = None,
            available_tools: list[dict] | None = None,
            **kwargs: Any,
        ) -> list[dict] | str:
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
            # Bare list, not a json string: CrewAI's native loop executes a list of tool-call
            # dicts; a string falls through to a final answer so the calls would never run.
            if tool_calls_seen:
                return merge_streaming_tool_calls(tool_calls_seen)
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
