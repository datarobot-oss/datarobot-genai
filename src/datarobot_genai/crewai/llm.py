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
from crewai.events.types.llm_events import LLMCallFailedEvent
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
    """Parse args from a recovered call body's markup (params block, JSON, or child tags)."""
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
    """Recursively drop null values + empty array-keywords (``anyOf: []``) that bedrock rejects."""
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
    """Recover a tool call a model leaked as TEXT markup into OpenAI tool-call dicts.

    Matches known wrapper tags, and — anchored on ``tool_names`` so prose can't trigger it —
    the bare tool-name-as-tag form. Duplicates are collapsed; returns ``[]`` if nothing matches.
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

    # Tool-name-as-tag form, anchored on real tool names. A real answer can contain such markup
    # too, so recover a block only when its body parses to arguments (else treat it as prose).
    for name in tool_names or []:
        block_re = re.compile(rf"<{re.escape(name)}\b[^>]*>(.*?)</{re.escape(name)}>", re.DOTALL)
        for body in block_re.findall(text):
            if _parse_tool_args(body):
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

    def _collect_chunk(
        self,
        chunk: Any,
        text: list[str],
        tool_calls: list[Any],
        call_id: str,
        callbacks: list | None,
    ) -> Any:
        """Gather a chunk's content + tool-call parts; return its usage (skips empty choices)."""
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                text.append(delta.content)
                event = LLMStreamChunkEvent(chunk=delta.content, call_id=call_id)
                crewai_event_bus.emit(self, event=event)
                for cb in callbacks or []:
                    if hasattr(cb, "on_llm_new_token"):
                        cb.on_llm_new_token(delta.content)
            if getattr(delta, "tool_calls", None):
                tool_calls.extend(delta.tool_calls)
        usage = getattr(chunk, "usage", None)
        return usage if not isinstance(usage, type) else None

    def _track_native_usage(self, usage: Any) -> None:
        """Track streamed usage in CrewAI's metrics (the native loop bypasses the base path)."""
        if usage:
            self._track_token_usage_internal(self._usage_to_dict(usage) or {})

    def _finalize_native(
        self, text: list[str], tool_calls: list[Any], tool_names: list[str] | None = None
    ) -> list[dict] | str:
        """Return tool calls (streamed or recovered) as the bare list CrewAI runs, else text."""
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
        """Stream and return native tool calls ourselves — CrewAI's handler drops them.

        Non-native calls delegate to the base, stop-word-truncating any text result.
        """
        if self._wants_native_tool_calls(kwargs):
            import litellm  # noqa: PLC0415

            tools = _sanitize_tool_schema(kwargs["tools"])
            params = self._prepare_completion_params(args[0], tools)
            params["stream"] = True
            call_id = str(uuid.uuid4())
            text: list[str] = []
            tool_calls: list[Any] = []
            usage: Any = None
            try:
                for chunk in litellm.completion(**params):
                    usage = (
                        self._collect_chunk(
                            chunk, text, tool_calls, call_id, kwargs.get("callbacks")
                        )
                        or usage
                    )
            except Exception as exc:
                crewai_event_bus.emit(
                    self, event=LLMCallFailedEvent(call_id=call_id, error=str(exc))
                )
                raise
            self._track_native_usage(usage)
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
            usage: Any = None
            try:
                async for chunk in await litellm.acompletion(**params):
                    usage = (
                        self._collect_chunk(
                            chunk, text, tool_calls, call_id, kwargs.get("callbacks")
                        )
                        or usage
                    )
            except Exception as exc:
                crewai_event_bus.emit(
                    self, event=LLMCallFailedEvent(call_id=call_id, error=str(exc))
                )
                raise
            self._track_native_usage(usage)
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
    """Return a CrewAI ``LLM`` routing calls through a ``litellm.Router`` (primary → fallbacks)."""
    from datarobot_genai.core.router import build_litellm_router  # noqa: PLC0415
    from datarobot_genai.core.router import merge_streaming_tool_calls  # noqa: PLC0415

    router = build_litellm_router(primary, fallbacks, router_settings)
    # The router fails over primary → fallbacks at runtime, so capability detection considers the
    # whole chain: a placeholder/retired primary shouldn't force the router onto the ReAct path
    # (where models can hallucinate the tool result) when a reachable fallback supports tools.
    chain_models = [cfg.to_litellm_params().get("model", "") for cfg in (primary, *fallbacks)]

    class RouterLitellmOnlyLLM(LLM):
        def __new__(cls, *args: Any, **kwargs: Any) -> "RouterLitellmOnlyLLM":
            return object.__new__(cls)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.is_litellm = True
            self._llm_router = router

        def supports_function_calling(self) -> bool:
            # self.model is a sentinel; report the first model in the failover chain that
            # litellm can resolve (the primary may be a placeholder the router never uses).
            for model in chain_models:
                supported = _model_supports_tool_calling(model)
                if supported is not None:
                    return supported
            return super().supports_function_calling()

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
