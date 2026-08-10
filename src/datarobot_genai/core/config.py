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

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import cast

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from datarobot.core.config import LLMConfig
from datarobot.core.config import LLMType  # noqa: F401  # re-exported for genai consumers
from datarobot.core.config import deployment_url
from datarobot.core.config import llm_gateway_url
from pydantic import Field

DEFAULT_MAX_HISTORY_MESSAGES = 20
DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM = "datarobot/datarobot-deployed-llm"
DEFAULT_DATAROBOT_ENDPOINT = "https://app.datarobot.com/api/v2"
DEFAULT_LLM_NAME = "llm"


def _with_datarobot_prefix(model_name: str) -> str:
    return model_name if model_name.startswith("datarobot/") else "datarobot/" + model_name


class Config(DataRobotAppFrameworkBaseSettings):
    """The single GLOBAL application config. There is exactly one.

    Finds variables in priority order: env vars (including Runtime Parameters),
    .env, file secrets, then Pulumi outputs. It holds the two ecosystem-wide
    globals, app-wide settings, and the default LLM instance's flat,
    instance-namespaced fields so a standalone genai (no app registered) can still
    resolve the default LLM from the environment.

    This is NOT an :class:`LLMConfig`. genai never reads LLM routing fields off it
    directly; those are mapped into an :class:`LLMConfig` by
    :func:`resolve_llm_config`, and the two globals through the base class's
    ``resolve_datarobot_endpoint`` / ``resolve_datarobot_api_token`` methods.
    """

    # True ecosystem-wide globals. Fixed names, shared by every LLM instance.
    datarobot_endpoint: str = DEFAULT_DATAROBOT_ENDPOINT
    datarobot_api_token: str | None = None

    # App-wide settings (genai-specific tunables).
    max_history_messages: int = Field(
        default=DEFAULT_MAX_HISTORY_MESSAGES, ge=0, alias="datarobot_genai_max_history_messages"
    )
    assume_native_tool_calling_when_unmapped: bool = Field(
        default=False,
        description=(
            "CrewAI only: when LiteLLM has no catalog entry for the NIM model, "
            "still report native tool-calling support so CrewAI uses API tool_calls "
            "instead of the ReAct text path."
        ),
    )

    # Default LLM instance ("llm") fields, namespaced by instance name, so a
    # standalone genai reads them from the environment. An app registers its own
    # config, which may namespace by a different instance name; see the seam below.
    llm_deployment_id: str | None = None
    llm_nim_deployment_id: str | None = None
    llm_use_datarobot_llm_gateway: bool = True
    llm_default_model: str | None = None


# --- App config injection seam ---------------------------------------------
#
# There are exactly two config objects and two resolvers, and they are NOT the
# same thing:
#
#   Config       - the single GLOBAL app config (endpoint, token, app-wide
#                  settings, and per-instance LLM fields). resolve_config() -> Config.
#   LLMConfig    - ONE LLM instance's routing config. resolve_llm_config(name) -> LLMConfig,
#                  mapped from the global config's {name}_* fields plus the two globals.
#
# The application (an af-component-* app) owns the authoritative global config: a
# DataRobotAppFrameworkBaseSettings subclass in its config.py. genai cannot import
# that class, so the app registers a provider (a zero-arg callable returning that
# global config) at import time. The app package is imported during NAT plugin
# discovery, before NAT validates the workflow config, so the provider is in place
# before genai first reads config. One flow: a registered provider means genai
# reads the app's global config; otherwise it falls back to its own env-only
# Config().
#
# The invariant that keeps this from getting twisted again: NOTHING in genai reads
# a config attribute directly. Everything is resolved off the config object through
# the datarobot.core base class methods (resolve_datarobot_endpoint /
# resolve_datarobot_api_token / resolve_llm_config); resolve_llm_config() below is
# the only genai wrapper, and only to inject the registered default instance name.

_provider_registry: dict[str, Any] = {"provider": None, "default_llm_name": DEFAULT_LLM_NAME}


def register_config_provider(
    provider: Callable[[], object | None] | None,
    default_llm_name: str = DEFAULT_LLM_NAME,
) -> None:
    """Register the app's authoritative GLOBAL config source, or clear it with ``None``.

    ``provider`` is a zero-arg callable returning the app's global config object
    (typically ``lambda: Config()`` over the app's own ``Config``). It is called
    each time genai resolves config, so values re-resolve through the app's
    settings sources (env / .env / secrets / Pulumi) on every read.

    ``default_llm_name`` is the app's default LLM instance name (the prefix on its
    per-LLM fields). ``resolve_llm_config()`` with no explicit name uses it, so a
    non-"llm" component name still works for a bare ``get_llm()``.
    """
    _provider_registry["provider"] = provider
    _provider_registry["default_llm_name"] = default_llm_name


def _validate_global_config(config: object) -> None:
    """Fail loud if an injected global config is not a settings object.

    The app's global config must be a :class:`DataRobotAppFrameworkBaseSettings`
    subclass: genai resolves the endpoint, API token, and per-LLM config off it
    through that base class's ``resolve_*`` methods. genai can now import the base
    class from ``datarobot.core``, so this is a real ``isinstance`` check rather
    than the duck-typed field check it used to be.
    """
    if not isinstance(config, DataRobotAppFrameworkBaseSettings):
        raise TypeError(
            f"Registered config provider returned {type(config).__name__}, which is not a "
            "DataRobotAppFrameworkBaseSettings. The app config must subclass it so genai can "
            "resolve the DataRobot endpoint, API token, and LLM config off it."
        )


def resolve_config() -> Config:
    """Return the single GLOBAL application config.

    The registered app config when a provider is registered, otherwise genai's own
    env-reading :class:`Config` (a standalone genai with no app around it).
    Everything is resolved off the returned config through the ``datarobot.core``
    base class methods (``resolve_datarobot_endpoint`` / ``resolve_datarobot_api_token``);
    for LLM routing use :func:`resolve_llm_config`.
    """
    provider = _provider_registry["provider"]
    if provider is not None:
        provided = provider()
        if provided is not None:
            _validate_global_config(provided)
            # provided is a DataRobotAppFrameworkBaseSettings subclass (validated
            # above); genai resolves everything off it through its resolve_* methods.
            # Cast to Config for typing since genai cannot import the app's class.
            return cast(Config, provided)
    return Config()


def resolve_llm_config(name: str | None = None) -> LLMConfig:
    """Resolve ONE LLM instance's config from the global config.

    Thin seam glue: it injects genai's global config (:func:`resolve_config`) and
    the app's registered default instance name, then defers the entire mapping to
    :meth:`DataRobotAppFrameworkBaseSettings.resolve_llm_config` in ``datarobot.core``.
    That base method folds the instance's ``{name}_*`` fields together with the two
    globals and applies the deprecated-name backwards-compat bridge, so genai holds
    no LLM-config logic of its own.

    ``name`` is the LLM component instance name; when omitted it is the app's
    registered default (``"llm"`` for a standalone genai). Core's base method
    defaults to ``"llm"`` too, but only genai knows the registered override, which
    is the one thing this wrapper still contributes.
    """
    instance = name if name is not None else cast(str, _provider_registry["default_llm_name"])
    return resolve_config().resolve_llm_config(instance)


def get_max_history_messages_default() -> int:
    """Return the default maximum number of history messages.

    This is a genai-internal tunable (``DATAROBOT_GENAI_MAX_HISTORY_MESSAGES``),
    read off genai's own :class:`Config`, not per-LLM config. Invalid values fall
    back to the built-in default; negative values are treated as 0 (disable history).
    """
    return max(Config().max_history_messages, 0)


def default_api_key() -> str | None:
    return resolve_config().resolve_datarobot_api_token()


def default_model_name() -> str | None:
    return resolve_llm_config().llm_default_model


def default_response_model() -> str:
    """Return the configured model to report in OpenAI ``chat/completions`` responses.

    dragent agents ignore the request's ``model`` and run the LLM configured in
    ``workflow.yaml`` / env, so the response should report that actual model, not
    echo the caller's string (which need not be sent at all) nor NAT's
    ``"unknown-model"`` placeholder. Resolves the same way the LLM client does
    (the default LLM's model), always ``datarobot/``-prefixed and never ``None`` so
    the response can never regress to ``"unknown-model"``.
    """
    return _with_datarobot_prefix(default_model_name() or "datarobot-deployed-llm")


def default_use_datarobot_llm_gateway() -> bool:
    return resolve_llm_config().llm_use_datarobot_llm_gateway


def default_deployment_url(deployment_id: str | None = None) -> str:
    resolved_id = deployment_id or resolve_llm_config().llm_deployment_id
    if resolved_id is None:
        raise ValueError("Neither deployment ID nor default deployment ID is set")

    return deployment_url(resolved_id, resolve_config().resolve_datarobot_endpoint())


def default_datarobot_llm_gateway_url() -> str:
    return llm_gateway_url(resolve_config().resolve_datarobot_endpoint())


def default_llm_deployment_id() -> str | None:
    return resolve_llm_config().llm_deployment_id


def default_nim_deployment_id() -> str | None:
    return resolve_llm_config().llm_nim_deployment_id


def default_assume_native_tool_calling_when_unmapped() -> bool:
    """Return the CrewAI native-tool-calling override for unmapped NIM models.

    Like :func:`get_max_history_messages_default`, this is a genai-internal tunable
    read off genai's own :class:`Config` rather than per-LLM config, so it does not
    go through the app config seam.
    """
    return Config().assume_native_tool_calling_when_unmapped
