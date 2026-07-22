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

import logging
from collections.abc import Callable
from enum import StrEnum
from typing import Any
from typing import cast

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from datarobot.core.config import getenv
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)

DEFAULT_MAX_HISTORY_MESSAGES = 20
DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM = "datarobot/datarobot-deployed-llm"
DEFAULT_DATAROBOT_ENDPOINT = "https://app.datarobot.com/api/v2"
DEFAULT_LLM_NAME = "llm"


class LLMType(StrEnum):
    GATEWAY = "gateway"
    DEPLOYMENT = "deployment"
    NIM = "nim"
    EXTERNAL = "external"


def _with_datarobot_prefix(model_name: str) -> str:
    return model_name if model_name.startswith("datarobot/") else "datarobot/" + model_name


class LLMConfig(BaseModel):
    """One LLM instance's resolved connection parameters. A value object.

    There can be MANY of these, one per configured LLM instance. It carries only
    per-LLM routing fields (which LLM, how to reach it), never app-wide settings.
    Produce one with :func:`resolve_llm_config` (mapped from the global
    :class:`Config`), or accept one directly (the router does this). The two
    ``datarobot_*`` globals are copied in for self-containment so the client
    builder does not have to reach back to the global config.

    This is deliberately NOT the same object as :class:`Config`. ``Config`` is the
    one global; ``LLMConfig`` is one-of-many. Keeping them separate is what lets
    ``resolve_config`` (the global) and ``resolve_llm_config`` (a single LLM) be
    genuinely different things.
    """

    datarobot_endpoint: str | None = None
    datarobot_api_token: str | None = None
    llm_deployment_id: str | None = None
    llm_nim_deployment_id: str | None = None
    llm_use_datarobot_llm_gateway: bool = True
    llm_default_model: str | None = None

    def get_llm_type(self) -> LLMType:
        if self.llm_use_datarobot_llm_gateway:
            return LLMType.GATEWAY
        elif self.llm_deployment_id:
            return LLMType.DEPLOYMENT
        elif self.llm_nim_deployment_id:
            return LLMType.NIM
        else:
            return LLMType.EXTERNAL

    def to_litellm_params(self) -> dict:
        """Return a litellm_params dict suitable for ``litellm.Router``'s model_list.

        Endpoint and api_key fall back to the resolved globals
        (:func:`resolve_datarobot_endpoint` / :func:`resolve_datarobot_api_token`)
        when they are not set on this instance directly.
        """
        api_key = (
            self.datarobot_api_token
            if self.datarobot_api_token is not None
            else resolve_datarobot_api_token()
        )
        endpoint = (
            self.datarobot_endpoint
            if self.datarobot_endpoint is not None
            else resolve_datarobot_endpoint()
        )
        model_name = (
            getattr(self, "model_name", None)
            or self.llm_default_model
            or default_model_name()
            or "datarobot-deployed-llm"
        )
        llm_type = self.get_llm_type()

        if llm_type == LLMType.GATEWAY:
            return {
                "model": _with_datarobot_prefix(model_name),
                "api_base": llm_gateway_url(endpoint),
                "api_key": api_key,
            }
        elif llm_type == LLMType.DEPLOYMENT:
            return {
                "model": _with_datarobot_prefix(model_name),
                "api_base": deployment_url(self.llm_deployment_id, endpoint),  # type: ignore[arg-type]
                "api_key": api_key,
            }
        elif llm_type == LLMType.NIM:
            return {
                "model": _with_datarobot_prefix(model_name),
                "api_base": deployment_url(self.llm_nim_deployment_id, endpoint),  # type: ignore[arg-type]
                "api_key": api_key,
            }
        else:  # EXTERNAL
            return {
                "model": model_name.removeprefix("datarobot/"),
                "api_key": api_key,
            }


class Config(DataRobotAppFrameworkBaseSettings):
    """The single GLOBAL application config. There is exactly one.

    Finds variables in priority order: env vars (including Runtime Parameters),
    .env, file secrets, then Pulumi outputs. It holds the two ecosystem-wide
    globals, app-wide settings, and the default LLM instance's flat,
    instance-namespaced fields so a standalone genai (no app registered) can still
    resolve the default LLM from the environment.

    This is NOT an :class:`LLMConfig`. genai never reads LLM routing fields off it
    directly; those are mapped into an :class:`LLMConfig` by
    :func:`resolve_llm_config`, and the two globals are read only through
    :func:`resolve_datarobot_endpoint` / :func:`resolve_datarobot_api_token`.
    """

    # True ecosystem-wide globals. Fixed names, shared by every LLM instance.
    datarobot_endpoint: str = DEFAULT_DATAROBOT_ENDPOINT
    datarobot_api_token: str | None = None

    # App-wide setting (genai-specific tunable).
    max_history_messages: int = Field(
        default=DEFAULT_MAX_HISTORY_MESSAGES, ge=0, alias="datarobot_genai_max_history_messages"
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
# a config attribute directly. The two globals go through resolve_datarobot_endpoint()
# / resolve_datarobot_api_token(); per-LLM routing goes through resolve_llm_config().

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
    """Fail loud if an injected global config is missing the required globals.

    genai cannot ``isinstance``-check the app's ``Config`` (a different class it
    cannot import), so this is a structural (duck-typed) check that the injected
    object at least exposes the two globals every consumer relies on.
    """
    missing = [
        name for name in ("datarobot_endpoint", "datarobot_api_token") if not hasattr(config, name)
    ]
    if missing:
        raise TypeError(
            f"Registered config provider returned {type(config).__name__}, which is "
            f"missing required global field(s): {', '.join(missing)}. The app config "
            "must expose datarobot_endpoint and datarobot_api_token."
        )


def resolve_config() -> Config:
    """Return the single GLOBAL application config.

    The registered app config when a provider is registered, otherwise genai's own
    env-reading :class:`Config` (a standalone genai with no app around it). Only
    the two globals are read off this, and only via
    :func:`resolve_datarobot_endpoint` / :func:`resolve_datarobot_api_token`. For
    LLM routing use :func:`resolve_llm_config`.
    """
    provider = _provider_registry["provider"]
    if provider is not None:
        provided = provider()
        if provided is not None:
            _validate_global_config(provided)
            # The app's Config is a structurally-compatible settings object; genai
            # only reads it through the resolver helpers (getattr), so treat it as
            # a Config for typing purposes.
            return cast(Config, provided)
    return Config()


def resolve_datarobot_endpoint() -> str:
    """Resolve the DataRobot endpoint global. The only place it is read off config."""
    endpoint: str | None = getattr(resolve_config(), "datarobot_endpoint", None)
    return endpoint or DEFAULT_DATAROBOT_ENDPOINT


def resolve_datarobot_api_token() -> str | None:
    """Resolve the DataRobot API token global. The only place it is read off config."""
    token: str | None = getattr(resolve_config(), "datarobot_api_token", None)
    return token or None


# --- Deprecated LLM config parameter bridge (REMOVE IN A FUTURE RELEASE) -------
#
# Two per-LLM params were renamed to the universal ``{instance}_`` namespace:
#   nim_deployment_id          -> {instance}_nim_deployment_id
#   use_datarobot_llm_gateway  -> {instance}_use_datarobot_llm_gateway
#
# Old deployments still carry the pre-rename bare runtime-parameter names. Those
# bare names are no longer declared config fields, and the settings base class is
# ``extra="ignore"``, so they never land on the resolved config object; they can
# only be read straight from the environment. This bridge does exactly that, then
# warns loudly. It WILL BE REMOVED IN A FUTURE RELEASE. New deployments set only
# the namespaced params and never reach this path.


def _coerce_bool(value: object) -> bool:
    """Coerce a runtime-parameter value (a bool, or a ``"1"``/``"0"`` string) to bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _deprecated_param(old_name: str, new_name: str) -> object | None:
    """Read a renamed LLM param from its pre-rename runtime-parameter name.

    Returns ``None`` when the old param is absent. When present, emits a loud,
    hard-to-miss deprecation warning and returns the old value. REMOVE IN A
    FUTURE RELEASE together with the rest of this bridge.
    """
    old_value = getenv(old_name)
    if old_value is None:
        return None
    banner = "!" * 88
    logger.warning(
        "\n%s\n"
        "DEPRECATED LLM CONFIG PARAMETER IN USE\n"
        "  Runtime parameter %r is DEPRECATED and WILL BE REMOVED IN A FUTURE RELEASE.\n"
        "  Rename it to %r as soon as possible.\n"
        "  Falling back to the deprecated value for now.\n"
        "%s",
        banner,
        old_name,
        new_name,
        banner,
    )
    return old_value


def resolve_llm_config(name: str | None = None) -> LLMConfig:
    """Resolve ONE LLM instance's config from the global config.

    ``name`` is the LLM component instance name; when omitted it is the app's
    registered default (``"llm"`` for a standalone genai). Reads the instance's
    ``{name}_*`` fields off :func:`resolve_config` and folds in the two globals,
    producing a self-contained :class:`LLMConfig`. This is the one machine that
    turns app config into an LLM config, and it is what ``get_llm`` runs on.
    """
    config = resolve_config()
    instance = name if name is not None else cast(str, _provider_registry["default_llm_name"])
    # Read everything off a single resolve_config() so the provider is invoked
    # exactly once per resolution (values still re-resolve on the next call).
    endpoint: str | None = getattr(config, "datarobot_endpoint", None)
    api_token: str | None = getattr(config, "datarobot_api_token", None)

    # --- deprecated-name backwards-compat bridge (REMOVE IN A FUTURE RELEASE) ---
    # Fall back to the pre-rename bare param names only when the namespaced field
    # was not explicitly provided. ``model_fields_set`` is what distinguishes an
    # explicit value from a default (required for the bool, whose default is True).
    set_fields = getattr(config, "model_fields_set", ())

    llm_nim_deployment_id = getattr(config, f"{instance}_nim_deployment_id", None)
    if f"{instance}_nim_deployment_id" not in set_fields:
        old = _deprecated_param("NIM_DEPLOYMENT_ID", f"{instance.upper()}_NIM_DEPLOYMENT_ID")
        if old is not None:
            llm_nim_deployment_id = str(old)

    llm_use_datarobot_llm_gateway = getattr(config, f"{instance}_use_datarobot_llm_gateway", True)
    if f"{instance}_use_datarobot_llm_gateway" not in set_fields:
        old = _deprecated_param(
            "USE_DATAROBOT_LLM_GATEWAY", f"{instance.upper()}_USE_DATAROBOT_LLM_GATEWAY"
        )
        if old is not None:
            llm_use_datarobot_llm_gateway = _coerce_bool(old)
    # --- end bridge ---

    return LLMConfig(
        datarobot_endpoint=endpoint or DEFAULT_DATAROBOT_ENDPOINT,
        datarobot_api_token=api_token or None,
        llm_deployment_id=getattr(config, f"{instance}_deployment_id", None),
        llm_nim_deployment_id=llm_nim_deployment_id,
        llm_use_datarobot_llm_gateway=llm_use_datarobot_llm_gateway,
        llm_default_model=getattr(config, f"{instance}_default_model", None),
    )


def get_max_history_messages_default() -> int:
    """Return the default maximum number of history messages.

    This is a genai-internal tunable (``DATAROBOT_GENAI_MAX_HISTORY_MESSAGES``),
    read off genai's own :class:`Config`, not per-LLM config. Invalid values fall
    back to the built-in default; negative values are treated as 0 (disable history).
    """
    return max(Config().max_history_messages, 0)


def default_api_key() -> str | None:
    return resolve_datarobot_api_token()


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


def deployment_url(deployment_id: str, datarobot_endpoint: str) -> str:
    return f"{datarobot_endpoint}/deployments/{deployment_id}/chat/completions"


def default_deployment_url(deployment_id: str | None = None) -> str:
    resolved_id = deployment_id or resolve_llm_config().llm_deployment_id
    if resolved_id is None:
        raise ValueError("Neither deployment ID nor default deployment ID is set")

    return deployment_url(resolved_id, resolve_datarobot_endpoint())


def llm_gateway_url(datarobot_endpoint: str) -> str:
    return datarobot_endpoint.removesuffix("/api/v2")


def default_datarobot_llm_gateway_url() -> str:
    return llm_gateway_url(resolve_datarobot_endpoint())


def default_llm_deployment_id() -> str | None:
    return resolve_llm_config().llm_deployment_id


def default_nim_deployment_id() -> str | None:
    return resolve_llm_config().llm_nim_deployment_id
