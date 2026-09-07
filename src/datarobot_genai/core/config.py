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

import re
import warnings
from collections.abc import Callable
from typing import Any
from typing import cast

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from datarobot.core.config import LLMConfig  # noqa: F401  # re-exported for genai consumers
from datarobot.core.config import LLMType  # noqa: F401  # re-exported for genai consumers
from datarobot.core.config import MCPServerRef
from datarobot.core.config import MCPServersSettingsSource
from datarobot.core.config import deployment_url
from datarobot.core.config import getenv
from datarobot.core.config import llm_gateway_url
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from pydantic_settings.exceptions import SettingsError

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
    directly; those are mapped into an :class:`LLMConfig` by its
    ``resolve_llm_config(name=...)`` method, and the two globals through the base
    class's ``resolve_datarobot_endpoint`` / ``resolve_datarobot_api_token`` methods.
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

    # Every MCP server a standalone genai can reach. Generic: adding a server is an
    # entry in MCP_SERVERS, never an edit here. An app that registers its own config
    # declares the same field; see the seam below.
    mcp_servers: list[MCPServerRef] = []

    # The singular pre-`mcp_servers` settings, each of which addressed the one server an
    # agent could reach. Declared so they keep resolving through all the normal sources,
    # as the entry named `default`; `resolve_mcp_servers()` performs that synthesis and
    # rejects setting two of them. Removed once the deprecation period ends.
    mcp_deployment_id: str | None = None
    mcp_workload_id: str | None = None
    mcp_server_port: int | None = None
    external_mcp_url: str | None = None
    external_mcp_headers: str | None = None
    external_mcp_transport: str | None = None

    # Restated rather than inherited: declaring model_config on a subclass replaces the
    # base class's entirely. `env_ignore_empty` is the part worth having -- without it an
    # empty `MCP_DEPLOYMENT_ID=` left in a container image is a *set* environment
    # variable, which outranks the runtime parameter infra deliberately set and resolves
    # the field to "".
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    @classmethod
    def settings_customise_sources(  # type: ignore[override]
        cls,
        settings_cls: type[Any],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        """Append the MCP servers source, so flat per-server variables resolve.

        A standalone genai -- no application, no registered provider -- must still be
        able to reach an MCP server, for the same reason this class holds the default
        LLM instance's flat fields. Without the source, `weather_mcp_deployment_id`
        resolves to nothing here and the symptom is an empty fleet rather than an error.

        An application that registers its own config adds the same source to it. Both
        must, or whether a server resolves depends on whether registration happened --
        which is the hardest class of configuration bug to reproduce.
        """
        base = super().settings_customise_sources(
            settings_cls,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
        return (*base, MCPServersSettingsSource(settings_cls))


# --- App config injection seam ---------------------------------------------
#
# There are exactly two config objects and two resolvers, and they are NOT the
# same thing:
#
#   Config       - the single GLOBAL app config (endpoint, token, app-wide
#                  settings, and per-instance LLM fields). resolve_config() -> Config.
#   LLMConfig    - ONE LLM instance's routing config, obtained by calling
#                  resolve_config().resolve_llm_config(name=...), which maps the
#                  global config's {name}_* fields plus the two globals.
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
# resolve_datarobot_api_token / resolve_llm_config).
#
# DO NOT add a module-level resolve_llm_config() wrapper here. This has been
# removed five separate times and keeps coming back through refactors and rebases.
# A module-level function that hands back an LLMConfig is an attractive nuisance:
# people import it, monkeypatch it, and grow LLM-routing logic inside it, all of
# which breaks the moment a real app registers its own config.py, because the
# override lives in genai instead of on the app's config object. Always resolve
# per-LLM config at the call site off the resolved global config, with an explicit
# instance name:
#
#     llm_name = <passed in, or registered_default_llm_name()>
#     config = resolve_config()
#     llm_config = config.resolve_llm_config(name=llm_name)
#
# Callers that already hold an LLMConfig (the NAT path builds one from
# workflow.yaml) should pass it down rather than re-resolving.

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
    per-LLM fields), read back by :func:`registered_default_llm_name`. Call sites
    with no name of their own pass it to ``config.resolve_llm_config(name=...)``, so
    a non-"llm" component name still works for a bare ``get_llm()``.
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


def apply_legacy_llm_params(config: Config, name: str) -> None:
    """Pre-rename parameter shim logic until datarobot>=3.19 ships.

    !!! REMOVE WHEN datarobot>=3.19 SHIPS !!!

    Fill ``{name}_*`` fields from the pre-rename bare names, warning when used.

    Only fields the config did not set explicitly are touched, so a namespaced value
    always wins over a legacy one. Touches at most the two fields below, on the single
    default instance, and nothing at all when ``name`` is not an LLM namespace.
    """
    declared = type(config).model_fields

    if not any(
        f"{name}_{marker}" in declared
        for marker in ("deployment_id", "default_model", "nim_deployment_id")
    ):
        return

    legacy_llm_params: dict[str, tuple[str, bool]] = {
        # namespaced suffix -> (pre-rename bare name, needs bool coercion)
        "nim_deployment_id": ("NIM_DEPLOYMENT_ID", False),
        "use_datarobot_llm_gateway": ("USE_DATAROBOT_LLM_GATEWAY", True),
    }

    for suffix, (legacy_name, is_bool) in legacy_llm_params.items():
        field = f"{name}_{suffix}"
        if field in config.model_fields_set:
            continue
        raw = getenv(legacy_name)
        if raw is None:
            continue
        value: object = raw
        if is_bool and not isinstance(raw, bool):
            value = str(raw).strip().lower() in {"1", "true", "yes", "on"}
        warnings.warn(
            f"{legacy_name} is deprecated and will stop being read in a future release. "
            f"Rename it to {field.upper()}.",
            DeprecationWarning,
            stacklevel=3,
        )
        # Bypasses pydantic validation on purpose: `field` is often not declared on
        # this class at all (the old template had no per-instance NIM field), and a
        # plain setattr rejects a name that is not a model field.
        object.__setattr__(config, field, value)
        config.model_fields_set.add(field)


def _reraise_with_parse_detail(exc: SettingsError) -> None:
    """Re-raise a settings-source parse failure with the text that failed to parse.

    pydantic-settings JSON-decodes complex fields *inside the settings source*, before
    any validator runs, and reports only::

        error parsing value for field "mcp_servers" from source "EnvSettingsSource"

    -- no parse error, no offending text, no position. A stray comma in a seven-server
    MCP_SERVERS is otherwise a scavenger hunt, so dig the underlying JSON error and the
    raw value out and put them in the message.
    """
    field = None
    if match := re.search(r'field "([^"]+)"', str(exc)):
        field = match.group(1)

    cause = exc.__cause__ or exc.__context__
    detail = f"{type(cause).__name__}: {cause}" if cause is not None else "no further detail"

    raw = None
    if field:
        raw = getenv(field.upper())

    message = f"{exc} -- {detail}."
    if raw is not None:
        message += f" The value was: {raw!r}"
    if field == "mcp_servers":
        message += (
            " MCP_SERVERS must be a JSON array of objects, each with a name and exactly "
            'one address, e.g. [{"name":"analytics","deployment_id":"<24-hex>"}].'
        )
    raise SettingsError(message) from exc


def resolve_config() -> Config:
    """Return the single GLOBAL application config.

    The registered app config when a provider is registered, otherwise genai's own
    env-reading :class:`Config` (a standalone genai with no app around it).
    Everything is resolved off the returned config through the ``datarobot.core``
    base class methods (``resolve_datarobot_endpoint`` / ``resolve_datarobot_api_token``);
    for LLM routing call ``resolve_llm_config(name=...)`` on the returned config, and for
    MCP servers ``resolve_mcp_server(name=...)`` / ``resolve_mcp_servers()``.
    """
    provider = _provider_registry["provider"]
    if provider is not None:
        try:
            provided = provider()
        except SettingsError as exc:
            _reraise_with_parse_detail(exc)
            raise  # unreachable; _reraise_with_parse_detail always raises
        if provided is not None:
            _validate_global_config(provided)
            # provided is a DataRobotAppFrameworkBaseSettings subclass (validated
            # above); genai resolves everything off it through its resolve_* methods.
            # Cast to Config for typing since genai cannot import the app's class.
            app_config = cast(Config, provided)
            apply_legacy_llm_params(app_config, registered_default_llm_name())
            return app_config
    try:
        config = Config()
    except SettingsError as exc:
        _reraise_with_parse_detail(exc)
        raise  # unreachable; _reraise_with_parse_detail always raises
    apply_legacy_llm_params(config, registered_default_llm_name())
    return config


def registered_default_llm_name() -> str:
    """Return the app's registered default LLM instance name (``"llm"`` if none).

    This is a plain lookup of the name registered with
    :func:`register_config_provider`, and nothing more. It exists so a call site
    with no instance name of its own can still target the app's namespace:

        config = resolve_config()
        llm_config = config.resolve_llm_config(name=registered_default_llm_name())

    Deliberately returns a ``str`` and not an :class:`LLMConfig`. See the DO NOT
    note above: genai must not have a function that resolves an LLM config for you.
    """
    return cast(str, _provider_registry["default_llm_name"])


def get_max_history_messages_default() -> int:
    """Return the default maximum number of history messages.

    This is a genai-internal tunable (``DATAROBOT_GENAI_MAX_HISTORY_MESSAGES``),
    read off genai's own :class:`Config`, not per-LLM config. Invalid values fall
    back to the built-in default; negative values are treated as 0 (disable history).
    """
    return max(Config().datarobot_genai_max_history_messages, 0)


def default_api_key() -> str | None:
    return resolve_config().resolve_datarobot_api_token()


def default_model_name() -> str | None:
    config = resolve_config()
    return config.resolve_llm_config(name=registered_default_llm_name()).llm_default_model


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
    config = resolve_config()
    llm_config = config.resolve_llm_config(name=registered_default_llm_name())
    return llm_config.llm_use_datarobot_llm_gateway


def default_deployment_url(deployment_id: str | None = None) -> str:
    config = resolve_config()
    resolved_id = (
        deployment_id
        or config.resolve_llm_config(name=registered_default_llm_name()).llm_deployment_id
    )
    if resolved_id is None:
        raise ValueError("Neither deployment ID nor default deployment ID is set")

    return deployment_url(resolved_id, config.resolve_datarobot_endpoint())


def default_datarobot_llm_gateway_url() -> str:
    return llm_gateway_url(resolve_config().resolve_datarobot_endpoint())


def default_llm_deployment_id() -> str | None:
    config = resolve_config()
    return config.resolve_llm_config(name=registered_default_llm_name()).llm_deployment_id


def default_nim_deployment_id() -> str | None:
    config = resolve_config()
    return config.resolve_llm_config(name=registered_default_llm_name()).llm_nim_deployment_id


def default_assume_native_tool_calling_when_unmapped() -> bool:
    """Return the CrewAI native-tool-calling override for unmapped NIM models.

    Like :func:`get_max_history_messages_default`, this is a genai-internal tunable
    read off genai's own :class:`Config` rather than per-LLM config, so it does not
    go through the app config seam.
    """
    return Config().assume_native_tool_calling_when_unmapped
