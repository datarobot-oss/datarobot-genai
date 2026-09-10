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

import re
import typing
import warnings
from collections.abc import AsyncGenerator

from a2a.types import AgentSkill
from nat.cli.register_workflow import register_front_end
from nat.data_models.api_server import GlobalTypeConverter
from nat.data_models.config import Config
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from ..constants import A2A_MOUNT_PATH
from ..cross_app_access_config import CrossApplicationAccessConfig
from .converters import convert_chat_request_to_run_agent_input
from .converters import convert_dragent_event_response_to_chat_response
from .converters import convert_dragent_event_response_to_chat_response_chunk
from .converters import convert_dragent_event_response_to_str
from .converters import convert_dragent_run_agent_input_to_chat_request
from .converters import convert_dragent_run_agent_input_to_chat_request_or_message
from .converters import convert_run_agent_input_to_chat_request_or_message
from .converters import convert_str_to_chat_response
from .converters import convert_tool_message_to_str
from .logging import logging_handler_setup

# Suppress specific non-actionable NAT warning messages by content.
# Patch Handler.handle (inherited by all subclasses - they only override emit)
# because root-logger filters are skipped during log propagation.
logging_handler_setup()

# Suppress UserWarning from langchain about non-default parameters (uses warnings.warn, not logging)
warnings.filterwarnings("ignore", message=".*stream_options is not default parameter.*")


#: Characters RFC 3986 §2.3 calls ``unreserved`` — the set that never needs percent-encoding
#: anywhere in a URI. ``a2a.mount_path`` segments are held to exactly this alphabet: the value
#: is interpolated raw into both a Starlette route pattern and the advertised agent card URL,
#: so anything requiring encoding (or carrying URI syntax like ``?``/``#``/``%``) would make one
#: of the two wrong. See :meth:`DRAgentA2AConfig._normalize_mount_path`.
_MOUNT_PATH_SEGMENT_RE = re.compile(r"[A-Za-z0-9._~-]+")

#: Generous ceiling; a mount path far shorter than this is already a design smell.
_MOUNT_PATH_MAX_LENGTH = 200

#: Closing advice shared by every ``mount_path`` rejection, so the messages carry the
#: fix once rather than each restating it.
_MOUNT_PATH_ADVICE = (
    'Use a plain path segment, e.g. "agent" or "api/a2a", or omit it to use the default "a2a".'
)


def _reject_mount_path(reason: str, got: str) -> typing.NoReturn:
    """Raise a ``mount_path`` rejection: what is wrong, what was seen, then shared advice.

    ``got`` is pre-rendered rather than a value to repr, so each rejection can show the
    part that is actually useful — the whole value, just the offending segment, or a
    length where echoing a too-long value would only add noise.
    """
    raise ValueError(f"a2a.mount_path {reason} (got {got}). {_MOUNT_PATH_ADVICE}")


class DRAgentA2AExternalConfig(BaseModel):
    """Customer-provided external identity and URL override for the agent card."""

    id: str | None = Field(
        default=None, description="External agent identifier for catalog discovery."
    )
    url: str | None = Field(
        default=None, description="Custom external URL override for the agent card endpoint."
    )


class DRAgentA2AConfig(BaseModel):
    """DR-owned wrapper around NAT's A2AFrontEndConfig with optional skill definitions."""

    server: A2AFrontEndConfig = Field(description="NAT A2A server configuration.")
    cross_application_access: CrossApplicationAccessConfig | None = Field(
        default=None,
        description=(
            "Configuration for Cross-Application Access utilizing a hybrid RFC 8693 / "
            "RFC 7523 flow."
        ),
    )
    skills: list[AgentSkill] = Field(
        default=[],
        description="Skills to advertise in the A2A agent card. "
        "If empty, a single default skill is generated from the agent name and description.",
    )
    external: DRAgentA2AExternalConfig | None = Field(
        default=None,
        description="External identity and URL override for the agent card.",
    )
    oauth_claim_validation: bool = Field(
        default=False,
        description=(
            "Opt in to L2 validation of the inbound IdP token's claims. Off unless set to "
            "true, so an agent never starts enforcing because ``cross_application_access`` "
            "was filled in for the agent card. Today this governs the ``aud`` claim, taken "
            "from ``cross_application_access.token_request.audience``; ``scope`` joins it "
            "under the same flag. Applies to every route, not only /a2a, because an inbound "
            "token reaches the workflow the same way whichever route it arrives on."
        ),
    )
    enable_unauthenticated_well_known_route: bool = Field(
        default=False,
        description=(
            "Per-agent developer opt-in for unauthenticated "
            "GET /.well-known/agent-card.json. Also requires platform-level "
            "opt-in per cluster to route unauthenticated traffic to the agent. "
            "When disabled (default), unauthenticated requests receive the same "
            "generic 404 as a nonexistent agent, so the refusal does not reveal "
            "that the agent exists. When enabled, anonymous callers receive a "
            "redacted agent card."
        ),
    )
    mount_path: str = Field(
        default=A2A_MOUNT_PATH,
        # Pydantic skips validators for defaults unless asked; without this the default
        # would be the one value never normalized or checked.
        validate_default=True,
        description=(
            "Path suffix the A2A server is mounted under, relative to the app root. "
            'Defaults to "a2a"; set to a different value, e.g. "agent" or "api/a2a", to '
            'mount it elsewhere. Leading and trailing slashes are stripped, so "/a2a/" '
            'and "a2a" are equivalent. The advertised agent card URL follows this value '
            "automatically, and the agent card is additionally served at the root "
            ".well-known/agent-card.json as a discovery fallback. "
            "Each slash-separated segment must be made of RFC 3986 unreserved characters "
            "(letters, digits, and - . _ ~) and must not begin with a dot. Mounting at "
            "the application root is not supported: an empty value is rejected. Mounting "
            "on a path already served by another route is rejected at startup."
        ),
    )

    @field_validator("mount_path")
    @classmethod
    def _normalize_mount_path(cls, value: str) -> str:
        """Normalize ``mount_path`` and reject values that cannot safely be a path suffix.

        This value is interpolated raw into two different places — a Starlette route
        pattern (``app.mount(f"/{mount_path}", ...)``) and the ``url`` advertised in the
        agent card — so it has to be valid as both. Nothing downstream re-checks it:
        ``a2a.types.AgentCard.url`` is an unvalidated ``str`` despite documenting that it
        must be a valid absolute URL, so this validator is the only enforcement point.

        Surrounding slashes are stripped, so ``"a2a"`` and ``"/a2a/"`` agree; interior
        slashes are kept for multi-segment mounts such as ``"api/a2a"``.

        Four rejections, each covering a failure that is otherwise silent:

        * **Empty** (``""``, ``"/"``): would mount A2A at the application root, where its
          catch-all ``Mount`` shadows any route registered after it, with no error.
        * **Characters outside RFC 3986 §2.3 ``unreserved``**: ``{``/``}`` are Starlette
          path-parameter syntax — ``"{id}"`` compiles to a wildcard that swallows unrelated
          paths, and ``"{path}"`` collides with ``Mount``'s own parameter and crashes at
          startup. ``?``, ``#``, ``%`` and whitespace are URI syntax or need encoding, so
          the advertised URL resolves somewhere other than the mount and A2A becomes
          silently undiscoverable.
        * **Empty interior segments** (``"a2a//nested"``): advertises a URL clients and
          proxies normalize differently than the route matches.
        * **Dot-leading segments** (``"."``, ``".."``, ``".well-known"``): ``.``/``..`` are
          relative-reference syntax that clients resolve away, and RFC 8615 reserves
          ``/.well-known/`` as a registry-controlled namespace which the A2A protocol
          itself uses for agent card discovery.
        """
        normalized = value.strip().strip("/")
        if not normalized:
            _reject_mount_path(
                "must not be empty; mounting A2A at the application root is not supported",
                repr(value),
            )
        if len(normalized) > _MOUNT_PATH_MAX_LENGTH:
            # Length only: echoing an over-long value back would bury the message in it.
            _reject_mount_path(
                f"must be at most {_MOUNT_PATH_MAX_LENGTH} characters",
                f"{len(normalized)}",
            )
        for segment in normalized.split("/"):
            if not segment:
                _reject_mount_path(
                    "must not contain an empty path segment; use a single slash between segments",
                    repr(value),
                )
            if segment.startswith("."):
                _reject_mount_path(
                    "segments must not start with a dot: dot segments are "
                    "relative-reference syntax that clients resolve away, and "
                    "/.well-known/ is reserved by RFC 8615 for discovery",
                    repr(value),
                )
            if not _MOUNT_PATH_SEGMENT_RE.fullmatch(segment):
                _reject_mount_path(
                    "segments must be made of letters, digits, and - . _ ~ (RFC 3986 "
                    "unreserved characters): the value is used both as the mount point "
                    "and in the agent card URL, so anything needing percent-encoding or "
                    "carrying URI syntax would make one of the two wrong",
                    # Names the offending segment, which a multi-segment value would
                    # otherwise leave the reader to spot.
                    repr(segment) if segment == normalized else f"{segment!r} in {value!r}",
                )
        return normalized


# Register frontend
class DRAgentFastApiFrontEndConfig(FastApiFrontEndConfig, name="dragent_fastapi"):  # type: ignore
    a2a: DRAgentA2AConfig | None = Field(
        default=None,
        description="Expose this agent via the Agent2Agent protocol. "
        "A2A server endpoints are mounted under /a2a/ by default; set a2a.mount_path "
        "to serve them from a different suffix.",
    )
    workflow: typing.Annotated[
        FastApiFrontEndConfig.EndpointBase,
        Field(description="Endpoint for the default workflow."),
    ] = FastApiFrontEndConfig.EndpointBase(
        method="POST",
        path="/v1/workflow",
        openai_api_v1_path="/chat/completions",
        legacy_path="/generate",
        legacy_openai_api_path="/chat",
        description="Executes the default NAT workflow from the loaded configuration ",
    )


@register_front_end(config_type=DRAgentFastApiFrontEndConfig)
async def dragent_fastapi_front_end(
    config: DRAgentFastApiFrontEndConfig, full_config: Config
) -> AsyncGenerator[typing.Any, None]:
    from .fastapi import DRAgentFastApiFrontEndPlugin

    yield DRAgentFastApiFrontEndPlugin(full_config=full_config)


# Register console frontend for `nat dragent run`
from .console import DRAgentConsoleFrontEndConfig  # noqa: E402


@register_front_end(config_type=DRAgentConsoleFrontEndConfig)
async def dragent_console_front_end(
    config: DRAgentConsoleFrontEndConfig, full_config: Config
) -> AsyncGenerator[typing.Any, None]:
    from .console import DRAgentConsoleFrontEndPlugin

    yield DRAgentConsoleFrontEndPlugin(full_config=full_config)


# Register converters
GlobalTypeConverter.register_converter(convert_dragent_run_agent_input_to_chat_request)
GlobalTypeConverter.register_converter(convert_chat_request_to_run_agent_input)
GlobalTypeConverter.register_converter(convert_dragent_run_agent_input_to_chat_request_or_message)
GlobalTypeConverter.register_converter(convert_run_agent_input_to_chat_request_or_message)
GlobalTypeConverter.register_converter(convert_tool_message_to_str)
GlobalTypeConverter.register_converter(convert_dragent_event_response_to_str)
GlobalTypeConverter.register_converter(convert_dragent_event_response_to_chat_response_chunk)
# Direct DRAgentEventResponse -> ChatResponse so the non-streaming /chat/completions and
# inline paths preserve ``datarobot_moderations`` instead of routing through the lossy
# DRAgentEventResponse -> str -> ChatResponse path.
GlobalTypeConverter.register_converter(convert_dragent_event_response_to_chat_response)
# Overrides NAT's built-in str -> ChatResponse so the non-streaming /chat/completions
# response reports the agent's configured model instead of NAT's "unknown-model" default.
GlobalTypeConverter.register_converter(convert_str_to_chat_response)
