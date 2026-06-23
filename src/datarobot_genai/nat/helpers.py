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
from contextlib import asynccontextmanager
from typing import Any

from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.runtime.runner import Context
from nat.runtime.session import SessionManager
from nat.utils.data_models.schema_validator import validate_schema
from nat.utils.io.yaml_tools import yaml_load
from nat.utils.type_utils import StrPath

from datarobot_genai.core.chat.auth import get_authorization_context_from_headers
from datarobot_genai.core.utils.auth import prepare_identity_header
from datarobot_genai.dragent.workflow_paths import publish_dragent_config_file_env

logger = logging.getLogger(__name__)

DATAROBOT_MODERATION_MIDDLEWARE_TYPE = "datarobot_moderation"


def remove_datarobot_moderation_middleware(config_yaml: dict[str, Any]) -> None:
    """Remove ``datarobot_moderation`` middleware entries from a NAT workflow config.

    DRUM applies LLM guardrails via ``moderation_config.yaml`` outside the NAT workflow.
    When ``NatAgent`` loads a shared ``workflow.yaml`` through DRUM, strip the NAT
    middleware so guardrails are not applied twice.
    """
    middleware_section = config_yaml.get("middleware")
    if not isinstance(middleware_section, dict):
        return

    removed_names = [
        name
        for name, cfg in middleware_section.items()
        if isinstance(cfg, dict) and cfg.get("_type") == DATAROBOT_MODERATION_MIDDLEWARE_TYPE
    ]
    if not removed_names:
        return

    for name in removed_names:
        del middleware_section[name]
    if not middleware_section:
        config_yaml.pop("middleware", None)

    removed = set(removed_names)
    _strip_middleware_references(config_yaml, removed)
    logger.info(
        "Removed datarobot_moderation middleware for DRUM execution: %s",
        ", ".join(sorted(removed)),
    )


def _strip_middleware_references(node: Any, removed: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key == "middleware" and isinstance(value, list):
                filtered = [name for name in value if name not in removed]
                if filtered:
                    node[key] = filtered
                else:
                    del node[key]
            else:
                _strip_middleware_references(value, removed)
    elif isinstance(node, list):
        for item in node:
            _strip_middleware_references(item, removed)


def load_config(
    config_file: StrPath,
    headers: dict[str, str] | None = None,
    *,
    disable_datarobot_moderation: bool = False,
) -> Config:
    """
    Load a NAT configuration file with injected headers. It ensures that all plugins are
    loaded and then validates the configuration file against the Config schema.

    Parameters
    ----------
    config_file : StrPath
        The path to the configuration file
    disable_datarobot_moderation : bool, optional
        When ``True``, strip ``datarobot_moderation`` middleware from the loaded config.
        Used by ``NatAgent`` on the DRUM path where guardrails are applied externally.

    Returns
    -------
    Config
        The validated Config object
    """
    # Ensure all of the plugins are loaded
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    config_yaml = yaml_load(config_file)

    if disable_datarobot_moderation:
        remove_datarobot_moderation_middleware(config_yaml)

    add_headers_to_datarobot_mcp_auth(config_yaml, headers)
    add_headers_to_datarobot_llm_deployment(config_yaml, headers)

    # Validate configuration adheres to NAT schemas
    validated_nat_config = validate_schema(config_yaml, Config)

    return validated_nat_config


def add_headers_to_datarobot_mcp_auth(config_yaml: dict, headers: dict[str, str] | None) -> None:
    if headers:
        if authentication := config_yaml.get("authentication"):
            keys_present = {k.lower() for k in headers.keys()}
            extra_api_token_keys = {"x-datarobot-api-token", "x-datarobot-api-key"}
            has_extra_api_token_header = extra_api_token_keys & keys_present
            # Do not mutate the caller's headers; give each auth config its own copy.
            if "Authorization" in headers and not has_extra_api_token_header:
                token = headers["Authorization"].removeprefix("Bearer ")
                headers_copy = {**headers, "x-datarobot-api-token": token}
            else:
                headers_copy = dict(headers)
            for auth_name in authentication:
                auth_config = authentication[auth_name]
                if auth_config.get("_type") == "datarobot_mcp_auth":
                    auth_config["headers"] = dict(headers_copy)


def add_headers_to_datarobot_llm_deployment(
    config_yaml: dict, headers: dict[str, str] | None
) -> None:
    if not headers:
        return

    config_types_to_update = ("datarobot-llm-deployment", "datarobot-llm-component")
    if identity_header := prepare_identity_header(headers):
        if llms := config_yaml.get("llms"):
            for llm_name in llms:
                llm_config = llms[llm_name]
                if llm_config.get("_type") in config_types_to_update:
                    llm_config["headers"] = identity_header


@asynccontextmanager
async def load_workflow(
    config_file: StrPath,
    max_concurrency: int = -1,
    headers: dict[str, str] | None = None,
    *,
    disable_datarobot_moderation: bool = False,
) -> AsyncGenerator[Workflow, None]:
    """
    Load the NAT configuration file and create a Runner object. This is the primary entry point for
    running NAT workflows with injected headers.

    Parameters
    ----------
    config_file : StrPath
        The path to the configuration file
    max_concurrency : int, optional
        The maximum number of parallel workflow invocations to support. Specifying 0 or -1 will
        allow an unlimited count, by default -1
    disable_datarobot_moderation : bool, optional
        When ``True``, strip ``datarobot_moderation`` middleware from the loaded config.
        Used by ``NatAgent`` on the DRUM path where guardrails are applied externally.
    """
    # Publish the workflow path so middleware (e.g. datarobot_moderation) can locate
    # ``moderation_config.yaml`` next to ``workflow.yaml`` without relying on CWD.
    publish_dragent_config_file_env(config_file)

    # Load the config object
    config = load_config(
        config_file,
        headers=headers,
        disable_datarobot_moderation=disable_datarobot_moderation,
    )

    # Must yield the workflow function otherwise it cleans up
    async with WorkflowBuilder.from_config(config=config) as builder:
        session_manager = await SessionManager.create(
            config=config, shared_builder=builder, max_concurrency=max_concurrency
        )

        try:
            yield session_manager
        finally:
            await session_manager.shutdown()


def extract_datarobot_headers_from_context() -> dict[str, str]:
    context = Context.get()
    headers = context.metadata.headers
    extracted_headers: dict[str, str] = {}
    if not headers:
        return extracted_headers

    for header in headers:
        # Already lowercase from NAT
        if header.startswith("x-datarobot-") or header.startswith("x-untrusted-"):
            extracted_headers[header] = headers[header]

    return extracted_headers


def extract_authorization_from_context(secret_key: str | None = None) -> dict[str, Any] | None:
    context = Context.get()
    headers = context.metadata.headers
    if not headers:
        return None

    return get_authorization_context_from_headers(headers, secret_key=secret_key)
