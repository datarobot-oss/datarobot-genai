# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from http import HTTPStatus
from logging import getLogger

from starlette.requests import Request
from starlette.responses import JSONResponse

from datarobot_genai import __version__
from datarobot_genai.drtools.core.auth import set_request_headers_for_context

from .config import get_config
from .dynamic_prompts.controllers import delete_registered_prompt_template
from .dynamic_prompts.controllers import refresh_registered_prompt_template
from .dynamic_prompts.controllers import register_prompt_from_prompt_template_id_and_version
from .dynamic_tools.deployment.controllers import delete_registered_tool_deployment
from .dynamic_tools.deployment.controllers import get_registered_tool_deployments
from .dynamic_tools.deployment.controllers import register_tool_for_deployment_id
from .exceptions import DynamicPromptRegistrationError
from .mcp_instance import DataRobotMCP
from .routes_utils import prefix_mount_path
from .tool_config import TOOL_CONFIGS
from .tool_config import ToolType
from .utils import get_prompt_tags
from .utils import get_resource_tags
from .utils import get_tool_tags

logger = getLogger(__name__)


def register_routes(mcp: DataRobotMCP) -> None:
    """Register all routes with the MCP server."""

    @mcp.custom_route(prefix_mount_path("/"), methods=["GET"])
    async def handle_health(_: Request) -> JSONResponse:
        return JSONResponse(
            status_code=HTTPStatus.OK,
            content={
                "status": "healthy",
                "message": "DataRobot MCP Server is running",
            },
        )

    @mcp.custom_route(prefix_mount_path("/metadata"), methods=["GET"])
    async def get_metadata(_: Request) -> JSONResponse:
        """Get metadata about tools, prompts, resources, and system configuration."""
        try:
            # Get tools with tags
            tools = await mcp.list_tools()
            tools_metadata = [
                {
                    "name": tool.name,
                    "tags": sorted(list(get_tool_tags(tool))),
                }
                for tool in tools
            ]

            # Get prompts with tags
            prompts = await mcp.list_prompts()
            prompts_metadata = [
                {
                    "name": prompt.name,
                    "tags": sorted(list(get_prompt_tags(prompt))),
                }
                for prompt in prompts
            ]

            # Get resources with tags
            resources = await mcp.list_resources()
            resources_metadata = [
                {
                    "name": resource.name,
                    "tags": sorted(list(get_resource_tags(resource))),
                }
                for resource in resources
            ]

            # Get safe configuration details
            config = get_config()

            # Build tool config status
            tool_config_status = {}
            for tool_type in ToolType:
                tool_config = TOOL_CONFIGS[tool_type]
                is_enabled = getattr(config.tool_config, tool_config["config_field_name"], False)
                oauth_check_fn = tool_config["oauth_check"]
                oauth_required = oauth_check_fn is not None
                oauth_configured = None
                if oauth_required and oauth_check_fn is not None:
                    oauth_configured = oauth_check_fn(config)

                tool_config_status[tool_type.value] = {
                    "enabled": is_enabled,
                    "oauth_required": oauth_required,
                    "oauth_configured": oauth_configured,
                }

            # Safe config details (excluding sensitive information)
            safe_config = {
                "server": {
                    "name": config.mcp_server_name,
                    "port": config.mcp_server_port,
                    "log_level": config.mcp_server_log_level,
                    "app_log_level": config.app_log_level,
                    "mount_path": config.mount_path,
                    "drmcp_genai_version": __version__,
                },
                "features": {
                    "register_dynamic_tools_on_startup": (
                        config.mcp_server_register_dynamic_tools_on_startup
                    ),
                    "register_dynamic_prompts_on_startup": (
                        config.mcp_server_register_dynamic_prompts_on_startup
                    ),
                    "tool_registration_allow_empty_schema": (
                        config.tool_registration_allow_empty_schema
                    ),
                    "tool_registration_duplicate_behavior": (
                        config.tool_registration_duplicate_behavior
                    ),
                    "prompt_registration_duplicate_behavior": (
                        config.prompt_registration_duplicate_behavior
                    ),
                },
                "tool_config": tool_config_status,
            }

            return JSONResponse(
                status_code=HTTPStatus.OK,
                content={
                    "tools": {
                        "items": tools_metadata,
                        "count": len(tools_metadata),
                    },
                    "prompts": {
                        "items": prompts_metadata,
                        "count": len(prompts_metadata),
                    },
                    "resources": {
                        "items": resources_metadata,
                        "count": len(resources_metadata),
                    },
                    "config": safe_config,
                },
            )
        except Exception as e:
            logger.exception("Failed to retrieve metadata")
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"error": f"Failed to retrieve metadata: {str(e)}"},
            )

    @mcp.custom_route(prefix_mount_path("/registeredDeployments/{deployment_id}"), methods=["PUT"])
    async def add_deployment(request: Request) -> JSONResponse:
        """Add or update a deployment with a known deployment_id."""
        deployment_id = request.path_params["deployment_id"]
        try:
            tool = await register_tool_for_deployment_id(deployment_id)
            return JSONResponse(
                status_code=HTTPStatus.CREATED,
                content={
                    "name": tool.name,
                    "description": tool.description,
                    "tags": list(tool.tags),
                    "deploymentId": deployment_id,
                },
            )
        except Exception as e:
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST,
                content={"error": f"Failed to add deployment: {str(e)}"},
            )

    @mcp.custom_route(prefix_mount_path("/registeredDeployments"), methods=["GET"])
    async def list_deployments(_: Request) -> JSONResponse:
        """List all deployments."""
        try:
            deployments = await get_registered_tool_deployments()
            formatted_deployments = [
                {"deploymentId": k, "toolName": v} for k, v in deployments.items()
            ]
            return JSONResponse(
                status_code=HTTPStatus.OK,
                content={
                    "deployments": formatted_deployments,
                    "count": len(deployments),
                },
            )
        except Exception as e:
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"error": f"Failed to retrieve deployments: {str(e)}"},
            )

    @mcp.custom_route(
        prefix_mount_path("/registeredDeployments/{deployment_id}"), methods=["DELETE"]
    )
    async def delete_deployment(request: Request) -> JSONResponse:
        """Delete (de-register) a deployment by deployment_id."""
        deployment_id = request.path_params["deployment_id"]
        try:
            deleted = await delete_registered_tool_deployment(deployment_id)
            if deleted is True:
                return JSONResponse(
                    status_code=HTTPStatus.OK,
                    content={
                        "message": f"Tool with deployment {deployment_id} deleted successfully"
                    },
                )
            return JSONResponse(
                status_code=HTTPStatus.NOT_FOUND,
                content={"error": f"Tool with deployment {deployment_id} not found"},
            )
        except Exception as e:
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"error": f"Failed to delete deployment: {str(e)}"},
            )

    @mcp.custom_route(prefix_mount_path("/registeredPrompts"), methods=["GET"])
    async def list_prompt_templates(_: Request) -> JSONResponse:
        """List all prompt templates."""
        try:
            prompts = await mcp.get_prompt_mapping()
            formatted_prompts = [
                {
                    "promptTemplateId": pt_id,
                    "promptTemplateVersionId": ptv_id,
                    "promptName": p_name,
                }
                for pt_id, (ptv_id, p_name) in prompts.items()
            ]
            return JSONResponse(
                status_code=HTTPStatus.OK,
                content={
                    "promptTemplates": formatted_prompts,
                    "count": len(formatted_prompts),
                },
            )
        except Exception as e:
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"error": f"Failed to retrieve promptTemplates: {str(e)}"},
            )

    @mcp.custom_route(
        prefix_mount_path("/registeredPrompts/{prompt_template_id}"), methods=["DELETE"]
    )
    async def delete_prompt_template(request: Request) -> JSONResponse:
        """Delete (de-register) a prompt by prompt_template_id."""
        prompt_template_id = request.path_params["prompt_template_id"]
        try:
            deleted = await delete_registered_prompt_template(prompt_template_id)
            if deleted:
                return JSONResponse(
                    status_code=HTTPStatus.OK,
                    content={
                        "message": f"Prompt with prompt template id {prompt_template_id} "
                        f"deleted successfully"
                    },
                )
            return JSONResponse(
                status_code=HTTPStatus.NOT_FOUND,
                content={"error": f"Prompt with prompt template id {prompt_template_id} not found"},
            )
        except Exception as e:
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"error": f"Failed to delete prompt: {str(e)}"},
            )

    @mcp.custom_route(
        prefix_mount_path("/registeredPrompts/{prompt_template_id}"),
        methods=["PUT"],
    )
    async def add_prompt_template(request: Request) -> JSONResponse:
        """Add or update prompt template."""
        prompt_template_id = request.path_params["prompt_template_id"]
        prompt_template_version_id = request.query_params.get("promptTemplateVersionId")
        # Ensure token resolution sees this REST request (not only the open MCP stream).
        try:
            set_request_headers_for_context({k.lower(): v for k, v in request.headers.items()})
        except (AttributeError, TypeError):
            pass
        try:
            prompt = await register_prompt_from_prompt_template_id_and_version(
                prompt_template_id,
                prompt_template_version_id,
                headers_auth_only=False,
            )
            return JSONResponse(
                status_code=HTTPStatus.CREATED,
                content={
                    "name": prompt.name,
                    "description": prompt.description,
                    "promptTemplateId": prompt_template_id,
                    "promptTemplateVersionId": prompt.meta["prompt_template_version_id"],
                },
            )
        except DynamicPromptRegistrationError as e:
            return JSONResponse(
                status_code=HTTPStatus.NOT_FOUND,
                content={"error": f"Failed to add prompt template: {str(e) or 'not found'}"},
            )
        except Exception as e:
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"error": f"Failed to add prompt template: {str(e)}"},
            )

    @mcp.custom_route(prefix_mount_path("/registeredPrompts"), methods=["PUT"])
    async def refresh_prompt_templates(_: Request) -> JSONResponse:
        """Refresh prompt templates."""
        try:
            await refresh_registered_prompt_template(headers_auth_only=True)
            return JSONResponse(
                status_code=HTTPStatus.OK,
                content={"message": "Prompts refreshed successfully"},
            )
        except Exception as e:
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"error": f"Failed to refresh prompt templates: {str(e)}"},
            )
