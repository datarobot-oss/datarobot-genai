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
from logging import getLogger

from botocore.exceptions import ClientError
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from .dynamic_tools.deployment.controllers import delete_registered_tool_deployment
from .dynamic_tools.deployment.controllers import get_registered_tool_deployments
from .dynamic_tools.deployment.controllers import register_tool_for_deployment_id
from .memory_management import get_memory_manager
from .shared import prefix_mount_path

logger = getLogger(__name__)


def register_routes(mcp: FastMCP) -> None:
    """Register all routes with the MCP server."""

    @mcp.custom_route(prefix_mount_path("/"), methods=["GET"])
    async def handle_health(_: Request) -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "message": "DataRobot MCP Server is running",
            },
        )

    # Custom endpoint to get all tags
    @mcp.custom_route(prefix_mount_path("/tags"), methods=["GET"])
    async def handle_tags(_: Request) -> JSONResponse:
        try:
            # TaggedFastMCP extends FastMCP with get_all_tags
            tags = await mcp.get_all_tags()  # type: ignore[attr-defined]
            return JSONResponse(
                status_code=200,
                content={
                    "tags": tags,
                    "count": len(tags),
                    "message": "All available tags retrieved successfully",
                },
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Failed to retrieve tags: {str(e)}",
                },
            )

    memory_manager = get_memory_manager()
    if memory_manager:
        # Route to initialize a new storage for an agent
        @mcp.custom_route(prefix_mount_path("/agent/{agent_id}/storage/{label}"), methods=["POST"])
        async def initialize_agent_storage(request: Request) -> JSONResponse:
            agent_id = request.path_params["agent_id"]
            label = request.path_params["label"]

            # Get storage name and config from request body
            body = await request.json()
            config = body.get("config")

            # Initialize storage
            storage_id = await memory_manager.initialize_storage(
                agent_identifier=agent_id, label=label, storage_config=config
            )

            return JSONResponse(
                status_code=200,
                content={
                    "agentId": agent_id,
                    "storageId": storage_id,
                    "label": label,
                },
            )

        # Route to list all storages for an agent
        @mcp.custom_route(prefix_mount_path("/agent/{agent_id}/storages"), methods=["GET"])
        async def list_agent_storages(request: Request) -> JSONResponse:
            agent_id = request.path_params["agent_id"]
            storages = await memory_manager.list_storages(agent_identifier=agent_id)

            if not storages:
                return JSONResponse(
                    status_code=200,
                    content={"agentId": agent_id, "storages": []},
                )

            storage_list = [
                {
                    "storageId": storage.id,
                    "label": storage.label,
                    "createdAt": storage.created_at.isoformat(),
                }
                for storage in storages
            ]

            return JSONResponse(
                status_code=200,
                content={"agentId": agent_id, "storages": storage_list},
            )

        # Route to get a specific storage by ID
        @mcp.custom_route(
            prefix_mount_path("/agent/{agent_id}/storages/{storage_id}"),
            methods=["GET"],
        )
        async def get_agent_storage(request: Request) -> JSONResponse:
            agent_id = request.path_params["agent_id"]
            storage_id = request.path_params["storage_id"]

            storage = await memory_manager.get_storage(
                agent_identifier=agent_id, memory_storage_id=storage_id
            )

            if storage:
                return JSONResponse(
                    status_code=200,
                    content={
                        "agentId": agent_id,
                        "storageId": storage.id,
                        "label": storage.label,
                        "createdAt": storage.created_at.isoformat(),
                        "storageConfig": storage.storage_config,
                    },
                )

            return JSONResponse(
                status_code=404,
                content={"error": f"Storage {storage_id} not found for agent {agent_id}"},
            )

        # Route to delete a specific storage
        @mcp.custom_route(
            prefix_mount_path("/agent/{agent_id}/storages/{storage_id}"),
            methods=["DELETE"],
        )
        async def delete_agent_storage(request: Request) -> JSONResponse:
            agent_id = request.path_params["agent_id"]
            storage_id = request.path_params["storage_id"]

            success = await memory_manager.delete_storage(
                memory_storage_id=storage_id, agent_identifier=agent_id
            )

            if success:
                return JSONResponse(
                    status_code=200,
                    content={"message": f"Storage {storage_id} deleted successfully"},
                )

            return JSONResponse(
                status_code=404,
                content={"error": f"Storage {storage_id} not found for agent {agent_id}"},
            )

        # Route to delete all storages for an agent
        @mcp.custom_route(prefix_mount_path("/agent/{agent_id}"), methods=["DELETE"])
        async def delete_agent(request: Request) -> JSONResponse:
            agent_id = request.path_params["agent_id"]

            success = await memory_manager.delete_agent(agent_identifier=agent_id)

            if success:
                return JSONResponse(
                    status_code=200,
                    content={"message": f"Agent {agent_id} and all storages deleted successfully"},
                )

            return JSONResponse(
                status_code=404,
                content={"error": f"Agent {agent_id} not found"},
            )

        # Route to set active storage for an agent
        @mcp.custom_route(
            prefix_mount_path("/agent/{agent_id}/storages/{storage_id}/activate"),
            methods=["POST"],
        )
        async def set_active_storage(request: Request) -> JSONResponse:
            agent_id = request.path_params["agent_id"]
            storage_id = request.path_params["storage_id"]

            # First verify the storage exists
            storage = await memory_manager.get_storage(
                agent_identifier=agent_id, memory_storage_id=storage_id
            )

            if not storage:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Storage {storage_id} not found for agent {agent_id}"},
                )

            # Set as active storage
            await memory_manager.set_storage_id_for_agent(
                agent_identifier=agent_id,
                storage_id=storage_id,
                label=storage.label,
            )

            return JSONResponse(
                status_code=200,
                content={
                    "agentId": agent_id,
                    "storageId": storage_id,
                    "label": storage.label,
                    "message": "Active storage set successfully",
                },
            )

        # Route to get active storage for an agent
        @mcp.custom_route(prefix_mount_path("/agent/{agent_id}/active-storage"), methods=["GET"])
        async def get_active_storage(request: Request) -> JSONResponse:
            agent_id = request.path_params["agent_id"]

            try:
                storage_id = await memory_manager.get_active_storage_id_for_agent(
                    agent_identifier=agent_id
                )
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return JSONResponse(
                        status_code=404,
                        content={"error": f"No active storage found for agent {agent_id}"},
                    )
                return JSONResponse(status_code=500, content={"error": str(e)})

            return JSONResponse(
                status_code=200,
                content={
                    "agentId": agent_id,
                    "storageId": storage_id,
                },
            )

        # Route to clear active storage for an agent
        @mcp.custom_route(prefix_mount_path("/agent/{agent_id}/active-storage"), methods=["DELETE"])
        async def clear_active_storage(request: Request) -> JSONResponse:
            agent_id = request.path_params["agent_id"]

            # Clear active storage
            try:
                await memory_manager.clear_storage_id_for_agent(agent_identifier=agent_id)
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return JSONResponse(
                        status_code=404,
                        content={"error": f"No active storage found for agent {agent_id}"},
                    )
                return JSONResponse(status_code=500, content={"error": str(e)})

            return JSONResponse(
                status_code=200,
                content={"message": f"Active storage cleared for agent {agent_id}"},
            )
    else:
        logger.info("Memory manager not initialized, skipping memory manager routes")

    @mcp.custom_route(prefix_mount_path("/registeredDeployments/{deployment_id}"), methods=["PUT"])
    async def add_deployment(request: Request) -> JSONResponse:
        """Add or update a deployment with a known deployment_id."""
        deployment_id = request.path_params["deployment_id"]
        try:
            tool = await register_tool_for_deployment_id(deployment_id)
            return JSONResponse(
                status_code=201,
                content={
                    "name": tool.name,
                    "description": tool.description,
                    "tags": list(tool.tags),
                    "deploymentId": deployment_id,
                },
            )
        except Exception as e:
            return JSONResponse(
                status_code=400,
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
                status_code=200,
                content={
                    "deployments": formatted_deployments,
                    "count": len(deployments),
                },
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
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
                    status_code=200,
                    content={
                        "message": f"Tool with deployment {deployment_id} deleted successfully"
                    },
                )
            return JSONResponse(
                status_code=404,
                content={"error": f"Tool with deployment {deployment_id} not found"},
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to delete deployment: {str(e)}"},
            )
