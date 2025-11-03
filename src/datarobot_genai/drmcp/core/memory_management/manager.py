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

import json
import logging
import re
import uuid
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError
from pydantic import BaseModel
from pydantic import Field

from datarobot_genai.drmcp.core.credentials import get_credentials

logger = logging.getLogger(__name__)


class MemoryError(Exception):
    """Base exception for memory management errors."""

    pass


class S3StorageError(MemoryError):
    """Exception raised for S3 storage related errors."""

    pass


class S3ConfigError(MemoryError):
    """Exception raised for S3 configuration related errors."""

    pass


class S3Config:
    def __init__(self) -> None:
        credentials = get_credentials()
        self.bucket_name = credentials.aws_predictions_s3_bucket

        aws_access_key_id, aws_secret_access_key, aws_session_token = (
            credentials.get_aws_credentials()
        )

        if not aws_access_key_id or not aws_secret_access_key:
            raise S3ConfigError(
                "AWS credentials not found. Please provide credentials or set environment "
                "variables."
            )

        try:
            # Initialize S3 client
            self.client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
            )

            # Verify bucket exists and is accessible
            self.client.head_bucket(Bucket=self.bucket_name)

            # Test all required S3 operations
            test_key = "_test_permissions"
            try:
                # Test PUT operation
                self.client.put_object(Bucket=self.bucket_name, Key=test_key, Body=b"test")

                # Test LIST operation
                self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=test_key, MaxKeys=1)

                # Test HEAD operation
                self.client.head_object(Bucket=self.bucket_name, Key=test_key)

                # Test GET operation
                self.client.get_object(Bucket=self.bucket_name, Key=test_key)

                # Test DELETE operation
                self.client.delete_object(Bucket=self.bucket_name, Key=test_key)

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                operation = e.operation_name
                if error_code == "403":
                    raise S3ConfigError(
                        f"Access denied: Missing {operation} permissions for bucket "
                        f"{self.bucket_name}"
                    )
                else:
                    raise S3ConfigError(
                        f"Error testing {operation} access to bucket {self.bucket_name}: {str(e)}"
                    )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise S3ConfigError(f"Bucket {self.bucket_name} does not exist")
            elif error_code == "403":
                raise S3ConfigError(f"Access denied to bucket {self.bucket_name}")
            else:
                raise S3ConfigError(f"Error accessing bucket {self.bucket_name}: {str(e)}")

        except BotoCoreError as e:
            raise S3ConfigError(f"Error initializing S3 client: {str(e)}")


def initialize_s3() -> S3Config:
    """Initialize S3 configuration with error handling and validation."""
    try:
        s3_config = S3Config()
        logger.info(
            f"Successfully initialized S3 configuration with bucket: {s3_config.bucket_name}"
        )
        return s3_config
    except (S3ConfigError, Exception) as e:
        logger.error(f"Failed to initialize S3 configuration: {str(e)}")
        raise


class ToolContext(BaseModel):
    name: str
    parameters: dict[str, Any]


class MemoryResource(BaseModel):
    id: str
    memory_storage_id: str | None = (
        None  # a memory resource can belong to a memory storage or it can be standalone act as a
        # temp session memory
    )
    prompt: str | None = None
    tool_context: ToolContext | None = None
    embedding_vector: list[float] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryStorage(BaseModel):
    id: str
    agent_identifier: str
    label: str
    storage_config: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ActiveStorageMapping(BaseModel):
    """Model for storing active storage mappings."""

    agent_identifier: str
    storage_id: str
    label: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


def get_memory_manager() -> Optional["MemoryManager"]:
    """Get the singleton instance of MemoryManager if it is initialized, otherwise return None."""
    if MemoryManager.is_initialized():
        return MemoryManager.get_instance()
    else:
        return None


class MemoryManager:
    """Manages memory operations."""

    _instance: Optional["MemoryManager"] = None
    _initialized = False
    s3_config: S3Config

    def __new__(cls) -> "MemoryManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not MemoryManager._initialized:
            self.s3_config = self._initialize()
            MemoryManager._initialized = True

    @classmethod
    def get_instance(cls) -> "MemoryManager":
        """Get the singleton instance of MemoryManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the MemoryManager is initialized."""
        return cls._initialized

    def _initialize(self) -> S3Config:
        """Initialize the MemoryManager with S3 configuration."""
        s3_config = initialize_s3()
        logger.info("MemoryManager initialized successfully")
        return s3_config

    @staticmethod
    def _generate_memory_storage_id() -> str:
        """Generate a unique memory ID."""
        return str(uuid.uuid4())[:8]

    @staticmethod
    def _get_resource_data_s3_key(
        resource_id: str,
        agent_identifier: str | None = None,
        storage_id: str | None = None,
    ) -> str:
        """Generate S3 key for a resource data."""
        if agent_identifier:
            if not storage_id:
                raise ValueError("Storage ID is required for agent memory scope")
            return f"agents/{agent_identifier}/storages/{storage_id}/resources/{resource_id}/data"

        return f"resources/{resource_id}/data"

    @staticmethod
    def _get_resource_metadata_s3_key(
        resource_id: str,
        agent_identifier: str | None = None,
        storage_id: str | None = None,
    ) -> str:
        """Generate S3 key for a resource metadata."""
        if agent_identifier:
            if not storage_id:
                raise ValueError("Storage ID is required for agent memory scope")
            return (
                f"agents/{agent_identifier}/storages/{storage_id}/resources/{resource_id}/"
                f"metadata.json"
            )

        return f"resources/{resource_id}/metadata.json"

    @staticmethod
    def _get_storage_metadata_s3_key(storage_id: str, agent_identifier: str) -> str:
        """Generate S3 key for a storage metadata."""
        return f"agents/{agent_identifier}/storages/{storage_id}/metadata.json"

    @staticmethod
    def _get_agent_identifier_s3_key(agent_identifier: str) -> str:
        """Generate S3 key for a agent identifier."""
        return f"agents/{agent_identifier}/"

    @staticmethod
    def _get_active_storage_mapping_key(agent_identifier: str) -> str:
        """Generate S3 key for active storage mapping."""
        return f"agents/{agent_identifier}/active_storage.json"

    @staticmethod
    def _handle_s3_error(operation: str, error: Exception, resource_id: str | None = None) -> None:
        """Handle S3 related errors with proper logging and re-raising."""
        error_msg = f"Error during {operation}"
        if resource_id:
            error_msg += f" for resource {resource_id}"

        if isinstance(error, ClientError):
            error_code = error.response["Error"]["Code"]
            error_msg += f": {error_code} - {str(error)}"
            logger.error(error_msg)
            if error_code in ["NoSuchKey", "404"]:
                return None
            raise S3StorageError(error_msg) from error
        else:
            error_msg += f": {str(error)}"
            logger.error(error_msg)
            raise S3StorageError(error_msg) from error

    def _validate_agent_identifier(self, agent_identifier: str) -> None:
        """Validate the agent identifier is a valid s3 string to be used as a key and the key does
        not already exist.
        """
        if not re.match(r"^[a-zA-Z0-9!-_.*\'()]+$", agent_identifier):
            raise ValueError("Agent identifier must be a valid s3 string to be used as a key")

        try:
            self.s3_config.client.head_object(
                Bucket=self.s3_config.bucket_name, Key=f"agents/{agent_identifier}"
            )
            # If we get here, the object exists
            raise ValueError(f"Agent identifier {agent_identifier} is already in use")
        except ClientError as e:
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                # This is good - means the agent identifier doesn't exist yet
                return
            # For any other error, raise it
            raise

    async def initialize_storage(
        self,
        agent_identifier: str,
        label: str,
        storage_config: dict[str, Any] | None = None,
    ) -> str:
        """Initialize a new memory storage instance."""
        self._validate_agent_identifier(agent_identifier)

        memory_storage_id = MemoryManager._generate_memory_storage_id()

        memory_storage = MemoryStorage(
            id=memory_storage_id,
            agent_identifier=agent_identifier,
            label=label,
            storage_config=storage_config,
            created_at=datetime.utcnow(),
        )

        # Store metadata in S3
        try:
            s3_key = MemoryManager._get_storage_metadata_s3_key(memory_storage_id, agent_identifier)
            self.s3_config.client.put_object(
                Bucket=self.s3_config.bucket_name,
                Key=s3_key,
                Body=memory_storage.model_dump_json(),
            )

            # Set this as the active storage for the agent
            await self.set_storage_id_for_agent(
                agent_identifier=agent_identifier,
                storage_id=memory_storage_id,
                label=label,
            )

            logger.info(f"Initialized memory storage in S3: {memory_storage_id}")
            return memory_storage_id
        except ClientError as e:
            MemoryManager._handle_s3_error("initialize_storage", e, memory_storage_id)
            return ""

    async def delete_storage(self, memory_storage_id: str, agent_identifier: str) -> bool:
        """Delete a memory storage and its resources."""
        try:
            # Check if this is the active storage
            active_storage_id = await self.get_active_storage_id_for_agent(agent_identifier)

            # List all objects with the storage ID prefix
            prefix = f"agents/{agent_identifier}/storages/{memory_storage_id}"
            response = self.s3_config.client.list_objects_v2(
                Bucket=self.s3_config.bucket_name, Prefix=prefix
            )

            # Delete all resources
            if "Contents" in response:
                for obj in response["Contents"]:
                    self.s3_config.client.delete_object(
                        Bucket=self.s3_config.bucket_name, Key=obj["Key"]
                    )

            # Clear the active storage mapping if this was the active storage
            if active_storage_id == memory_storage_id:
                await self.clear_storage_id_for_agent(agent_identifier)

            logger.info(f"Deleted memory storage from S3: {memory_storage_id}")
            return True

        except ClientError as e:
            MemoryManager._handle_s3_error("delete_storage", e, memory_storage_id)
            return False

    async def delete_agent(self, agent_identifier: str) -> bool:
        """Delete an agent and all its memory storages."""
        try:
            # List all contents for the agent
            prefix = f"agents/{agent_identifier}"
            response = self.s3_config.client.list_objects_v2(
                Bucket=self.s3_config.bucket_name, Prefix=prefix
            )

            # Delete all contents
            if "Contents" in response:
                for obj in response["Contents"]:
                    self.s3_config.client.delete_object(
                        Bucket=self.s3_config.bucket_name, Key=obj["Key"]
                    )

            # Clear the active storage mapping
            await self.clear_storage_id_for_agent(agent_identifier)

            logger.info(f"Deleted agent and all its memory storages from S3: {agent_identifier}")
            return True
        except ClientError as e:
            MemoryManager._handle_s3_error("delete_agent", e, agent_identifier)
            return False

    async def list_storages(
        self,
        agent_identifier: str,
    ) -> list[MemoryStorage]:
        """List available memory storages for an agent."""
        try:
            prefix = f"agents/{agent_identifier}/storages/"

            response = self.s3_config.client.list_objects_v2(
                Bucket=self.s3_config.bucket_name, Prefix=prefix
            )

            storages = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    if not obj["Key"].endswith("metadata.json"):
                        continue

                    result = self.s3_config.client.get_object(
                        Bucket=self.s3_config.bucket_name, Key=obj["Key"]
                    )
                    storage_data = json.loads(result["Body"].read().decode("utf-8"))
                    storage = MemoryStorage.model_validate_json(json.dumps(storage_data))
                    storages.append(storage)

            return storages

        except ClientError as e:
            MemoryManager._handle_s3_error("list_storages", e)
            return []

    async def get_storage(
        self,
        agent_identifier: str,
        memory_storage_id: str,
    ) -> MemoryStorage | None:
        """Get a memory storage by ID."""
        try:
            metadata_key = MemoryManager._get_storage_metadata_s3_key(
                memory_storage_id, agent_identifier
            )

            result = self.s3_config.client.get_object(
                Bucket=self.s3_config.bucket_name, Key=metadata_key
            )
            storage_data = json.loads(result["Body"].read().decode("utf-8"))
            return MemoryStorage.model_validate_json(json.dumps(storage_data))
        except ClientError as e:
            MemoryManager._handle_s3_error("get_storage", e, memory_storage_id)
            return None

    async def store_resource(
        self,
        data: Any,  # the data to stored it could be a string, a json or binary data
        memory_storage_id: str | None = None,
        agent_identifier: str | None = None,
        prompt: str | None = None,
        tool_context: ToolContext | None = None,
        embedding_vector: list[float] | None = None,
    ) -> str:
        """Store a resource in the memory storage.

        Args:
            data: The data to store (string, json or binary)
            memory_storage_id: Optional storage ID to associate the resource with
            agent_identifier: Required if memory_storage_id is provided, the agent that owns the
                storage
            prompt: Optional prompt used to generate this resource
            tool_context: Optional tool context associated with this resource
            embedding_vector: Optional embedding vector for the resource

        Returns
        -------
            str: The ID of the stored resource

        Raises
        ------
            ValueError: If memory_storage_id is provided without agent_identifier or vice versa
            S3StorageError: If there are S3 related errors
        """
        if (agent_identifier and not memory_storage_id) or (
            memory_storage_id and not agent_identifier
        ):
            raise ValueError("Agent identifier and memory storage ID must be provided together")

        resource_id = str(uuid.uuid4())
        resource = MemoryResource(
            id=resource_id,
            memory_storage_id=memory_storage_id,
            prompt=prompt,
            tool_context=tool_context,
            embedding_vector=embedding_vector,
            created_at=datetime.utcnow(),
        )

        try:
            # Store resource metadata
            metadata_key = MemoryManager._get_resource_metadata_s3_key(
                resource_id,
                agent_identifier=agent_identifier,
                storage_id=memory_storage_id,
            )
            self.s3_config.client.put_object(
                Bucket=self.s3_config.bucket_name,
                Key=metadata_key,
                Body=resource.model_dump_json(),
            )

            data_key = MemoryManager._get_resource_data_s3_key(
                resource_id,
                agent_identifier=agent_identifier,
                storage_id=memory_storage_id,
            )

            # Store actual data
            self.s3_config.client.put_object(
                Bucket=self.s3_config.bucket_name, Key=data_key, Body=data
            )

            return resource_id

        except ClientError as e:
            MemoryManager._handle_s3_error("store_resource", e, resource_id)
            return ""

    async def get_resource(
        self,
        resource_id: str,
        memory_storage_id: str | None = None,
        agent_identifier: str | None = None,
    ) -> MemoryResource | None:
        """Get a resource from the memory storage.

        Args:
            resource_id: The ID of the resource to retrieve
            memory_storage_id: Optional storage ID the resource belongs to
            agent_identifier: Required if memory_storage_id is provided, the agent that owns the
                storage

        Returns
        -------
            Optional[MemoryResource]: The resource if found, None otherwise

        Raises
        ------
            ValueError: If memory_storage_id is provided without agent_identifier or vice versa
            S3StorageError: If there are S3 related errors
        """
        if (agent_identifier and not memory_storage_id) or (
            memory_storage_id and not agent_identifier
        ):
            raise ValueError("Agent identifier and memory storage ID must be provided together")

        try:
            metadata_key = MemoryManager._get_resource_metadata_s3_key(
                resource_id,
                agent_identifier=agent_identifier,
                storage_id=memory_storage_id,
            )
            result = self.s3_config.client.get_object(
                Bucket=self.s3_config.bucket_name, Key=metadata_key
            )
            resource_data = json.loads(result["Body"].read().decode("utf-8"))
            return MemoryResource.model_validate_json(json.dumps(resource_data))
        except ClientError as e:
            MemoryManager._handle_s3_error("get_resource", e, resource_id)
            return None

    async def list_resources(
        self, agent_identifier: str, memory_storage_id: str | None = None
    ) -> list[MemoryResource]:
        """List all resources from the memory storage.

        Args:
            agent_identifier: Agent identifier to scope the search
            memory_storage_id: Optional Storage ID to filter resources
        """
        try:
            prefix = (
                f"agents/{agent_identifier}/storages/{memory_storage_id}/resources/"
                if memory_storage_id
                else f"agents/{agent_identifier}/storages/"
            )

            response = self.s3_config.client.list_objects_v2(
                Bucket=self.s3_config.bucket_name, Prefix=prefix
            )

            resources = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    if not obj["Key"].endswith("metadata.json"):
                        continue

                    result = self.s3_config.client.get_object(
                        Bucket=self.s3_config.bucket_name, Key=obj["Key"]
                    )
                    resource_data = json.loads(result["Body"].read().decode("utf-8"))
                    resource = MemoryResource.model_validate_json(json.dumps(resource_data))

                    if memory_storage_id and resource.memory_storage_id != memory_storage_id:
                        continue

                    resources.append(resource)

            return resources

        except ClientError as e:
            MemoryManager._handle_s3_error("list_resources", e)
            return []

    async def get_resource_data(
        self,
        resource_id: str,
        memory_storage_id: str | None = None,
        agent_identifier: str | None = None,
    ) -> bytes | None:
        """Get the data of a resource by resource id.

        Args:
            resource_id: The ID of the resource to retrieve data for
            memory_storage_id: Optional storage ID the resource belongs to
            agent_identifier: Required if memory_storage_id is provided, the agent that owns the
                storage

        Returns
        -------
            Optional[bytes]: The resource data if found, None otherwise

        Raises
        ------
            ValueError: If memory_storage_id is provided without agent_identifier or vice versa
            S3StorageError: If there are S3 related errors
        """
        if (agent_identifier and not memory_storage_id) or (
            memory_storage_id and not agent_identifier
        ):
            raise ValueError("Agent identifier and memory storage ID must be provided together")

        try:
            data_key = MemoryManager._get_resource_data_s3_key(
                resource_id,
                agent_identifier=agent_identifier,
                storage_id=memory_storage_id,
            )
            result = self.s3_config.client.get_object(
                Bucket=self.s3_config.bucket_name, Key=data_key
            )
            data = result["Body"].read()
            return data if isinstance(data, bytes) else None
        except ClientError as e:
            MemoryManager._handle_s3_error("get_resource_data", e, resource_id)
            return None

    async def delete_resource(
        self,
        resource_id: str,
        memory_storage_id: str | None = None,
        agent_identifier: str | None = None,
    ) -> bool:
        """Delete a resource from the memory storage.

        Args:
            resource_id: The ID of the resource to delete
            memory_storage_id: Optional storage ID the resource belongs to
            agent_identifier: Required if memory_storage_id is provided, the agent that owns the
                storage

        Returns
        -------
            bool: True if deletion was successful

        Raises
        ------
            ValueError: If memory_storage_id is provided without agent_identifier or vice versa
            S3StorageError: If there are S3 related errors
        """
        if (agent_identifier and not memory_storage_id) or (
            memory_storage_id and not agent_identifier
        ):
            raise ValueError("Agent identifier and memory storage ID must be provided together")

        try:
            # Delete metadata
            metadata_key = MemoryManager._get_resource_metadata_s3_key(
                resource_id,
                agent_identifier=agent_identifier,
                storage_id=memory_storage_id,
            )
            self.s3_config.client.delete_object(Bucket=self.s3_config.bucket_name, Key=metadata_key)

            # Delete data
            data_key = MemoryManager._get_resource_data_s3_key(
                resource_id,
                agent_identifier=agent_identifier,
                storage_id=memory_storage_id,
            )
            self.s3_config.client.delete_object(Bucket=self.s3_config.bucket_name, Key=data_key)

            return True

        except ClientError as e:
            MemoryManager._handle_s3_error("delete_resource", e, resource_id)
            return False

    async def clear_all_temp_resources(self, older_than_by_days: int = 1) -> bool:
        """Clear all temp resources older than a given number of days.

        Args:
            older_than_by_days: Optional number of days to compare against. If not provided,
                       defaults to 1 day ago.
        """
        older_than = datetime.now(timezone.utc) - timedelta(days=older_than_by_days)

        try:
            prefix = "resources/"
            response = self.s3_config.client.list_objects_v2(
                Bucket=self.s3_config.bucket_name, Prefix=prefix
            )

            if "Contents" in response:
                for obj in response["Contents"]:
                    last_modified = obj["LastModified"]
                    if last_modified < older_than:
                        self.s3_config.client.delete_object(
                            Bucket=self.s3_config.bucket_name, Key=obj["Key"]
                        )

            return True
        except ClientError as e:
            MemoryManager._handle_s3_error("clear_all_temp_resources", e)
            return False

    async def list_temp_resources(self) -> list[MemoryResource]:
        """List all temp resources."""
        prefix = "resources/"
        response = self.s3_config.client.list_objects_v2(
            Bucket=self.s3_config.bucket_name, Prefix=prefix
        )

        resources = []
        if "Contents" in response:
            for obj in response["Contents"]:
                if not obj["Key"].endswith("metadata.json"):
                    continue
                result = self.s3_config.client.get_object(
                    Bucket=self.s3_config.bucket_name, Key=obj["Key"]
                )
                resource_data = json.loads(result["Body"].read().decode("utf-8"))
                resource = MemoryResource.model_validate_json(json.dumps(resource_data))
                resources.append(resource)
        return resources

    async def find_agent_identifier_for_storage(self, memory_storage_id: str) -> str:
        """Find agent identifier from storage ID."""
        prefix = "agents/"
        response = self.s3_config.client.list_objects_v2(
            Bucket=self.s3_config.bucket_name, Prefix=prefix
        )

        agent_identifier = None
        if "Contents" in response:
            for obj in response["Contents"]:
                if memory_storage_id in obj["Key"]:
                    # Extract agent identifier from key pattern:
                    # agents/{agent_id}/storages/{storage_id}/
                    parts = obj["Key"].split("/")
                    if len(parts) > 1:
                        agent_identifier = parts[1]
                        break

        if not agent_identifier:
            raise ValueError(f"Memory storage {memory_storage_id} not found")

        return str(agent_identifier)

    async def set_storage_id_for_agent(
        self, agent_identifier: str, storage_id: str, label: str
    ) -> None:
        """Set the active storage ID for an agent in S3."""
        try:
            mapping = ActiveStorageMapping(
                agent_identifier=agent_identifier,
                storage_id=storage_id,
                label=label,
                updated_at=datetime.now(timezone.utc),
            )

            key = self._get_active_storage_mapping_key(agent_identifier)
            self.s3_config.client.put_object(
                Bucket=self.s3_config.bucket_name,
                Key=key,
                Body=mapping.model_dump_json(),
            )
        except ClientError as e:
            MemoryManager._handle_s3_error("set_storage_id_for_agent", e)

    async def get_active_storage_id_for_agent(self, agent_identifier: str) -> str | None:
        """Get the active storage ID for an agent from S3."""
        try:
            key = self._get_active_storage_mapping_key(agent_identifier)
            result = self.s3_config.client.get_object(Bucket=self.s3_config.bucket_name, Key=key)
            mapping_data = json.loads(result["Body"].read().decode("utf-8"))
            mapping = ActiveStorageMapping.model_validate_json(json.dumps(mapping_data))
            return mapping.storage_id
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return None
            MemoryManager._handle_s3_error("get_active_storage_id_for_agent", e)
            return None

    async def clear_storage_id_for_agent(self, agent_identifier: str) -> None:
        """Clear the active storage ID for an agent from S3."""
        try:
            key = self._get_active_storage_mapping_key(agent_identifier)
            self.s3_config.client.delete_object(Bucket=self.s3_config.bucket_name, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                MemoryManager._handle_s3_error("clear_storage_id_for_agent", e)
