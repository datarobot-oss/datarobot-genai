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

from datetime import datetime
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError

from datarobot_genai.drmcp.core.memory_management.manager import ActiveStorageMapping
from datarobot_genai.drmcp.core.memory_management.manager import MemoryManager
from datarobot_genai.drmcp.core.memory_management.manager import MemoryResource
from datarobot_genai.drmcp.core.memory_management.manager import MemoryStorage
from datarobot_genai.drmcp.core.memory_management.manager import S3Config
from datarobot_genai.drmcp.core.memory_management.manager import S3ConfigError
from datarobot_genai.drmcp.core.memory_management.manager import S3StorageError
from datarobot_genai.drmcp.core.memory_management.manager import ToolContext
from datarobot_genai.drmcp.core.memory_management.manager import get_memory_manager
from datarobot_genai.drmcp.core.memory_management.manager import initialize_s3


class TestS3Config:
    """Test cases for S3Config class."""

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_success(self, mock_boto3_client, mock_get_credentials):
        """Test successful S3Config initialization."""
        mock_credentials = Mock()
        mock_credentials.aws_predictions_s3_bucket = "test-bucket"
        mock_credentials.get_aws_credentials.return_value = (
            "test_key",
            "test_secret",
            "test_token",
        )
        mock_get_credentials.return_value = mock_credentials
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        config = S3Config()

        assert config.bucket_name == "test-bucket"
        assert config.client is not None
        mock_boto3_client.assert_called_once_with(
            "s3",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            aws_session_token="test_token",
        )
        # Verify S3 operations were called
        mock_s3_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
        mock_s3_client.put_object.assert_called_once()
        mock_s3_client.list_objects_v2.assert_called_once()
        mock_s3_client.head_object.assert_called_once()
        mock_s3_client.get_object.assert_called_once()
        mock_s3_client.delete_object.assert_called_once()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_with_credentials_error(
        self, mock_boto3_client, mock_get_credentials
    ):
        """Test S3Config initialization with credentials error."""
        mock_get_credentials.side_effect = Exception("Credentials error")

        with pytest.raises(Exception, match="Credentials error"):
            S3Config()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_with_boto_error(
        self, mock_boto3_client, mock_get_credentials
    ):
        """Test S3Config initialization with boto3 error."""
        mock_credentials = Mock()
        mock_credentials.aws_predictions_s3_bucket = "test-bucket"
        mock_credentials.get_aws_credentials.return_value = (
            "test_key",
            "test_secret",
            "test_token",
        )
        mock_credentials.region_name = "us-east-1"
        mock_get_credentials.return_value = mock_credentials
        mock_boto3_client.side_effect = BotoCoreError()

        with pytest.raises(S3ConfigError, match="Error initializing S3 client:"):
            S3Config()


class TestInitializeS3:
    """Test cases for initialize_s3 function."""

    @patch("datarobot_genai.drmcp.core.memory_management.manager.S3Config")
    def test_initialize_s3_success(self, mock_s3_config_class):
        """Test successful S3 initialization."""
        mock_config = Mock()
        mock_config.bucket_name = "test-bucket"
        mock_s3_config_class.return_value = mock_config

        result = initialize_s3()

        assert result == mock_config
        mock_s3_config_class.assert_called_once()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.S3Config")
    def test_initialize_s3_with_error(self, mock_s3_config_class):
        """Test S3 initialization with error."""
        mock_s3_config_class.side_effect = S3ConfigError("Config error")

        with pytest.raises(S3ConfigError, match="Config error"):
            initialize_s3()


class TestToolContext:
    """Test cases for ToolContext class."""

    def test_tool_context_creation(self):
        """Test ToolContext creation."""
        context = ToolContext(name="test_tool", parameters={"param1": "value1"})

        assert context.name == "test_tool"
        assert context.parameters == {"param1": "value1"}


class TestMemoryResource:
    """Test cases for MemoryResource class."""

    def test_memory_resource_creation(self):
        """Test MemoryResource creation."""
        tool_context = ToolContext(name="test_tool", parameters={"param1": "value1"})
        resource = MemoryResource(
            id="resource123",
            memory_storage_id="storage123",
            prompt="test prompt",
            tool_context=tool_context,
            embedding_vector=[0.1, 0.2, 0.3],
        )

        assert resource.id == "resource123"
        assert resource.memory_storage_id == "storage123"
        assert resource.prompt == "test prompt"
        assert resource.tool_context == tool_context
        assert resource.embedding_vector == [0.1, 0.2, 0.3]
        assert isinstance(resource.created_at, datetime)

    def test_memory_resource_creation_minimal(self):
        """Test MemoryResource creation with minimal fields."""
        resource = MemoryResource(id="resource123")

        assert resource.id == "resource123"
        assert resource.memory_storage_id is None
        assert resource.prompt is None
        assert resource.tool_context is None
        assert resource.embedding_vector is None
        assert isinstance(resource.created_at, datetime)


class TestMemoryStorage:
    """Test cases for MemoryStorage class."""

    def test_memory_storage_creation(self):
        """Test MemoryStorage creation."""
        storage = MemoryStorage(
            id="storage123",
            agent_identifier="agent123",
            label="test storage",
            storage_config={"config": "value"},
        )

        assert storage.id == "storage123"
        assert storage.agent_identifier == "agent123"
        assert storage.label == "test storage"
        assert storage.storage_config == {"config": "value"}
        assert isinstance(storage.created_at, datetime)


class TestActiveStorageMapping:
    """Test cases for ActiveStorageMapping class."""

    def test_active_storage_mapping_creation(self):
        """Test ActiveStorageMapping creation."""
        mapping = ActiveStorageMapping(
            agent_identifier="agent123", storage_id="storage123", label="test mapping"
        )

        assert mapping.agent_identifier == "agent123"
        assert mapping.storage_id == "storage123"
        assert mapping.label == "test mapping"
        assert isinstance(mapping.updated_at, datetime)


class TestGetMemoryManager:
    """Test cases for get_memory_manager function."""

    @patch("datarobot_genai.drmcp.core.memory_management.manager.MemoryManager")
    def test_get_memory_manager_initialized(self, mock_memory_manager_class):
        """Test get_memory_manager when MemoryManager is initialized."""
        mock_instance = Mock()
        mock_memory_manager_class.is_initialized.return_value = True
        mock_memory_manager_class.get_instance.return_value = mock_instance

        result = get_memory_manager()

        assert result == mock_instance

    @patch("datarobot_genai.drmcp.core.memory_management.manager.MemoryManager")
    def test_get_memory_manager_not_initialized(self, mock_memory_manager_class):
        """Test get_memory_manager when MemoryManager is not initialized."""
        mock_memory_manager_class.is_initialized.return_value = False

        result = get_memory_manager()

        assert result is None


class TestMemoryManager:
    """Test cases for MemoryManager class."""

    def setup_method(self):
        """Reset MemoryManager state before each test."""
        MemoryManager._instance = None
        MemoryManager._initialized = False

    @patch("datarobot_genai.drmcp.core.memory_management.manager.initialize_s3")
    def test_memory_manager_singleton(self, mock_initialize_s3):
        """Test MemoryManager singleton pattern."""
        mock_config = Mock()
        mock_initialize_s3.return_value = mock_config

        manager1 = MemoryManager()
        manager2 = MemoryManager()

        assert manager1 is manager2
        assert MemoryManager._instance is manager1

    @patch("datarobot_genai.drmcp.core.memory_management.manager.initialize_s3")
    def test_memory_manager_get_instance(self, mock_initialize_s3):
        """Test MemoryManager.get_instance method."""
        mock_config = Mock()
        mock_initialize_s3.return_value = mock_config

        manager = MemoryManager.get_instance()

        assert isinstance(manager, MemoryManager)
        assert MemoryManager._instance is manager

    @patch("datarobot_genai.drmcp.core.memory_management.manager.initialize_s3")
    def test_memory_manager_is_initialized(self, mock_initialize_s3):
        """Test MemoryManager.is_initialized method."""
        mock_config = Mock()
        mock_initialize_s3.return_value = mock_config

        assert MemoryManager.is_initialized() is False

        MemoryManager()

        assert MemoryManager.is_initialized() is True

    @patch("datarobot_genai.drmcp.core.memory_management.manager.initialize_s3")
    def test_memory_manager_initialization(self, mock_initialize_s3):
        """Test MemoryManager initialization."""
        mock_config = Mock()
        mock_initialize_s3.return_value = mock_config

        manager = MemoryManager()

        assert manager.s3_config == mock_config
        mock_initialize_s3.assert_called_once()

    def test_generate_memory_storage_id(self):
        """Test _generate_memory_storage_id method."""
        storage_id = MemoryManager._generate_memory_storage_id()

        assert isinstance(storage_id, str)
        assert len(storage_id) == 8

    def test_get_resource_data_s3_key_with_agent(self):
        """Test _get_resource_data_s3_key with agent identifier."""
        key = MemoryManager._get_resource_data_s3_key("resource123", "agent123", "storage123")

        expected = "agents/agent123/storages/storage123/resources/resource123/data"
        assert key == expected

    def test_get_resource_data_s3_key_without_agent(self):
        """Test _get_resource_data_s3_key without agent identifier."""
        key = MemoryManager._get_resource_data_s3_key("resource123")

        expected = "resources/resource123/data"
        assert key == expected

    def test_get_resource_data_s3_key_agent_without_storage(self):
        """Test _get_resource_data_s3_key with agent but no storage ID."""
        with pytest.raises(ValueError, match="Storage ID is required for agent memory scope"):
            MemoryManager._get_resource_data_s3_key("resource123", "agent123")

    def test_get_resource_metadata_s3_key_with_agent(self):
        """Test _get_resource_metadata_s3_key with agent identifier."""
        key = MemoryManager._get_resource_metadata_s3_key("resource123", "agent123", "storage123")

        expected = "agents/agent123/storages/storage123/resources/resource123/metadata.json"
        assert key == expected

    def test_get_resource_metadata_s3_key_without_agent(self):
        """Test _get_resource_metadata_s3_key without agent identifier."""
        key = MemoryManager._get_resource_metadata_s3_key("resource123")

        expected = "resources/resource123/metadata.json"
        assert key == expected

    def test_get_resource_metadata_s3_key_agent_without_storage(self):
        """Test _get_resource_metadata_s3_key with agent but no storage ID."""
        with pytest.raises(ValueError, match="Storage ID is required for agent memory scope"):
            MemoryManager._get_resource_metadata_s3_key("resource123", "agent123")

    def test_get_storage_metadata_s3_key(self):
        """Test _get_storage_metadata_s3_key method."""
        key = MemoryManager._get_storage_metadata_s3_key("storage123", "agent123")

        expected = "agents/agent123/storages/storage123/metadata.json"
        assert key == expected

    def test_get_agent_identifier_s3_key(self):
        """Test _get_agent_identifier_s3_key method."""
        key = MemoryManager._get_agent_identifier_s3_key("agent123")

        expected = "agents/agent123/"
        assert key == expected

    def test_get_active_storage_mapping_key(self):
        """Test _get_active_storage_mapping_key method."""
        key = MemoryManager._get_active_storage_mapping_key("agent123")

        expected = "agents/agent123/active_storage.json"
        assert key == expected

    def test_handle_s3_error_with_resource_id(self):
        """Test _handle_s3_error with resource ID."""
        error = ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject")

        # Should not raise exception for NoSuchKey
        result = MemoryManager._handle_s3_error("test operation", error, "resource123")
        assert result is None

    def test_handle_s3_error_with_resource_id_other_error(self):
        """Test _handle_s3_error with other ClientError."""
        error = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}, "GetObject"
        )

        with pytest.raises(
            S3StorageError,
            match="Error during test operation for resource resource123: AccessDenied",
        ):
            MemoryManager._handle_s3_error("test operation", error, "resource123")

    def test_handle_s3_error_without_resource_id(self):
        """Test _handle_s3_error without resource ID."""
        error = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}, "GetObject"
        )

        with pytest.raises(S3StorageError, match="Error during test operation: AccessDenied"):
            MemoryManager._handle_s3_error("test operation", error)

    def test_handle_s3_error_404(self):
        """Test _handle_s3_error with 404 error."""
        error = ClientError({"Error": {"Code": "404", "Message": "Not found"}}, "GetObject")

        # Should not raise exception for 404
        result = MemoryManager._handle_s3_error("test operation", error, "resource123")
        assert result is None

    def test_handle_s3_error_non_client_error(self):
        """Test _handle_s3_error with non-ClientError."""
        error = Exception("Generic error")

        with pytest.raises(
            S3StorageError, match="Error during test operation for resource resource123"
        ):
            MemoryManager._handle_s3_error("test operation", error, "resource123")


class TestMemoryManagementErrorScenarios:
    """Test cases for memory management error scenarios."""

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_bucket_not_found(
        self, mock_boto3_client, mock_get_credentials
    ):
        """Test S3Config initialization when bucket doesn't exist."""
        mock_credentials = Mock()
        mock_credentials.aws_predictions_s3_bucket = "nonexistent-bucket"
        mock_credentials.get_aws_credentials.return_value = (
            "test_key",
            "test_secret",
            "test_token",
        )
        mock_get_credentials.return_value = mock_credentials

        mock_s3_client = Mock()
        mock_s3_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
        )
        mock_boto3_client.return_value = mock_s3_client

        with pytest.raises(S3ConfigError, match="Bucket nonexistent-bucket does not exist"):
            S3Config()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_access_denied(self, mock_boto3_client, mock_get_credentials):
        """Test S3Config initialization when access is denied."""
        mock_credentials = Mock()
        mock_credentials.aws_predictions_s3_bucket = "restricted-bucket"
        mock_credentials.get_aws_credentials.return_value = (
            "test_key",
            "test_secret",
            "test_token",
        )
        mock_get_credentials.return_value = mock_credentials

        mock_s3_client = Mock()
        mock_s3_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadBucket"
        )
        mock_boto3_client.return_value = mock_s3_client

        with pytest.raises(S3ConfigError, match="Access denied to bucket restricted-bucket"):
            S3Config()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_general_error(self, mock_boto3_client, mock_get_credentials):
        """Test S3Config initialization with general error."""
        mock_credentials = Mock()
        mock_credentials.aws_predictions_s3_bucket = "test-bucket"
        mock_credentials.get_aws_credentials.return_value = (
            "test_key",
            "test_secret",
            "test_token",
        )
        mock_get_credentials.return_value = mock_credentials

        mock_s3_client = Mock()
        mock_s3_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "500", "Message": "Internal Error"}}, "HeadBucket"
        )
        mock_boto3_client.return_value = mock_s3_client

        with pytest.raises(S3ConfigError, match="Error accessing bucket test-bucket"):
            S3Config()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_put_object_permission_error(
        self, mock_boto3_client, mock_get_credentials
    ):
        """Test S3Config initialization with PUT object permission error."""
        mock_credentials = Mock()
        mock_credentials.aws_predictions_s3_bucket = "test-bucket"
        mock_credentials.get_aws_credentials.return_value = (
            "test_key",
            "test_secret",
            "test_token",
        )
        mock_get_credentials.return_value = mock_credentials

        mock_s3_client = Mock()
        mock_s3_client.head_bucket.return_value = {}
        mock_s3_client.put_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "PutObject"
        )
        mock_boto3_client.return_value = mock_s3_client

        with pytest.raises(S3ConfigError, match="Access denied: Missing PutObject permissions"):
            S3Config()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_list_objects_permission_error(
        self, mock_boto3_client, mock_get_credentials
    ):
        """Test S3Config initialization with LIST objects permission error."""
        mock_credentials = Mock()
        mock_credentials.aws_predictions_s3_bucket = "test-bucket"
        mock_credentials.get_aws_credentials.return_value = (
            "test_key",
            "test_secret",
            "test_token",
        )
        mock_get_credentials.return_value = mock_credentials

        mock_s3_client = Mock()
        mock_s3_client.head_bucket.return_value = {}
        mock_s3_client.put_object.return_value = {}
        mock_s3_client.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "ListObjectsV2"
        )
        mock_boto3_client.return_value = mock_s3_client

        with pytest.raises(S3ConfigError, match="Access denied: Missing ListObjectsV2 permissions"):
            S3Config()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_head_object_permission_error(
        self, mock_boto3_client, mock_get_credentials
    ):
        """Test S3Config initialization with HEAD object permission error."""
        mock_credentials = Mock()
        mock_credentials.aws_predictions_s3_bucket = "test-bucket"
        mock_credentials.get_aws_credentials.return_value = (
            "test_key",
            "test_secret",
            "test_token",
        )
        mock_get_credentials.return_value = mock_credentials

        mock_s3_client = Mock()
        mock_s3_client.head_bucket.return_value = {}
        mock_s3_client.put_object.return_value = {}
        mock_s3_client.list_objects_v2.return_value = {}
        mock_s3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
        )
        mock_boto3_client.return_value = mock_s3_client

        with pytest.raises(S3ConfigError, match="Access denied: Missing HeadObject permissions"):
            S3Config()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_get_object_permission_error(
        self, mock_boto3_client, mock_get_credentials
    ):
        """Test S3Config initialization with GET object permission error."""
        mock_credentials = Mock()
        mock_credentials.aws_predictions_s3_bucket = "test-bucket"
        mock_credentials.get_aws_credentials.return_value = (
            "test_key",
            "test_secret",
            "test_token",
        )
        mock_get_credentials.return_value = mock_credentials

        mock_s3_client = Mock()
        mock_s3_client.head_bucket.return_value = {}
        mock_s3_client.put_object.return_value = {}
        mock_s3_client.list_objects_v2.return_value = {}
        mock_s3_client.head_object.return_value = {}
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "GetObject"
        )
        mock_boto3_client.return_value = mock_s3_client

        with pytest.raises(S3ConfigError, match="Access denied: Missing GetObject permissions"):
            S3Config()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_delete_object_permission_error(
        self, mock_boto3_client, mock_get_credentials
    ):
        """Test S3Config initialization with DELETE object permission error."""
        mock_credentials = Mock()
        mock_credentials.aws_predictions_s3_bucket = "test-bucket"
        mock_credentials.get_aws_credentials.return_value = (
            "test_key",
            "test_secret",
            "test_token",
        )
        mock_get_credentials.return_value = mock_credentials

        mock_s3_client = Mock()
        mock_s3_client.head_bucket.return_value = {}
        mock_s3_client.put_object.return_value = {}
        mock_s3_client.list_objects_v2.return_value = {}
        mock_s3_client.head_object.return_value = {}
        mock_s3_client.get_object.return_value = {}
        mock_s3_client.delete_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "DeleteObject"
        )
        mock_boto3_client.return_value = mock_s3_client

        with pytest.raises(S3ConfigError, match="Access denied: Missing DeleteObject permissions"):
            S3Config()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_general_operation_error(
        self, mock_boto3_client, mock_get_credentials
    ):
        """Test S3Config initialization with general operation error."""
        mock_credentials = Mock()
        mock_credentials.aws_predictions_s3_bucket = "test-bucket"
        mock_credentials.get_aws_credentials.return_value = (
            "test_key",
            "test_secret",
            "test_token",
        )
        mock_get_credentials.return_value = mock_credentials

        mock_s3_client = Mock()
        mock_s3_client.head_bucket.return_value = {}
        mock_s3_client.put_object.side_effect = ClientError(
            {"Error": {"Code": "500", "Message": "Internal Error"}}, "PutObject"
        )
        mock_boto3_client.return_value = mock_s3_client

        with pytest.raises(
            S3ConfigError, match="Error testing PutObject access to bucket test-bucket"
        ):
            S3Config()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.get_credentials")
    @patch("datarobot_genai.drmcp.core.memory_management.manager.boto3.client")
    def test_s3_config_initialization_boto_core_error(
        self, mock_boto3_client, mock_get_credentials
    ):
        """Test S3Config initialization with BotoCoreError."""
        mock_credentials = Mock()
        mock_credentials.aws_predictions_s3_bucket = "test-bucket"
        mock_credentials.get_aws_credentials.return_value = (
            "test_key",
            "test_secret",
            "test_token",
        )
        mock_get_credentials.return_value = mock_credentials

        mock_boto3_client.side_effect = BotoCoreError()

        with pytest.raises(S3ConfigError, match="Error initializing S3 client"):
            S3Config()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.S3Config")
    def test_initialize_s3_success(self, mock_s3_config_class):
        """Test successful S3 initialization."""
        mock_s3_config = Mock()
        mock_s3_config_class.return_value = mock_s3_config

        result = initialize_s3()

        assert result == mock_s3_config
        mock_s3_config_class.assert_called_once()

    @patch("datarobot_genai.drmcp.core.memory_management.manager.S3Config")
    def test_initialize_s3_error(self, mock_s3_config_class):
        """Test S3 initialization with error."""
        mock_s3_config_class.side_effect = S3ConfigError("S3 initialization failed")

        with pytest.raises(S3ConfigError, match="S3 initialization failed"):
            initialize_s3()

    def test_memory_resource_creation_with_all_fields(self):
        """Test MemoryResource creation with all fields."""
        tool_context = ToolContext(name="test_tool", parameters={"param1": "value1"})

        resource = MemoryResource(
            id="resource123",
            memory_storage_id="storage123",
            prompt="test prompt",
            tool_context=tool_context,
            embedding_vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
        )

        assert resource.id == "resource123"
        assert resource.memory_storage_id == "storage123"
        assert resource.prompt == "test prompt"
        assert resource.tool_context == tool_context
        assert resource.embedding_vector == [0.1, 0.2, 0.3]
        assert resource.created_at is not None

    def test_memory_storage_creation_with_all_fields(self):
        """Test MemoryStorage creation with all fields."""
        storage = MemoryStorage(
            id="storage123",
            agent_identifier="agent123",
            label="test_storage",
            created_at=datetime.now(),
            storage_config={"key": "value"},
        )

        assert storage.id == "storage123"
        assert storage.label == "test_storage"
        assert storage.created_at is not None
        assert storage.storage_config == {"key": "value"}

    def test_active_storage_mapping_creation(self):
        """Test ActiveStorageMapping creation."""
        mapping = ActiveStorageMapping(
            agent_identifier="agent123",
            storage_id="storage123",
            label="test_storage",
            updated_at=datetime.now(),
        )

        assert mapping.agent_identifier == "agent123"
        assert mapping.storage_id == "storage123"
        assert mapping.label == "test_storage"
        assert mapping.updated_at is not None

    def test_tool_context_creation(self):
        """Test ToolContext creation."""
        context = ToolContext(name="test_tool", parameters={"param1": "value1"})

        assert context.name == "test_tool"
        assert context.parameters == {"param1": "value1"}

    @patch("datarobot_genai.drmcp.core.memory_management.manager.initialize_s3")
    def test_get_memory_manager_success(self, mock_initialize_s3):
        """Test successful memory manager initialization."""
        mock_s3_config = Mock()
        mock_initialize_s3.return_value = mock_s3_config

        # First initialize the manager
        manager = MemoryManager()
        manager._initialize()

        # Now get the manager
        retrieved_manager = get_memory_manager()

        assert retrieved_manager is not None
        assert isinstance(retrieved_manager, MemoryManager)
        assert mock_initialize_s3.call_count >= 1

    def test_get_memory_manager_error(self):
        """Test memory manager when not initialized."""
        # Reset the singleton state
        MemoryManager._instance = None
        MemoryManager._initialized = False

        manager = get_memory_manager()

        assert manager is None

    @patch("datarobot_genai.drmcp.core.memory_management.manager.initialize_s3")
    def test_memory_manager_generate_memory_storage_id(self, mock_initialize_s3):
        """Test MemoryManager _generate_memory_storage_id method."""
        mock_s3_config = Mock()
        mock_initialize_s3.return_value = mock_s3_config

        manager = MemoryManager()
        storage_id = manager._generate_memory_storage_id()

        assert isinstance(storage_id, str)
        assert len(storage_id) > 0

    @patch("datarobot_genai.drmcp.core.memory_management.manager.initialize_s3")
    def test_memory_manager_get_resource_data_s3_key(self, mock_initialize_s3):
        """Test MemoryManager get_resource_data_s3_key method."""
        mock_s3_config = Mock()
        mock_initialize_s3.return_value = mock_s3_config

        manager = MemoryManager()
        key = manager._get_resource_data_s3_key("resource123", "agent123", "storage123")

        assert isinstance(key, str)
        assert "agent123" in key
        assert "storage123" in key
        assert "resource123" in key

    @patch("datarobot_genai.drmcp.core.memory_management.manager.initialize_s3")
    def test_memory_manager_get_resource_metadata_s3_key(self, mock_initialize_s3):
        """Test MemoryManager get_resource_metadata_s3_key method."""
        mock_s3_config = Mock()
        mock_initialize_s3.return_value = mock_s3_config

        manager = MemoryManager()
        key = manager._get_resource_metadata_s3_key("resource123", "agent123", "storage123")

        assert isinstance(key, str)
        assert "agent123" in key
        assert "storage123" in key
        assert "resource123" in key

    @patch("datarobot_genai.drmcp.core.memory_management.manager.initialize_s3")
    def test_memory_manager_get_storage_metadata_s3_key(self, mock_initialize_s3):
        """Test MemoryManager _get_storage_metadata_s3_key method."""
        mock_s3_config = Mock()
        mock_initialize_s3.return_value = mock_s3_config

        manager = MemoryManager()
        key = manager._get_storage_metadata_s3_key("agent123", "storage123")

        assert isinstance(key, str)
        assert "agent123" in key
        assert "storage123" in key

    @patch("datarobot_genai.drmcp.core.memory_management.manager.initialize_s3")
    def test_memory_manager_get_agent_identifier_s3_key(self, mock_initialize_s3):
        """Test MemoryManager _get_agent_identifier_s3_key method."""
        mock_s3_config = Mock()
        mock_initialize_s3.return_value = mock_s3_config

        manager = MemoryManager()
        key = manager._get_agent_identifier_s3_key("agent123")

        assert isinstance(key, str)
        assert "agent123" in key

    @patch("datarobot_genai.drmcp.core.memory_management.manager.initialize_s3")
    def test_memory_manager_get_active_storage_mapping_key(self, mock_initialize_s3):
        """Test MemoryManager _get_active_storage_mapping_key method."""
        mock_s3_config = Mock()
        mock_initialize_s3.return_value = mock_s3_config

        manager = MemoryManager()
        key = manager._get_active_storage_mapping_key("agent123")

        assert isinstance(key, str)
        assert "agent123" in key

    @patch("datarobot_genai.drmcp.core.memory_management.manager.initialize_s3")
    def test_memory_manager_handle_s3_error(self, mock_initialize_s3):
        """Test MemoryManager handle_s3_error method."""
        mock_s3_config = Mock()
        mock_initialize_s3.return_value = mock_s3_config

        manager = MemoryManager()

        # Test with ClientError
        client_error = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
        # Test with ClientError 404 - this should return None
        result = manager._handle_s3_error("test_operation", client_error)
        assert result is None

        # Test with ClientError 500 - this should raise an exception
        server_error = ClientError(
            {"Error": {"Code": "500", "Message": "Internal Error"}}, "GetObject"
        )
        with pytest.raises(Exception):  # S3StorageError or similar
            manager._handle_s3_error("test_operation", server_error)

        # Test with general exception - this will also raise an exception
        general_error = Exception("General error")
        with pytest.raises(Exception):  # S3StorageError or similar
            manager._handle_s3_error("test_operation", general_error)
