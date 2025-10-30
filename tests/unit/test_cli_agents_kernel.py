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

from unittest.mock import Mock
from unittest.mock import patch

import pytest
from datarobot_genai.cli.agent_kernel import Kernel
from openai.types.chat import ChatCompletion


class TestKernel:
    def test_headers_property(self):
        """Test headers property returns correct authorization header."""
        kernel = Kernel(api_token="api-123456", base_url="https://test.example.com")

        headers = kernel.headers

        assert headers == {"Authorization": "Token api-123456"}

    def test_construct_prompt_with_verbose(self):
        """Test construct_prompt with verbose set to True."""
        # Setup
        kernel = Kernel(api_token="test-key", base_url="https://test.example.com")
        user_prompt = "Hello, how are you?"

        # Execute
        result_dict = kernel.construct_prompt(user_prompt, verbose=True)

        # Assert
        assert result_dict["model"] == "datarobot-deployed-llm"
        assert len(result_dict["messages"]) == 2
        assert result_dict["messages"][0]["content"] == "You are a helpful assistant"
        assert result_dict["messages"][0]["role"] == "system"
        assert result_dict["messages"][1]["content"] == "Hello, how are you?"
        assert result_dict["messages"][1]["role"] == "user"
        assert result_dict["n"] == 1
        assert result_dict["temperature"] == 1
        assert result_dict["extra_body"]["api_key"] == "test-key"
        assert result_dict["extra_body"]["api_base"] == "https://test.example.com"
        assert result_dict["extra_body"]["verbose"] is True

    def test_construct_prompt_without_verbose(self):
        """Test construct_prompt with verbose set to False."""
        # Setup
        kernel = Kernel(api_token="test-key", base_url="https://test.example.com")
        user_prompt = "Tell me about Python"

        # Execute
        result_dict = kernel.construct_prompt(user_prompt, verbose=False)

        # Assert
        assert result_dict["model"] == "datarobot-deployed-llm"
        assert len(result_dict["messages"]) == 2
        assert result_dict["messages"][0]["content"] == "You are a helpful assistant"
        assert result_dict["messages"][0]["role"] == "system"
        assert result_dict["messages"][1]["content"] == "Tell me about Python"
        assert result_dict["messages"][1]["role"] == "user"
        assert result_dict["n"] == 1
        assert result_dict["temperature"] == 1
        assert result_dict["extra_body"]["api_key"] == "test-key"
        assert result_dict["extra_body"]["api_base"] == "https://test.example.com"
        assert result_dict["extra_body"]["verbose"] is False

    @patch("datarobot_genai.cli.agent_kernel.OpenAI")
    def test_deployment_basic_functionality(self, mock_openai):
        """
        Test deployment method creates OpenAI client and calls
        chat.completions.create correctly.
        """
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        deployment_id = "test-deployment-id"
        user_prompt = "Hello, assistant!"

        # Mock the OpenAI client and its methods
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_completions = Mock()
        mock_client.chat.completions = mock_completions
        mock_completion_obj = Mock(spec=ChatCompletion)
        mock_completions.create.return_value = mock_completion_obj

        # Execute
        result = kernel.deployment(deployment_id, user_prompt)

        # Assert
        # Verify OpenAI client was created with correct parameters
        mock_openai.assert_called_once_with(
            base_url=f"https://test.example.com/api/v2/deployments/{deployment_id}/",
            api_key="test-token",
            _strict_response_validation=False,
        )

        # Verify chat.completions.create was called with correct parameters
        mock_completions.create.assert_called_once_with(
            model="datarobot-deployed-llm",
            messages=[
                {"content": "You are a helpful assistant", "role": "system"},
                {"content": "Hello, assistant!", "role": "user"},
            ],
            n=1,
            temperature=1,
            extra_body={
                "api_key": "test-token",
                "api_base": "https://test.example.com",
                "verbose": True,
            },
        )

        # Verify the result is the completion object
        assert result == mock_completion_obj

    @patch("datarobot_genai.cli.agent_kernel.OpenAI")
    def test_deployment_streaming(self, mock_openai):
        """
        Test deployment method creates OpenAI client and calls
        chat.completions.create correctly with streaming.
        """
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        deployment_id = "test-deployment-id"
        user_prompt = "Hello, assistant!"

        # Mock the OpenAI client and its methods
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_completions = Mock()
        mock_client.chat.completions = mock_completions
        mock_completion_obj = Mock(spec=ChatCompletion)
        mock_completions.create.return_value = mock_completion_obj

        # Execute
        result = kernel.deployment(deployment_id, user_prompt, stream=True)

        # Assert
        # Verify OpenAI client was created with correct parameters
        mock_openai.assert_called_once_with(
            base_url=f"https://test.example.com/api/v2/deployments/{deployment_id}/",
            api_key="test-token",
            _strict_response_validation=False,
        )

        # Verify chat.completions.create was called with correct parameters
        mock_completions.create.assert_called_once_with(
            model="datarobot-deployed-llm",
            messages=[
                {"content": "You are a helpful assistant", "role": "system"},
                {"content": "Hello, assistant!", "role": "user"},
            ],
            n=1,
            temperature=1,
            extra_body={
                "api_key": "test-token",
                "api_base": "https://test.example.com",
                "verbose": True,
            },
            stream=True,
        )

        # Verify the result is the completion object
        assert result == mock_completion_obj

    @patch("datarobot_genai.cli.agent_kernel.OpenAI")
    @patch("builtins.print")
    def test_deployment_prints_debug_info(self, mock_print, mock_openai):
        """Test deployment method prints debug info."""
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        deployment_id = "test-deployment-id"
        user_prompt = "Hello, assistant!"

        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_completions = Mock()
        mock_client.chat.completions = mock_completions
        mock_completions.create.return_value = Mock(spec=ChatCompletion)

        # Execute
        kernel.deployment(deployment_id, user_prompt)

        # Assert print statements were called with expected arguments
        expected_api_url = "https://test.example.com/api/v2/deployments/test-deployment-id/"
        mock_print.assert_any_call(expected_api_url)

    @patch("datarobot_genai.cli.agent_kernel.OpenAI")
    def test_deployment_error_handling(self, mock_openai):
        """Test deployment method propagates errors from OpenAI client."""
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        deployment_id = "test-deployment-id"
        user_prompt = "Hello, assistant!"

        # Mock the OpenAI client to raise an exception
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_completions = Mock()
        mock_client.chat.completions = mock_completions
        mock_completions.create.side_effect = ValueError("Test error")

        # Execute and Assert
        with pytest.raises(ValueError, match="Test error"):
            kernel.deployment(deployment_id, user_prompt)

    @patch("datarobot_genai.cli.agent_kernel.OpenAI")
    def test_local_basic_functionality(self, mock_openai):
        """
        Test local method creates OpenAI client and calls
        chat.completions.create correctly.
        """
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        user_prompt = "Hello, assistant!"

        # Mock the OpenAI client and its methods
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_completions = Mock()
        mock_client.chat.completions = mock_completions
        mock_completion_obj = Mock(spec=ChatCompletion)
        mock_completions.create.return_value = mock_completion_obj

        # Execute
        result = kernel.local(user_prompt)

        # Assert
        # Verify OpenAI client was created with correct parameters
        mock_openai.assert_called_once_with(
            base_url="http://localhost:8842",
            api_key="test-token",
            _strict_response_validation=False,
        )

        # Verify chat.completions.create was called with correct parameters
        mock_completions.create.assert_called_once_with(
            model="datarobot-deployed-llm",
            messages=[
                {"content": "You are a helpful assistant", "role": "system"},
                {"content": "Hello, assistant!", "role": "user"},
            ],
            n=1,
            temperature=1,
            extra_body={
                "api_key": "test-token",
                "api_base": "https://test.example.com",
                "verbose": True,
            },
        )

        # Verify the result is the completion object
        assert result == mock_completion_obj

    @patch("datarobot_genai.cli.agent_kernel.OpenAI")
    def test_local_streaming(self, mock_openai):
        """
        Test local method creates OpenAI client and calls
        chat.completions.create correctly with streaming.
        """
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        user_prompt = "Hello, assistant!"

        # Mock the OpenAI client and its methods
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_completions = Mock()
        mock_client.chat.completions = mock_completions
        mock_completion_obj = Mock(spec=ChatCompletion)
        mock_completions.create.return_value = mock_completion_obj

        # Execute
        result = kernel.local(user_prompt, stream=True)

        # Assert
        # Verify OpenAI client was created with correct parameters
        mock_openai.assert_called_once_with(
            base_url="http://localhost:8842",
            api_key="test-token",
            _strict_response_validation=False,
        )

        # Verify chat.completions.create was called with correct parameters
        mock_completions.create.assert_called_once_with(
            model="datarobot-deployed-llm",
            messages=[
                {"content": "You are a helpful assistant", "role": "system"},
                {"content": "Hello, assistant!", "role": "user"},
            ],
            n=1,
            temperature=1,
            extra_body={
                "api_key": "test-token",
                "api_base": "https://test.example.com",
                "verbose": True,
            },
            stream=True,
        )

        # Verify the result is the completion object
        assert result == mock_completion_obj

    @patch("datarobot_genai.cli.agent_kernel.OpenAI")
    @patch("builtins.print")
    def test_local_prints_debug_info(self, mock_print, mock_openai):
        """Test local method prints debug info."""
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        user_prompt = "Hello, assistant!"

        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_completions = Mock()
        mock_client.chat.completions = mock_completions
        mock_completions.create.return_value = Mock(spec=ChatCompletion)

        # Execute
        kernel.local(user_prompt)

        # Assert print statements were called with expected arguments
        expected_api_url = "http://localhost:8842"
        mock_print.assert_any_call(expected_api_url)

    @patch("datarobot_genai.cli.agent_kernel.OpenAI")
    def test_local_error_handling(self, mock_openai):
        """Test local method propagates errors from OpenAI client."""
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        user_prompt = "Hello, assistant!"

        # Mock the OpenAI client to raise an exception
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_completions = Mock()
        mock_client.chat.completions = mock_completions
        mock_completions.create.side_effect = ValueError("Test error")

        # Execute and Assert
        with pytest.raises(ValueError, match="Test error"):
            kernel.local(user_prompt)

    @patch("datarobot_genai.cli.agent_kernel.requests.post")
    @patch("datarobot_genai.cli.agent_kernel.requests.get")
    @patch("datarobot_genai.cli.agent_kernel.time.sleep")
    @patch(
        "os.environ",
        {
            "DATAROBOT_API_TOKEN": "test-api-token",
            "DATAROBOT_ENDPOINT": "https://test.example.com",
        },
    )
    def test_custom_model_basic_functionality(
        self, mock_sleep, mock_requests_get, mock_requests_post
    ):
        """Test custom_model method makes HTTP requests to DataRobot API correctly."""
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        custom_model_id = "test-custom-model-id"
        user_prompt = "Hello, assistant!"

        # Mock the initial POST response
        mock_post_response = Mock()
        mock_post_response.ok = True
        mock_post_response.headers = {"Location": "https://test.example.com/status/123"}
        mock_requests_post.return_value = mock_post_response

        # Mock the status check response (first call returns status, second redirects)
        mock_status_response = Mock()
        mock_status_response.ok = True
        mock_status_response.status_code = 200
        mock_status_response.json.return_value = {"status": "RUNNING"}

        mock_redirect_response = Mock()
        mock_redirect_response.ok = True
        mock_redirect_response.status_code = 303
        mock_redirect_response.headers = {"Location": "https://test.example.com/result/123"}

        # Mock the final result response
        mock_result_response = Mock()
        mock_result_response.json.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help you?"}}]
        }

        mock_requests_get.side_effect = [
            mock_status_response,
            mock_redirect_response,
            mock_result_response,
        ]

        # Execute
        result = kernel.custom_model(custom_model_id, user_prompt)

        # Assert
        # Verify POST request was made with correct parameters
        mock_requests_post.assert_called_once_with(
            "https://test.example.com/api/v2/genai/agents/fromCustomModel/test-custom-model-id/chat/",
            headers={
                "Authorization": "Bearer test-api-token",
                "Content-Type": "application/json",
            },
            json={"messages": [{"role": "user", "content": "Hello, assistant!"}]},
        )

        # Verify status polling was done
        assert mock_requests_get.call_count == 3

        # Verify the result content
        assert result == "Hello! How can I help you?"

    @patch("datarobot_genai.cli.agent_kernel.requests.post")
    @patch("datarobot_genai.cli.agent_kernel.time.sleep")
    @patch(
        "os.environ",
        {
            "DATAROBOT_API_TOKEN": "test-api-token",
            "DATAROBOT_ENDPOINT": "https://test.example.com",
        },
    )
    def test_custom_model_initial_request_failure(self, mock_sleep, mock_requests_post):
        """Test custom_model handles initial POST request failure."""
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        custom_model_id = "test-custom-model-id"
        user_prompt = "Hello, assistant!"

        # Mock the initial POST response to fail
        mock_post_response = Mock()
        mock_post_response.ok = False
        mock_post_response.content = b"API request failed with status 500"
        mock_requests_post.return_value = mock_post_response

        # Execute and Assert
        with pytest.raises(Exception):
            kernel.custom_model(custom_model_id, user_prompt)

        # Verify the correct request was attempted
        mock_requests_post.assert_called_once_with(
            "https://test.example.com/api/v2/genai/agents/fromCustomModel/test-custom-model-id/chat/",
            headers={
                "Authorization": "Bearer test-api-token",
                "Content-Type": "application/json",
            },
            json={"messages": [{"role": "user", "content": "Hello, assistant!"}]},
        )

    @patch("datarobot_genai.cli.agent_kernel.requests.post")
    @patch("datarobot_genai.cli.agent_kernel.requests.get")
    @patch("datarobot_genai.cli.agent_kernel.time.sleep")
    @patch(
        "os.environ",
        {
            "DATAROBOT_API_TOKEN": "test-api-token",
            "DATAROBOT_ENDPOINT": "https://test.example.com",
        },
    )
    def test_custom_model_missing_location_header(
        self, mock_sleep, mock_requests_get, mock_requests_post
    ):
        """Test custom_model handles missing Location header in successful response."""
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        custom_model_id = "test-custom-model-id"
        user_prompt = "Hello, assistant!"

        # Mock the initial POST response with missing Location header
        mock_post_response = Mock()
        mock_post_response.ok = True
        mock_post_response.headers = {}  # No Location header
        mock_post_response.content = b"No Location header provided"
        mock_requests_post.return_value = mock_post_response

        # Execute and Assert
        with pytest.raises(Exception):
            kernel.custom_model(custom_model_id, user_prompt)

    @patch("datarobot_genai.cli.agent_kernel.requests.post")
    @patch("datarobot_genai.cli.agent_kernel.requests.get")
    @patch("datarobot_genai.cli.agent_kernel.time.sleep")
    @patch(
        "os.environ",
        {
            "DATAROBOT_API_TOKEN": "test-api-token",
            "DATAROBOT_ENDPOINT": "https://test.example.com",
        },
    )
    def test_custom_model_status_error(self, mock_sleep, mock_requests_get, mock_requests_post):
        """Test custom_model handles ERROR status from status endpoint."""
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        custom_model_id = "test-custom-model-id"
        user_prompt = "Hello, assistant!"

        # Mock the initial POST response
        mock_post_response = Mock()
        mock_post_response.ok = True
        mock_post_response.headers = {"Location": "https://test.example.com/status/123"}
        mock_requests_post.return_value = mock_post_response

        # Mock the status check response with ERROR status
        mock_status_response = Mock()
        mock_status_response.ok = True
        mock_status_response.status_code = 200
        mock_status_response.json.return_value = {
            "status": "ERROR",
            "errorMessage": "Model execution failed",
        }
        mock_requests_get.return_value = mock_status_response

        # Execute and Assert
        with pytest.raises(Exception) as exc_info:
            kernel.custom_model(custom_model_id, user_prompt)

        # Verify the error contains the status response
        assert "status" in str(exc_info.value)
        assert "ERROR" in str(exc_info.value)

    @patch("datarobot_genai.cli.agent_kernel.requests.post")
    @patch("datarobot_genai.cli.agent_kernel.requests.get")
    @patch("datarobot_genai.cli.agent_kernel.time.sleep")
    @patch(
        "os.environ",
        {
            "DATAROBOT_API_TOKEN": "test-api-token",
            "DATAROBOT_ENDPOINT": "https://test.example.com",
        },
    )
    def test_custom_model_error_in_response(
        self, mock_sleep, mock_requests_get, mock_requests_post
    ):
        """Test custom_model handles error message in agent response."""
        # Setup
        kernel = Kernel(
            api_token="test-token",
            base_url="https://test.example.com",
        )
        custom_model_id = "test-custom-model-id"
        user_prompt = "Hello, assistant!"

        # Mock the initial POST response
        mock_post_response = Mock()
        mock_post_response.ok = True
        mock_post_response.headers = {"Location": "https://test.example.com/status/123"}
        mock_requests_post.return_value = mock_post_response

        # Mock the status check responses
        mock_status_response = Mock()
        mock_status_response.ok = True
        mock_status_response.status_code = 200
        mock_status_response.json.return_value = {"status": "RUNNING"}

        mock_redirect_response = Mock()
        mock_redirect_response.ok = True
        mock_redirect_response.status_code = 303
        mock_redirect_response.headers = {"Location": "https://test.example.com/result/123"}

        # Mock the final result response with an error message
        mock_result_response = Mock()
        mock_result_response.json.return_value = {
            "errorMessage": "Failed to process request",
            "errorDetails": "Invalid input format",
        }

        mock_requests_get.side_effect = [
            mock_status_response,
            mock_redirect_response,
            mock_result_response,
        ]

        # Execute
        result = kernel.custom_model(custom_model_id, user_prompt)

        # Assert the result contains the error message
        assert "Error: " in result
        assert "Failed to process request" in result
        assert "Error details:" in result
        assert "Invalid input format" in result
