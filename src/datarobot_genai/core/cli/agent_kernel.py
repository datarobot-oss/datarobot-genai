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
import os
import time
from typing import Any
from typing import cast

import requests
from openai import OpenAI
from openai import Stream
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionSystemMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming


class AgentKernel:
    def __init__(
        self,
        api_token: str,
        base_url: str,
    ):
        self.base_url = base_url
        self.api_token = api_token

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Token {self.api_token}",
        }

    def load_completion_json(self, completion_json: str) -> CompletionCreateParamsNonStreaming:
        """Load the completion JSON from a file or return an empty prompt."""
        if not os.path.exists(completion_json):
            raise FileNotFoundError(f"Completion JSON file not found: {completion_json}")

        with open(completion_json) as f:
            completion_data = json.load(f)

        completion_create_params = CompletionCreateParamsNonStreaming(
            **completion_data,  # type: ignore[typeddict-item]
        )
        return cast(CompletionCreateParamsNonStreaming, completion_create_params)

    def construct_prompt(
        self, user_prompt: str, verbose: bool, stream: bool = False
    ) -> CompletionCreateParamsNonStreaming | CompletionCreateParamsStreaming:
        extra_body = {
            "api_key": self.api_token,
            "api_base": self.base_url,
            "verbose": verbose,
        }
        if stream:
            return CompletionCreateParamsStreaming(
                model="datarobot-deployed-llm",
                messages=[
                    ChatCompletionSystemMessageParam(
                        content="You are a helpful assistant",
                        role="system",
                    ),
                    ChatCompletionUserMessageParam(
                        content=user_prompt,
                        role="user",
                    ),
                ],
                n=1,
                temperature=1,
                stream=True,
                extra_body=extra_body,  # type: ignore[typeddict-unknown-key]
            )
        else:
            return CompletionCreateParamsNonStreaming(
                model="datarobot-deployed-llm",
                messages=[
                    ChatCompletionSystemMessageParam(
                        content="You are a helpful assistant",
                        role="system",
                    ),
                    ChatCompletionUserMessageParam(
                        content=user_prompt,
                        role="user",
                    ),
                ],
                n=1,
                temperature=1,
                extra_body=extra_body,  # type: ignore[typeddict-unknown-key]
            )

    def local(
        self,
        user_prompt: str,
        completion_json: str = "",
        stream: bool = False,
        config: Any | None = None,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        if config is not None:
            chat_api_url = f"http://localhost:{config.local_dev_port}"
        else:
            chat_api_url = self.base_url
        print(chat_api_url)

        return self._do_chat_completion(chat_api_url, user_prompt, completion_json, stream=stream)

    def custom_model(self, custom_model_id: str, user_prompt: str, timeout: float = 300) -> str:
        chat_api_url = (
            f"{self.base_url}/api/v2/genai/agents/fromCustomModel/{custom_model_id}/chat/"
        )
        print(chat_api_url)

        headers = {
            "Authorization": f"Bearer {os.environ['DATAROBOT_API_TOKEN']}",
            "Content-Type": "application/json",
        }
        data = {"messages": [{"role": "user", "content": user_prompt}]}

        print(f'Querying custom model with prompt: "{data}"')
        print(
            "Please wait... This may take 1-2 minutes the first time "
            "you run this as a codespace is provisioned "
            "for the custom model to execute."
        )
        response = requests.post(
            chat_api_url,
            headers=headers,
            json=data,
        )

        if not response.ok or not response.headers.get("Location"):
            raise Exception(response.text)
        # Wait for the agent to complete
        status_location = response.headers["Location"]
        while response.ok:
            time.sleep(1)
            response = requests.get(
                status_location, headers=headers, allow_redirects=False, timeout=timeout
            )
            if response.status_code == 303:
                agent_response = requests.get(response.headers["Location"], headers=headers).json()
                # Show the agent response
                break
            else:
                status_response = response.json()
                if status_response["status"] in ["ERROR", "ABORTED"]:
                    raise Exception(status_response)
        else:
            raise Exception(response.content)

        if "errorMessage" in agent_response and agent_response["errorMessage"]:
            return (
                f"Error: "
                f"{agent_response.get('errorMessage', 'No error message available')}"
                f"Error details:"
                f"{agent_response.get('errorDetails', 'No details available')}"
            )
        elif "choices" in agent_response:
            return str(agent_response["choices"][0]["message"]["content"])
        else:
            return str(agent_response)

    def deployment(
        self,
        deployment_id: str,
        user_prompt: str,
        completion_json: str = "",
        stream: bool = False,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        chat_api_url = f"{self.base_url}/api/v2/deployments/{deployment_id}/"
        print(chat_api_url)

        return self._do_chat_completion(chat_api_url, user_prompt, completion_json, stream=stream)

    def _do_chat_completion(
        self,
        url: str,
        user_prompt: str,
        completion_json: str = "",
        stream: bool = False,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        if len(user_prompt) > 0:
            completion_create_params = self.construct_prompt(
                user_prompt, stream=stream, verbose=True
            )
        else:
            completion_create_params = self.load_completion_json(completion_json)

        openai_client = OpenAI(
            base_url=url,
            api_key=self.api_token,
            _strict_response_validation=False,
        )

        print(f'Querying deployment with prompt: "{completion_create_params}"')
        print(
            "Please wait for the agent to complete the response. "
            "This may take a few seconds to minutes "
            "depending on the complexity of the agent workflow."
        )

        completion = openai_client.chat.completions.create(**completion_create_params)
        return completion
