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
import os
import subprocess
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
import requests

from tests.drmcp.integration.helper import create_prompt_template
from tests.drmcp.integration.helper import delete_prompt_template
from tests.drmcp.integration.helper import get_or_create_prompt_template
from tests.drmcp.integration.helper import get_or_create_prompt_template_version


@pytest.fixture(scope="session")
def prompt_template_name_without_version() -> str:
    return "drmcp-integration-test-prompt-without-version"


@pytest.fixture(scope="session")
def prompt_template_name_with_version_without_variables() -> str:
    return "drmcp-integration-test-prompt-with-version-without-variables"


@pytest.fixture(scope="session")
def prompt_template_text_without_variables() -> str:
    return "Prompt text without any variables."


@pytest.fixture(scope="session")
def prompt_template_name_with_version_with_variables() -> str:
    return "drmcp-integration-test-prompt-with-variables"


@pytest.fixture(scope="session")
def prompt_template_name_duplicate() -> str:
    random_suffix = str(uuid.uuid4())
    return f"drmcp-integration-test-prompt-duplicate-{random_suffix}"


@pytest.fixture(scope="session")
def prompt_template_text_with_2_variables() -> str:
    return "Prompt text to greet {{name}} in max {{sentences}} sentences."


@pytest.fixture(scope="session")
def prompt_template_without_versions(prompt_template_name_without_version: str) -> dict[str, Any]:
    return get_or_create_prompt_template(prompt_template_name_without_version)


@pytest.fixture(scope="session")
def prompt_template_with_version_without_variables(
    prompt_template_name_with_version_without_variables: str,
    prompt_template_text_without_variables: str,
) -> dict[str, Any]:
    prompt_template = get_or_create_prompt_template(
        prompt_template_name_with_version_without_variables
    )
    prompt_template_version = get_or_create_prompt_template_version(
        prompt_template_id=prompt_template["id"],
        prompt_text=prompt_template_text_without_variables,
        variables=[],
    )
    return {
        "id": prompt_template["id"],
        "name": prompt_template_name_with_version_without_variables,
        "version_id": prompt_template_version["id"],
        "prompt_text": prompt_template_version["prompt_text"],
    }


@pytest.fixture(scope="session")
def prompt_template_with_version_with_variables(
    prompt_template_name_with_version_with_variables: str,
    prompt_template_text_with_2_variables: str,
) -> dict[str, Any]:
    prompt_template = get_or_create_prompt_template(
        prompt_template_name_with_version_with_variables
    )
    prompt_template_version = get_or_create_prompt_template_version(
        prompt_template_id=prompt_template["id"],
        prompt_text=prompt_template_text_with_2_variables,
        variables=["name", "sentences"],
    )
    return {
        "id": prompt_template["id"],
        "name": prompt_template_name_with_version_with_variables,
        "version_id": prompt_template_version["id"],
        "prompt_text": prompt_template_version["prompt_text"],
    }


@pytest.fixture(scope="session")
def prompt_templates_with_duplicates(
    prompt_template_name_duplicate: str,
    prompt_template_text_without_variables: str,
    prompt_template_text_with_2_variables: str,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    prompt_template_1 = create_prompt_template(prompt_template_name_duplicate)
    prompt_template_version_1 = get_or_create_prompt_template_version(
        prompt_template_id=prompt_template_1["id"],
        prompt_text=prompt_template_text_without_variables,
        variables=[],
    )
    prompt_template_2 = create_prompt_template(prompt_template_name_duplicate)
    prompt_template_version_2 = get_or_create_prompt_template_version(
        prompt_template_id=prompt_template_2["id"],
        prompt_text=prompt_template_text_with_2_variables,
        variables=["name", "sentences"],
    )

    yield (
        {
            "id": prompt_template_1["id"],
            "name": prompt_template_name_duplicate,
            "version_id": prompt_template_version_1["id"],
            "prompt_text": prompt_template_version_1["prompt_text"],
        },
        {
            "id": prompt_template_2["id"],
            "name": prompt_template_name_duplicate,
            "version_id": prompt_template_version_2["id"],
            "prompt_text": prompt_template_version_2["prompt_text"],
        },
    )

    # Cleanup
    delete_prompt_template(prompt_template_1["id"])
    delete_prompt_template(prompt_template_2["id"])


@pytest.fixture(scope="session")
def http_mcp_server() -> Iterator[str]:
    """
    Start an HTTP MCP server for integration tests that need HTTP transport.
    The server automatically imports integration test tools from ete_test_server.py.
    Yields the server URL.
    """
    port = 8080
    server_url = f"http://localhost:{port}/mcp/"

    # Check if server is already running
    try:
        response = requests.get(f"http://localhost:{port}/", timeout=2)
        if response.status_code == 200:
            # Server already running, use it
            yield server_url
            return
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        pass

    # Start server
    env = os.environ.copy()
    env["MCP_SERVER_PORT"] = str(port)
    env["OTEL_ENABLED"] = "false"
    env["MCP_SERVER_LOG_LEVEL"] = "WARNING"
    env["APP_LOG_LEVEL"] = "WARNING"

    process = subprocess.Popen(
        ["uv", "run", "tests/drmcp/acceptance/ete_test_server.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    max_attempts = 10
    for i in range(max_attempts):
        try:
            response = requests.get(f"http://localhost:{port}/", timeout=1)
            if response.status_code == 200:
                break
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            pass

        # Check if process died
        if process.poll() is not None:
            stdout, _ = process.communicate()
            error_msg = stdout.decode() if stdout else "Unknown error"
            raise RuntimeError(f"Server process died before starting. Output: {error_msg}")

        if i == max_attempts - 1:
            process.terminate()
            process.wait()
            raise RuntimeError(
                f"Server failed to start on port {port} within {max_attempts} seconds"
            )

        time.sleep(1)

    try:
        yield server_url
    finally:
        # Cleanup
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
