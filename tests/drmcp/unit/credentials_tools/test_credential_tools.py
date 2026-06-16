# Copyright 2026 DataRobot, Inc.
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

"""Unit tests for credential tools."""

from collections.abc import Iterator
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.credentials import tools as credential_tools


@pytest.fixture
def mock_rest_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def patched_dr_client(mock_rest_client: MagicMock) -> Iterator[MagicMock]:
    with patch(
        "datarobot_genai.drtools.core.clients.datarobot.request_user_dr_client"
    ) as mock_cm:
        mock_cm.return_value.__enter__.return_value = mock_rest_client
        mock_cm.return_value.__exit__.return_value = False
        yield mock_rest_client


# ------------------------------------------------------------------ #
# credential_list                                                       #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_credential_list_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [
                {"credentialId": "cred-1", "name": "my-s3", "credentialType": "s3"},
                {"credentialId": "cred-2", "name": "my-basic", "credentialType": "basic"},
            ],
            "count": 2,
            "totalCount": 2,
            "next": None,
            "previous": None,
        }
    )

    result = await credential_tools.credential_list(limit=50)

    patched_dr_client.get.assert_called_once_with("credentials/", params={"limit": 50, "offset": 0})
    assert len(result["credentials"]) == 2
    assert result["credentials"][0]["credentialId"] == "cred-1"


@pytest.mark.asyncio
async def test_credential_list_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await credential_tools.credential_list(offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_credential_list_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("403", status_code=403, json={})
    with pytest.raises(ToolError) as exc_info:
        await credential_tools.credential_list()
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# credential_get                                                        #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_credential_get_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "credentialId": "cred-1",
            "name": "my-s3",
            "credentialType": "s3",
            "description": "prod S3 bucket",
        }
    )

    result = await credential_tools.credential_get(credential_id="cred-1")

    patched_dr_client.get.assert_called_once_with("credentials/cred-1/")
    assert result["credentialType"] == "s3"


@pytest.mark.asyncio
async def test_credential_get_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await credential_tools.credential_get(credential_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_credential_get_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await credential_tools.credential_get(credential_id="cred-missing")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# credential_keys                                                       #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_credential_keys_basic() -> None:
    result = await credential_tools.credential_keys(credential_type="basic")
    assert result["credentialType"] == "basic"
    assert "login" in result["keys"]
    assert "password" in result["keys"]


@pytest.mark.asyncio
async def test_credential_keys_s3() -> None:
    result = await credential_tools.credential_keys(credential_type="s3")
    assert "awsAccessKeyId" in result["keys"]
    assert "awsSecretAccessKey" in result["keys"]


@pytest.mark.asyncio
async def test_credential_keys_empty_type_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await credential_tools.credential_keys(credential_type="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_credential_keys_unknown_type_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await credential_tools.credential_keys(credential_type="foobar")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_credential_keys_all_known_types() -> None:
    from datarobot_genai.drmcputils.constants import CREDENTIAL_TYPE_KEYS

    for ctype in CREDENTIAL_TYPE_KEYS:
        result = await credential_tools.credential_keys(credential_type=ctype)
        assert result["keys"], f"Empty keys for type {ctype}"
