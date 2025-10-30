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
from typing import Any

from pydantic_core import PydanticUseDefault


def extract_datarobot_runtime_param_payload(v: Any) -> Any:
    """Extract payload from DataRobot runtime parameter JSON format.

    DataRobot runtime parameters come in the format:
    {"type":"string","payload":"value"} or {"type":"boolean","payload":false}

    If payload is None, raises PydanticUseDefault so Pydantic uses the field default.

    This function extracts the payload value for simple types (strings, booleans, numbers).
    For dict types with nested JSON, use extract_datarobot_dict_runtime_param_payload instead.

    Args:
        v: The input value (may be a raw value, JSON string, or DataRobot runtime param format)

    Returns
    -------
        The extracted payload value

    Raises
    ------
        PydanticUseDefault: When payload is None, signaling Pydantic to use field default
    """
    # If it's a string, try to parse as JSON
    if isinstance(v, str):
        # Handle Python-style boolean strings (True/False) by converting to lowercase
        v_normalized = v.lower() if v.lower() in ("true", "false") else v

        try:
            parsed = json.loads(v_normalized)
            if isinstance(parsed, dict) and "payload" in parsed:
                payload = parsed["payload"]
                if payload is not None:
                    return payload
                # If payload is None, use field default
                raise PydanticUseDefault
            # If it's a plain JSON value (boolean, number, etc.), return it
            # This handles cases like "true", "false", "123", etc.
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return v


def extract_datarobot_dict_runtime_param_payload(v: Any) -> Any:  # noqa: PLR0911
    r"""Extract and parse dict from DataRobot runtime parameter JSON format.

    DataRobot runtime parameters for dict fields come in the format:
    {"type":"string","payload":"{\\"key\\":\\"value\\"}"}

    The payload itself is a JSON string that needs to be parsed into a dict.
    If payload is None, raises PydanticUseDefault so Pydantic uses the field default.

    Args:
        v: The input value (may be a dict, JSON string, or DataRobot runtime param format)

    Returns
    -------
        The extracted and parsed dict

    Raises
    ------
        PydanticUseDefault: When payload is None, signaling Pydantic to use field default
    """
    # If it's already a dict, check if it's in DataRobot format
    if isinstance(v, dict):
        # If it has "payload" key, it's in DataRobot format - extract it
        if "payload" in v:
            payload = v["payload"]
            if payload is None:
                raise PydanticUseDefault
            # If payload is a string, parse it as JSON
            if isinstance(payload, str):
                try:
                    return json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    return {}
            return payload
        # Otherwise, return as-is (already a plain dict)
        return v

    # If it's a string, try to parse as JSON
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, dict) and "payload" in parsed:
                payload = parsed["payload"]
                if payload is not None:
                    # If payload is a string, parse it as JSON
                    if isinstance(payload, str):
                        try:
                            return json.loads(payload)
                        except (json.JSONDecodeError, ValueError):
                            return {}
                    return payload
                raise PydanticUseDefault
            # If it's already a dict, return it
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: return empty dict
    return {}


def extract_datarobot_credential_runtime_param_payload(v: Any) -> Any:
    """Extract credential payload from DataRobot runtime parameter JSON format.

    DataRobot credential runtime parameters come in the format:
    {"type":"credential","payload":{...}} where the payload is the full credential object.

    For credential types, the entire dict is preserved.
    If payload is None, raises PydanticUseDefault so Pydantic uses the field default.

    Args:
        v: The input value (may be a dict, JSON string, or DataRobot runtime param format)

    Returns
    -------
        The extracted credential dict

    Raises
    ------
        PydanticUseDefault: When payload is None, signaling Pydantic to use field default
    """
    # If it's already a dict (credential object or wrapped payload)
    if isinstance(v, dict):
        # If it's a wrapped payload, extract it
        if "payload" in v:
            payload = v["payload"]
            if payload is not None:
                return payload
            # If payload is None, use field default
            raise PydanticUseDefault
        # Otherwise return the dict as-is (it's the credential object)
        return v

    # If it's a string, try to parse as JSON
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                # If it's a wrapped payload, extract it
                if "payload" in parsed:
                    payload = parsed["payload"]
                    if payload is not None:
                        return payload
                    # If payload is None, use field default
                    raise PydanticUseDefault
                # Otherwise return the dict as-is
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return v
