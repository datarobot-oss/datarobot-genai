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

from datarobot_genai.drtools.sandbox import SandboxSecurityContext


def test_defaults_are_restrictive() -> None:
    ctx = SandboxSecurityContext()
    assert ctx.read_only_root_filesystem is True
    assert ctx.allow_privilege_escalation is False
    assert ctx.capabilities_drop == ["ALL"]
    assert ctx.capabilities_add == []
    assert ctx.seccomp_profile_type == "RuntimeDefault"


def test_default_serializes_to_workload_api_shape() -> None:
    payload = SandboxSecurityContext().to_workload_api_dict()
    assert payload == {
        "readOnlyRootFilesystem": True,
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }


def test_custom_caps_serialize_with_camel_case() -> None:
    ctx = SandboxSecurityContext(
        capabilities_drop=["NET_RAW"],
        capabilities_add=["NET_BIND_SERVICE"],
        seccomp_profile_type="Localhost",
    )
    payload = ctx.to_workload_api_dict()
    assert payload["capabilities"] == {
        "drop": ["NET_RAW"],
        "add": ["NET_BIND_SERVICE"],
    }
    assert payload["seccompProfile"] == {"type": "Localhost"}


def test_empty_caps_omitted() -> None:
    ctx = SandboxSecurityContext(capabilities_drop=[], capabilities_add=[])
    payload = ctx.to_workload_api_dict()
    assert "capabilities" not in payload
