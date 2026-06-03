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

"""Tests for ``LocalDockerSandbox`` — the docker CLI is mocked at the
``asyncio.create_subprocess_exec`` boundary, so these run without Docker.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.drtools.sandbox import LocalDockerSandbox
from datarobot_genai.drtools.sandbox import SandboxError
from datarobot_genai.drtools.sandbox import SandboxSecurityContext
from datarobot_genai.drtools.sandbox import SandboxTimeout

IMAGE = "datarobotdev/datarobot-user-models:sandbox-test"
SPAWN = "datarobot_genai.drtools.sandbox.local_docker.asyncio.create_subprocess_exec"


def _proc(
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
    communicate_exc: BaseException | None = None,
) -> MagicMock:
    proc = MagicMock()
    if communicate_exc is not None:
        proc.communicate = AsyncMock(side_effect=communicate_exc)
    else:
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    proc.returncode = returncode
    return proc


def _run_argv(spawn: MagicMock) -> list[str]:
    """Positional argv of the first (main `docker run`) spawn."""
    return list(spawn.call_args_list[0].args)


async def test_happy_path_parses_marker() -> None:
    main = _proc(stdout=b"hello\n__DR_SANDBOX_RESULT__:42\n", returncode=0)
    with patch(SPAWN, side_effect=[main, _proc()]) as spawn:
        result = await LocalDockerSandbox(IMAGE).run("_return = 42", inputs={"xs": [1]})

    assert result.return_value == 42
    assert result.exit_code == 0
    assert result.stdout == "hello"
    assert "__DR_SANDBOX_RESULT__" not in result.stdout
    assert result.duration_s >= 0.0

    argv = _run_argv(spawn)
    assert IMAGE in argv
    assert any(a.startswith("DR_SANDBOX_CODE_B64=") for a in argv)
    assert any(a.startswith("DR_SANDBOX_INPUTS_B64=") for a in argv)


async def test_run_command_shape_and_default_isolation_flags() -> None:
    with patch(
        SPAWN, side_effect=[_proc(stdout=b"__DR_SANDBOX_RESULT__:null\n"), _proc()]
    ) as spawn:
        await LocalDockerSandbox(IMAGE).run("x = 1")

    argv = _run_argv(spawn)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--name" in argv
    assert "--network" in argv and "none" in argv
    assert "--memory" in argv and "512m" in argv
    assert argv[-1] == IMAGE


async def test_disabling_isolation_omits_flags() -> None:
    sb = LocalDockerSandbox(IMAGE, network_disabled=False, memory_limit=None)
    with patch(
        SPAWN, side_effect=[_proc(stdout=b"__DR_SANDBOX_RESULT__:null\n"), _proc()]
    ) as spawn:
        await sb.run("x = 1")

    argv = _run_argv(spawn)
    assert "--network" not in argv
    assert "--memory" not in argv


async def test_security_context_flags_passed() -> None:
    sb = LocalDockerSandbox(IMAGE, security_context=SandboxSecurityContext())
    with patch(
        SPAWN, side_effect=[_proc(stdout=b"__DR_SANDBOX_RESULT__:null\n"), _proc()]
    ) as spawn:
        await sb.run("x = 1")

    argv = _run_argv(spawn)
    assert "--read-only" in argv
    assert "--cap-drop" in argv and "ALL" in argv
    assert "no-new-privileges=true" in argv


async def test_extra_run_args_appended() -> None:
    sb = LocalDockerSandbox(IMAGE, extra_run_args=["--label", "team=daria"])
    with patch(
        SPAWN, side_effect=[_proc(stdout=b"__DR_SANDBOX_RESULT__:null\n"), _proc()]
    ) as spawn:
        await sb.run("x = 1")

    argv = _run_argv(spawn)
    assert "--label" in argv and "team=daria" in argv


async def test_nonzero_exit_raises_sandbox_error() -> None:
    main = _proc(stdout=b"", stderr=b"boom traceback", returncode=1)
    with patch(SPAWN, side_effect=[main, _proc()]):
        with pytest.raises(SandboxError) as excinfo:
            await LocalDockerSandbox(IMAGE).run("raise ValueError('x')")
    assert "boom" in str(excinfo.value)


async def test_runner_timeout_exit_code_maps_to_timeout() -> None:
    # Runner's in-process SIGALRM cap fired first -> exit 124.
    main = _proc(stdout=b"__DR_SANDBOX_RESULT__:null\n", returncode=124)
    with patch(SPAWN, side_effect=[main, _proc()]):
        with pytest.raises(SandboxTimeout):
            await LocalDockerSandbox(IMAGE).run("while True:\n    pass", timeout_s=1)


async def test_client_timeout_kills_and_raises_timeout() -> None:
    main = _proc(communicate_exc=TimeoutError())
    with patch(SPAWN, side_effect=[main, _proc()]):
        with pytest.raises(SandboxTimeout):
            await LocalDockerSandbox(IMAGE).run("while True:\n    pass", timeout_s=1)
    main.kill.assert_called_once()


async def test_externals_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        await LocalDockerSandbox(IMAGE).run("x = 1", externals={"f": lambda: 1})


async def test_missing_docker_binary_raises_sandbox_error() -> None:
    with patch(SPAWN, side_effect=FileNotFoundError()):
        with pytest.raises(SandboxError) as excinfo:
            await LocalDockerSandbox(IMAGE, docker_bin="nope").run("x = 1")
    assert "not found" in str(excinfo.value)


async def test_teardown_force_removes_container() -> None:
    with patch(
        SPAWN, side_effect=[_proc(stdout=b"__DR_SANDBOX_RESULT__:null\n"), _proc()]
    ) as spawn:
        await LocalDockerSandbox(IMAGE).run("x = 1")

    rm_argv = list(spawn.call_args_list[1].args)
    assert rm_argv[:3] == ["docker", "rm", "-f"]
    # the same container name is created and torn down
    assert _run_argv(spawn)[_run_argv(spawn).index("--name") + 1] == rm_argv[3]


async def test_teardown_swallows_errors() -> None:
    # `docker rm -f` failing must not mask a successful run.
    with patch(
        SPAWN, side_effect=[_proc(stdout=b"__DR_SANDBOX_RESULT__:1\n"), OSError("rm failed")]
    ):
        result = await LocalDockerSandbox(IMAGE).run("_return = 1")
    assert result.return_value == 1


async def test_no_marker_yields_none_return() -> None:
    main = _proc(stdout=b"just some logs\n", returncode=0)
    with patch(SPAWN, side_effect=[main, _proc()]):
        result = await LocalDockerSandbox(IMAGE).run("print('hi')")
    assert result.return_value is None
    # No marker line -> stdout is returned unchanged (trailing newline kept).
    assert result.stdout == "just some logs\n"
