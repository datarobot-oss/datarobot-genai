# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

_RUN_LOCAL_PATH = Path(__file__).parents[2] / "e2e-tests" / "scripts" / "run_local.py"
_SCRIPTS_PATH = _RUN_LOCAL_PATH.parent

if str(_SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_PATH))

_CASES_STUB = ModuleType("cases")
_CASES_STUB.Combination = object
_CASES_STUB.expand = None
_CASES_STUB.load_cases = None
_CASES_STUB.resolve_case_file = None
sys.modules.setdefault("cases", _CASES_STUB)

_SPEC = importlib.util.spec_from_file_location("run_local", _RUN_LOCAL_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
run_local = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_local)


def test_main_stops_custom_metric_server_when_health_check_fails(monkeypatch, capsys):
    # GIVEN an ootb-custom combo whose custom metric server starts but never becomes healthy
    proc = object()
    stopped: list[object] = []
    combo = SimpleNamespace(
        name="ootb-custom-nat",
        case="ootb-custom",
        agent="nat",
        env={},
        tests=[],
        pytest_args=[],
    )

    monkeypatch.setattr(run_local, "resolve_case_file", lambda _arg: Path("cases.yaml"))
    monkeypatch.setattr(run_local, "load_cases", lambda _path: [object()])
    monkeypatch.setattr(
        run_local,
        "expand",
        lambda _cases, *, case_name, overrides: [combo],
    )
    monkeypatch.setattr(run_local, "_start_guards_server", lambda: proc)
    monkeypatch.setattr(run_local, "_wait_for_health", lambda *, url, timeout: False)

    def stop_server(stopped_proc):
        stopped.append(stopped_proc)

    monkeypatch.setattr(run_local, "_stop_server", stop_server)
    monkeypatch.setattr(
        run_local,
        "_uv_sync",
        lambda _agent: pytest.fail("main should return before installing dependencies"),
    )
    monkeypatch.setattr(
        run_local,
        "_run_one",
        lambda _combo, *, no_server: pytest.fail("main should return before running combos"),
    )

    # WHEN main exits early because the custom metric health check fails
    rc = run_local.main(["cases.yaml"])

    # THEN the process is still stopped by the surrounding finally block
    assert rc == 1
    assert stopped == [proc]
    assert "error: guards server failed to start" in capsys.readouterr().err
