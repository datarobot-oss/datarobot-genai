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
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.eval.runner import _resolve_benchmark_module
from datarobot_genai.eval.runner import run_byob


def _cfg() -> dict[str, Any]:
    return {
        "benchmark": {
            "module": "datarobot_genai/eval/benchmarks/answer_quality.py",
            "name": "answer_quality",
        },
        "target": {"model_type": "chat", "model_id": "unknown"},
        "judge": {
            "url": "https://judge.example.com",
            "model_id": "gpt-4o",
            "api_key_name": "JUDGE_KEY",
        },
        "run": {
            "parallelism": 2,
            "max_tokens": 512,
            "temperature": 0.0,
            "timeout_per_sample": 60,
        },
    }


def test_run_byob_invokes_subprocess(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("datarobot_genai.eval.runner.subprocess.run", return_value=mock_result) as mock_run:
        run_byob(_cfg(), "http://agent/v1", "/tmp/dataset.jsonl", "/tmp/output", tmp_path)

    assert mock_run.called
    cmd = mock_run.call_args[0][0]
    assert "-m" in cmd
    assert "nemo_evaluator.contrib.byob.runner" in cmd


def test_run_byob_passes_required_flags(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("datarobot_genai.eval.runner.subprocess.run", return_value=mock_result) as mock_run:
        run_byob(_cfg(), "http://agent/v1", "/tmp/ds.jsonl", "/tmp/out", tmp_path)

    cmd = mock_run.call_args[0][0]
    assert "--benchmark-module" in cmd
    assert "--benchmark-name" in cmd
    assert "--dataset" in cmd
    assert "--model-url" in cmd
    assert "--output-dir" in cmd
    assert "--save-predictions" in cmd


def test_run_byob_sets_judge_env_vars(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("datarobot_genai.eval.runner.subprocess.run", return_value=mock_result) as mock_run:
        run_byob(_cfg(), "http://agent/v1", "/tmp/ds.jsonl", "/tmp/out", tmp_path)

    env = mock_run.call_args[1]["env"]
    assert env["JUDGE_URL"] == "https://judge.example.com"
    assert env["JUDGE_MODEL_ID"] == "gpt-4o"
    assert env["JUDGE_API_KEY_NAME"] == "JUDGE_KEY"


def test_run_byob_omits_judge_env_when_judge_free(tmp_path: Path) -> None:
    """A judge-free pipeline (no judge: block) exports no JUDGE_* env vars."""
    cfg = _cfg()
    del cfg["judge"]
    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("datarobot_genai.eval.runner.subprocess.run", return_value=mock_result) as mock_run:
        run_byob(cfg, "http://agent/v1", "/tmp/ds.jsonl", "/tmp/out", tmp_path)

    env = mock_run.call_args[1]["env"]
    assert "JUDGE_URL" not in env
    assert "JUDGE_MODEL_ID" not in env
    assert "JUDGE_API_KEY_NAME" not in env


def test_run_byob_clears_inherited_judge_env_when_judge_free(tmp_path: Path) -> None:
    """Inherited JUDGE_* vars are scrubbed so a judge-free pipeline can't be accidentally
    activated.
    """
    cfg = _cfg()
    del cfg["judge"]
    mock_result = MagicMock()
    mock_result.returncode = 0

    inherited = {
        "JUDGE_URL": "https://old-judge.example.com",
        "JUDGE_MODEL_ID": "old-model",
        "JUDGE_API_KEY_NAME": "OLD_KEY",
    }
    with (
        patch("datarobot_genai.eval.runner.subprocess.run", return_value=mock_result) as mock_run,
        patch.dict("os.environ", inherited),
    ):
        run_byob(cfg, "http://agent/v1", "/tmp/ds.jsonl", "/tmp/out", tmp_path)

    env = mock_run.call_args[1]["env"]
    assert "JUDGE_URL" not in env
    assert "JUDGE_MODEL_ID" not in env
    assert "JUDGE_API_KEY_NAME" not in env


def test_run_byob_raises_on_nonzero_exit(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1

    with patch("datarobot_genai.eval.runner.subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError, match="BYOB runner exited"):
            run_byob(_cfg(), "http://agent/v1", "/tmp/ds.jsonl", "/tmp/out", tmp_path)


def test_run_byob_does_not_pass_api_key_flag_when_env_var_unset(tmp_path: Path) -> None:
    """api_key_name is only forwarded if the env var is actually set."""
    cfg = _cfg()
    cfg["target"]["api_key_name"] = "AGENT_API_KEY"
    mock_result = MagicMock()
    mock_result.returncode = 0

    with (
        patch("datarobot_genai.eval.runner.subprocess.run", return_value=mock_result) as mock_run,
        patch.dict("os.environ", {}, clear=True),
    ):
        run_byob(cfg, "http://agent/v1", "/tmp/ds.jsonl", "/tmp/out", tmp_path)

    cmd = mock_run.call_args[0][0]
    assert "--api-key-name" not in cmd


# ---------------------------------------------------------------------------
# _resolve_benchmark_module
# ---------------------------------------------------------------------------


def test_resolve_local_file_wins(tmp_path: Path) -> None:
    """A local file takes priority over the installed package."""
    module_dir = tmp_path / "evaluator" / "benchmarks"
    module_dir.mkdir(parents=True)
    local_file = module_dir / "my_benchmark.py"
    local_file.write_text("# custom")
    result = _resolve_benchmark_module("evaluator/benchmarks/my_benchmark.py", tmp_path)
    assert result == str(local_file.absolute())


def test_resolve_falls_back_to_installed_package(tmp_path: Path) -> None:
    """When no local file exists, resolves via the installed package."""
    result = _resolve_benchmark_module(
        "datarobot_genai/eval/benchmarks/answer_correctness.py", tmp_path
    )
    assert "answer_correctness" in result
    assert Path(result).exists()


def test_resolve_strips_py_extension_for_import(tmp_path: Path) -> None:
    """The .py suffix is stripped before the importlib lookup."""
    result = _resolve_benchmark_module(
        "datarobot_genai/eval/benchmarks/pii_leakage.py", tmp_path
    )
    assert Path(result).exists()


def test_resolve_raises_on_unresolvable_module(tmp_path: Path) -> None:
    """Neither a local file nor an importable module raises ImportError."""
    with pytest.raises(ImportError):
        _resolve_benchmark_module("totally/nonexistent/benchmark.py", tmp_path)
