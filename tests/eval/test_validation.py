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
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.error import URLError

import pytest
import yaml

from datarobot_genai.eval.validation import health_check
from datarobot_genai.eval.validation import load_pipeline
from datarobot_genai.eval.validation import validate_inputs

# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


def test_health_check_returns_none_on_success() -> None:
    with patch("datarobot_genai.eval.validation.urlopen"):
        assert health_check("http://localhost:8842/v1") is None


def test_health_check_returns_none_on_http_error() -> None:
    # Any HTTP response (even 4xx) means the server is up
    with patch(
        "datarobot_genai.eval.validation.urlopen",
        side_effect=HTTPError(None, 404, "Not Found", {}, None),
    ):  # type: ignore[arg-type]
        assert health_check("http://localhost:8842/v1") is None


def test_health_check_returns_error_on_url_error() -> None:
    with patch(
        "datarobot_genai.eval.validation.urlopen", side_effect=URLError("connection refused")
    ):
        result = health_check("http://localhost:8842/v1")
    assert result is not None
    assert "not reachable" in result


def test_health_check_returns_error_on_unexpected_exception() -> None:
    with patch("datarobot_genai.eval.validation.urlopen", side_effect=OSError("timeout")):
        result = health_check("http://localhost:8842/v1")
    assert result is not None


# ---------------------------------------------------------------------------
# load_pipeline
# ---------------------------------------------------------------------------


def test_load_pipeline_valid(pipeline_yaml_path: Path) -> None:
    cfg = load_pipeline(pipeline_yaml_path)
    assert "benchmark" in cfg
    assert "target" in cfg
    assert "judge" in cfg


def test_load_pipeline_missing_section(tmp_path: Path) -> None:
    # `target` is required; omitting it must raise.
    incomplete = {"benchmark": {"module": "b.py", "name": "b"}}
    p = tmp_path / "incomplete.yaml"
    p.write_text(yaml.dump(incomplete))
    with pytest.raises(ValueError, match="missing required section"):
        load_pipeline(p)


def test_load_pipeline_judge_optional(tmp_path: Path) -> None:
    # Judge-free pipelines omit the judge: block entirely and must still load.
    cfg = {
        "benchmark": {
            "module": "datarobot_genai/eval/benchmarks/pii_leakage.py",
            "name": "pii_leakage",
        },
        "target": {"model_type": "chat", "model_id": "unknown"},
        "run": {"parallelism": 4},
    }
    p = tmp_path / "judge_free.yaml"
    p.write_text(yaml.dump(cfg))
    loaded = load_pipeline(p)
    assert "judge" not in loaded
    assert loaded["benchmark"]["name"] == "pii_leakage"


def test_load_pipeline_not_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="did not parse to a mapping"):
        load_pipeline(p)


# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------


def test_validate_inputs_all_pass(
    tmp_path: Path, pipeline_yaml_path: Path, dataset_path: Path
) -> None:
    pipelines_dir = pipeline_yaml_path.parent
    with patch("datarobot_genai.eval.validation.health_check", return_value=None):
        errors = validate_inputs(
            "http://localhost/v1",
            "test_pipeline.yaml",
            str(dataset_path),
            pipelines_dir,
            tmp_path,
        )
    assert errors == []


def test_validate_inputs_endpoint_unreachable(
    tmp_path: Path, pipeline_yaml_path: Path, dataset_path: Path
) -> None:
    pipelines_dir = pipeline_yaml_path.parent
    with patch("datarobot_genai.eval.validation.health_check", return_value="not reachable"):
        errors = validate_inputs(
            "http://bad",
            "test_pipeline.yaml",
            str(dataset_path),
            pipelines_dir,
            tmp_path,
        )
    assert any("Health check failed" in e for e in errors)


def test_validate_inputs_missing_pipeline(tmp_path: Path, dataset_path: Path) -> None:
    pipelines_dir = tmp_path / "pipelines"
    pipelines_dir.mkdir()
    with patch("datarobot_genai.eval.validation.health_check", return_value=None):
        errors = validate_inputs(
            "http://localhost/v1",
            "nonexistent.yaml",
            str(dataset_path),
            pipelines_dir,
            tmp_path,
        )
    assert any("not found" in e for e in errors)


def test_validate_inputs_missing_dataset(tmp_path: Path, pipeline_yaml_path: Path) -> None:
    pipelines_dir = pipeline_yaml_path.parent
    with patch("datarobot_genai.eval.validation.health_check", return_value=None):
        errors = validate_inputs(
            "http://localhost/v1",
            "test_pipeline.yaml",
            str(tmp_path / "missing.json"),
            pipelines_dir,
            tmp_path,
        )
    assert any("Dataset not found" in e for e in errors)


def test_validate_inputs_missing_benchmark_module(tmp_path: Path, dataset_path: Path) -> None:
    # Pipeline references a benchmark module that doesn't exist
    import yaml as _yaml

    pipelines_dir = tmp_path / "pipelines"
    pipelines_dir.mkdir()
    cfg = {
        "benchmark": {"module": "benchmarks/missing.py", "name": "x"},
        "target": {},
        "judge": {},
    }
    (pipelines_dir / "p.yaml").write_text(_yaml.dump(cfg))

    with patch("datarobot_genai.eval.validation.health_check", return_value=None):
        errors = validate_inputs(
            "http://localhost/v1",
            "p.yaml",
            str(dataset_path),
            pipelines_dir,
            tmp_path,
        )
    assert any("Benchmark module not found" in e for e in errors)


def test_validate_inputs_collects_multiple_errors(tmp_path: Path) -> None:
    pipelines_dir = tmp_path / "pipelines"
    pipelines_dir.mkdir()
    with patch("datarobot_genai.eval.validation.health_check", return_value="bad endpoint"):
        errors = validate_inputs(
            "http://bad",
            "missing.yaml",
            str(tmp_path / "missing.json"),
            pipelines_dir,
            tmp_path,
        )
    assert len(errors) >= 2


# ---------------------------------------------------------------------------
# preflight_judge
# ---------------------------------------------------------------------------

import io
from unittest.mock import MagicMock

from datarobot_genai.eval.validation import preflight_judge


def test_preflight_judge_raises_when_env_var_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)
    cfg = {
        "url": "https://example.com/llmgw",
        "model_id": "azure/gpt-4o",
        "api_key_name": "DATAROBOT_API_TOKEN",
    }
    with pytest.raises(RuntimeError, match="is not set"):
        preflight_judge(cfg)


def test_preflight_judge_succeeds_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TOKEN", "tok-abc")
    cfg = {
        "url": "https://example.com/llmgw",
        "model_id": "azure/gpt-4o",
        "api_key_name": "MY_TOKEN",
    }
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("datarobot_genai.eval.validation.urlopen", return_value=mock_resp):
        preflight_judge(cfg)  # should not raise


def test_preflight_judge_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TOKEN", "tok-abc")
    cfg = {
        "url": "https://example.com/llmgw",
        "model_id": "azure/gpt-4o",
        "api_key_name": "MY_TOKEN",
    }
    err = HTTPError(
        "https://example.com/llmgw/chat/completions",
        401,
        "Unauthorized",
        {},
        io.BytesIO(b"invalid token"),
    )
    with patch("datarobot_genai.eval.validation.urlopen", side_effect=err):
        with pytest.raises(RuntimeError, match="HTTP 401"):
            preflight_judge(cfg)


def test_preflight_judge_raises_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TOKEN", "tok-abc")
    cfg = {
        "url": "https://example.com/llmgw",
        "model_id": "azure/gpt-4o",
        "api_key_name": "MY_TOKEN",
    }
    with patch(
        "datarobot_genai.eval.validation.urlopen", side_effect=URLError("connection refused")
    ):
        with pytest.raises(RuntimeError, match="cannot reach"):
            preflight_judge(cfg)
