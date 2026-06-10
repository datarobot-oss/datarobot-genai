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
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.eval.generator import CaseGenerator

_URL = "https://app.datarobot.com/api/v2"
_MODEL = "my-llm-model"


def _mock_completion(cases: list[dict[str, Any]]) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = json.dumps(cases)
    return response


def _valid_case(case_id: str = "gen-001", behavior: str = "good") -> dict[str, Any]:
    return {
        "id": case_id,
        "source": "synthetic",
        "input": "What does this agent do?",
        "expected_behavior": behavior,
        "ideal_response": None,
        "notes": "Should explain capabilities",
    }


def _make_gen() -> CaseGenerator:
    return CaseGenerator(url=_URL, model_id=_MODEL)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_raises_without_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATAROBOT_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="url is required"):
        CaseGenerator(model_id=_MODEL)


def test_init_raises_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_DEFAULT_MODEL", raising=False)
    with pytest.raises(ValueError, match="model_id is required"):
        CaseGenerator(url=_URL)


def test_init_uses_env_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAROBOT_ENDPOINT", _URL)
    gen = CaseGenerator(url=None, model_id=_MODEL)
    assert gen._api_base == "https://app.datarobot.com"


def test_init_uses_env_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_DEFAULT_MODEL", _MODEL)
    gen = CaseGenerator(url=_URL)
    assert gen._model == f"datarobot/{_MODEL}"


def test_init_strips_api_v2_suffix() -> None:
    gen = _make_gen()
    assert gen._api_base == "https://app.datarobot.com"


def test_init_sets_model_with_prefix() -> None:
    gen = _make_gen()
    assert gen._model == f"datarobot/{_MODEL}"


def test_init_reads_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok-123")
    gen = _make_gen()
    assert gen._api_key == "tok-123"


def test_init_explicit_api_key_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "env-token")
    gen = CaseGenerator(url=_URL, model_id=_MODEL, api_key="explicit-token")
    assert gen._api_key == "explicit-token"


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def test_generate_returns_cases() -> None:
    cases = [_valid_case("gen-001", "good"), _valid_case("gen-002", "bad")]
    with patch("datarobot_genai.eval.generator.litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_completion(cases)
        result = _make_gen().generate("test agent", n_good=1, n_bad=1)
    assert len(result) == 2
    assert result[0]["id"] == "gen-001"
    assert result[1]["expected_behavior"] == "bad"


def test_generate_calls_litellm_with_model() -> None:
    cases = [_valid_case()]
    with patch("datarobot_genai.eval.generator.litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_completion(cases)
        _make_gen().generate("test agent", n_good=1, n_bad=0)

    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["model"] == f"datarobot/{_MODEL}"
    assert call_kwargs["api_base"] == "https://app.datarobot.com"


def test_generate_raises_on_missing_fields() -> None:
    incomplete = [{"id": "gen-001", "source": "synthetic"}]
    with patch("datarobot_genai.eval.generator.litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_completion(incomplete)
        with pytest.raises(ValueError, match="missing fields"):
            _make_gen().generate("test agent", n_good=1, n_bad=0)


def test_generate_raises_on_invalid_behavior() -> None:
    bad_case = {**_valid_case(), "expected_behavior": "maybe"}
    with patch("datarobot_genai.eval.generator.litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_completion([bad_case])
        with pytest.raises(ValueError, match="invalid expected_behavior"):
            _make_gen().generate("test agent", n_good=1, n_bad=0)


def test_generate_raises_on_non_list_response() -> None:
    with patch("datarobot_genai.eval.generator.litellm.completion") as mock_completion:
        response = MagicMock()
        response.choices[0].message.content = '{"id": "gen-001"}'
        mock_completion.return_value = response
        with pytest.raises(ValueError, match="Expected a JSON array"):
            _make_gen().generate("test agent", n_good=1, n_bad=0)


def test_generate_raises_on_non_string_content() -> None:
    with patch("datarobot_genai.eval.generator.litellm.completion") as mock_completion:
        response = MagicMock()
        response.choices[0].message.content = None
        mock_completion.return_value = response
        with pytest.raises(ValueError, match="Unexpected response content type"):
            _make_gen().generate("test agent", n_good=1, n_bad=0)


def test_generate_warns_on_count_mismatch() -> None:
    with patch("datarobot_genai.eval.generator.litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_completion([_valid_case("gen-001")])
        with pytest.warns(UserWarning, match="Requested 2 cases"):
            _make_gen().generate("test agent", n_good=1, n_bad=1)


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


def test_save_writes_file(tmp_path: Path) -> None:
    cases = [_valid_case()]
    gen = _make_gen()
    out = tmp_path / "output.json"
    gen.save(cases, out)
    assert out.exists()
    written = json.loads(out.read_text())
    assert len(written) == 1
    assert written[0]["id"] == "gen-001"


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    cases = [_valid_case()]
    out = tmp_path / "nested" / "dir" / "cases.json"
    _make_gen().save(cases, out)
    assert out.exists()


def test_save_overwrites_by_default(tmp_path: Path) -> None:
    gen = _make_gen()
    out = tmp_path / "cases.json"
    gen.save([_valid_case("gen-001")], out)
    gen.save([_valid_case("gen-002")], out)
    written = json.loads(out.read_text())
    assert len(written) == 1
    assert written[0]["id"] == "gen-002"


def test_save_append_merges(tmp_path: Path) -> None:
    gen = _make_gen()
    out = tmp_path / "cases.json"
    gen.save([_valid_case("gen-001")], out)
    result = gen.save([_valid_case("gen-002")], out, append=True)
    assert len(result) == 2
    ids = {c["id"] for c in result}
    assert ids == {"gen-001", "gen-002"}


def test_save_returns_final_list(tmp_path: Path) -> None:
    out = tmp_path / "cases.json"
    returned = _make_gen().save([_valid_case("gen-001"), _valid_case("gen-002")], out)
    assert len(returned) == 2
