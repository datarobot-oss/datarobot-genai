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
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from datarobot_genai.eval.generator import CaseGenerator


def _make_mock_client(cases: list[dict[str, Any]]) -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns the given cases as JSON."""
    client = MagicMock(spec=anthropic.Anthropic)
    response = MagicMock()

    # Patch TextBlock in the generator module so isinstance check passes
    block = MagicMock()
    block.text = json.dumps(cases)
    response.content = [block]
    client.messages.create.return_value = response
    return client


def _valid_case(case_id: str = "gen-001", behavior: str = "good") -> dict[str, Any]:
    return {
        "id": case_id,
        "source": "synthetic",
        "input": "What does this agent do?",
        "expected_behavior": behavior,
        "ideal_response": None,
        "notes": "Should explain capabilities",
    }


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def test_generate_returns_cases() -> None:
    cases = [_valid_case("gen-001", "good"), _valid_case("gen-002", "bad")]
    client = _make_mock_client(cases)

    with patch(
        "datarobot_genai.eval.generator.anthropic.types.TextBlock",
        type(client.messages.create.return_value.content[0]),
    ):
        gen = CaseGenerator(client=client)
        result = gen.generate("test agent", n_good=1, n_bad=1)

    assert len(result) == 2
    assert result[0]["id"] == "gen-001"
    assert result[1]["expected_behavior"] == "bad"


def test_generate_calls_api_with_model(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [_valid_case()]
    client = _make_mock_client(cases)

    with patch(
        "datarobot_genai.eval.generator.anthropic.types.TextBlock",
        type(client.messages.create.return_value.content[0]),
    ):
        gen = CaseGenerator(client=client, model="claude-haiku-4-5-20251001")
        gen.generate("test agent", n_good=1, n_bad=0)

    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"


def test_generate_raises_on_missing_fields() -> None:
    incomplete = [{"id": "gen-001", "source": "synthetic"}]  # missing required fields
    client = _make_mock_client(incomplete)

    with patch(
        "datarobot_genai.eval.generator.anthropic.types.TextBlock",
        type(client.messages.create.return_value.content[0]),
    ):
        gen = CaseGenerator(client=client)
        with pytest.raises(ValueError, match="missing fields"):
            gen.generate("test agent", n_good=1, n_bad=0)


def test_generate_raises_on_invalid_behavior() -> None:
    bad_case = {**_valid_case(), "expected_behavior": "maybe"}
    client = _make_mock_client([bad_case])

    with patch(
        "datarobot_genai.eval.generator.anthropic.types.TextBlock",
        type(client.messages.create.return_value.content[0]),
    ):
        gen = CaseGenerator(client=client)
        with pytest.raises(ValueError, match="invalid expected_behavior"):
            gen.generate("test agent", n_good=1, n_bad=0)


def test_generate_raises_on_unexpected_block_type() -> None:
    client = MagicMock(spec=anthropic.Anthropic)
    response = MagicMock()
    response.content = [MagicMock()]  # plain MagicMock — not a TextBlock
    client.messages.create.return_value = response

    gen = CaseGenerator(client=client)
    with pytest.raises(ValueError, match="Unexpected response content type"):
        gen.generate("test agent", n_good=1, n_bad=0)


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


def test_save_writes_file(tmp_path: Path) -> None:
    cases = [_valid_case()]
    gen = CaseGenerator(client=MagicMock())
    out = tmp_path / "output.json"
    gen.save(cases, out)
    assert out.exists()
    written = json.loads(out.read_text())
    assert len(written) == 1
    assert written[0]["id"] == "gen-001"


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    cases = [_valid_case()]
    gen = CaseGenerator(client=MagicMock())
    out = tmp_path / "nested" / "dir" / "cases.json"
    gen.save(cases, out)
    assert out.exists()


def test_save_overwrites_by_default(tmp_path: Path) -> None:
    gen = CaseGenerator(client=MagicMock())
    out = tmp_path / "cases.json"
    gen.save([_valid_case("gen-001")], out)
    gen.save([_valid_case("gen-002")], out)
    written = json.loads(out.read_text())
    assert len(written) == 1
    assert written[0]["id"] == "gen-002"


def test_save_append_merges(tmp_path: Path) -> None:
    gen = CaseGenerator(client=MagicMock())
    out = tmp_path / "cases.json"
    gen.save([_valid_case("gen-001")], out)
    result = gen.save([_valid_case("gen-002")], out, append=True)
    assert len(result) == 2
    ids = {c["id"] for c in result}
    assert ids == {"gen-001", "gen-002"}


def test_save_returns_final_list(tmp_path: Path) -> None:
    gen = CaseGenerator(client=MagicMock())
    out = tmp_path / "cases.json"
    returned = gen.save([_valid_case("gen-001"), _valid_case("gen-002")], out)
    assert len(returned) == 2
