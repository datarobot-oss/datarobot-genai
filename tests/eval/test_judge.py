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
"""Unit tests for the provider-compatible judge wrapper.

The wrapper drops the redundant ``top_p`` that NeMo's judge always sends, which
Anthropic/Bedrock Claude models reject when ``temperature`` is also present.
These tests pin the sanitizing logic without a judge, a server, or a network.
"""

import warnings
from typing import Any

import pytest

from datarobot_genai.eval import judge


@pytest.fixture(autouse=True)
def _reset_warn_guard() -> Any:
    """The "warn once" guard is module-level; reset it around each test."""
    judge._warned = False
    yield
    judge._warned = False


# ---------------------------------------------------------------------------
# _strip_redundant_top_p
# ---------------------------------------------------------------------------


def test_strip_drops_top_p_when_both_present() -> None:
    payload = {"temperature": 0.0, "top_p": 0.9, "max_tokens": 16}
    dropped = judge._strip_redundant_top_p(payload)
    assert dropped == 0.9
    assert "top_p" not in payload
    assert payload == {"temperature": 0.0, "max_tokens": 16}


def test_strip_keeps_top_p_when_temperature_absent() -> None:
    payload = {"top_p": 0.9, "max_tokens": 16}
    assert judge._strip_redundant_top_p(payload) is judge._UNSET
    assert payload["top_p"] == 0.9


def test_strip_noop_when_no_top_p() -> None:
    payload = {"temperature": 0.0, "max_tokens": 16}
    assert judge._strip_redundant_top_p(payload) is judge._UNSET
    assert payload == {"temperature": 0.0, "max_tokens": 16}


def test_strip_ignores_non_dict() -> None:
    assert judge._strip_redundant_top_p(None) is judge._UNSET


# ---------------------------------------------------------------------------
# _JudgeCompatSession.post — sanitizes payload + warns on meaningful drop
# ---------------------------------------------------------------------------


class _RecordingSession(judge._JudgeCompatSession):
    """Capture the payload instead of issuing a real HTTP request."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: dict[str, Any] = {}

    def request(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self.sent = kwargs.get("json") or {}
        return "sentinel-response"


def test_session_strips_top_p_before_send() -> None:
    session = _RecordingSession()
    with warnings.catch_warnings(record=True):  # warning path covered separately
        warnings.simplefilter("always")
        session.post(
            "https://judge/chat/completions", json={"temperature": 0.0, "top_p": 0.5}
        )
    assert "top_p" not in session.sent
    assert session.sent["temperature"] == 0.0


def test_session_warns_on_meaningful_top_p() -> None:
    session = _RecordingSession()
    with pytest.warns(UserWarning, match="dropped top_p"):
        session.post(
            "https://judge/chat/completions", json={"temperature": 0.0, "top_p": 0.5}
        )


def test_session_silent_for_noop_top_p() -> None:
    session = _RecordingSession()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an exception
        session.post(
            "https://judge/chat/completions", json={"temperature": 0.0, "top_p": 1.0}
        )
    assert "top_p" not in session.sent  # still stripped, just not warned about


def test_session_warns_only_once() -> None:
    session = _RecordingSession()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session.post("https://judge/x", json={"temperature": 0.0, "top_p": 0.5})
        session.post("https://judge/x", json={"temperature": 0.0, "top_p": 0.3})
    assert len(caught) == 1


# ---------------------------------------------------------------------------
# judge_score wrapper — installs the sanitizing session and delegates
# ---------------------------------------------------------------------------


class _FakeSample:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {}


def test_wrapper_injects_session_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_judge_score(sample: Any, *args: Any, **kwargs: Any) -> dict:
        captured["session"] = sample.config.get("_judge_session")
        captured["args"] = (args, kwargs)
        return {"judge_score": 1.0, "judge_grade": "C"}

    monkeypatch.setattr(judge, "_judge_score", fake_judge_score)

    sample = _FakeSample()
    result = judge.judge_score(sample, template="binary_qa", criteria="x")

    assert result == {"judge_score": 1.0, "judge_grade": "C"}
    assert captured["session"] is judge._SESSION
    assert captured["args"] == ((), {"template": "binary_qa", "criteria": "x"})


def test_wrapper_preserves_caller_supplied_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(judge, "_judge_score", lambda sample, *a, **k: {})
    sample = _FakeSample()
    sentinel = object()
    sample.config["_judge_session"] = sentinel
    judge.judge_score(sample)
    assert sample.config["_judge_session"] is sentinel
