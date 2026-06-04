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
"""Provider-compatible judge scoring.

A thin wrapper over ``nemo_evaluator.contrib.byob.judge.judge_score`` that fixes
one cross-provider incompatibility: NeMo's judge always sends BOTH ``temperature``
and ``top_p`` in the chat payload (``judge.judge_call`` hardcodes ``top_p`` with no
way to suppress it). Anthropic models — Claude on Bedrock, Vertex, or the direct
API — reject that combination with a hard 400:

    "`temperature` and `top_p` cannot both be specified for this model.
     Please use only one."

The agent/target path in NeMo's own ``runner.py`` already guards this
(``if top_p is not None``); only the judge path does not. We close the gap here
without forking the library: NeMo honours ``sample.config["_judge_session"]`` as
the request session, so we inject a session that drops the redundant ``top_p``
before the request leaves the process.

This is lossless. Every judge here runs at ``temperature=0.0`` for determinism,
and ``top_p`` has no effect at temperature 0 — so dropping it changes nothing on
providers that accept both (OpenAI/Azure) while unblocking those that don't.

Usage — benchmarks import ``judge_score`` from here instead of from NeMo::

    from datarobot_genai.eval.judge import judge_score
"""

from __future__ import annotations

import functools
import warnings
from typing import Any
from typing import cast

import requests
from nemo_evaluator.contrib.byob.judge import judge_score as _judge_score

# Sentinel so we can tell "no top_p present" apart from a real ``top_p`` value.
_UNSET = object()

# ``top_p=1.0`` is the no-op default NeMo injects when the judge config sets no
# top_p of its own. Dropping that is invisible, so we don't warn about it; we
# only warn when a *meaningful* top_p (one that would have changed sampling) is
# discarded, so a user who deliberately set it learns why it had no effect.
_NOOP_TOP_P = 1.0

# Warn at most once per process — the judge is called once per sample, and a
# per-sample warning would bury the signal under hundreds of identical lines.
# Single-element list avoids a `global` statement (PLW0603).
_warned: list[bool] = [False]


def _strip_redundant_top_p(payload: Any) -> Any:
    """Drop ``top_p`` from a chat payload when ``temperature`` is also present.

    Mutates ``payload`` in place. Returns the removed ``top_p`` value, or
    ``_UNSET`` when nothing was removed (not a dict, or not both params set).
    """
    if isinstance(payload, dict) and "temperature" in payload and "top_p" in payload:
        return payload.pop("top_p")
    return _UNSET


def _warn_top_p_dropped(value: Any) -> None:
    if _warned[0]:
        return
    _warned[0] = True
    warnings.warn(
        f"Judge request dropped top_p={value!r} because temperature was also set. "
        "Many providers (e.g. Anthropic / Bedrock Claude) reject temperature and "
        "top_p together, and at temperature=0 top_p has no effect. Set only one of "
        "the two in your judge config if you need top_p sampling.",
        UserWarning,
        stacklevel=2,
    )


class _JudgeCompatSession(requests.Session):
    """``requests.Session`` that sanitizes judge payloads before sending.

    NeMo's ``judge_call`` issues ``session.post(endpoint, json=payload, ...)``;
    intercepting ``post`` lets us fix the payload for any provider without
    touching the library's prompt/parse/score logic.
    """

    def post(self, url: str, **kwargs: Any) -> requests.Response:  # type: ignore[override]
        dropped = _strip_redundant_top_p(kwargs.get("json"))
        if dropped is not _UNSET and dropped != _NOOP_TOP_P:
            _warn_top_p_dropped(dropped)
        return super().post(url, **kwargs)


# One shared session is fine: it carries no per-judge state and NeMo already
# pools a single session per judge URL.
_SESSION = _JudgeCompatSession()


@functools.wraps(_judge_score)
def judge_score(sample: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """``nemo_evaluator`` ``judge_score`` with cross-provider payload sanitizing.

    Drop-in replacement: identical signature and return value. It just installs
    the sanitizing session on ``sample.config`` before delegating, so the judge
    works against Anthropic/Bedrock models in addition to OpenAI/Azure.
    """
    # setdefault: never clobber a session the caller deliberately supplied.
    sample.config.setdefault("_judge_session", _SESSION)
    # cast: nemo_evaluator ships untyped, so its return is Any to mypy.
    return cast(dict[str, Any], _judge_score(sample, *args, **kwargs))
