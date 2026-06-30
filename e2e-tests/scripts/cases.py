#!/usr/bin/env -S uv run --script
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

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2",
#     "pyyaml",
# ]
# ///

"""Case-file parser and matrix generator for the e2e test suite.

Reads a YAML case file (see ``e2e-tests/cases/*.yaml``), validates it with
pydantic, expands each case's ``matrix`` block into a flat list of
combinations, and either prints them as a GitHub Actions matrix include or as
a human-readable table.

The CLI is invoked from two places:

* ``.github/workflows/e2e.yml`` ``prepare-matrix`` job runs ``generate-matrix``
  to produce the strategy matrix for the downstream ``e2e`` job.
* ``e2e-tests/scripts/run_local.py`` imports :func:`load_cases` and
  :func:`expand` to build the same combination list before running pytest.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

DEFAULT_PYTHON_VERSION = "3.12"
DEFAULT_PYTEST_ARGS: tuple[str, ...] = (
    "-vv",
    "-s",
    "--timeout=60",
    "--tb=short",
    "--reruns=1",
    "--reruns-delay=5",
)

# AGENT is the only matrix dim the rest of the pipeline (CI filtering,
# `uv sync --group dragent-<agent>`, `task dragent:run-<agent>`) treats as
# special; every case must declare it.
AGENT_DIM = "AGENT"

_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize(value: str) -> str:
    """Make *value* safe for use in matrix names and artifact paths."""
    return _NAME_SAFE_RE.sub("-", value).strip("-")


# Allow callers to pass either a bare file name (resolved under
# ``e2e-tests/cases/``) or any explicit path (relative or absolute). The
# convention is "no slash → look in cases/".
DEFAULT_CASES_DIR = Path(__file__).resolve().parent.parent / "cases"


def resolve_case_file(arg: str) -> Path:
    """Resolve ``arg`` to an existing case YAML, looking in cases/ for bare names."""
    candidates: list[Path] = []
    p = Path(arg)
    if p.is_absolute() or "/" in arg or "\\" in arg:
        candidates.append(p)
    else:
        candidates.append(p)  # cwd-relative
        candidates.append(DEFAULT_CASES_DIR / arg)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"case file {arg!r} not found (searched: {searched})")


def _stringify_scalar(v: object) -> str:
    """Coerce YAML scalar (int/bool/float/str) to a string env value."""
    if isinstance(v, bool):
        # YAML 'true'/'false' would round-trip as 'True'/'False' otherwise;
        # tests usually expect lowercase. Be explicit.
        return "true" if v else "false"
    if v is None:
        return ""
    return str(v)


class Case(BaseModel):
    """One entry in the YAML ``cases:`` list."""

    model_config = ConfigDict(extra="forbid")

    name: str
    env: dict[str, str] = Field(default_factory=dict)
    tests: list[str]
    matrix: dict[str, list[str]]
    python_version: str = DEFAULT_PYTHON_VERSION
    pytest_args: list[str] = Field(
        default_factory=lambda: list(DEFAULT_PYTEST_ARGS),
    )

    # Env values are env vars, so coerce YAML scalars (e.g. integers like
    # ``USE_DATAROBOT_LLM_GATEWAY: 0``) to strings rather than forcing the
    # author to quote everything.
    @field_validator("env", mode="before")
    @classmethod
    def _coerce_env(cls, raw: object) -> dict[str, str]:
        if not isinstance(raw, dict):
            raise TypeError(f"env must be a mapping, got {type(raw).__name__}")
        return {str(k): _stringify_scalar(v) for k, v in raw.items()}

    @field_validator("matrix", mode="before")
    @classmethod
    def _coerce_matrix(cls, raw: object) -> dict[str, list[str]]:
        if not isinstance(raw, dict):
            raise TypeError(
                f"matrix must be a mapping, got {type(raw).__name__}"
            )
        coerced: dict[str, list[str]] = {}
        for k, values in raw.items():
            if not isinstance(values, list):
                raise TypeError(
                    f"matrix dim {k!r} must be a list, got "
                    f"{type(values).__name__}"
                )
            coerced[str(k)] = [_stringify_scalar(v) for v in values]
        return coerced


class CasesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: list[Case]


class Combination(BaseModel):
    """One concrete (case, matrix combo) pair ready to execute."""

    model_config = ConfigDict(extra="forbid")

    name: str
    case: str
    agent: str
    python_version: str
    tests: list[str]
    pytest_args: list[str]
    env: dict[str, str]

    def to_matrix_entry(self) -> dict[str, object]:
        """Flatten for a GitHub Actions ``strategy.matrix.include`` entry.

        ``tests`` and ``pytest_args`` are joined with spaces because they get
        interpolated directly into shell commands (``uv run pytest
        ${{ matrix.tests }} ${{ matrix.pytest_args }}``).
        """
        return {
            "name": self.name,
            "case": self.case,
            "agent": self.agent,
            "python_version": self.python_version,
            "tests": " ".join(self.tests),
            "pytest_args": " ".join(self.pytest_args),
            "env": dict(self.env),
        }


def load_cases(path: Path) -> list[Case]:
    """Parse and validate a case YAML file."""
    raw = yaml.safe_load(path.read_text())
    return CasesFile.model_validate(raw).cases


def _validate_case(case: Case) -> None:
    if AGENT_DIM not in case.matrix:
        raise ValueError(
            f"case {case.name!r}: matrix must include {AGENT_DIM!r}"
        )
    overlap = set(case.env) & set(case.matrix)
    if overlap:
        raise ValueError(
            f"case {case.name!r}: keys {sorted(overlap)} appear in both env "
            "and matrix"
        )
    for dim, values in case.matrix.items():
        if not values:
            raise ValueError(
                f"case {case.name!r}: matrix dim {dim!r} is empty"
            )


def _expand_case(
    case: Case,
    *,
    agents: Iterable[str] | None,
    overrides: dict[str, str] | None,
) -> list[Combination]:
    _validate_case(case)
    agent_set = set(agents) if agents is not None else None
    overrides = overrides or {}

    dims = list(case.matrix.keys())
    value_lists = [case.matrix[d] for d in dims]

    combos: list[Combination] = []
    for values in itertools.product(*value_lists):
        combo_env = dict(zip(dims, values))

        # KEY=VAL filter: drop combos that don't match every override pinned
        # by the caller. Unknown keys raise — we want to fail loudly rather
        # than silently emit zero combos.
        skip = False
        for k, v in overrides.items():
            if k not in case.matrix:
                raise ValueError(
                    f"case {case.name!r}: override {k}={v} is not a matrix "
                    f"dim (have {sorted(case.matrix)})"
                )
            if combo_env[k] != v:
                skip = True
                break
        if skip:
            continue

        agent = combo_env[AGENT_DIM]
        if agent_set is not None and agent not in agent_set:
            continue

        env = dict(case.env)
        env.update(combo_env)

        suffix = "--".join(_sanitize(v) for v in values)
        combos.append(
            Combination(
                name=f"{_sanitize(case.name)}--{suffix}",
                case=case.name,
                agent=agent,
                python_version=case.python_version,
                tests=list(case.tests),
                pytest_args=list(case.pytest_args),
                env=env,
            )
        )
    return combos


def expand(
    cases: list[Case],
    *,
    case_name: str | None = None,
    agents: Iterable[str] | None = None,
    overrides: dict[str, str] | None = None,
) -> list[Combination]:
    """Expand cases into combinations, applying optional filters.

    Parameters
    ----------
    cases:
        Cases as loaded by :func:`load_cases`.
    case_name:
        If set, restrict to the single case with this ``name``. Raises
        ``ValueError`` if no such case exists.
    agents:
        If set, drop combinations whose ``AGENT`` value is not in the
        allowlist. ``None`` disables agent filtering (all agents pass).
    overrides:
        ``KEY=VAL`` filter from the local CLI; combos that don't match every
        pinned dim are dropped. Keys must be matrix dims; unknown keys raise.
    """
    selected = cases
    if case_name is not None:
        selected = [c for c in cases if c.name == case_name]
        if not selected:
            raise ValueError(f"no case named {case_name!r}")

    out: list[Combination] = []
    for case in selected:
        out.extend(
            _expand_case(case, agents=agents, overrides=overrides)
        )
    return out


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items


def _emit_matrix(combos: list[Combination]) -> None:
    payload = {"include": [c.to_matrix_entry() for c in combos]}
    text = json.dumps(payload)
    print(text)
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(f"matrix={text}\n")


def _format_table(combos: list[Combination]) -> str:
    if not combos:
        return "(no combinations)"

    headers = ("name", "case", "agent", "tests", "env")
    rows: list[tuple[str, str, str, str, str]] = []
    for c in combos:
        env = " ".join(f"{k}={v}" for k, v in sorted(c.env.items()))
        rows.append((c.name, c.case, c.agent, " ".join(c.tests), env))

    widths = [
        max(len(h), max(len(r[i]) for r in rows))
        for i, h in enumerate(headers)
    ]

    def line(parts: Iterable[str]) -> str:
        return "  ".join(p.ljust(w) for p, w in zip(parts, widths))

    out = [line(headers), line(["-" * w for w in widths])]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("case_file",
                   help="Case YAML to read. A bare file name (e.g. "
                        "'pr-tests.yaml') is resolved against e2e-tests/cases/; "
                        "anything containing a slash is used as-is.")
    p.add_argument("--case", default=None,
                   help="Limit to a single case by name.")
    p.add_argument(
        "--agents",
        default=None,
        help="Comma-separated allowlist of agents (e.g. 'nat,langgraph'). "
             "Pass an empty string to drop everything; omit for no filter.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cases.py")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_common_args(sub.add_parser(
        "generate-matrix",
        help="Print a GitHub Actions matrix include JSON.",
    ))
    _add_common_args(sub.add_parser(
        "list",
        help="Print expanded combinations as a table.",
    ))
    args = parser.parse_args(argv)

    try:
        case_path = resolve_case_file(args.case_file)
        cases = load_cases(case_path)
        agents = _parse_csv(args.agents)
        combos = expand(cases, case_name=args.case, agents=agents)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "generate-matrix":
        _emit_matrix(combos)
    else:
        print(_format_table(combos))
    return 0


if __name__ == "__main__":
    sys.exit(main())
