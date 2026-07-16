# Copyright 2025 DataRobot, Inc. and its affiliates.
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

"""Generate the shipped uv consumer config from pyproject.toml.

``pyproject.toml`` ``[tool.uv]`` is the single source of truth for the
override / exclude / constraint lists. uv reads those lists ONLY from the
workspace-root pyproject.toml and never publishes them in package metadata, so
consumers must copy them by hand. To keep that copy honest we ship it in the
wheel at ``src/datarobot_genai/_packaging/uv_consumer_config.toml`` and generate
it from the authoritative lists rather than hand-maintaining a parallel file.

    task sync-consumer-config     # rewrite the shipped file from pyproject.toml
    task check-consumer-config    # fail if the shipped file is out of date (CI)

The generated file is the three arrays copied verbatim (comments included),
under the fixed consumer-facing header below. Editing it by hand is pointless:
CI regenerates it and diffs, so any manual change shows up as drift.
"""

from __future__ import annotations

import difflib
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
TARGET = REPO_ROOT / "src" / "datarobot_genai" / "_packaging" / "uv_consumer_config.toml"

# Emitted in this order regardless of how they appear in pyproject.toml.
ARRAYS = (
    "override-dependencies",
    "exclude-dependencies",
    "constraint-dependencies",
)

HEADER = """\
# =============================================================================
# datarobot-genai -- canonical uv configuration for CONSUMERS
# =============================================================================
# AUTO-GENERATED from pyproject.toml [tool.uv] by scripts/gen_consumer_config.py.
# Do NOT edit by hand: run `task sync-consumer-config` after changing the
# [tool.uv] lists in pyproject.toml, and commit the result. CI (Packaging CI)
# regenerates this file and fails the build if it drifts.
# -----------------------------------------------------------------------------
# These three lists CANNOT be delivered through package metadata. uv reads
# override-dependencies / exclude-dependencies / constraint-dependencies ONLY
# from the *workspace-root* pyproject.toml, and they are never published in a
# wheel's Requires-Dist. Any project that depends on datarobot-genai and pulls
# the crewai / nvidia-nat / llama-index stacks MUST copy the entries below into
# its own [tool.uv] section, or dependency resolution will either fail outright
# or silently install CVE-vulnerable versions.
#
#   Print this block:   datarobot-genai-uv-config
#
# Requirements / notes:
#   * Needs uv >= 0.9.8 for first-class `exclude-dependencies`.
#   * Re-copy this block whenever you bump the datarobot-genai version.
#   * CVE-safe *floors* for packages datarobot-genai depends on are intentionally
#     NOT listed here: those ship as real dependency metadata and are inherited
#     automatically, so you do not need to restate them.
#   * If you also install Jupyter / notebook tooling (e.g. an agentic playground
#     extra), add `mistune>=3.3.0` to constraint-dependencies yourself -- that is
#     pulled by nbconvert, not by datarobot-genai.
# -----------------------------------------------------------------------------
"""


def _uv_section_lines(text: str) -> list[str]:
    """Return the lines of the ``[tool.uv]`` table (header line excluded)."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "[tool.uv]":
            start = i + 1
            break
    if start is None:
        raise SystemExit("no [tool.uv] section found in pyproject.toml")
    # The table ends at the next top-level table header. Table headers start
    # with '[' in column 0; array closing brackets start with ']', and array
    # items are indented, so this cleanly bounds the section.
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("["):
            end = j
            break
    return lines[start:end]


def _extract_array(section: list[str], name: str) -> str:
    """Return the verbatim ``name = [ ... ]`` block, including inline comments."""
    out: list[str] = []
    capturing = False
    for line in section:
        if not capturing and line.startswith(f"{name} = ["):
            capturing = True
            out.append(line)
            continue
        if capturing:
            out.append(line)
            if line.rstrip() == "]":
                break
    if not out or out[-1].rstrip() != "]":
        raise SystemExit(f"could not extract complete array `{name}` from [tool.uv]")
    return "\n".join(out)


def render() -> str:
    """Build the consumer config text from pyproject.toml [tool.uv]."""
    text = PYPROJECT.read_text(encoding="utf-8")
    section = _uv_section_lines(text)
    blocks = [_extract_array(section, name) for name in ARRAYS]
    rendered = HEADER + "\n" + "\n\n".join(blocks) + "\n"
    _validate(rendered, text)
    return rendered


def _validate(rendered: str, pyproject_text: str) -> None:
    """Confirm the rendered file parses and matches pyproject's values exactly."""
    got = tomllib.loads(rendered)
    src = tomllib.loads(pyproject_text)["tool"]["uv"]
    for name in ARRAYS:
        if got.get(name) != src.get(name):
            raise SystemExit(
                f"generated `{name}` does not match pyproject.toml after render; "
                "the extractor and pyproject formatting have diverged"
            )


def main() -> int:
    """Write (or, with --check, verify) the shipped consumer config."""
    check = "--check" in sys.argv[1:]
    rendered = render()
    rel = TARGET.relative_to(REPO_ROOT)

    if check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current == rendered:
            print(f"OK: {rel} is in sync with pyproject.toml [tool.uv].")
            return 0
        sys.stdout.writelines(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                fromfile=f"{rel} (on disk)",
                tofile=f"{rel} (regenerated)",
            )
        )
        print(
            f"\nERROR: {rel} is OUT OF SYNC with pyproject.toml [tool.uv].\n"
            "Fix it with:  task sync-consumer-config   (then commit the change)."
        )
        return 1

    TARGET.write_text(rendered, encoding="utf-8")
    print(f"Wrote {rel} from pyproject.toml [tool.uv].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
