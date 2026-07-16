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

"""Regression tests for the published documentation layout and nav.

Only files under ``docs/src`` are published (``properdocs.yml`` sets
``docs_dir: src``).  These tests pin the Application Utils section — the
memory-ORM/chat-history split — and guard the whole nav against references 
to files that do not exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Repo root: this file is <root>/tests/docs/test_docs_structure.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs"


def _load_nav_config() -> dict:
    """Parse ``docs/properdocs.yml`` into a dict."""
    return yaml.safe_load((_DOCS / "properdocs.yml").read_text())


def _nav_leaves(node: object) -> list[str]:
    """Return every string leaf (page path) reachable from a nav *node*."""
    leaves: list[str] = []
    if isinstance(node, dict):
        for value in node.values():
            leaves.extend(_nav_leaves(value))
    elif isinstance(node, list):
        for value in node:
            leaves.extend(_nav_leaves(value))
    elif isinstance(node, str):
        leaves.append(node)
    return leaves


def test_application_utils_docs_are_published_and_split() -> None:
    """
    GIVEN the docs move/split.

    WHEN the published ``docs/src/application_utils`` directory is inspected.

    THEN it contains the section-index README plus the memory-ORM and
    chat-history pages.
    """
    section = _DOCS / "src" / "application_utils"

    assert (section / "README.md").is_file()
    assert (section / "memory_orm.md").is_file()
    assert (section / "chat_history.md").is_file()


def test_old_unpublished_application_utils_dir_is_removed() -> None:
    """
    GIVEN the prior PR mistakenly placed the docs outside ``docs/src``.

    WHEN the repository is inspected after the application_utils move.

    THEN the unpublished ``docs/application_utils`` directory no longer exists.
    """
    assert not (_DOCS / "application_utils").exists()


def test_application_utils_nav_block_follows_section_index_convention() -> None:
    """
    GIVEN the ``properdocs.yml`` navigation.

    WHEN the "Application Utils" block is located.

    THEN the section-index README is listed first, then the memory-ORM and
    chat-history pages in order.
    """
    nav = _load_nav_config()["nav"]

    app_utils_block = next(
        (
            entry["Application Utils"]
            for entry in nav
            if isinstance(entry, dict) and "Application Utils" in entry
        ),
        None,
    )

    assert app_utils_block is not None, "properdocs.yml is missing an 'Application Utils' nav block"
    assert app_utils_block[0] == "application_utils/README.md"
    assert {"Memory Service ORM": "application_utils/memory_orm.md"} in app_utils_block
    assert {"Chat History": "application_utils/chat_history.md"} in app_utils_block


def test_every_nav_page_resolves_to_a_published_file() -> None:
    """
    GIVEN the full ``properdocs.yml`` navigation.

    WHEN each page reference is resolved against ``docs_dir``.

    THEN every entry points at a real file (the auto-generated ``api/`` tree,
    produced at build time by ``gen_ref_pages.py``, is exempt).
    """
    config = _load_nav_config()
    docs_dir = _DOCS / config.get("docs_dir", "src")

    missing = [
        page
        for page in _nav_leaves(config["nav"])
        if not page.endswith("/") and not (docs_dir / page).is_file()
    ]

    assert missing == [], f"nav references non-existent pages: {missing}"


def test_stale_persistence_integration_path_is_corrected() -> None:
    """
    GIVEN the old README pointed at a non-existent ``persistence/integration`` dir.

    WHEN the split memory-ORM page is read.

    THEN it points at the real ``persistence/acceptance -m integration`` suite.
    """
    memory_orm = (_DOCS / "src" / "application_utils" / "memory_orm.md").read_text()

    assert "persistence/integration" not in memory_orm
    assert "persistence/acceptance -m integration" in memory_orm


@pytest.mark.parametrize(
    "needle",
    [
        # Run-outcome status semantics carried forward from the storage agent.
        "`complete`",
        "`interrupted`",
        "`errored`",
        "_finalize_errored",
        "StreamPersistenceManager",
        "RunHandle",
    ],
)
def test_chat_history_documents_status_semantics_and_quickstart(needle: str) -> None:
    """
    GIVEN the new chat-history page.

    WHEN it is read.

    THEN it documents the run-outcome status semantics and the
    ``StreamPersistenceManager`` / ``RunHandle`` quick-start surface.
    """
    chat_history = (_DOCS / "src" / "application_utils" / "chat_history.md").read_text()

    assert needle in chat_history
