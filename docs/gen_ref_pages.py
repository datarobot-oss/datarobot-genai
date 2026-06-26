"""Generate API reference pages from the datarobot_genai package source tree.

Runs at mkdocs build time via mkdocs-gen-files.  For every public .py module
found under src/datarobot_genai/ it writes a matching Markdown page containing
a single mkdocstrings ::: directive, and accumulates a SUMMARY.md consumed by
mkdocs-literate-nav to build the nav automatically.
"""

import sys
from pathlib import Path

import mkdocs_gen_files

# Make sure the library src is importable by griffe/mkdocstrings at build time
SRC_ROOT = Path(__file__).parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PKG_ROOT = SRC_ROOT / "datarobot_genai"
# All generated .md files live under api/ (relative to docs_dir/src)
API_DIR = Path("api")


def _is_private(path: Path) -> bool:
    return any(part.startswith("_") for part in path.parts)


nav = mkdocs_gen_files.Nav()

for py_file in sorted(PKG_ROOT.rglob("*.py")):
    if _is_private(py_file.relative_to(PKG_ROOT)):
        continue

    # e.g. datarobot_genai/core/config.py -> datarobot_genai.core.config
    module_path = py_file.relative_to(SRC_ROOT).with_suffix("")
    module_name = ".".join(module_path.parts)

    # Relative path inside the package, e.g. core/config.py
    rel = py_file.relative_to(PKG_ROOT)
    parts = list(rel.with_suffix("").parts)

    # __init__.py -> use parent as the page, skip bare top-level __init__
    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            continue
        doc_rel = Path(*parts, "index.md")    # e.g. core/index.md
    else:
        doc_rel = rel.with_suffix(".md")      # e.g. core/config.md

    # Full path from docs root: api/core/config.md
    full_doc_path = API_DIR / doc_rel

    # Nav value must be relative to the SUMMARY.md location (api/),
    # so use doc_rel directly (e.g. "core/config.md"), not the full path.
    nav[tuple(parts)] = doc_rel.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fh:
        fh.write(f"# `{module_name}`\n\n")
        fh.write(f"::: {module_name}\n")

    mkdocs_gen_files.set_edit_path(
        full_doc_path,
        # edit_uri is "edit/main/docs/src/"; traverse back to repo root first
        Path("../../") / py_file.relative_to(SRC_ROOT.parent),
    )

with mkdocs_gen_files.open(API_DIR / "SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

