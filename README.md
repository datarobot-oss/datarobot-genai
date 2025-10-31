# datarobot-genai
Repository for the DataRobot GenAI Library

## Releases (tag-driven)

- Set the release version in `pyproject.toml`.
- Push a tag `vX.Y.Z` (e.g., `v0.3.0`).
- CI validates the tag matches `project.version`, builds, and publishes.

### Where releases go
- PRs: CI publishes dev builds to TestPyPI (`X.Y.Z.dev<run>`), for validation.
- Tags: CI publishes to PyPI when a `vX.Y.Z` tag is pushed.

### Install from TestPyPI (quick check)
```bash
VERSION=X.Y.Z.dev123   # replace with the run number shown in the PR workflow
pip install --upgrade pip
pip install -i https://test.pypi.org/simple/ datarobot-genai=="$VERSION"
python -c "import datarobot_genai as drg; print(drg.__version__)"
```

### Secrets
- `TEST_PYPI_API_TOKEN` (username `__token__`)
- `PYPI_API_TOKEN` (username `__token__`)

## Local development (quick start)

```bash
# install dev dependencies (uses uv)
uv sync --all-extras --dev

# activate virtualenv if not auto-activated
source .venv/bin/activate

# enable git hooks
pre-commit install

# run unit tests
make test

# run acceptance tests
make test-acceptance
```

Python requirement: >= 3.11,< 3.13

## Optional dependencies (extras)

Install specific integrations only when needed:

```bash
# CrewAI
pip install "datarobot-genai[crewai]"

# LangGraph
pip install "datarobot-genai[langgraph]"

# LlamaIndex
pip install "datarobot-genai[llamaindex]"

# NVIDIA NAT
pip install "datarobot-genai[nat]"

# PydanticAI
pip install "datarobot-genai[pydanticai]"

# DataRobot MCP
pip install "datarobot-genai[drmcp]"


# Combine extras
pip install "datarobot-genai[crewai,nat]"
pip install "datarobot-genai[crewai,langgraph,llamaindex,nat,drmcp]"
```
