# datarobot-genai
Generic helpers for GenAI.

## Features
- Utilities for common GenAI workflows
- Integrations: CrewAI, LangGraph, LlamaIndex, NAT, MCP

## Installation
- Requires Python 3.10–3.12.
- Install:
```bash
pip install --upgrade pip
pip install "datarobot-genai"
```
- Optional extras:
```bash
pip install "datarobot-genai[crewai]"
pip install "datarobot-genai[langgraph]"
pip install "datarobot-genai[llamaindex]"
# Multiple extras
pip install "datarobot-genai[crewai,langgraph,llamaindex]"
```
  Available extras include: `crewai`, `langgraph`, `llamaindex`, `nat`, `drmcp`, `pydanticai`.

## Development
Prerequisites: Python 3.10–3.12, uv, Task CLI, pre-commit.
```bash
uv sync --all-extras --dev
pre-commit install
task test
```

## Publishing
- PRs (same-repo): dev builds are auto-published to TestPyPI (`.devN`).
- Merge to `main`: tags `v{version}` and publishes to PyPI automatically.
- Pushing a `v*` tag also triggers PyPI publish.
- Optional: `task release:tag-and-push` creates and pushes `v{version}` locally.

## Links
- Home: https://github.com/datarobot-oss/datarobot-genai
- PyPI: https://pypi.org/project/datarobot-genai/
- TestPyPI: https://test.pypi.org/project/datarobot-genai/

## License
Apache-2.0
