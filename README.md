<p align="center">
  <a href="https://github.com/datarobot-oss/datarobot-genai">
    <img src="docs/img/datarobot_logo.avif" width="600px" alt="DataRobot Logo"/>
  </a>
</p>
<h3 align="center">DataRobot GenAI Library</h3>

<p align="center">
  <a href="https://www.datarobot.com/">Homepage</a>
  ·
  <a href="https://pypi.org/project/datarobot-genai/">PyPI</a>
  ·
  <a href="https://docs.datarobot.com/en/docs/get-started/troubleshooting/general-help.html">Support</a>
</p>

<p align="center">
  <a href="/LICENSE">
    <img src="https://img.shields.io/github/license/datarobot-oss/datarobot-genai" alt="License">
  </a>
  <a href="https://pypi.org/project/datarobot-genai/">
    <img src="https://img.shields.io/pypi/v/datarobot-genai" alt="PyPI version">
  </a>
</p>


## Features
- Utilities for common GenAI workflows
- Integrations: CrewAI, LangGraph, LlamaIndex, NAT, MCP

## Installation
- Requires Python 3.11–3.13.
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

## Excluded Dependencies

Some transitive dependencies are excluded via `exclude-dependencies` in `pyproject.toml` because they are unused by this project. Do not re-add them.

| Package | Pulled in by | Reason for exclusion |
|---|---|---|
| `uv` | build tooling | Not a runtime dependency |
| `langchain-milvus` | langchain ecosystem | Unused vector store integration |
| `pymilvus` | langchain-milvus | Transitive dep of langchain-milvus |
| `flask` | nvidia-nat-core 1.6.0 | Only used in NAT examples, not core library code ([ref](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/main/packages/nvidia_nat_core/pyproject.toml#L66)) |

## Development
Prerequisites: Python 3.11–3.14, uv, Task CLI, pre-commit.
```bash
uv sync --all-extras --dev
pre-commit install
task test
```

### Semantic versioning
Every change requires a patch version change. It also requires an entry to CHANGELOG.md.
Changes breaking backward compatibility requires a minor version change.

### Test pypi
To build and publish a dev version of a package, comment `/build` on a PR.

## Publishing
- PRs (same-repo): comment `/build` to publish dev builds to TestPyPI (`.devN`).
- Merge to `main`: tags `v{version}` and publishes to PyPI automatically.
- Pushing a `v*` tag also triggers PyPI publish.
- Optional: `task release:tag-and-push` creates and pushes `v{version}` locally.

## Links
- Home: https://github.com/datarobot-oss/datarobot-genai
- PyPI: https://pypi.org/project/datarobot-genai/
- TestPyPI: https://test.pypi.org/project/datarobot-genai/

## License
Apache-2.0
