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

- **AG-UI integration**&mdash;agents expose a standard **AG-UI** event stream (`RunAgentInput` in, lifecycle + text + tool-call events out), so UIs and the DataRobot platform can render runs consistently without bespoke adapters per framework.
- **Multi-agent systems out of the box**&mdash;first-class patterns for **planner/writer crews**, **LangGraph** multi-node graphs, and **LlamaIndex** `AgentWorkflow` handoffs; wrap them with one helper and keep the same streaming contract.
- **Unified LLM layer (DataRobot-compatible)**&mdash;one **`get_llm()`** entry point per integration (**LangGraph**, **LlamaIndex**, **CrewAI**, **NAT**), all backed by the same **LiteLLM**-based routing to the **DataRobot LLM Gateway**, **LLM deployments**, **NIM**, or external providers&mdash;driven by the same environment and `Config`, so every component speaks to DataRobot consistently.
- Utilities for common GenAI workflows.
- **Orchestration**&mdash;build agents from universal pieces in the low-code `workflow.yaml` interface. Combine and reuse LLMs, tools, agents, and evaluators. The design is compatible with and inspired by [NeMo Agentic Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit).
- **Evaluating and serving with DRAgent**&mdash;use a front-end server to plug your agent into a real-world application. It supports distributed tracing, generation and evaluation endpoints, async generations, and two-way communication over WebSockets.

User-facing walkthrough: [docs/README.md](docs/README.md).

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
| `uv` | build tooling | Not a runtime dependency. |
| `langchain-milvus` | langchain ecosystem | Unused vector store integration. |
| `pymilvus` | langchain-milvus | Transitive dependency of langchain-milvus. |
| `flask` | nvidia-nat-core 1.6.0 | Only used in NAT examples, not core library code ([ref](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/main/packages/nvidia_nat_core/pyproject.toml#L66)). |

## Development
Prerequisites: Python 3.11–3.14, uv, Task CLI, pre-commit.
```bash
uv sync --all-extras --dev
pre-commit install
task test
```

### Semantic versioning
Every change requires a patch version change and an entry in `CHANGELOG.md`.
Changes that break backward compatibility require a minor version bump.

### TestPyPI
Comment `/build` on a PR to build and publish a dev version of the package to TestPyPI.

## Publishing

- **Same-repo PRs**&mdash;comment `/build` to publish dev builds to TestPyPI (`.devN`).
- **Merge to `main`**&mdash;creates tag `v{version}` and publishes to PyPI automatically.
- **Version tags**&mdash;pushing a `v*` tag also triggers PyPI publish.
- **Local release**&mdash;optional `task release:tag-and-push` creates and pushes `v{version}` locally.

## Links

- [Repository](https://github.com/datarobot-oss/datarobot-genai)&mdash;source and issues.
- [PyPI](https://pypi.org/project/datarobot-genai/)&mdash;released packages.
- [TestPyPI](https://test.pypi.org/project/datarobot-genai/)&mdash;dev builds.

## License

Apache-2.0&mdash;see [LICENSE](LICENSE).
