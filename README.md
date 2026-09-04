<p align="center">
  <a href="https://github.com/datarobot-oss/datarobot-genai">
    <img src="docs/src/img/datarobot_logo.avif" width="600px" alt="DataRobot Logo"/>
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

A toolkit for building agents on DataRobot.

- **Unified LLM layer (DataRobot-compatible)**&mdash;one **`get_llm()`** entry point per integration (**LangGraph**, **LlamaIndex**, **CrewAI**, **NAT**), all backed by the same **LiteLLM**-based routing to the **DataRobot LLM Gateway**, **LLM deployments**, **NIM**, or external providers.
- **Library of agentic tools and DataRobot-compatible MCP server**&mdash;use `drtools` and `drmcp` to give agents first-class capabilities to interact with the world.
- **AG-UI integration**&mdash;agents expose a standard **AG-UI** event stream (`RunAgentInput` in, lifecycle + text + tool-call events out), so UIs and the DataRobot platform render runs consistently without bespoke adapters per framework.
- **Multi-agent systems out of the box**&mdash;first-class patterns for **planner/writer crews**, **LangGraph** multi-node graphs, and **LlamaIndex** `AgentWorkflow` handoffs; wrap them with one helper and keep the same streaming contract.
- **Orchestration**&mdash;build agents from universal pieces in the low-code `workflow.yaml` interface. Combine and reuse LLMs, tools, agents, and evaluators. The design stays compatible with and draws inspiration from [NeMo Agentic Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit).
- **Serving and evaluating with DRAgent**&mdash;run a front-end server to plug an agent into a real-world application. DRAgent supports distributed tracing, generation and evaluation endpoints, async generations, and two-way communication over WebSockets.

# Table of contents

- [Use](#use)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Credentials](#credentials)
  - [Standalone end-to-end examples](#standalone-end-to-end-examples)
  - [In-depth documentation](#in-depth-documentation)
- [Develop](#develop)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)
- [Publishing](#publishing)
- [Contributing and support](#contributing-and-support)
- [Links](#links)
- [License](#license)

# Use

## Prerequisites

The following requirements apply before installing or running samples:

| Requirement | Notes |
|---|---|
| Python 3.11–3.13 | Required for all installs |
| DataRobot account | Required for gateway, deployments, MCP, and platform features |
| Agent framework extra | One of `crewai`, `langgraph`, or `llamaindex`; use `dragent` for NAT/DRAgent |

For local development, also install [`uv`](https://docs.astral.sh/uv/), [Task](https://taskfile.dev/), and [pre-commit](https://pre-commit.com/).

## Installation

Requires Python 3.11–3.13. Install the extra that matches the target framework:
```bash
pip install "datarobot-genai[crewai]"
pip install "datarobot-genai[langgraph]"
pip install "datarobot-genai[llamaindex]"
pip install "datarobot-genai[nat]"
```

Optional extras:

* `datarobot-genai[dragent]`&mdash;serve and orchestrate agents with `DRAgent`.
* `datarobot-genai[drtools]`&mdash;use the standard library of agentic tools DataRobot provides.
* `datarobot-genai[drmcpbase]`&mdash;Base class to derive FastMCP servers.
* `datarobot-genai[drmcputils]`&mdash;shared MCP utilities consumed by `drmcpbase` and `drtools`.
* `datarobot-genai[drmcp]`&mdash;host a custom MCP server in DataRobot (includes `drmcpbase`, `drtools`, and template-server dependencies).
* `datarobot-genai[eval]`&mdash;agent evaluation utilities built on the NeMo Evaluator launcher.

## Credentials

A DataRobot account is required for DataRobot-backed features. Export these environment variables:

```bash
# Set the DataRobot API token (replace the placeholder).
export DATAROBOT_API_TOKEN=YOUR_DATAROBOT_API_TOKEN
export DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
```

## Standalone end-to-end examples

Follow [quickstart.ipynb](e2e-tests/examples/quickstart.ipynb) to walk through setting up a LangGraph agent with DataRobot:

* LLM Gateway
* `drtools`
* Prompt Management
* Conversion to DataRobot agent format
* Running the agent with an AG-UI interface.

## In-depth documentation

See [docs/README.md](docs/README.md) for guides on every framework and feature in `datarobot-genai`. The published site is at [datarobot-oss.github.io/datarobot-genai](https://datarobot-oss.github.io/datarobot-genai/).

# Develop

Requires Python 3.11–3.13, uv, Task CLI, and pre-commit.
```bash
uv sync --all-extras --dev
pre-commit install
task test
```

### Semantic versioning

Bump the patch version and add an entry to `CHANGELOG.md` for each library change.
Bump the minor version for backward-incompatible changes.

### TestPyPI

Comment `/build` on a PR to build and publish a dev version of the package to TestPyPI.

## Excluded upstream dependencies

Several transitive dependencies pulled in by upstream packages are not used by this library at runtime.
These are explicitly excluded via `[tool.uv] exclude-dependencies` in `pyproject.toml` to reduce install size and CVE surface area.

| Package | Pulled in by | Reason excluded |
|---------|-------------|-----------------|
| `build` | `crewai` | Python build system; runtime unnecessary |
| `flask` | `nvidia-nat 1.6.0` | Web framework; not used |
| `kubernetes` | `crewai-tools` | K8s client; not used |
| `lancedb` | `crewai` | Optional vector DB backend; not used |
| `langchain-milvus` | `nvidia-nat-langchain` | Milvus vector DB adapter; not used |
| `leptonai` | `nemo-evaluator-launcher` | Lepton AI cloud backend; not used, pins `httpx==0.27.2` which conflicts with the `auth` extra |
| `llama-index-cli` | `llama-index` | CLI tool; not needed at runtime |
| `openpyxl` | `crewai-tools` | Excel parser; not used |
| `pymilvus` | `langchain-milvus` | Milvus client; not used |
| `python-docx` | `crewai-tools` | Word doc parser; not used |
| `pytube` | `crewai-tools` | YouTube downloader; not used |
| `stagehand` | `crewai-tools` | Playwright web automation; not used |
| `uv` | `crewai` | Package manager bundled as a runtime dep; not needed |
| `youtube-transcript-api` | `crewai-tools` | YouTube transcripts; not used |

# Troubleshooting

Common setup and runtime issues and where to find answers:

| Symptom | Where to look |
|---|---|
| LLM auth or routing errors | [docs/src/llm.md](docs/src/llm.md) — verify `DATAROBOT_API_TOKEN` and gateway flags |
| MCP tools missing at runtime | Framework MCP guides under [docs/README.md](docs/README.md) — merge injected tools with local tools |
| DRAgent tracing empty | [docs/src/dragent/tracing.md](docs/src/dragent/tracing.md) — OTLP env vars and `instrument()` in `register.py` |
| E2E test failures locally | [e2e-tests/README.md](e2e-tests/README.md) |

# Next steps

After installation, continue with these resources:

- Pick a framework guide from [docs/README.md](docs/README.md) and start from the matching sample under `e2e-tests/dragent/`.
- Read [DRAgent CLI](docs/src/dragent/README.md) for `serve`, `run`, and `query`.
- Browse the [API reference](https://datarobot-oss.github.io/datarobot-genai/api/) on the published docs site.

# Publishing

Release and distribution options for the package.

- **Same-repo PRs**&mdash;comment `/build` on a PR to publish dev builds to TestPyPI (`.devN`).
- **Merge to `main`**&mdash;the release flow creates tag `v{version}` and publishes to PyPI automatically.
- **Version tags**&mdash;pushing a `v*` tag also triggers PyPI publish.
- **Local release**&mdash;optionally run `task release:tag-and-push` to create and push `v{version}` locally.

# Contributing and support

See [CONTRIBUTING.md](CONTRIBUTING.md) for issue guidelines, changelog expectations, and versioning. Report bugs via [GitHub Issues](https://github.com/datarobot-oss/datarobot-genai/issues). For security issues, email oss-community-management@datarobot.com.

# Links

External references for the package and source repository:

- [Repository](https://github.com/datarobot-oss/datarobot-genai)&mdash;source and issues.
- [PyPI](https://pypi.org/project/datarobot-genai/)&mdash;released packages.
- [TestPyPI](https://test.pypi.org/project/datarobot-genai/)&mdash;dev builds.

# License

Apache-2.0&mdash;see [LICENSE](LICENSE).
