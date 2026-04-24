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

A toolkit for building agents on DataRobot.

- **Unified LLM layer (DataRobot-compatible)**&mdash;you use one **`get_llm()`** entry point per integration (**LangGraph**, **LlamaIndex**, **CrewAI**, **NAT**), all backed by the same **LiteLLM**-based routing to the **DataRobot LLM Gateway**, **LLM deployments**, **NIM**, or external providers.
- **Library of agentic tools and DataRobot-compatible MCP server**&mdash;use `drtools` and `drmcp` to give your agent first-class capabilities to interact with the world.
- **AG-UI integration**&mdash;your agents expose a standard **AG-UI** event stream (`RunAgentInput` in, lifecycle + text + tool-call events out), so your UIs and the DataRobot platform render runs consistently without bespoke adapters per framework.
- **Multi-agent systems out of the box**&mdash;you get first-class patterns for **planner/writer crews**, **LangGraph** multi-node graphs, and **LlamaIndex** `AgentWorkflow` handoffs; wrap them with one helper and keep the same streaming contract.
- **Orchestration**&mdash;you build agents from universal pieces in the low-code `workflow.yaml` interface. Combine and reuse LLMs, tools, agents, and evaluators. The design stays compatible with and draws inspiration from [NeMo Agentic Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit).
- **Serving and evaluating with DRAgent**&mdash;you run a front-end server to plug your agent into a real-world application. DRAgent supports distributed tracing, generation and evaluation endpoints, async generations, and two-way communication over WebSockets.

# Use

## Installation
- You need Python 3.11–3.13.
- Install the extra that matches the framework you use:
```bash
pip install "datarobot-genai[crewai]"
pip install "datarobot-genai[langgraph]"
pip install "datarobot-genai[llamaindex]"
pip install "datarobot-genai[nat]"
```

You can also install:
* `datarobot-genai[dragent]`&mdash;serve and orchestrate your agent with `DRAgent`.
* `datarobot-genai[drtools]`&mdash;use the standard library of agentic tools DataRobot provides.
* `datarobot-genai[drmcp]`&mdash;host a custom MCP server in DataRobot.

## Credentials
You need a DataRobot account to use DataRobot-backed features. Export these environment variables:

```bash
# Set your DataRobot API token (replace the placeholder).
export DATAROBOT_API_TOKEN=YOUR_DATAROBOT_API_TOKEN
export DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
```

## Standalone end-to-end examples
Follow [quickstart.ipynb](e2e-tests/examples/quickstart.ipynb) to walk through an experience of setting a LangGraph agent with DataRobot:
* LLM Gateway
* `drtools`
* Prompt Management
* Conversion to DataRobot agent format
* Running the agent with an AG-UI interface.

## In-depth documentation
See [docs/README.md](docs/README.md) for guides on every framework and feature in `datarobot-genai`.

# Develop
You need Python 3.11–3.13, uv, Task CLI, and pre-commit.
```bash
uv sync --all-extras --dev
pre-commit install
task test
```

### Semantic versioning
When you change the library, bump the patch version and add an entry to `CHANGELOG.md`.
When you introduce a backward-incompatible change, bump the minor version.

### TestPyPI
Comment `/build` on your PR to build and publish a dev version of the package to TestPyPI.

## Publishing

- **Same-repo PRs**&mdash;comment `/build` on your PR to publish dev builds to TestPyPI (`.devN`).
- **Merge to `main`**&mdash;the release flow creates tag `v{version}` and publishes to PyPI automatically.
- **Version tags**&mdash;when you push a `v*` tag, PyPI publish runs as well.
- **Local release**&mdash;optionally run `task release:tag-and-push` to create and push `v{version}` from your machine.

## Links

- [Repository](https://github.com/datarobot-oss/datarobot-genai)&mdash;source and issues.
- [PyPI](https://pypi.org/project/datarobot-genai/)&mdash;released packages.
- [TestPyPI](https://test.pypi.org/project/datarobot-genai/)&mdash;dev builds.

## License

Apache-2.0&mdash;see [LICENSE](LICENSE).
