# E2E test matrix

The repository supports several agent frameworks, and each framework can use different LLM options. Test cases are described in YAML files under `cases/` (e.g. [cases/pr-tests.yaml](cases/pr-tests.yaml)) — each case declares an `env:` block, a `tests:` list, and a `matrix:` of dimensions (always including `AGENT`) that gets expanded into one job per combination.

The same case files drive both:

- The GitHub Actions `e2e` job ([.github/workflows/e2e.yml](../.github/workflows/e2e.yml)), via the `prepare-matrix` step.
- Local execution via [scripts/run_local.py](scripts/run_local.py).

## Environment

Credentials/endpoints shared by every case (DataRobot platform, MCP deployment, Azure for the external-LLM case, etc.) live in `e2e-tests/.env*` and are loaded automatically by the `dotenv:` chain in [Taskfile.yaml](Taskfile.yaml). Copy [.env.sample](.env.sample) and fill in the blanks. LLM-specific knobs (`LLM`, `LLM_DEFAULT_MODEL`, `WORKFLOW_FILE`, `USE_DATAROBOT_LLM_GATEWAY`) are NOT placed in `.env` — they come from the case YAML and the matrix combination selected for the run.

The `task cases-*` commands below all run through Task, so they pick up the dotenv chain automatically. The runner scripts (`scripts/cases.py`, `scripts/run_local.py`) deliberately do not load `.env` themselves.

### Dependencies

Some tests require additional resources deployed in DataRobot. They are injected as environment variables. Make sure to include said variables before running a specific test case.

```
# Necessary for all tests using an MCP server
MCP_DEPLOYMENT_ID=

# Necessary for configurations using DataRobot Deployment as LLM
LLM_DEPLOYMENT_ID=

# Necessary for OOTB custom_metric:
OOTB_CUSTOM_METRIC_ENDPOINT=

# Necessary for configurations using the Emotion Classifier custom model:
EMOTION_CLASSIFIER_DEPLOYMENT_ID=
```

## Inspect what would run

The case file is the first arg after `--` and is required. A bare file name is resolved against `e2e-tests/cases/`; pass an explicit path (e.g. `./tmp.yaml`) to point elsewhere.

```shell
task cases -- pr-tests.yaml                                    # list every combination
task cases -- pr-tests.yaml --case primary-test                # filter to one case
task cases -- pr-tests.yaml --agents nat,langgraph             # agent allowlist
task cases-matrix -- pr-tests.yaml --case primary-test         # emit GH Actions matrix JSON
```

## Run locally

`task cases-run` does the full lifecycle sequentially for every matched combination: `uv sync --group dragent-<agent>`, start the agent server, wait on `/health`, run pytest, stop the server.

```shell
# Run a whole case across all its agents:
task cases-run -- pr-tests.yaml --case primary-test

# Run one specific combination (KEY=VAL overrides pin matrix dims):
task cases-run -- pr-tests.yaml --case primary-test AGENT=nat

# Iterate fast: skip dependency install and reuse an already-running agent
# (started in another shell via `task run-dragent`):
task cases-run -- pr-tests.yaml --case primary-test AGENT=nat --no-install --no-server

# Dry-run: print the env exports + pytest command for each combo and exit:
task cases-print -- pr-tests.yaml --case extended-llmgw \
    AGENT=crewai LLM_DEFAULT_MODEL=datarobot/azure/gpt-5-2-2025-12-11

# Run the entire file and don't bail on first failure:
task cases-run -- pr-tests.yaml --keep-going
```

Combinations execute one at a time on port 8080. Use `--no-server` if you have an agent already running; the runner will skip the start/stop dance and just point pytest at the existing process.

## Run the agent and tests by hand

The original two-shell flow still works when you want to watch agent logs interactively:

```shell
# Shell 1: start the agent (load LLM-specific env from a case file by hand or
# set the variables directly).
AGENT=llamaindex LLM=external WORKFLOW_FILE=workflow.yaml task run-dragent

# Shell 2: run tests against it.
AGENT=llamaindex LLM=external task test-dragent
```

Tests vary by agent and LLM context; some cases are skipped or interpreted differently depending on configuration.

## DataRobot Memory Service

[cases/memory.yaml](cases/memory.yaml) runs [dragent_tests/test_memory.py](dragent_tests/test_memory.py) against each dragent agent framework wrapped with `streaming_memory_agent` and `dr_mem0_memory` (per-agent [workflow-memory.yaml](dragent/nat/workflow-memory.yaml) overlays). The case sets `E2E_PROVISION_MEMORY_SPACE=true`, which creates an ephemeral `MemorySpace` via the DataRobot SDK before the dragent server starts and deletes it afterward.

```shell
# Full lifecycle (provision MemorySpace → start dragent → pytest → cleanup):
task cases-run -- memory.yaml

# Or trigger the same matrix in CI:
task cases-ci -- memory.yaml
```

When iterating with `--no-server`, export `AGENT_MEMORY_SPACE_ID` yourself before starting dragent (the runner skips provisioning in that mode unless the variable is already set).

## Trigger a custom matrix in CI

The `E2E Tests` workflow accepts `workflow_dispatch` with a `case_file` input (defaults to `pr-tests.yaml`). Pick any file under `e2e-tests/cases/` to fire that whole matrix without changeset-based filtering.
