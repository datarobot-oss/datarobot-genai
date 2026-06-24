# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.19.4
- Added x-datarobot-external-access-token header support for okta integration

## 0.19.3
- Fixed OTEL traces endpoint resolution.

## 0.19.2
- Updated dependency constraints to fix CVEs.

## 0.19.1
- `crewai`: an empty `agent_role` streaming chunk (CrewAI's `[] Working on task:` task boundary) no longer opens an AG-UI step that is never closed → fixes the `RUN_FINISHED while steps are still active` verifier error.
- `e2e-tests` (crewai): lower `max_iter` to 5 in the dragent crewai workflow config (was 20) so a runaway ReAct loop is forced to a final answer instead of stalling the stream past the 60s httpx read timeout (the pytest global timeout is 300s).
- `crewai`: added a logging event listener that logs the agent/task/tool lifecycle in real time (tool calls with args, results, and attempt count at INFO; failures at WARNING; reasoning at DEBUG), so a dragent run shows what the agent is doing instead of only a stream of LiteLLM calls.

## 0.19.0
- LLM clients and LLM providers moved out from subpackage `nat` to `dragent` and respective framework folders.

## 0.18.14
- `drmcputils/panels`: **fix** panel Files-API tags used `-` (e.g. `dr-panel-source:…`), which the DataRobot Files API rejects with a 422 (`Tag cannot contain '-'`), so every panel create/list failed against the real backend (only ever exercised against an in-memory store in tests). Tags now use `_` (`dr_panel`, `dr_panel_payload`, `dr_panel_source:…`, `dr_panel_type:…`). Verified end to end against staging.
- `drmcpbase/panels` + `drmcp`: panels are now served by DRMCP — exposed as read-only MCP resources (`panels://{source}`, `panels://{source}/{id}`, `panels://{source}/{id}/content`) with the panel tool domain enabled via `enable_panels_tools`; adds `inspect_panel` and `view_json_panel` review tools.
- `drmcpbase/panels`: the panel resource handlers live in `drmcpbase` (the shared resources layer) and are registered onto a server's FastMCP instance via `register_panel_resources(mcp)`, so both DRMCP and global-mcp can reuse them without reaching either server's mcp singleton.
- `drtools/panels` + `drtools/sandbox`: **fix** the entitlement the panel/sandbox gate evaluates — it was `MCP_SANDBOX`, which the platform rejects as an invalid entitlement name (422), so the fail-closed gate denied every user. Corrected to the registered `ENABLE_MCP_SANDBOX`.
- `drtools/sandbox`: **fix** the workload-api request to match the current API — submit to `workloads/` (not `console/workloads/`), declare the service artifact `type`/named container groups + a primary-container `port`, and carry resources as `runtime.containerGroups[].containers[].resourceAllocation` instead of the now-rejected per-container `resourceRequest` / `runtime.replicaCount`. Verified end to end against staging (workload now schedules).

## 0.18.13
- drmcp - removed the api key enforcment on startup, we get it through the headers and if not provided we will skip the startup functionality, still required for deploy through pulumi.
- drtools - Removed the local_path from file_write.

## 0.18.12
- `eval`: moved the evaluation CLI into the package so the thin `run.py`/`generate.py`/`summarize.py` component wrappers just call into it.
  - **CLI** (`eval/cli.py`): `run_main` (validate → run BYOB → normalize, with `--dry-run`), `generate_main` (synthetic generation or CSV→JSON `--convert`), and `summarize_main`. Each takes an optional `argv` and `repo_root` (defaults to cwd).
  - **Runner** (`eval/eval.py`): new `EvalRunner` orchestrating a single batch run — input validation, judge preflight, status writes, BYOB execution, and output normalization to the fixed `output/eval_status.json` / `output/eval_results.json` paths.
  - **Generator** (`eval/generator.py`): `generate()` now takes an optional `benchmark_name` that tailors good/bad case guidance and enforces benchmark-required fields (e.g. `canary`, `context`, `constraints`); without one it uses a generic, non-safety-biased context.

## 0.18.11
- Fixed flaky NAT test. Writer being set as `return_direct` prevented reasoning models output to be ignored

## 0.18.10
Added to `e2e-tests` for moderations:
- ootb custom_metric guard
- Custom model guard
- All NeMo Evaluator guards

## 0.18.9
- `core`/`drtools`: upgraded `pyarrow` from `21.0.0` to `>=23.0.1,<24.0.0` to fix CVE-2026-25087 (HIGH). `pyarrow` is not imported directly; it backs the polars→pandas conversion in `drtools/predictive` and other Arrow boundaries. The full unit suite and a polars/pandas/pyarrow round-trip pass on 23.0.1.

## 0.18.8
- `drtools/files_api`: local-disk upload — `file_upload` streams files, directory trees, or globs from the server's filesystem into a catalog path (no inline size cap; batched `put`), and `file_write` accepts an optional `local_path` for small files still bounded by `MAX_INLINE_SIZE`. Both require `FILES_API_LOCAL_ALLOWED_ROOTS` (comma-separated allowlist; empty disables local access). Backed by new `DataRobotFileSystemStore.upload`.
- `drmcp`: `MCPServerConfig` and `MCPToolConfig` now extend `DataRobotAppFrameworkBaseSettings`, consolidating env, `.env`, file secrets, pulumi config, and `MLOPS_RUNTIME_PARAM_` resolution; removed `drtools.core.config_utils` and its re-exports from `datarobot_genai.drmcp`, config attributes renamed to `mcp_server_tool_registration_allow_empty_schema`, `mcp_server_tool_registration_duplicate_behavior`, and `mcp_server_prompt_registration_duplicate_behavior`.

## 0.18.7
- `drtools/files_api`: async import tools for large or remote files — `file_import` (background ingest from a URL or data source, returns a `status_id`) and `file_get_status` (single non-blocking status fetch with optional `target_status` / `target_reached`, raises on terminal failure). Backed by new store methods on `DataRobotFileSystemStore` (`import_from_url`, `import_from_data_source`, `get_status`).

## 0.18.6
- `drtools/files_api`: write and structural tools for the DataRobot Files API filesystem — `file_write` (inline UTF-8/base64 content, `overwrite`/`create` modes, capped at `MAX_INLINE_SIZE`) and `file_manage` (consolidated `create_dir` | `delete` | `copy` | `move` | `clone` lifecycle actions). Backed by new store methods on `DataRobotFileSystemStore` (`write`, `create_dir`, `delete`, `copy`, `move`, `clone`); shared path/content helpers live in `common_utils.py`.

## 0.18.5
- `e2e-tests`: overrided WORKFLOW_FILE when run agent inline and in CLI
- `dragent`: ensured that conventional OTLP_EXPORTER environment variables are used by default to configure exporter, and fixed OTEL context propagation in `execute_dragent_inline`

## 0.18.4
- `llama_index`: omit `temperature` when unconfigured so the model uses its own default (matching langgraph/crewai/nat). LlamaIndex's `LiteLLM` baked in `temperature=0.1`, breaking Anthropic extended thinking. Explicit values are still forwarded.
- `e2e-tests`: set `temperature: 0` on the non-reasoning dragent configs for deterministic runs; reasoning overlays unset it (`temperature: ~`) because extended thinking is incompatible with any temperature modification.

## 0.18.3
- `e2e-tests`: test cases to test different LLM scenarios with all tests: NIM, external, variety of LLMs in LLMGW.

## 0.18.2
- `drtools/files_api`: new read-only DataRobot Files API tool surface for browsing the hierarchical `dr://<catalog_id>/path` filesystem — `file_list` (ls/recursive/glob/tree with pagination), `file_info` (single file/directory metadata), `file_read` (inline UTF-8/base64 content with byte-range reads, capped at `MAX_INLINE_SIZE`), and `file_sign` (temporary signed download URLs for large files). Gated behind the new `enable_files_api_tools` config flag (`ENABLE_FILES_API_TOOLS`), disabled by default, and registered as the `files_api` tool type. Moved the `datarobot-early-access[fs]` dependency into `drmcputils` and dropped the now-redundant `datarobot` pins from `drtools`/`drmcpbase`.

## 0.18.1
- Fixing runtime error thrown during E2E tests

## 0.18.0
-  CrewAI + anthropic/claude-sonnet-4-6: tool calling issues fix

## 0.17.9
- `e2e-tests`: nemo-guardrails moderation — dropped the invalid `datarobot/` prefix from the guard `llm_gateway_model_id` (the LLM Gateway catalog keys on bare `provider/model`; the prefixed id 404'd and the "stay on topic" guard silently failed open). Added a `dragent_tests` test that asserts the guard blocks disallowed input, so a fail-open guard turns the suite red.

## 0.17.8
- `crewai`: `CrewAIAgent.invoke` now calls `crew.akickoff` instead of the deprecated `kickoff_async`.
- `crewai`: apply client-side stop-word truncation in ``LitellmStopWordLLM.acall`` so native async kickoff preserves ReAct tool-loop behavior, including inline hallucinations after ``Action Input``.
- `crewai`: emit ``LLMStreamChunkEvent`` from router ``acall`` so ``Crew.akickoff`` streaming receives text chunks.

## 0.17.7
- `drtools/workload`: consolidated the workload/artifact tool surface from 39 tools to 21 for agent ergonomics, following MCP tool-design guidance. No functionality was lost; every tool still issues a single non-blocking REST call (no client-side polling or waiting).

## 0.17.6
- `drtools/predictive`: raised the `catalog_query_datastore` datastore-preview cap (`_PREVIEW_QUERY_MAX_ROWS`) from 999 to 9999, matching the `externalDataStores/<id>/previewQuery/` route's server-side `maxRows` limit (the `DataStorePreviewQueryValidator` allows up to 10000). Lets `offset`-based paging reach deeper before requiring SQL-level paging.

## 0.17.5
- `e2e`: fixed NeMo-guardrails dragent tests crashing (SIGILL) on non-AVX-512 runners — excluded `annoy` (an sdist-only `-march=native` AVX-512 build) from the e2e resolution.

## 0.17.4
- `drtools/workload`: submit-and-poll lifecycle workflow — `workload_start` and `workload_stop` return immediately after the request is accepted (202) with `accepted` and a `note` directing the agent to poll status; `workload_stop` no longer accepts `wait_stopped` or `timeout_seconds`. **`workload_wait_for_status` is replaced by `workload_get_status`** (`workload_id`, optional `target_status`): a lightweight single status fetch returning `status`, `target_reached`, and `raw`, raising on terminal `errored` without blocking. Removed `WorkloadApiClient.wait_for_workload_status` server-side polling.

## 0.17.3
- `drtools/panels`: filter and transform Dataset panels with sandboxed code execution (`filter_panel`, `transform_panel`), saving results as derived child panels with lineage.

## 0.17.2
- `llamaindex`: ``LlamaIndexAgent.invoke`` now prepends the ``streaming_memory_agent`` memory injection (system message immediately before the latest user turn) to the processed user prompt so retrieved memory reaches the workflow.

## 0.17.1
- Added E2E test cases for moderations: OOTB and NeMo Guardrails

## 0.17.0
- **Breaking** `core/agents`: removed opt-in long-term memory from framework base agents (`BaseAgent`, LangGraph, LlamaIndex, CrewAI, and `NatAgent`). The `memory_client` constructor argument, `{memory}` prompt placeholder handling, and automatic retrieve/store hooks are gone. Use NAT `auto_memory_agent` or `streaming_memory_agent` with the `dr_mem0_memory` provider instead (see `docs/nat/memory.md`).

## 0.16.19
- `drtools/workload`: artifact replacement / rolling update.
  - **Client** (`WorkloadApiClient`): added `get_workload_replacement`, `create_workload_replacement`, `delete_workload_replacement` against `GET/POST/DELETE /api/v2/workloads/{id}/replacement`.
  - **Tools** (`replacement_tools`): `workload_replacement_get` — fetch current replacement status (candidate artifact, proton ids, config, timestamps); `workload_replacement_create` — start a rolling update by deploying a new artifact alongside the running version with optional warmup/retention config and runtime override; `workload_replacement_delete` — cancel an in-progress replacement and revert traffic to the original version.

## 0.16.18
- `nat/datarobot_moderation_middleware`: streaming moderation now attaches prescore guard metrics to `TEXT_MESSAGE_START` chunks (matching DRUM/dome first-chunk semantics) and keeps postscore metrics on moderated `TEXT_MESSAGE_CONTENT` chunks.

## 0.16.17
- `crewai`: fixed a file-descriptor leak that crashed long-running `nat dragent serve` with `[Errno 24] Too many open files`. crewai's kickoff-outputs SQLite storage leaks connections (unclosed `with sqlite3.connect(...)`); the agent now runs crews with an in-process no-op task-output handler, so no database is opened.

## 0.16.16
- `drtools/workload`: artifact builds and repositories.
  - **Client** (`WorkloadApiClient`): added `list_artifact_builds`, `trigger_artifact_build`,
    `get_artifact_build`, `get_artifact_build_logs` (text/plain response), `delete_artifact_build`
    targeting `GET/POST /artifacts/{id}/builds` and `GET/DELETE /artifacts/{id}/builds/{build_id}`;
    added `list_artifact_repositories`, `get_artifact_repository`, `delete_artifact_repository`
    targeting `GET /artifactRepositories` and `GET/DELETE /artifactRepositories/{id}`.
  - **Build Tools** (`drtools/workload/build_tools.py`): `artifact_build_list`, `artifact_build_trigger`,
    `artifact_build_get`, `artifact_build_logs`, `artifact_build_delete`.
  - **Repo Tools** (`drtools/workload/repository_tools.py`): `artifact_repository_list` (filterable by
    search/type), `artifact_repository_get`, `artifact_repository_delete`.

## 0.16.15
- `drtools/workload`: artifact core management.
  - **Client** (`WorkloadApiClient`): added `list_artifacts`, `get_artifact`, `create_artifact`,
    `put_artifact`, `patch_artifact`, `delete_artifact`, and `clone_artifact` methods targeting
    `GET/POST /artifacts/`, `GET/PUT/PATCH/DELETE /artifacts/{id}`, and
    `POST /artifacts/{id}/clone`.
  - **Tools** (`drtools/workload/artifact_tools.py`): `artifact_list` (paginated, filterable by
    status/type/repository/search), `artifact_get`, `artifact_create`, `artifact_update` (PATCH
    helper for name/description/spec), `artifact_lock` (PATCH shortcut to set status→locked),
    `artifact_clone`, and `artifact_delete`.

## 0.16.14
- Fix `ResultsSummarizer.print_summary` crash when a case has `expected_behavior: null` or `id: null`

## 0.16.13
- `dragent/frontends`: `POST /chat/completions` now reports the agent's configured LLM model (via `core/config.default_response_model`) instead of NAT's `"unknown-model"`, on the non-streaming body and streaming content chunks. The request's `model` is ignored (the agent runs its `workflow.yaml`/env-configured LLM) and need not be sent; moderation's `MODERATION_MODEL_NAME` is preserved. Known gap: NAT's terminal `finish_reason="stop"` streaming chunk still reports `"unknown-model"`.

## 0.16.12
- A2A per-user workflow keys use gateway identity headers instead of caller-supplied `context_id`; invalid auth context is rejected.

## 0.16.11
- Fix datarobot_genai.eval.benchmarks pipeline YAML reference retrievals

## 0.16.10
- `drtools/panels`: create Dataset panels from any saved datastore connection via SQL (`create_dataset_panel_from_connector`) and preview their contents (`preview_dataset_panel`).
- `drtools/predictive`: fixed `catalog_query_datastore` to use the supported `previewQuery` API route.

## 0.16.9
- `drtools/panels`: added a server-side panel store — typed panel models (Dataset, Chart, Text, Json) persisted via the Files API, with CRUD and schema-validation tools.
  - Fix import sorting (ruff I001) in drtools modules left behind by the 0.16.8 merge.

## 0.16.8
- `drmcputils/files`: added a `BlobStore` protocol with a DataRobot Files API backend for storing and retrieving blobs (`put`/`get`/`delete`/`list`), in the shared base so both tools and resources can use it.

## 0.16.7
- `drmcputils`: moved the shared, fastmcp-free base (DataRobot client, auth, credentials, errors, feature flags) here from `drtools.core`, so both the tools and MCP-server layers depend on one foundation.

## 0.16.6
- Updated CODEOWNERS to include common files like .gitignore

## 0.16.5
- `drtools/workload`: Proton inspection and OTel log tools.
  - **New tools** (`proton_tools.py`): `proton_list` (paginated list of proton instances for a workload), `proton_get` (single proton by id), `proton_status_details` (per-replica pod status — CrashLoopBackOff, OOMKilled, container readiness; returns `{status: pending}` when no update received yet), `workload_logs` (OTel log lines with level, time-window, includes/excludes, span/trace-id filters).
  - **Client additions** (`WorkloadApiClient`): `list_protons`, `get_proton`, `get_proton_status_details` (204 → `None`), `list_workload_logs` (GET `/otel/workload/{id}/logs/`).

## 0.16.4
- Add `drmcputils` subpackage for shared drtools and drmcpbase utilities e.g clients and common code

## 0.16.3
- Migrate benchmark helper classes to genai[eval] package

## 0.16.2
- Replace `anthropic` with `litellm` calls in the eval package

## 0.16.1
- `core/agents`: a prior-turn reasoning message is folded into the following assistant message's `content` as `<reasoning>…</reasoning>` text during history extraction, so chain-of-thought round-trips to the model across all agent frameworks and ingress paths (AG-UI `AssistantMessage` has no reasoning field). The text `{chat_history}` summary and the langgraph/llama_index structured converters both surface it; a reasoning turn with no following assistant turn is dropped. Consumer note: turns that carry reasoning now replay their full chain-of-thought into history, which adds tokens — tune `max_history_messages` if context budget is tight.

## 0.16.0
- `core/agents/events.py`: `events_to_messages` folds an AG-UI event stream back into `Message` objects (assistant text + its tool calls on one `AssistantMessage`, paired `ToolMessage` results, reasoning) for replay as history — the Python port of the TS client's `defaultApplyEvents` (messages slice).
- `llama_index`: tool-call events now carry `parent_message_id`, so a client folding the stream keeps a turn's text and tool calls on one assistant message.
- `dragent` e2e-tests: multi-turn conversation test (tool calls + reasoning) for langgraph/nat/llama_index; langgraph + llama_index replay structured history. The langgraph e2e agent drops `{chat_history}` and only interrupts for the interrupt/resume case.
- `core/agents`: the text `{chat_history}` summary now keeps tool calls even when the assistant turn also has text (previously dropped), so all frameworks surface prior tool steps.
- `langgraph`/`llama_index`: prior turns now replay to the model as native messages with tool calls preserved (`structured_history`), default **on** when the prompt has no `{chat_history}` (opt out with `structured_history=False`). Breaking: such agents now replay prior turns (bounded by `max_history_messages`) where before they got none.

## 0.15.127
- `drtools/workload`: settings and observability tools.
  - **New tools** (`observability_tools.py`): `workload_settings_get`, `workload_settings_update` (triggers rolling replacement via PATCH /settings), `workload_stats` (aggregated perf stats with quantile + slow-request controls), `workload_history` (artifact deployment history), `workload_events` (status-change and error events), `workload_promote` (lock running draft artifact), `workload_related` (linked artifacts and related entities).
  - **Client additions** (`WorkloadApiClient`): `get_workload_settings`, `update_workload_settings`, `get_workload_stats`, `list_workload_history`, `list_workload_events`, `promote_workload_artifact`, `get_workload_related`.

## 0.15.126
- Removed user MCP lineage feature flag logic

## 0.15.125
- `dragent/frontends/converters`: registered `convert_run_agent_input_to_chat_request_or_message` so plain `RunAgentInput` from the DRUM `NatAgent.invoke` / `streaming_memory_agent` passthrough boundary converts to NAT `ChatRequestOrMessage` for inner `per_user_tool_calling_agent` workflows.

## 0.15.124
- Moved `drmcp` dynamic tools core functionality to `drmcpbase` to be used by the global MCP

## 0.15.123
- `drtools/workload`: lifecycle tools.
  - **New tools**: `workload_create_payload` (builds and validates a create payload without an API call; supports both existing `artifactId` and inline artifact via `artifact_name`, `image_uri`, `port`, `cpu`, `memory_bytes`), `workload_create`, `workload_start`, `workload_stop` (with optional `wait_stopped` polling), `workload_delete`, `workload_update` (PATCH name/description/importance), `workload_wait_for_status`.
  - **Client additions** (`WorkloadApiClient`): `create_workload`, `start_workload`, `stop_workload`, `delete_workload`, `patch_workload`, and `wait_for_workload_status` polling method (raises `RuntimeError` on `errored`, `TimeoutError` on deadline).
  - Runtime payload uses the canonical `runtime.containerGroups[].resourceBundles` schema from the OpenAPI spec.

## 0.15.122
- `nat/helpers`: `NatAgent` (DRUM path) now strips `datarobot_moderation` middleware from the loaded `workflow.yaml` automatically. DRUM applies guardrails via `moderation_config.yaml` outside NAT; keeping the middleware in YAML for DRAgent deployments no longer requires a separate DRUM copy of the file. DRAgent entry points (`load_workflow` default, inline runner, CLI) are unchanged.

## 0.15.121
- `drmcp/core/config`: `MCPServerConfig` assembles `otel_exporter_otlp_headers` dynamically from `OTEL_ENTITY_ID` + `DATAROBOT_API_TOKEN` when the header is not explicitly set, avoiding stale API tokens baked at `pulumi up` time.
- `dragent/cli/commands`: `_bridge_pulumi_otel_env()` reads `OTEL_ENTITY_ID` from `pulumi_config.json` and assembles OTel headers with the live `DATAROBOT_API_TOKEN`.

## 0.15.120
- Test cases runner for `dragent` e2e-tests.

## 0.15.119
- `drtools/workload`: workload lifecycle tools — `workload_create_payload` (payload builder helper), `workload_create`, `workload_start`, `workload_stop` (with optional `wait_stopped` polling), `workload_delete`, `workload_update` (PATCH name/description/importance), and `workload_wait_for_status` (async polling with terminal-status detection and configurable timeout). Client gains `create_workload`, `start_workload`, `stop_workload`, `delete_workload`, `patch_workload` methods. Payload builder supports both existing-artifact (`artifactId`) and inline-artifact modes, fixed and autoscaling replica configurations, and resource bundle selection — following the source-of-truth `WorkloadRuntime.containerGroups` schema.

## 0.15.118
- `eval`: migrated third-party dependent modules from `af-component-evaluation` into `datarobot_genai.eval` — `validation` (pyyaml), `generator` (anthropic), `judge` (nemo_evaluator). `judge` fixes a cross-provider incompatibility where NeMo always sends `temperature` + `top_p` together, which Anthropic/Bedrock Claude rejects. Judge sessions are now thread-local to be safe under `parallelism > 1`. Full test coverage added; `nemo_evaluator` is stubbed in `conftest.py` so tests run without flask.

## 0.15.117
- Initialize `_dask_client` to exit cleanly and not throw error during shutdown

## 0.15.116
- `nat/datarobot_mem0_memory`: when no memory backend is configured (no `agent_memory_space_id` + `DATAROBOT_API_TOKEN`, and no `api_key` / `MEM0_API_KEY`), the provider yields an `UnconfiguredMemoryEditor` no-op instead of raising at startup. Workflows can declare `dr_mem0_memory` unconditionally and enable memory later via runtime parameters or env vars. The mutually-exclusive guardrail against setting both `agent_memory_space_id` and `api_key` is unchanged.
- `dragent/plugins/streaming_memory_agent`: passes through to the inner agent when the referenced memory backend is unconfigured (`is_memory_editor_configured` returns false), so a fixed `memory_name` in `workflow.yaml` works with or without credentials wired at deploy time.

## 0.15.115
- `drmcpbase`: added `class UserMCPProvider` to support user MCP proxy

## 0.15.114
- `eval`: migrated stdlib foundation layer from `af-component-evaluation` into `datarobot_genai.eval` — `utils`, `status`, `output`, `converter`, `dataset`, `summarize`, `runner`, and JSON schemas. Full test coverage added under `tests/eval/`. Third-party modules (`validation`, `generator`, `judge`), benchmarks subpackage, and top-level orchestrator follow in subsequent PRs.

## 0.15.113
- `nat/datarobot_mem0_memory`: renamed the `memory_space_id` config field to `agent_memory_space_id` (endpoint path follows: `{datarobot_endpoint}/memory/{agent_memory_space_id}`), and added a default factory that reads `AGENT_MEMORY_SPACE_ID` from env via `DataRobotAppFrameworkBaseSettings`. This lets a minimal `workflow.yaml` memory block target the DataRobot Memory Service when the recipe's agent runtime wires the env var, without requiring an explicit field in YAML. Error messages, docstrings, and the mutually-exclusive guardrail against `api_key` were updated to reference the new field name.

## 0.15.112
- Upgrade github actions to release 0.0.9

## 0.15.111
- Bump `datarobot-moderations` to 11.2.33 to fix a bug with `ModerationIterator`

## 0.15.110
- `nat/datarobot_mem0_memory`: emit OpenTelemetry GenAI memory spans (`update_memory`, `search_memory`, `delete_memory`) for Mem0/DataRobot Memory Service access through `DRMem0Editor`, with `gen_ai.memory.store.*`, query/result counts, and per-user scope attributes. Spans export through the same OTel SDK bootstrap used by `instrument()` in `register.py`.
- `dragent/datarobot_otelcollector`: bridge NAT intermediate-step span context into the OTel SDK so memory and framework spans share the workflow trace instead of exporting as a separate tree. Falls back to NAT `workflow_trace_id` when the exporter bridge is unavailable.
- `core/telemetry_nat_tracer`: patch the SDK `TracerProvider` installed by `bootstrap_otel_provider_for_datarobot()` so LangChain/LangGraph, HTTP client, and other auto-instrumentor spans join the active NAT workflow trace. Adds a single-active-run fallback when NAT `Context` is unavailable in framework worker threads.

## 0.15.109
- `drtools/sandbox`: added a `Sandbox` protocol and `DataRobotWorkloadSandbox` (workload-api backend) plus the `execute_code` function; credentials come from the request/config helpers (not `os.environ`), container stderr is surfaced from OTEL logs, and the security context is gated by `ENABLE_WORKLOAD_API_SECURITY_CONTEXT`.
- `drtools.core.feature_flags`: added `is_tool_feature_enabled(flag, *, evaluator)`, the shared tool-gating policy reused by `drmcp` and global-mcp registries.

## 0.15.108
- `drtools` Atlassian (Jira/Confluence): added `AtlassianAuth` with OAuth Bearer (HTTP) and API token Basic auth (config). Config fields: `ATLASSIAN_API_TOKEN`, optional `ATLASSIAN_EMAIL` and `ATLASSIAN_SITE_URL` for Basic auth (cloud ID from `/_edge/tenant_info`); token alone is treated as a static OAuth Bearer token.
- Docs: added `docs/drtools/auth-atlassian.md` for Atlassian config auth.
- Tests: added Atlassian API token Basic auth tests for Jira/Confluence clients.

## 0.15.107
- `drmcpbase`: MCP catalog transforms live in `datarobot_genai.drmcpbase.fastmcp_transforms` (`DataRobotMCPCatalogTransform`, `register_mcp_catalog_transform`). Removed `conditional_code_mode` and `mcp_catalog_transform` modules.
- `drmcpbase`: in tools mode (`x-datarobot-mcp-mode=tools` or unset), optional header `x-datarobot-mcp-tools` (comma-separated tool names, exact match) filters `tools/list` and `tools/call` resolution; header names are matched case-insensitively; unknown tool names are logged and skipped.

## 0.15.106
- `drtools`: added `auth_resolution_strategy` on `ToolsAuthCredentials` (`AUTH_RESOLUTION_STRATEGY`: `http` or `config`, default `http`). `AuthResolutionStrategy` is a `StrEnum` so env values parse correctly.
- `drtools.core.auth`: runtime adapters inject per-request data via `set_request_headers` / `set_auth_context`; resolvers `resolve_datarobot_token`, `resolve_secret`, and `get_oauth_access_token_with_header_fallback` honor `auth_resolution_strategy`. Removed legacy helpers `set_request_headers_for_context`, `resolve_token_from_headers`, and `get_api_key_from_headers`. `get_datarobot_access_token(headers_auth_only=False)` falls back to the server `DATAROBOT_API_TOKEN` when strategy is `http` and no request headers are present (dynamic tool/prompt registration at startup).
- `drmcpbase`: added FastMCP middleware (`read_http_headers`, `OAuthMiddleWare`, `RequestHeadersMiddleware`, `register_oauth_middleware`) with injectable callbacks so `drmcpbase` stays free of `drtools` imports.
- `drmcp`: thin `core.middleware` wires `drmcpbase` middleware to `drtools.core.auth` (`initialize_oauth_middleware`, `create_oauth_middleware`). Tool clients use `resolve_datarobot_token` / `resolve_secret`.
- Docs: added `docs/drtools/auth.md` (MCP/http, LangChain/http, LangChain/config examples) and `AUTH_RESOLUTION_STRATEGY` to `docs/README.md`.
- Tests: added `tests/drmcp/unit/test_resolve_auth.py` for strategy-aware token/secret resolution; updated middleware, config, and OAuth fallback tests for the new injection model.

## 0.15.105
- `langgraph`: fix `APIConnectionError: 'str' object has no attribute 'get'` when re-sending streamed reasoning-model history. To align with AG-UI model, reasoning content is sent back to the model as usual text.
- `langgraph`/`llamaindex`: emit AG-UI Reasoning chunks under their own message id (derived from the text id) so frontends render reasoning as its own block instead of folding it into the assistant text bubble.

## 0.15.104
- Added new standalone `eval` extra (`datarobot-genai[eval]`) with `nemo-evaluator-launcher`, `anthropic`, and `pyyaml`, and a new `datarobot_genai.eval` subpackage for agent evaluation utilities.
- Excluded `leptonai` (Lepton AI cloud backend pulled in by `nemo-evaluator-launcher`); it is unused and pins `httpx==0.27.2`, which conflicts with the `auth` extra.
- Added `eval` to the CI per-module test matrix with a temporary stub test (`tests/eval/`); the full eval implementation will follow in a separate PR.

## 0.15.103
- `drtools`: refactored tool credentials to `ToolsAuthCredentials` and nested `DataRobotCredentials` using `DataRobotAppFrameworkBaseSettings` (env, runtime params, `.env`, file secrets, `pulumi_config.json`). Renamed fields to `datarobot_api_token` and `datarobot_endpoint` (`DATAROBOT_API_TOKEN`, `DATAROBOT_ENDPOINT`). Replaced `MCPServerCredentials`; added third-party config fields (`tavily_api_key`, `perplexity_api_key`, `atlassian_api_token`, `atlassian_email`, `atlassian_site_url`). Public export is `ToolsAuthCredentials`.

## 0.15.102
- `drtools` Jira: `Issue` model tolerates real Jira payloads—optional `emailAddress`, optional unassigned `assignee`, and `as_flat_dict()` falls back to `displayName` or `accountId` when email is missing.

## 0.15.101
- `drmcp`: removed OAuth provider startup gating (`IS_*_OAUTH_PROVIDER_CONFIGURED`, `is_*_oauth_configured`, and `oauth_check` in `tool_config`). Tool enablement is controlled only by `ENABLE_*_TOOLS` flags; OAuth tokens are resolved at request time.
- `drmcp`: `enable_predictive_tools` default is now `false` (opt in via env, `MCP_CLI_CONFIGS`, or runtime params). Integration and acceptance test harnesses set `ENABLE_PREDICTIVE_TOOLS=true` so predictive tool suites keep running.
- `drmcp`: `/metadata` tool config reports `enabled` only (dropped `oauth_required` / `oauth_configured`).

## 0.15.100
- Renamed MCP tools and updated unit, integration and acceptance tests

## 0.15.99
- `dragent/workflow_paths`: added `discover_workflow_yaml()` and `publish_dragent_config_file_env()` to locate `workflow.yaml` and set `DRAGENT_CONFIG_FILE` (from the env var, `$CODE_DIR/workflow.yaml`, or a walk up from CWD). Wired into the DRAgent CLI, FastAPI frontend startup, and `load_workflow()` so middleware can resolve guard assets without relying on the process working directory.
- `nat/datarobot_moderation_middleware`: default `model_dir` is now the directory containing `workflow.yaml` (via `DRAGENT_CONFIG_FILE`) instead of CWD, so `moderation_config.yaml` loads correctly in DataRobot custom-model deployments where CWD is not the agent code root. The middleware retries discovery on first use if `workflow.yaml` was not available at startup (e.g. gunicorn parent process).

## 0.15.98
- Added `datarobot_genai.drmcpbase.fastmcp_transforms.conditional_code_mode.ConditionalCodeMode`: This allows users to switch to fast mcp's CodeMode (tools are limited to {"search", "get_schema", "execute"}) if they pass the header: `x-datarobot-mcp-mode=code_execute`

## 0.15.97
- `nat/datarobot_moderation_middleware`: fixed moderated DRAgent streams that ended on a moderation `content_filter` without emitting `TEXT_MESSAGE_END` for open assistant text segments; the middleware now synthesizes those end events so AG-UI clients can close text messages cleanly.
- `nat/datarobot_moderation_middleware`: closed upstream and moderation async generators when the consumer stopped early (e.g. client disconnect), avoiding leaked iterators during stream teardown.

## 0.15.96
- Fix tool call sequense of ag-ui events for dragent+nat

## 0.15.95
- Surface reasoning/thinking from reasoning models as AG-UI Reasoning events (LangGraph, LlamaIndex DRAgent adapters) and fix the LangGraph crash on list-form `AIMessage.content`.

## 0.15.94
- CI: cache only `~/.cache/uv`, not `.venv`, so each job rebuilds a clean venv. Caching `.venv` under a shared key could restore a stale environment, intermittently breaking the `drmcpbase` test job (`ModuleNotFoundError: No module named 'datarobot'`).

## 0.15.93
- Per-user DataRobot API access (MODEL-23521): added `request_user_dr_client` and `request_user_dr_sdk` in `drtools.core.clients.datarobot`, both scoped via `client_configuration()` (ContextVar) instead of the global `dr.Client()`, so concurrent MCP tool requests do not share tokens.
- Removed `drtools.core.rest_client`; consolidated token resolution into `get_datarobot_access_token(*, headers_auth_only=...)` alongside the new context managers.
- `ThreadSafeDataRobotClient.request_user_client()` replaces `get_client_context_with_token_from_request_header`; predictive, use case, and VDB tools now call the scoped client context.
- Removed `drmcp.core.clients.get_sdk_client()` and `get_api_client()`; drmcp dynamic tool/prompt registration and deployment controllers use `request_user_dr_sdk` / `request_user_dr_client` from drtools. Public export is `request_user_dr_sdk` (was `get_sdk_client`). `drmcp.core.clients` still provides `RequestHeadersMiddleware` and `setup_and_return_dr_api_client_with_static_config_in_container()` for the container application account (lineage).
- Refactored `DataRobotClient.get_client()` to use `client_configuration()` (ContextVar-based) instead of the global `dr.Client()`, preventing token mixing between concurrent MCP tool invocations.
- Added `dr_client()` async context manager to eliminate repeated two-line boilerplate across predictive tool functions.

## 0.15.92
- `nat/datarobot_moderation_middleware`: `DataRobotModerationMiddleware` loads guard configuration from the inline `moderation` block in `workflow.yaml` when present, otherwise from `moderation_config.yaml` in `model_dir` (defaults to the process working directory). The middleware is a no-op when neither source has guards configured.

## 0.15.91
- LangGraph `dr_fs_checkpointer`: renamed `DataRobotFileSystemSaver` to `DataRobotFileSystemCheckpointSaver`.
- Updated `hitl.md` and comments/doc strings in `dr_fs_checkpointer` to be more descriptive.
- Removed `use_datarobot_fs_checkpointer` and its mentions

## 0.15.90
- `e2e-tests`: Enabled A2A server in all agent workflows and added A2A protocol end-to-end tests (agent card, `message/send`).

## 0.15.89
- Added `drtools.core.rest_client.request_user_dr_client`: a request-user-scoped DataRobot REST client reachable from `drtools` alone, so consumers pinning `datarobot-genai[drtools]` (e.g. global-mcp) and agents importing `drtools` directly can call the DataRobot API as the requesting user without depending on `drmcp`. It is a context manager backed by `client_configuration()` (ContextVar-scoped), so it does **not** mutate the global `dr.Client()` and won't mix tokens across concurrent requests (MODEL-23521). Also exposes `resolve_request_user_token`.
- Added `drtools.core.feature_flags.FeatureFlag`: per-user, entitlements-backed feature-flag evaluation keyed by `(flag, principal)` with a TTL cache (`ttl_seconds=0` bypasses the cache for live checks). Reachable from `drtools`, it is the building block for per-user, live tool gating (e.g. hiding a tool unless an entitlement is enabled). Distinct from `drmcp.core.feature_flags`, which evaluates the application-static MCP-container account.

## 0.15.88
- `core/datarobot_otel`: `bootstrap_otel_provider_for_datarobot()` now attaches a DataRobot-pointed `BatchSpanProcessor` to a pre-existing SDK `TracerProvider` instead of skipping. The `dragent_fastapi` server installs its own provider at startup before the agent module loads; under the previous skip behaviour, framework auto-instrumentor spans (CrewAI, Langchain, LlamaIndex) bound to that provider and never reached the DataRobot OTel ingest. Framework spans now show up in the deployment's Tracing tab alongside the existing exporter's output.
- `dragent/plugins/datarobot_otelcollector`: new NAT telemetry exporter that sends OTLP traces to the DataRobot OTel ingest endpoint with `X-DataRobot-Api-Key` / `X-DataRobot-Entity-Id` headers (NAT's built-in `otelcollector` doesn't support headers). `endpoint`, `datarobot_api_key`, and `datarobot_entity_id` auto-derive from deployment env (`MLOPS_DEPLOYMENT_ID`, `DATAROBOT_API_TOKEN`, `DATAROBOT_(PUBLIC_)ENDPOINT`), so the minimal `workflow.yaml` block is `_type: datarobot_otelcollector` + `project`. `project` maps to `service.name`.
- `core/telemetry_agent`: `instrument()` installs a global `TracerProvider` wired to the same DataRobot OTel endpoint when deployment env is present, so framework auto-instrumentors (Langchain, CrewAI, LlamaIndex) emit spans to the deployment's Tracing tab alongside the NAT exporter. No-ops when env is incomplete or another component has already set a `TracerProvider`. Shared env-resolution helpers live in `core/datarobot_otel`.

## 0.15.87
- `dragent/plugins/streaming_memory_agent`: registered via `register_per_user_function` (instead of `register_function`) so the wrapper builds lazily inside a `PerUserWorkflowBuilder` and `builder.get_function(inner_agent_name)` resolves per-user inner agents from the per-user cache (shared inner agents still resolve via fall-through). Switched the wrapper's I/O from NAT `ChatRequest` / `ChatResponseChunk` to AG-UI `RunAgentInput` and `DRAgentEventResponse`, so inner agents' native AG-UI events pass straight through without the intermediate `convert_chunks_to_agui_events` step. Added `stream_to_single_fn` so the function is also usable in non-streaming contexts.

## 0.15.86
- `nat/datarobot_mem0_memory`: added `default_ttl_seconds` to `dr_mem0_memory` config (defaults from the `AGENT_MEMORY_TTL_SECONDS` env var / DataRobot runtime parameter via `DataRobotAppFrameworkBaseSettings`). When set to a positive value, `DRMem0Editor.add_items` sends `expiration_date = today + ttl` (UTC, `YYYY-MM-DD`) to Mem0's `add` API so memories auto-expire on the platform's expiration sweep. A per-call `expiration_date` in `add_params` overrides the default; `None` / `0` leaves the field unset (no expiration), matching prior behavior.

## 0.15.85
- Expanded the `e2e-dragent-llmgw` job in `.github/workflows/e2e.yml` to cover multiple model providers. Matrix is now loaded from a new `e2e-tests/llmgw_matrix.yaml` file (5 agents × 4 models). The `default` model (bedrock) runs the full `dragent_tests` suite; other models (gpt, sonnet, gemini) run `test_streaming.py` only as a fast cross-model smoke check. This pre-empts model-specific framework bugs (like the LlamaIndex `tool_choice` issue below) before they reach downstream consumers.
- Tightened the planner/writer system prompts in `e2e-tests/dragent/{langgraph,crewai,llamaindex,nat}/` to reduce input tokens, output verbosity, and TTFT during e2e runs. Dropped the `make_system_prompt` boilerplate wrapper from test agents and shortened outputs to "1 bullet" + "1 short sentence". Test contracts (tool calls, HITL interrupt, multi-agent handoff) are preserved.
- `llama_index/llm.py`: Fixed `DataRobotLiteLLM` sending `tool_choice` and `parallel_tool_calls` in requests when no tools are provided. LlamaIndex's `_prepare_chat_with_tools` unconditionally emits both fields, which the DR LLM gateway rejects for Azure/GPT backends. Override strips both from the request when `tools` is absent.

## 0.15.84
- Refactored DataRobot feature flag logic and moved it to drmcpbase
- Added datarobot api client with async API in drmcpbase

## 0.15.83
- `dragent`: CLI now reads env vars `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` from `pulumi_config.json` at startup, so local OTel tracing works without manual env var setup.

## 0.15.82
- `drmcp`: removed the `memory_management` package (`MemoryManager`, S3-backed agent storage, and memory MCP tools), the `enable_memory_management` / `ENABLE_MEMORY_MANAGEMENT` config flag, memory-aware tool wrapping (`agent_id` / `storage_id` injection from `X-Agent-Id`), and the `/agent/...` storage REST routes.
- `drtools`: removed AWS/S3 credential fields and helpers from `MCPServerCredentials` (`aws_credential`, `aws_access_key_id`, `has_aws_credentials`, `get_aws_credentials`, `aws_predictions_s3_*`); they were only used by memory management.
- `drmcp` extra: dropped the `boto3` dependency.

## 0.15.81
`drmcp`: `MCPServerConfig` now reads `pulumi_config.json` via `PulumiConfigSettingsSource` (lowest priority) and accepts standard `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` fields. Telemetry setup bridges these to `os.environ` so local OTel tracing works without manual env var configuration.

## 0.15.80
- `nat/datarobot_moderation_middleware`: `DataRobotModerationMiddleware` is now a no-op when the `moderation` block is omitted or has no guards configured, so it can be listed unconditionally in `workflow.yaml` without requiring DataRobot credentials or emitting a warning. `load_llm_moderation_pipeline` returns `None` in those cases and skips `ModerationPipeline.from_config`.

## 0.15.79
- Added `drmcpbase` subpackage and standalone extra `datarobot-genai[drmcpbase]` (`fastmcp` only, no core). The `drmcp` extra now composes `drmcpbase` + `drtools` + template-server dependencies. Documented both extras in `README.md`.
- Import lint (`scripts/check_imports.py`): scans `drmcpbase`; `drmcp` may import `drtools`, `drmcp`, and `drmcpbase`; `drmcpbase` may only import `drmcpbase` (must not import `drtools`, `drmcp`, or `core`); `drtools` forbids `drmcpbase`.
- CI / tests: `drmcpbase` added to the `test-module` matrix; `tests/drmcpbase/` smoke test runs with `--confcutdir` so the root `tests/conftest.py` (core) is not loaded under the `drmcpbase` extra.
- `e2e-tests/uv.lock` regenerated to include the `drmcpbase` extra.
- `drtools` Jira/Confluence: replaced `get_atlassian_access_token` with `get_jira_access_token` and `get_confluence_access_token` (OBO provider types `jira` / `confluence`; fallbacks `x-datarobot-jira-access-token` and `x-datarobot-confluence-access-token`).
- `drtools`: `get_api_key_from_headers` now performs case-insensitive header lookup.
- `drtools`: `list_use_cases`, `list_vector_databases`, and `query_vector_database` map `ClientError` to `ToolError` via `raise_tool_error_for_client_error`.
- `drmcp`: `set_prompt_mapping` removes superseded prompt versions via FastMCP 3.x `local_provider.remove_prompt` instead of `prompt.disable()`.
- DRMCP tests: shared stub `DATAROBOT_*` constants for integration subprocesses; integration tests force `MCP_USE_CLIENT_STUBS=true` so a developer `.env` is not used; acceptance tests set `MCP_USE_CLIENT_STUBS=false` and require real credentials.

## 0.15.78
- `dragent/frontends/converters`: fixed dropped `datarobot_moderations` in dragent workflow chunk conversion paths by preserving moderation metadata on both NAT `ChatResponseChunk` and OpenAI `ChatCompletionChunk` streaming outputs.

## 0.15.77
- A non-existent `deployment_id` or `external_id` in the agent card registry now returns an actionable error message instead of a generic JSON-RPC `-32603 Internal error`.

## 0.15.76
- Pinned `starlette>=1.0.1` on the `drmcp` extra and switched MCP middleware to `request.scope["path"]` to harden against CVE-2026-48710 (BadHost)

## 0.15.75
- Upgrade to `nvidia-nat` 1.7.0, and pin `starlette>=1.0.1` to mitigate CVE-2026-48710

## 0.15.74
- Fixed `datarobot_api_key` auth provider not forwarding `Authorization: Bearer` header on A2A RPC calls when the agent card has no `security_schemes`.
- Fixed `asyncio.isasyncgenfunction` error on Python 3.12+

## 0.15.73
- Unhandled exceptions in A2A remote calls (auth failures, network errors, timeouts) no longer crash the agent. Errors are caught and sanitised.
- Fixed agent card registry returning at most 25 cards by adding pagination support.

## 0.15.72
- `dragent/plugins/streaming_memory_agent`: new `streaming_memory_agent` function (registered on the `nat.plugins` entry point) that wraps an inner agent with NAT's mem0 capture/retrieve semantics while preserving its `ChatResponseChunk` stream. The wrapper `astream`s the inner agent and pipes chunks through `convert_chunks_to_agui_events` so token deltas and tool-call deltas surface as AG-UI `TextMessage*` / `ToolCall*` events (NAT's upstream `auto_memory_agent` collapses the stream to a single `DEFAULT_NAT_RESPONSE`). `StreamingMemoryAgentConfig` inherits from upstream `AutoMemoryAgentConfig`, so the configuration surface (`memory_name`, `inner_agent_name`, `save_user_messages_to_memory`, `retrieve_memory_for_every_response`, `save_ai_messages_to_memory`, `search_params`, `add_params`) is identical to `auto_memory_agent`; switching wrappers is a `_type` rename in `workflow.yaml`. mem0 errors are logged and swallowed so partial output still reaches the client. Includes unit-test coverage for config registration/inheritance/defaults, the helper functions, all-flags-off streaming, user-message capture and AI-response persistence, memory retrieval and system-message injection, param forwarding, and the swallow-and-log error semantics.

## 0.15.71
- Fixed issue with empty `chunk.content` value for CrewAI.

## 0.15.70
- Added `prompt.py` for prompt templates integration in `langgraph` and `llamaindex`.

## 0.15.69
- `nat/datarobot_mem0_memory`: `_UserManagerShim.get_id()` now reads `Context.user_id` instead of re-decoding the `X-DataRobot-Authorization-Context` header. Identity resolution already happens upstream in `DRAgentAGUISessionManager` (via `DRAgentUserManager`, added in 0.15.60) and is stored on `ContextState.user_id`, so the shim just forwards it. Removed `_memory_user_uuid()` and the `AuthContextHeaderHandler` / `UserInfo` imports from the module. Per-user-workflow `default-user` fallback now flows through to the editor when no identity is present (previously the shim returned `None` and the editor fell back to the api-key owner).

## 0.15.68
- `nat/datarobot_moderation_middleware`: refactored DRAgent and NAT chat moderation to use OpenAI `ChatCompletionChunk` at the dome streaming boundary only. Shared AG-UI delta extraction and NAT↔OpenAI chunk converters live in `dragent/frontends/converters.py` (`convert_dragent_event_response_to_openai_chat_completion_chunk`, `convert_nat_chat_response_chunk_to_openai_chat_completion_chunk`).
- Prescore prompt extraction now delegates to `get_chat_prompt` from `datarobot_moderation_interface` (via `workflow_input_to_completion_dict`), matching DRUM integration behavior for multimodal content and tool context.
- Per-invoke moderation state stores the moderated prompt string instead of a prescore `DataFrame`; postscore reads `state.prompt`. Invoke context is not set when prescore blocks the prompt (post_invoke and streaming are skipped).
- `pre_invoke` fails closed with `TypeError` when the workflow argument is not `RunAgentInput`, `ChatRequest`, or `ChatRequestOrMessage`.

## 0.15.67
- Bump `datarobot-moderations` to 11.2.30 to use the async interface with `DataRobotModerationMiddleware`

## 0.15.66
- Added `datarobot_genai.dragent.execute_dragent_inline` (plus an async variant) — an in-process runner so `datarobot-user-models`'s `run_agent.py` can route between DRUM and dragent with a single env-var-gated branch. Workflow YAML is taken from the `config_file` argument when supplied, otherwise from `<custom_model_dir>/workflow.yaml`. Always returns a single aggregated OpenAI `ChatCompletion`; the `stream` flag on the request is ignored because the agentic playground only renders the final assistant message.

## 0.15.65
- Fix `extra_body` passthrough for `workflow.yaml` LLM configs (e.g. `mock_response`). PR #274 switched from `ChatOpenAI` to `ChatLiteLLM`/`LiteLLM` which silently drop unknown kwargs; `extra_body` is now routed through `model_kwargs` (langgraph) and `additional_kwargs` (llamaindex) so it reaches the underlying DataRobot LLM gateway API call.

## 0.15.64
- `dragent`: agent card XAA extension params now use camelCase (`tokenExchange`, `tokenRequest`, `tokenEndpointAuthMethod`, etc.) for consistency with the rest of the agentCard API response. The parser accepts both camelCase and snake_case for backward compatibility with previously generated cards.

## 0.15.63
- `nat/datarobot_mem0_memory`: the `dr_mem0_memory` provider now routes on config. When `memory_space_id` is set, requests go to the DataRobot Memory Service's mem0-compatible endpoint at `{datarobot_endpoint}/memory/{memory_space_id}` (authenticated with `datarobot_api_token` / `DATAROBOT_API_TOKEN`); otherwise `api_key` / `MEM0_API_KEY` is used against Mem0 SaaS. Both routes share the same `DRMem0Editor` because the DR endpoint is API-compatible with mem0 (PBMP-7431). New config fields: `memory_space_id`, `datarobot_endpoint`, `datarobot_api_token`.
- `nat/datarobot_mem0_memory`: providing both `memory_space_id` and `api_key` now raises `RuntimeError("...mutually exclusive...")` at factory time. The fields target different services with different tokens, so silently picking one would mask misconfiguration (e.g. a stray `MEM0_API_KEY` in env hydrating `api_key` via its default factory). The error message documents the `api_key=None` escape hatch for the env-contamination case.

## 0.15.61
- Added central agent card registry support to `authenticated_a2a_client`. Set `registry.deployment_id` or `registry.external_id` instead of `url` to resolve agent cards from the tenant-wide DataRobot registry.

## 0.15.60
- `dragent`: replace the `UserManager` monkey-patch with a `DRAgentUserManager` subclass that resolves `user_id` from the signed `X-DataRobot-Authorization-Context` header (then NAT's standard extractors). `DRAgentAGUISessionManager.session()` invokes it explicitly and, for per-user workflows only, falls back to a constant `default-user` key when no identity is present so the workflow does not crash (e.g. direct API-token calls to a deployed agent). The identity resolver and the per-user workflow fallback are kept separate so callers that need real identity are not silently handed a default.

## 0.15.59
- [MODEL-23506] `drmcp` dynamic tools: route chat-capable deployments to `/chat/completions` instead of `/predictions`. `DrumMetadataAdapter` now honors a `supports_chat_api` flag in metadata (sourced from the deployment's `/capabilities/` API by `get_mcp_tool_metadata`). When the flag is true the adapter returns endpoint `/chat/completions`, drops the `text/csv` Content-Type header, and uses the agentic (messages) fallback input schema. Defaults to `False` so legacy TextGeneration custom models served on `/predictions` retain their current routing. Fixes the 503 "Inference server is starting" reported when registering Guarded RAG / LLM-blueprint / NIM-served TextGeneration deployments as dynamic MCP tools.

## 0.15.58
- Update User ID for mem0 client to be per user

## 0.15.57
- Registered DataRobot moderation middleware on the `nat.plugins` entry point `datarobot_moderation_middleware` so `_type: datarobot_moderation` is available when NAT loads plugins.
- NAT / dragent moderation: `DataRobotModerationConfig` no longer uses `model_dir` to locate guard YAML. Configure guards with the optional `moderation` field (`ModerationConfig` from `datarobot_dome`). In `workflow.yaml`, nest guard definitions under `middleware.datarobot_guardrails.moderation` instead of setting `model_dir` to a directory that contained `moderation_config.yaml`.
- The `core` extra declares `datarobot-moderations[all]>=11.2.29,<12.0.0` (full moderation extras). Stack extras built on core (`nat`, `langgraph`, `crewai`, `llamaindex`, `dragent`, and `[core]` itself) install it; it is no longer listed only on the `dragent` extra. Standalone extras `auth`, `drtools`, and `drmcp` are unchanged.
- The `llamaindex` extra now pins `llama-index-llms-langchain` to `>=0.8.0,<1.0.0` (previously `>=0.6.1,<0.8.0`); 0.6.x and 0.7.x are no longer in range.

## 0.15.56
- `AgentKernel.custom_model`: HTTP 4xx/5xx now raise `requests.HTTPError` (via `response.raise_for_status()`) instead of plain `Exception`, so callers can classify failures by status code.

## 0.15.55
- Fixes related to the agent card parsing and mapping to the xaa (cross-application access) token exchange flow.
- Added two versions of xaa (cross-application access) token exchange flow: one using the `okta-client-python` SDK. Set `XAA_TOKEN_EXCHANGE_IMPL=okta_sdk` (default) or `http` for implementation making direct HTTP calls.
- Renamed XAA environment variables: `PRINCIPAL_ID` → `IDP_AGENT_ID` and `PRIVATE_JWK` → `IDP_AGENT_PRIVATE_KEY_JWK`. The old names are still accepted for backward compatibility but are deprecated.

## 0.15.54
- `langgraph/mcp.py`: Fixed `RuntimeError: generator didn't stop after athrow()` in `mcp_tools_context` when a connection-type exception (`ConnectionError`, `OSError`, `TimeoutError`, `ExceptionGroup`) is raised by the consumer inside the `async with` block. A `connected` flag now distinguishes setup-phase failures (graceful fallback to empty tools) from consumer exceptions (re-raised so the caller sees them). Without this guard the except clause would execute a second `yield []`, violating the `@asynccontextmanager` single-yield contract.

## 0.15.53
- LangGraph `LangGraphAgent`: DR FS checkpointing is opt-in via `use_datarobot_fs_checkpointer=True` when `checkpointer` is omitted (no longer automatic). Optional `langgraph_checkpoint_base` sets the `dr://` prefix for the default saver (typically from application settings); when omitted, the default root is `dr://`. Process exit cleanup removes only `<prefix>/checkpoints`, not the entire prefix (so other DR FS objects under the same root are preserved).
- `DataRobotFileSystemSaver` (`dr_fs_checkpointer`): checkpoint files use length-prefixed binary (`struct`, `.bin` suffix) without pickle or per-file magic headers; layout is implied by directory (`blobs/`, `cpts/`, `writes/`).

## 0.15.52
- `langgraph/agent.py`: Fixed `ValueError: Invalid message event` crash in `_stream_generator` when an intermediate LangGraph node (e.g. a planner-to-writer relay) emits a `HumanMessage` as a state update. `HumanMessage` events from relay nodes are now silently skipped rather than raising, which also prevents the cascading `RuntimeError: generator didn't stop after athrow()` from the MCP tools context manager failing to clean up.

## 0.15.51
- Bump ragas to "ragas>=0.4.3,<0.5.0" to align with execution environments

## 0.15.50
- `crewai/mcp.py`: Fixed `BadRequestError` from Azure OpenAI when an MCP tool has no input schema. `MCPServerAdapter` would leave `args_schema = None` on such tools; litellm then serialized `"parameters": null`, which Azure rejects. Tools with a `None` args schema now fall back to an empty-object schema (`_EmptyArgsSchema`) so the function-calling payload is always valid.

## 0.15.49
- `is_eligible_for_timeseries_training`: Surfaced median timestep, per-series gap percentage, and max-gap-seconds as a `cadence` field so agents can pick between TS and row-based partitioning before calling `start_autopilot`. Treated an entirely-null target as a scoring dataset (downgraded from blocking error to INFO). Detected row-level duplicates per (datetime, series_id) and reported up to three offending keys. Updated error messages to follow what + why + how-to-fix format, listed available columns on column-name mismatches, and showed sample bad values on unparseable datetimes.

## 0.15.48
- Added vector database tools: list_vector_databases and query_vector_database (MODEL-22811)
- Fixed `test_list_vector_databases_success` mock to return only deployments matching API `modelTargetType=VectorDatabase` filtering
- Refactored VDB tools to use `tool_metadata` and plain dict returns (no fastmcp/drmcp imports in drtools)
- Fixed mypy in `dr_client_stubs` deployment list filter (`model` dict narrowing)

## 0.15.47
- Fixed default `okta_token_header` value in `OAuth2CrossApplicationAccessAuthProviderConfig`: renamed `x-datarobot-okta-access-token` → `x-datarobot-external-access-token` to match the actual header name used by the DataRobot API gateway when forwarding Okta access tokens.

## 0.15.46
- Fixed CrewAI tool calling by enforcing client-side stop-word truncation when upstream APIs ignore the `stop` parameter

## 0.15.45
- `drtools/predictive/predict.py`: **Submit-and-poll batch workflow** — `predict_by_ai_catalog` and `predict_from_project_data` return immediately after submit (removed `timeout` and server-side `wait_for_completion` plus download-link polling); responses include `job_id`, `batch_job_status`, optional early `url`, and a `note` for follow-up instead of only completed-job metadata. **New tool `get_batch_prediction_job_status`** (`job_id`) returns status, optional download `url`, and progress fields without fetching CSV. **`get_batch_prediction_results`** is documented and used after polling for completion; passes `download_timeout` / `download_read_timeout` through to the SDK download. **`BatchPredictionJob.score` / `get`** use the same configured SDK client as `Dataset.get`.
- `drtools/predictive/training.py` (`get_exploratory_insights`): optional `feature_col` plus `include_feature_histogram` add a DataRobot catalog API-backed column profile (allFeaturesDetails statistics and optional feature histogram) alongside existing EDA output; helpers resolve catalog `DatasetFeature` rows by name and serialize them for the tool response.
- `drtools/predictive/predict_realtime.py`: clarified `predict_by_ai_catalog_rt` tool metadata so async batch scoring is described as submit-and-poll via `predict_by_ai_catalog` and `get_batch_prediction_job_status`.

## 0.15.44
- Added HTTP request headers forwarding into the NAT `Context` for A2A JSON-RPC routes.
- Renamed `OktaCrossApplicationAccessAuthProvider` → `OAuth2CrossApplicationAccessOAuth2AuthProvider` and `OktaCrossApplicationAccessAuthProviderConfig` → `OAuth2CrossApplicationAccessAuthProviderConfig` to satisfy the NAT SDK's name-based OAuth2 compatibility check.

## 0.15.43
- Fixed dragent A2A + per-user workflows when no Bearer JWT is present: `DRAgentAGUISessionManager.session` now forwards a preset `ContextState.user_id` (set from the A2A `context_id` by the FastAPI executor) into NAT’s explicit `user_id` argument. NAT 1.6+ otherwise replaced the context value with `None`, causing per-user workflows to fail in local dev and message-only A2A scenarios.

## 0.15.42
- Added NAT middleware for DataRobot LLM guardrails (`datarobot_genai.nat.datarobot_moderation_middleware`), ported from the agent application recipe. The `dragent` extra includes `datarobot-moderations` (there is no separate `moderation` extra); import the middleware module in your NAT workflow registration so `@register_middleware` runs (same pattern as `import agent.datarobot_moderation_middleware` in app templates). Extended `DRAgentEventResponse` with optional `datarobot_moderations` for serialized pipeline metadata.
- Declared `uv` `override-dependencies` for OpenTelemetry (`opentelemetry-api` / `sdk` / `instrumentation` and OTLP exporters at 1.39.x / 0.60b1) so `datarobot-moderations` can coexist with optional extras such as CrewAI when resolving `datarobot-genai[dragent]` together with other stacks.

## 0.15.41
- Added a NAT `dr_mem0_memory` provider that adapts `datarobot-genai[memory]`'s Mem0 client to NAT's `MemoryEditor` interface for `auto_memory_agent`.
- Documented Mem0 automatic-memory workflow configuration for NAT.

## 0.15.40
- Simplified cross-application access `workflow.yaml` config: IETF URNs and auth method defaults are now injected by the AgentCard generator rather than declared by the developer.
- Added `urn:datarobot:agent:identity:internal` and `urn:datarobot:agent:identity:external` AgentCard extensions, and an optional external URL override (`general.frontend.a2a.external.url`).

## 0.15.39
- Added documentation `docs/nat/a2a-auth.md` on how to configure various auth options for A2A agents.

## 0.15.38
- Fixed NAT MCP tool calls failing with `Enum` field `ValidationError` (e.g. Tavily) by setting `use_enum_values=True` on tool input schemas.

## 0.15.37
- Added an optional `model` identifier to `BaseAgent`. Updated CrewAI, LangGraph, and LlamaIndex agents to use explicit named `__init__` parameters (instead of forwarding arbitrary kwargs). Possible breaking change moving away from `kwargs` to named parameters.

## 0.15.36
- Fixed AG-UI tool call lifecycle in dragent frontends.

## 0.15.35
- Added Okta Cross-Application Access (XAA) support for A2A agent-to-agent calls via the new `okta_cross_app_access` auth provider.
- Server-side XAA parameters are declared in `workflow.yaml` under `cross_application_access` and published on the AgentCard: OpenAPI `clientCredentials` security scheme plus a JWT Bearer capability extension containing the two-step flow parameters (RFC 8693 → RFC 7523).
- Client-side: `OktaCrossApplicationAccessAuthProvider` reads the AgentCard extension at runtime, performs the two-step token exchange via `okta-client-python`, and requires only `PRINCIPAL_ID` / `PRIVATE_JWK` env vars.
- Added `okta-client-python` dependency into the `auth` extra.
- Server-side config tag was renamed from `oauth_token_exchange` into `cross_application_access`.

## 0.15.34
- Fixed AG-UI event lifecycle in the LlamaIndex agent adapter: each agent step emits its own message bubble and a matching `STEP_FINISHED`.

## 0.15.33
- Fixed CrewAI AG-UI streaming so each agent role in a multi-agent crew emits its own assistant message with a unique `messageId`.

## 0.15.32
- Implemented pagination for predictive model MCP tool

## 0.15.31
- Improved MCP lineage sync logic and made it always run during user MCP startup.

## 0.15.30
- Implemented pagination for predictive data MCP tools

## 0.15.29
- Added LangGraph human-in-the-loop (HITL) support and parameterization of LangGraph agent.

## 0.15.28
- Added `datarobot_genai.core.agents.verify` for validating AG-UI event streams against the protocol state machine. Re-enabled the previously skipped sequence checks in dragent e2e tests.

## 0.15.27
- Fixed LangGraph MCP integration so a single failing MCP tool no longer aborts the entire agent run.

## 0.15.26
- Categorized ToolErrors, OAuth access tokens with x-datarobot-*-access-token fallback, MCP logging that surfaces kinds to FastMCP, SDK ClientError → tool errors in predictive tools and improved third party APIs tool_metadata descriptions.

## 0.15.25
- Improved predictive drtools for MCP agents: rich tool_metadata descriptions, robust batch download polling and async-safe waits, safer CSV/JSON parsing for realtime predict, and more resilient deployment CSV validation (importance + whitespace/empty rows).

## 0.15.24
- Fixed NAT `per_user_tool_calling_agent` leaking prior assistant messages from chat history back into the response stream as a single trailing chunk.

## 0.15.23
- OAuth2 token exchange configuration now use a nested `subject_token_constraints` and `token_exchange_request`, with AgentCard extension.

## 0.15.22
- Removed `posthog` and `qdrant-client` from `e2e-tests/pyproject.toml` exclude list to mirror the main pyproject.toml fix in 0.15.21; both are imported eagerly by upstream libs and excluding them would break e2e test collection.

## 0.15.21
- Restored `posthog` and `qdrant-client` as runtime dependencies. Both are imported eagerly at module load by `deepeval/telemetry.py` and `mem0`'s `Memory` class respectively; excluding them broke pytest collection and any memory-enabled agent on startup.

## 0.15.20
- Added `DataRobotLLMRouterConfig` (`datarobot-llm-router`) to NAT to configure primary and fallback LLM providers via `litellm.Router`.
- Added `get_router_llm` for CrewAI and `RouterLLM` support for LangGraph and LlamaIndex to expose the same fallback router behavior across frameworks.
- Added fallback E2E coverage and updated docs for workflow-router fallback behavior.

## 0.15.19
- Removed unnecessary packages with exclude to reduce the dependency footprint

## 0.15.18
- Refactored event rendering: decouple it from NAT console and make it all go to stdout
- Added Agent.invoke_simple: method to simply call an agent class with a single request, and output a stream of rendered colored events

## 0.15.17
- Added option to configure OAuth2 token exchange flow for server

## 0.15.16
- Pinned ag-ui-protocol to version 0.1.15

## 0.15.15
- E2E testing for example notebooks
- Promp Management added to the quickstart example notebook

## 0.15.15
- Fixed interleaved event ordering in the stream converter to emit sequential text and tool call blocks
- Fixed input converter to handle tool and reasoning role messages during replay

## 0.15.14
- Removed custom chat completions implementations and set the default workflow config to generate it instead

## 0.15.13
- Changed CLI model placeholder to `unknown` so the agent resolves the model from config (LLM_DEFAULT_MODEL)

## 0.15.12
- Updated documentation for agents: added quickstart guide
- Updated documentation for LLMs: getting models for LLM GW, and additional details for each route
- Fixed an issue with `datarobot-deployed-llm`: this one should only be the default for the deployment case

## 0.15.11
- Added backward compatible route `/chat/completions` route to `dragent`
- Revised documentation based on feedback
- Added example Jupyter Notebook walktrough of setting a LangGraph agent in DataRobot

## 0.15.10
- Fixed `NatAgent` text extraction for `DRAgentEventResponse` objects.

## 0.15.9
- Adding parameters to Crew AI and LlamaIndex Agents

## 0.15.8
- **Dependencies**: Removed the python<3.13 restriction from mem0ai when including the memory dependency.

## 0.15.7
- **Dependencies**: Moved `datarobot-early-access` from the `drtools` extra to `drmcp`.

## 0.15.6
- Added cli.py-compatible aliases (`--user_prompt`, `--deployment_id`) to `nat dragent run` and `query` for Taskfile passthrough.
- Added `--file` / `--input-file` option to `nat dragent run` and `query`: reads a text file and uses its contents as the prompt.

## 0.15.5
- **Security**: Raised minimum `pypdf` to `>=6.10.1` for CVE-2026-40260 (fixed in 6.10.0) and GHSA-jj6c-8h6c-hppx (fixed in 6.10.1).

## 0.15.4
- Added README.md and standalone documentation
- Documented AG-UI integration, multi-agent patterns, and the unified DataRobot-compatible LLM layer (`get_llm()`, shared `Config` / environment) in the root and docs READMEs

## 0.15.3
- Wired `LangchainProfilerHandler` into LangGraph agent config so `nat dragent run` streams intermediate LLM events to the console frontend.
- Broadened CrewAI MCP exception handling to catch all exceptions on connection failure, so `nat dragent run` continues without MCP tools instead of crashing.

## 0.15.2
- Removed the dedicated S3 client module from `drtools.core.clients` (S3 is no longer exposed as a first-class tool client here).
- Removed the `predict_by_file_path` predictive tool; use catalog or dataset-based prediction flows instead.
- `predict_realtime` now accepts inline prediction payload data only (for example CSV/JSON text via `dataset`); local `file_path` is no longer supported.
- `validate_prediction_data` now takes inline CSV content (`csv_string`) only; local file paths are no longer supported.
- `upload_dataset_to_ai_catalog` uploads via `file_content_base64` and `dataset_filename`, or `file_url`; local filesystem paths are no longer accepted for remote-safe MCP usage.
- `score_dataset_with_model` takes an AI Catalog `dataset_id`, copies it into the project with `Project.upload_dataset_from_catalog`, then runs `Model.request_predictions` on the prediction dataset (catalog datasets are not valid for `request_predictions` alone; `Model.score` does not exist); dataset URL parameters were removed.
- Refactored deployment helper logic into `drtools.core.deployment_utils` and trimmed `deployment.py` to orchestration on top of it.
- Adjusted the DataRobot SDK client wrapper, shared constants, and tool error messaging in `utils.py` to align with the updated prediction and upload flows.
- Updated tests to align with the changes.

## 0.15.1
- Added opt-in memory retrieval and storage to the NAT base agent when prompts declare `{memory}`.

## 0.15.0
- Migrated NAT dependencies from 1.4.1 to 1.6.0
    - Updated import path for tool_calling_agent (nat.agent -> nat.plugins.langchain.agent)
    - Monkey-patched UserManager.extract_user_from_connection for DR auth context user_id resolution
    - Registered health routes in build_app() for NAT 1.6 compatibility
    - Stripped internal NAT params (verify_ssl) from crewai LLM config
    - Wrapped tool_calling_agent stream_fn for AG-UI event conversion
    - Removed CrewAI callback handler compatibility patch (fixed upstream in NAT 1.6)
    - Excluded flask transitive dependency from nvidia-nat-core 1.6.0

## 0.14.6
- Fixed MCP tool calls to deployments silently swallowing HTTP error responses instead of surfacing them as tool errors.

## 0.14.5
- Relaxed `langgraph` and `langgraph-prebuilt` version constraints from `<1.1.0` to `<2.0.0`

## 0.14.4
- Pinned crewai to 1.11.0: fixed is_litellm issues and CrewAI attempts to load a native provider

## 0.14.3
- Added `nat dragent` shell frontend for NAT - provides `nat dragent serve`, `nat dragent run`, and `nat dragent query` commands
- MCP tool loading now gracefully falls back to empty tools on connection errors (CrewAI, LangGraph, LlamaIndex)

## 0.14.2
- Implemented functions to instantiate LLM classes for LLM Gateway, deployment, NIM, and external

## 0.14.1
- Added opt-in memory retrieval and storage to the LlamaIndex base agent when input messages declare `{memory}`.

## 0.14.0
- Reworked agents to only accept `llm`: its not the job of an agent to instantiate the LLM
- Implemented functions to convert native agentic primitives into DataRobot agents

## 0.13.3
- Enabled functionalities of synchronizing MCP item metadata within MCP server (still behind the feature flag).

## 0.13.2
- Updated to remove prints, but instead use `logging` instead in our agents outputs

## 0.13.1
- Added opt-in memory retrieval and storage to the CrewAI base agent when kickoff inputs declare `memory`.

## 0.13.0
- Upgraded fastmcp from 2.x to 3.2.0+ (CVE-2026-32871, CVE-2026-27124, CVE-2025-64340)
- Replaced `on_duplicate_tools`/`on_duplicate_prompts` with unified `on_duplicate` (fastmcp 3.x API)
- Replaced `enabled` parameter with `mcp.disable()` for tool/prompt/resource registration (fastmcp 3.x API)
- Made `Context.set_state`/`get_state` calls async (fastmcp 3.x API)

## 0.12.0
- **Breaking** Update Langgraph template to support mem0 store and retrieve long term memory

## 0.11.0
- Reworked legacy interface to support mcp_tool_context being external to an agent.

## 0.10.3
- Updated `openai` and `litellm`, and removed unused transitive dependencies.

## 0.10.2
- Fixed multi-turn tool-calling by preserving assistant `tool_calls` on `AssistantMessage` and retaining tool-call metadata in chat history summaries.

## 0.10.1
- Fixed multi-turn tool-calling: preserving tool_calls on AssistantMessage and passing structured native messages to all agent frameworks when conversation history is present.

## 0.10.0
- **Breaking**: Decoupled agents and MCP tools. Now mcp_tools_context has to be called explicitly outside of an agent invoke.
- Reworked `mcp_tools_context`arguments to use `MCPConfig` explicitly.
- Fixed issue with `RuntimeError: generator didn't stop after athrow()` in MCP context.
- Fix CrewAI streaming steps and reasoning: enable CrewAI streaming by default
- Unify logging for CrewAI

## 0.9.2
- Added AG-UI Events for CrewAI

## 0.9.1
- Suppressed known-harmless NAT warnings

## 0.9.0
- Removed side-effects in MCPConfig
- Renamed datarobot_genai.core.mcp.common to datarobot_genai.core.mcp.config

## 0.8.17
- Fixed prompt for calculator to make output for LlamaIndex agents stable

## 0.8.16
- **Security**: Upgraded `aiohttp>=3.13.3` to fix CVE-2025-69229 (DoS via chunked messages) and CVE-2025-69230 (cookie parser warning storm)
- **Security**: Upgraded `pypdf>=6.9.2` to fix CVE-2026-33699 (infinite loop in DictionaryObject recovery) and CVE-2026-33123 (inefficient stream decoding)
- **Security**: Upgraded `pyjwt>=2.12.0` to fix CVE-2026-32597 in core and drmcp dependencies
- **Security**: Verified authlib CVE-2026-27962 is patched via fastmcp transitive dependency (1.6.9+)

## 0.8.15
- Did a major refactor to decouple `drtools` from `drmcp`
- Added dependency lint check task to the ci
- Moved auth/token extraction (`_extract_token_from_headers`, `_extract_token_from_auth_context`, `AuthContextHeaderHandler`) from `drmcp.core.clients` to `drtools.core.auth`; exposed a single `resolve_token_from_headers()` entrypoint consumed by `get_sdk_client`
- Updated all `drmcp` tests to align with the `drtools` refactor: corrected import paths, exception types, async mock targets, and direct error propagation expectations
- Fixed `MCPToolConfig` loading `.env` independently of `MCPServerConfig(_env_file=None)`; introduced `_MCPToolConfigNoEnvFile` subclass so config default assertions reflect true code defaults

## 0.8.14
- Isolated publish secrets to an environment

## 0.8.13
- Added base agent for retrieving and storing memory

## 0.8.12
- Removed fastmcp dependency from drtools
- Fixed all unit tests to handle dict returns instead of ToolResult objects after refactoring
- Removed ToolResult dependencies from test assertions and mock setups
- Fixed import paths and lint errors across all test files
- Updated test expectations to work with plain dictionary responses from tools
- Updated helpers.py to use FastMCP's get_http_headers with safe import handling

## 0.8.11
- Made build_workflow async

## 0.8.10
- Added GitHub Actions workflow `integration.yml`: path-filtered **Integration Tests** job for drmcp (runs when `src/datarobot_genai/drmcp`, `src/datarobot_genai/drtools`, `setup.py`, `tests/drmcp/integration`, or the workflow file changes; aligned with the e2e workflow pattern)
- DRMCP integration/ETE: `tests/drmcp/stub_credentials.py` plus `ete_test_server.py` default stub `DATAROBOT_*` env so `task drmcp-integration` can start the MCP server without a real API token in `.env`; shared stub token constant wired from `tests/drmcp/conftest.py`
- `test_interactive.py`: read `DR_LLM_GATEWAY_MODEL` and optional `LLM_TEMPERATURE` into the LLM Gateway client config (consistent with ETE helpers)

## 0.8.9
- Pin LiteLLM to safe version to prevent exploit (see https://github.com/BerriAI/litellm/issues/24518)

## 0.8.8
- When constructing the agent card, prefer the DATAROBOT_PUBLIC_API_ENDPOINT over DATAROBOT_API_ENDPOINT, avoiding connection issues in onprem environments.

## 0.8.7
- Rework NAT AG-UI integration
- Do not return final response from NAT twice

## 0.8.6
- Locked upperbound for dragent dependencies (fastapi, starlette) to avoid compatibility issues
- Locked lowerbound for AG-UI because of a change in reasoning events validation

## 0.8.5
- Added logic to sync MCP server deployment associated metadata with items in MCP server.

## 0.8.4
- Added use case tools: list_use_cases and list_use_case_assets (MODEL-22810)
- Added ToolType registrations for use_case, vdb, code_execution, and optimization
- Added corresponding config fields and CLI options for new tool types

## 0.8.3
- Added AG-UI Events for Llamaindex

## 0.8.2
- Added model and deployment tools: get_model_details (with optional feature impact and ROC curve), is_eligible_for_timeseries_training, get_prediction_history (MODEL-22809)
- CODEOWNERS: MCP team owns drtools; default Buzok with MCP overrides (last match wins)

## 0.8.1
- Fixed issue with NAT profiler interacting with MCP tools for langgraph
- Enabled MCP and tool tests for all agents
- Implemented decorator `nat_tool` which allows registering a function in NAT with a single line

## 0.8.0
- Allowed configuring step adaptor
- Reorganized `dragent` to submodules
- Disabled step adaptor for custom DataRobot models to prevert reporting events twice
- Added example of producing a single (non-streaming) response

## 0.7.8
- Updated enum values under DataRobotMCPPromptCategory and DataRobotMCPToolCategory to make them align with accepted enums of DataRobot public API.
- Removed UNKNOWN enum value from DataRobotMCPPromptCategory, DataRobotMCPToolCategory, and DataRobotMCPResourceCategory.

## 0.7.7
- Forwarded `x-untrusted-*` headers alongside `x-datarobot-*` headers in NAT `extract_datarobot_headers_from_context()`

## 0.7.6
- Added base agent e2e example under `e2e-tests/dragent/base/` demonstrating how to extend `BaseAgent` directly with litellm

## 0.7.5
- Added `authenticated_a2a_client` function group to dragent, to authenticate all api calls including calls to `/.well-known/agent-card.json`.

## 0.7.4
- Allowed running nat agents with per-user workflows with drum

## 0.7.3
- Added `temperature` parameter support to LLM MCP clients (`BaseLLMMCPClient`, `DRLLMGatewayMCPClient`): read from config dict and forwarded to `chat.completions.create`
- Added `LLM_TEMPERATURE` env var support in `get_openai_llm_client_config()` and `get_dr_llm_gateway_client_config()` to control LLM temperature in acceptance tests
- Switched tool parameter matching in `ToolBaseE2E` from exact equality to subset matching (`_check_dict_params_match`) to reduce test flakiness

## 0.7.2
- Added `nvidia-nat-crewai` support with crewai >= 1.1.0 compatibility patches

## 0.7.1
- Migrate Data MCP tools from wren mcp (MODEL-22804)

## 0.7.0
- **Breaking**: drop Python 3.10 support because of using NAT in all agents
- Pass forwarded DataRobot headers to agents, MCP clients, and LLMs
- Pass authorization context to agents and MCP clients
- Register MCP function group per user as it depends on the current user authorization context
- Rework dependencies in order to install only necessary libraries

## 0.6.21
- Added `x-datarobot-authorization` to `HEADER_TOKEN_CANDIDATE_NAMES` to fix auth when connecting through the API gateway

## 0.6.20
- Fixed an issue where the API token loaded via an environment variable was not properly serialized in NAT

## 0.6.19
- Enable A2A endpoints for per-user workflows with configurable skills via `DRAgentA2AConfig`
- **Breaking**: `DRAgentFastApiFrontEndConfig.a2a` type changed from `A2AFrontEndConfig` to `DRAgentA2AConfig`; update `workflow.yaml` by nesting the existing A2A fields under `server:`
- Added new data tools: get_dataset_details, list_datastores, browse_datastore, query_datastore
- Added "daria" tag to existing overlapping tools

## 0.6.17
- Fixed CVE-2026-25580: removed unused `pydantic-ai-slim` dependency and `pydanticai` install extra
- Added e2e tests for dragent server covering streaming, tool use, and MCP integration
- Added CI workflow for e2e tests with path-based triggers across langgraph, crewai, llamaindex, and nat

## 0.6.16
- Align MCP OpenTelemetry spans with OTel semantic conventions.

## 0.6.15
- Added Agent2Agent (A2A) server endpoints to `DRAgentFastApiFrontEndPluginWorker`, mounted at `/a2a`.
- Extended DRAgentFastApiFrontEndConfig with configuration options for the A2A server.
- A2A endpoints can be enabled by the `expose_a2a_server_endpoints` setting in the workflow.yaml file.
- Added `per_user_tool_calling_agent` workflow type
- Fixed `ToolCallArgsEvent.delta` encoding

## 0.6.14
- Fix loading JSON schemas from the package directory in DRUM adapter to work from wheel or source
- Fix dynamic tool deployment registration to filter deployments with `tool` tag name and value using strict AND logic
- Fix configuration parsing to correctly disable predictive tools when `MCP_CLI_CONFIGS` is empty

## 0.6.13
- Added `x-datarobot-identity-token` to `HEADER_TOKEN_CANDIDATE_NAMES`

## 0.6.12
- Add Mem0 for storing agent memory capabilities

## 0.6.11
- Fixed AG-UI event serialization

## 0.6.10
- Updated cursorbot review instructions with correct names

## 0.6.9
- Added cursorbot review instructions

## 0.6.8
- Make model required for MCP clients acceptance tests

## 0.6.7
- When getting api key in perplexity/tavily try to get it from 2 different headers (as fallback)

## 0.6.6
- Refactor drtools to use Polars for data handling, removing pandas dependency
- Enhance acceptance tests for realtime predictions with inline CSV dataset support

## 0.6.5
- Update task file to enable integration tests

## 0.6.4
- Created prompt stubs for MCP integration tests

## 0.6.3
- Created model stubs for MCP integration tests

## 0.6.2
- Created MCP predict test stubs and added stub predict client attribute integrated with MCP server

## 0.6.1
- Add e2e test scaffolding for CrewAI, LangGraph, and LlamaIndex agent frameworks

## 0.6.0
- Unify `InvokeReturn` type from `str | Event` to `Event` across all agent implementations
- All agents (CrewAI, LangGraph, LlamaIndex, NAT) now emit AG-UI lifecycle events
- Remove `str` code path from streaming/completions layer

## 0.5.15
- Restructure the tools and move them to drtools instead of drmcp.tools

## 0.5.14
- Fix unit tests to not depend on .env
- Update MCP dependencies to remove core extra
- Fix root conftest loads without error for drmcp tests by using inline imports.

## 0.5.13
- Add `dragent`: frontserver for DataRobot Agents
- Add placeholder for E2E tests of `dragent` and `drmcp`

## 0.5.12
Update MCP item metadata related enums
- DataRobotMCPToolCategory
- DataRobotMCPPromptCategory
- DataRobotMCPResourceCategory

## 0.5.11
- Created DR MCP test stubs and added stub `client` attribute integrated with MCP server

## 0.5.10
- Add DataRobotMCPServer APIs to retrieve registered tools/prompts/resources.

## 0.5.9
- Add MCP tool `deploy_custom_model` for deploying custom inference models (e.g. `.pkl`, `.joblib`) to DataRobot MLOps
- Custom model deployment: validation for prediction servers (consistent with `deploy_model`), optional execution environment, model file discovery in folder.

## 0.5.8
- Added "DR docs" tools: a tool for searching DataRobot Agentic AI docs and returning most relevant doc pages (includes title, URL, content) using TF-IDF, and a tool for fetching any DataRobot docs page. Note: only supported for English documentation, not Japanese.

## 0.5.7

- Updated NAT MCP client for 1.4.1 changes
- Update default transport for NAT MCP client to `streamable_http`
- Log error and fall back to empty function group when NAT MCP client is misconfigured

## 0.5.6

- Bump NAT libraries to 1.4.1
- Add `nvidia-nat-a2a` as a dependency for the `nat` extra

## 0.5.5

## 0.5.5
- Added upper bound to `crewai` dependency (`>=1.1.0,<2.0.0`)

## 0.5.4

- Added dr_mcp_prompt as prompt function decorator
- Added dr_mcp_resource as resource function decorator

## 0.5.3

- Added chat history support for all agent types (CrewAI, LangGraph, LlamaIndex, NAT)
- History is opt-in per agent; configurable via `max_history_messages` constructor param or `DATAROBOT_GENAI_MAX_HISTORY_MESSAGES` env var (default: 20)

## 0.5.2

- Dependency groups are converted into the optional `extra` options for installation, without list of default dependencies.

## 0.5.1

- Added the CHANGELOG.md file
- GitHub Action to verify changelog entries in pull requests

## 0.5.0

- No dependencies installed by default
- Optional `extra` options were converted into dependency groups and require `uv` for installation
- Added `auth` dependency group for authentication utilities
- Pandas dependency moved from the `core` to the `drmcp` dependency group
