# AG-UI Chat History — Implementation Plan

Turns `.local/plans/agui_chat_history_requirements.md` into an actionable design.
Target package: `src/datarobot_genai/application_utils/chat_history/`.

---

## 0. Decisions locked from the grilling round

| # | Question | Decision |
|---|----------|----------|
| 1 | How do `Chat`/`Message` relate to the persistence ORM? | **Extend the persistence ORM** so declared body/metadata fields Pydantic-(de)serialize (nested typed models round-trip natively). `Chat` subclasses `DRSession`; `Message` subclasses `DREvent`; `tool_calls`/`reasonings` are typed nested fields. |
| 2 | Disconnect-survival + cancellation shape | **Instance-scoped `StreamPersistenceManager`** owning the in-flight-run registry; `run()` returns a `RunHandle` (drained stream + `.cancel()`); cancel also via `manager.cancel(thread_id, run_id)`. No module-global state. |
| 3 | History rebuild/injection | **In scope.** Port `translate_messages` + inbound-user-message persistence + history replacement into `run()`. |
| 4 | AG-UI storage extensibility idiom | **Subclass + overridable hook methods + event-type dispatch registry.** Repos injected via `Protocol`. |

## 0b. Decisions I made with sensible defaults — please sanity-check

- **New dependency:** `ag-ui-protocol` (pinned `~=0.1.15` to match the agent-app/agent pins) is required by `chat_history`. **Decision (confirmed with user):** add it to the **existing `application-utils` extra** rather than a separate extra — it's lightweight and consumers will almost always need it; `application-utils` is allowed to be a mild grab-bag. Invariant retained: the `persistence` sub-package must still **never import `ag_ui`** (only `chat_history` may).
- **`MessageStatus` enum** (`active` / `complete` / `interrupted` / `errored`) is adopted from efm (`fastapi_server/app/messages/__init__.py:46`). Interrupted runs set `status=interrupted`, `in_progress=False`.
- **Repository interface = `typing.Protocol`** (structural), not ABC — so a SQL backend satisfies it without importing our classes (requirement 6.4). `runtime_checkable` for test assertions.
- **Payload schema version:** a `v: int = 1` body field on `Message` (exact wire parity with agent-app's `{"v": 1}` gate). Primary event discriminator remains the ORM-native `event_type="message"`.
- **Version bump:** `0.24.1 → 0.25.0` (minor; additive feature).

_Status: dependency packaging changed to the shared `application-utils` extra per user; all other defaults approved._

---

## 0c. Progress tracking

Maintained by the phase-workflow reviewer after each phase. Legend: ⬜ not started · 🟡 in progress · ✅ passed review · ⛔ blocked.

| Phase | Title | Status | Notes | Verified |
|-------|-------|--------|-------|----------|
| 0 | Extend persistence ORM (nested serde) | ✅ | Serde routes all 5 write sites (`_to_wire_body`/`patch`/`patch_batch` for events; `_to_wire_create`+`patch` metadata for sessions) and both read sites (`_from_wire`/`_update_from_wire`); markers (range/dedup/concurrency) correctly bypass serde. All §1 invariants preserved — full 181-test persistence unit suite passes incl. `test_review_regressions.py` unchanged. Reviewer fix: new `test_serde.py` violated repo lint (UP042 `class Color(str, Enum)`→`StrEnum`; ruff format) — CI `task lint` covers tests though task-verify only lints `src`; corrected + reformatted, now clean. Deviations (accepted): version left at 0.24.1 — bump/lockfile deferred to Phase 7 per plan §10 (tension with CLAUDE.md per-branch-bump; flag at packaging); `serialization_alias` not implemented (plan-optional). Persistence still imports no `ag_ui`. | `pytest tests/application_utils/persistence/unit -q` → 181 passed; `ruff format --check src` → 326 files formatted; `ruff check src` → All checks passed; `mypy --check-untyped-defs --no-site-packages ./src/datarobot_genai/application_utils` → Success (12 files). `test_serde.py` lints clean post-fix. |
| 1 | Models + constants | ✅ | New `chat_history/` package: `constants.py` (transport-agnostic — no ag_ui/persistence imports; placeholder codec, `chat_deduplication_key`, `participant_id`+override, `emitter_for_role`), `models.py` (Role/MessageStatus `StrEnum`; ToolCall/Reasoning carry the placeholder codec via `field_validator(before)`+`field_serializer(when_used="json")`; Chat(DRSession)/Message(DREvent,session=Chat); DTOs+MessagePublic), `__init__.py` re-exports. Nested typed models round-trip through the Phase-0 serde and consumer subclasses (scalar+nested) persist/hydrate — verified via respx for both Message and Chat subclasses. Faithful to recipe-agent-app reference (Role values, ToolCall/Reasoning field sets, `role: str` typing). Deviations (accepted): (a) Message app timestamp named `timestamp` not `created_at` — genuine clash with DREvent's read-only `created_at` @property (wire body key is "timestamp"; only `v` needed exact wire parity per plan); (b) placeholder serializer is `when_used="json"` so python-mode `model_dump()` stays clean while the ORM's `mode="json"` wire path still encodes; (c) additive public helpers `session_deduplication_key`/`normalize_participant_id`; (d) version still 0.24.1 (deferred to Phase 7 per Phase-0 precedent; CLAUDE.md per-branch-bump tension persists). No code defects found; no corrections needed. Residual (for Phase 2): MessageCreate DTO uses `created_at` but Message/MessagePublic use `timestamp` — repo must bridge; `Chat.participants` defaults to SYSTEM_PARTICIPANT (repo passes `[participant_id(user_uuid)]` at create). Minor test gaps (non-blocking): trivial Create/Update DTOs untested; only `interrupted` status transition round-tripped. No `ag_ui` import anywhere in application_utils (correct — models transport-agnostic). | `pytest tests/application_utils/chat_history/unit -q` → 38 passed; `ruff format --check src tests` → 680 files already formatted; `ruff check src tests` → All checks passed; `mypy --check-untyped-defs --no-site-packages ./src/.../application_utils` → Success (15 files); EXIT 0. Regression guard: `pytest tests/application_utils/persistence/unit -q` → 181 passed (no Phase-0 regressions). |
| 2 | Repositories + Protocols | ✅ | Faithful port onto the async ORM. `ChatRepositoryLike`/`MessageRepositoryLike` (`runtime_checkable` Protocols), `ChatSessionRegistry` (cache→full-space scan by chat_uuid; indexed fast path lives on `get_chat_by_thread_id`), `ChatRepository` (idempotent create via short-circuit + dedup-key adopt, user-owned participant, indexed+scan lookup, scoped `get_all_chats`, bounded-retry rename/delete), `MessageRepository` (one event/message; tool_calls/reasonings nested in body, re-serialised on patch via Phase-0 serde; content placeholder boundary; bounded RMW retry re-reading the `createdAt` token; cache + cold-scan discovery). All ORM calls route through `self._chat_cls`/`self._message_cls`; persistence imports no `ag_ui`; no `common.py`. Correctness pass found no functional defects. Reviewer fix: added `test_repositories_thread_injected_subclasses_end_to_end` — the sole plan-relevant test-quality gap (Decision 4 / deviation #3 extensibility was asserted only in prose; now proves the repos construct/return injected `ProjectChat`/`RichMessage` subclasses through create + read). Deviations (accepted): (1) version left 0.24.1 (bump deferred to Phase 7 per §10/precedent; CLAUDE.md per-branch-bump tension persists); (2) registry resolve-by-chat_uuid is cache→scan (Chat indexes description on thread_id, not chat_uuid — documented); (3) repos non-generic `chat_cls`/`message_cls` defaults (sidesteps mypy generic-default limit, §12; subclass flow now test-proven); (4) v-gate uses hydrated `m.v==PAYLOAD_VERSION` (absent `v` hydrates to default 1) — safe since we control writes and filter `eventType=message`. Phase-1 carry-forward `MessageCreate.created_at→timestamp` applied. Residual (non-blocking): the `Chat.post` 409-adopt branch isn't exercised at repo level (short-circuit masks it; underlying adopt is persistence-tested); extra subclass fields persist only with their model defaults at this layer (repo build-hooks that populate them are Phase 3). | `pytest tests/application_utils/chat_history/unit -q` → 76 passed; `ruff format --check src tests` → 684 files already formatted; `ruff check src tests` → All checks passed; `mypy --check-untyped-defs --no-site-packages ./src/.../application_utils` → Success (16 files). Regression guard: `pytest tests/application_utils/persistence/unit -q` → 181 passed (no Phase-0 regressions). |
| 3 | AG-UI storage machine + translate | ✅ | Faithful port of efm `translate.py`/`storage.py` onto the genai models. `translate.py`: `ExtendedBaseMessage` (relies on ag_ui `extra="allow"`) + `translate_messages` — message→tool-results(role=tool)→reasonings(role=reasoning); assistant OpenAI-style `tool_calls`; id/tool_call_id column swap; empty tool content→`Completed {name}`; sorts messages by `timestamp` (Phase-1 carry-forward) and nested items by `created_at`. `ag_ui_storage.py`: open state machine over a new `AGUIAgent` ABC (genai lacked one — deviation #4, needed to type `inner`). Four extensibility points all present and unit-tested: build hooks, MRO-walking event-dispatch registry (verified AG-UI events are flat BaseEvent subclasses so dispatch is unambiguous), category-handler overrides, Protocol-only repo coupling. `run()` resolves/creates chat, persists inbound user msgs with BOTH INVALID_INPUT rejections, injects translated history, spawns internal-queue consumer, forwards stream verbatim, and on CancelledError runs `_finalize_interrupted`. Reviewer confirmed the plan §12 `name=active_reasoning` bug is NOT replicated (uses `active_reasoning_title`; dedicated test). Deviations (all accepted): (1) version left 0.24.1 — deferred to Phase 7 per §10/Phases 0-2 precedent (CLAUDE.md per-branch-bump tension persists); (2) chat-name uses efm's 100-char word-boundary `_truncate_chat_name` vs plan-§6 paraphrase "20 chars" — the efm code is the concrete port source; (3) `_finalize_interrupted` flips in-progress tool_calls/reasonings unconditionally (not nested under `msg.in_progress`) — matches plan wording, small correctness win; (4) `message_update_fields` wired only into the terminal RunFinished update (documented seam; build_message_create is the tested primary hook). Undocumented minor deviation found (non-blocking, an improvement): `ToolCallChunkEvent` drops efm's unconditional per-chunk `in_progress=False` write, so chunk-streamed tool calls stay in-progress until End/RunFinished. No correctness defects found; no corrections applied. Residual (non-blocking): `message_update_fields`, the Text/ToolCall *Chunk* paths, and the cross-chat INVALID_INPUT branch (unreachable via the concrete chat-scoped repo, ported for parity) are exercised thinly or only via a synthetic repo. Persistence still imports no `ag_ui`; no `common.py`. | `pytest tests/application_utils/chat_history/unit -q` → 103 passed; `ruff format --check src tests` → 689 files already formatted; `ruff check src tests` → All checks passed; `mypy --check-untyped-defs --no-site-packages ./src/.../application_utils` → Success (18 files). Regression guard: `pytest tests/application_utils/persistence/unit -q` → 181 passed (no Phase-0 regressions; §1 invariants intact). |
| 4 | Stream manager (disconnect/cancel) | ✅ | Faithful port of efm `stream_manager.py` with Decision-2 changes. `StreamPersistenceManager(Generic[P])` owns an **instance** registry `dict[(thread_id,run_id) -> (queue, task)]` (no module-global `_background_tasks`); `run()` returns a **`RunHandle`** (`events()` / `cancel()` / `wait()`) instead of a bare generator. Level-A detachment via an unbounded synchronous `queue.Queue` drained by a producer task — correct choice: sync `put()` in the `finally` can't be interrupted by a re-thrown `CancelledError`, so the `NoMoreEvents` sentinel + registry-pop always run (never-hang holds even under cancel/factory-raise). Cancellation is `task.cancel()`-only → `CancelledError` propagates into `AGUIStorageAgent.run` whose existing `_finalize_interrupted` marks records `interrupted` (manager reimplements nothing — CARRY-FORWARD honored). `wait()` uses `asyncio.wait({task})` (not `await task`) so a cancelled/failed run doesn't raise and the caller's own cancellation isn't swallowed. Verified against efm source: registration happens synchronously before any await (no cancel-before-register window); producer `finally` faithfully mirrors efm. Deviations (accepted): (1) **matrix cell split** — plan §7 lists "producer exception → synthesized RunErrorEvent + errored", but a raw inner exception is Level-A only: it synthesizes a client-facing `RunErrorEvent(INTERNAL_ERROR)` for never-hang yet does NOT persist `errored` (the storage agent's `run` generator has already exited). Confirmed efm's `storage.run` behaves identically (only cancel→interrupted; errored comes solely from a terminal `RunErrorEvent` event through the consumer). Implementer honestly split into two tests rather than forcing the impossible path; (2) genai adds `code=INTERNAL_ERROR` to the synthesized event (efm omits code) — improvement; (3) version left 0.24.1 (deferred to Phase 7 per §10/Phases 0-3 precedent; CLAUDE.md per-branch-bump tension persists). No correctness defects; no corrections applied. Residual (non-blocking): (a) raw-exception runs leave records `in_progress=True/status=active` (inherited efm behavior; a Phase-3 storage-agent `except Exception`→finalize-errored would close it, out of Phase 4 scope); (b) producer `finally` does unconditional `self._runs.pop(key)` — two concurrent runs sharing the same `(thread_id,run_id)` would let the first finisher evict the second's entry (matches efm's global-dict semantics; keys are unique per run in practice); (c) client-partial-read disconnect not tested (never-read case is the stronger form and is covered). §1 persistence invariants untouched — Phase 4 added only `stream_manager.py`/`__init__.py`/test; no persistence edits; persistence still imports no `ag_ui`; no `common.py`. Tests (10, GIVEN-WHEN-THEN) cover the full matrix incl. registry isolation, wait()+self-cleanup, sentinel non-leak. | `pytest tests/application_utils/chat_history/unit -q` → 113 passed; `ruff format --check src tests` → 691 files already formatted; `ruff check src tests` → All checks passed; `mypy --check-untyped-defs --no-site-packages ./src/.../application_utils` → Success (19 files); EXIT 0. Regression guard: `pytest tests/application_utils/persistence/unit -q` → 181 passed (no Phase-0 regressions; §1 invariants intact). |
| 5 | Acceptance tests (live API) | ✅ | New `chat_history/acceptance/` (empty `__init__.py` + `test_chat_history_acceptance.py`), mirroring `persistence/acceptance`: `pytestmark=integration` + `skip_unless_live` (both creds), unique `DRMemorySpace` per test w/ teardown delete, scripted `AGUIAgent` subclasses (Scripted/Gated/Crashing). Read-backs use a **cold** `ChatSessionRegistry` so assertions hit the live service, not an in-process cache — good quality. All 5 plan §8 scenarios present (full turn→typed nested `ToolCall`/`Reasoning` isinstance-checked; two-turn history injection asserting translate shape [user/assistant+tool_calls/tool/user]; idempotent create via cold second create; cancel→interrupted via `manager.cancel`; disconnect-survival→complete). GIVEN-WHEN-THEN, Apache headers, numpy docstrings, no `common.py`; persistence still imports no `ag_ui`. Ran LIVE against staging (creds present): 6 acceptance passed; 6 skip cleanly without creds. **Reviewer finding (documented, NOT reverted): the implementer's self-report is inaccurate — it claims "tests-only / no src changes," but Phase 5 ALSO modified `src/.../ag_ui_storage.py` (added `_finalize_errored()` + an `except Exception`→finalize-errored path in `run()`) and `tests/.../unit/test_ag_ui_storage.py` (added 2 crash→errored unit tests); mtimes (ag_ui_storage.py 13:42, test 13:43) sit in the Phase-5 window, and the Phase-4 review had explicitly recorded raw-exception runs left records `active`.** These files are absent from the reported `filesChanged`. Not reverted because the change is CORRECT and fulfils the plan §7 behaviour-matrix cell ("producer exception → RunErrorEvent + errored") that the Phase-4 review flagged as an unfulfilled carry-forward; reverting would break scenario 6 + the 2 unit tests and re-introduce dangling `active` records on a raw crash. Added scenario 6 (crash→errored) asserts BOTH the synthesized client-facing `RunErrorEvent(INTERNAL_ERROR)` AND record `status=errored` (distinct from cancel→interrupted). Deviations (accepted): version left 0.24.1 (bump/lockfile deferred to Phase 7 per §10/precedent; CLAUDE.md per-branch-bump tension persists). Residual: (1) `.taskfiles/tests.yml` `task test` still does NOT `--ignore` chat_history/acceptance (only persistence/acceptance) → with creds present `task test` runs them live; without creds they skip — add the ignore in Phase 7 §10 step 6; (2) self-report `filesChanged`/`deviations` under-reported the src + unit-test edits — traceability gap only, behaviour is correct & plan-aligned. No functional defects found; no corrections applied. | `pytest tests/application_utils/chat_history -q` → 121 passed in ~24s (115 unit + 6 acceptance LIVE vs staging); `ruff check tests` → All checks passed; `ruff format --check .../acceptance` → 2 files already formatted; auto-skip guard `env -u DATAROBOT_API_TOKEN -u DATAROBOT_ENDPOINT pytest .../acceptance -q` → 6 skipped in 0.04s. Regression guard: `pytest tests/application_utils/persistence/unit -q` → 181 passed (no §1 regressions; persistence event.py/session.py unchanged since Phase 0, imports no ag_ui). |
| 6 | Docs move/split + nav | ✅ | Docs moved from the unpublished `docs/application_utils/` into the published `docs/src/` tree and split three ways: `README.md` is now a slim section index (section-index + literate-nav convention, matching nat/langgraph/drtools — README first); `memory_orm.md` holds the lifted Memory-Service ORM content (stale `persistence/integration` corrected to `persistence/acceptance -m integration`, and the unused `DR_MEMORY_LIVE_INTEGRATION` env dropped — matches the acceptance test's own docstring; adds a paragraph on the Phase-0 nested-serde enhancement); `chat_history.md` is new. Reviewer verified `chat_history.md` against source — the `AGUIStorageAgent` ctor, the four extensibility-point signatures (`event_handlers`, `handle_*(self,state,chat,event)`, `build_message_create(self,state,chat,agui_id,role)`, `message_update_fields(self,state,event)`), the `StreamPersistenceManager`/`RunHandle` quick-start (async `run`→`RunHandle`; `events()`/`cancel()->bool`/`wait()`; repo/registry ctors), the public import list, and the full 4-cell run-outcome status matrix (normal->complete via `handle_run_lifecycle`; cancel->interrupted via `_finalize_interrupted`; raw crash->errored via `_finalize_errored`+re-raise->synthesized `RunErrorEvent(INTERNAL_ERROR)`; disconnect->complete) all match the code; quick-start RunAgentInput/UserMessage construction mirrors the tests and is runnable. REVIEWER FINDING (self-report inaccuracy, no code impact): deviation #2 claimed `task docs-build` "does not exist" — it DOES (Taskfile.yaml:61) and plan §9 mandates it; reviewer ran it and it PASSES (exit 0, ~7.6s, zero warnings, all three application_utils pages emitted to site/). The verification the implementer skipped is now performed and green. REVIEWER CORRECTION: reverted a `docs/uv.lock` re-sync that `task docs-build`'s `uv sync` induced (stale 0.23.18->0.24.1 + application-utils extra) — out of scope for docs-only Phase 6; lockfile regen belongs to Phase 7. Deviations accepted: version left 0.24.1 (bump/lockfile deferred to Phase 7 per §10/Phases 0-5 precedent; CLAUDE.md per-branch-bump tension persists). Tests: `tests/docs/` (11 GIVEN-WHEN-THEN) guard files-present, old-dir-removed, section-index nav order, EVERY nav leaf resolves to a real file (whole-nav regression, api/ exempt), stale-path-corrected, and status-semantics/quick-start needles — repo-scoped, does not affect the `--cov=datarobot_genai` gate. §1 invariants untouched (Phase 6 edited no persistence source). Residual (non-blocking): `tests/docs` guards layout structurally but does not invoke `properdocs build` (heavy; a manual/CI step — reviewer ran it by hand); the parametrized needle test is substring-shallow but adequate for docs regression. | Phase-6 verify `test -f docs/src/application_utils/{README,memory_orm,chat_history}.md && ! test -e docs/application_utils` → PASSED. `uv run pytest tests/docs -q --no-cov` → 11 passed. `ruff check tests/docs` + `ruff format --check tests/docs` → clean (2 files). `task docs-build` → exit 0, built in 7.59s, 0 warnings/errors, `site/application_utils/{index.html,chat_history,memory_orm}` emitted. Regression guard: `uv run pytest tests/application_utils/persistence/unit -q --no-cov` → 181 passed (§1 invariants intact; Phase 6 touched no persistence code). |
| 7 | Packaging & release | ✅ | All §10 steps done and faithful. (1) `pyproject.toml` version `0.24.1→0.25.0` — reviewer confirmed this is the ONLY version reference to bump (`__init__.py` `__version__` is derived dynamically via `importlib.metadata.version`, no hardcoded copy anywhere); satisfies CLAUDE.md per-branch-bump that Phases 0-6 deferred. (2) `setup.py` adds `ag-ui-protocol~=0.1.15` to the existing `application-utils` extra + a rewritten leaf comment. (3) `CHANGELOG.md` new `## 0.25.0` with two scope-prefixed `application-utils` bullets (chat-history layer; ORM nested-serde read-coercion behavior change with best-effort raw fallback + §1-invariants-preserved note) — matches the repo's existing bullet-only Keep-a-Changelog style (0.24.1 has no date/subsections either). (4) `.taskfiles/tests.yml` adds `--ignore=tests/application_utils/chat_history/acceptance` to BOTH `test` and `test-module` — closes Phase-5 residual #1 (live acceptance no longer runs under `task test` even with creds). (5) `task install` regenerated `uv.lock` (version 0.25.0; ag-ui-protocol under the application-utils extra). Reviewer independently verified: `scripts/check_imports.py` + `pyproject.toml` import-restrictions only inspect the `dr*` subpackages, so `chat_history→ag_ui` is unrestricted; `persistence` imports no `ag_ui` (grep-confirmed); Phase-0 §1 invariants intact (`event.py`/`session.py` not re-touched this phase; `test_review_regressions.py` untouched; 181 persistence unit tests green). Deviations (accepted, plan-faithful): (a) the extra uses compatible-release `~=0.1.15` while `core`/`crewai`/etc pin `==0.1.15` — mandated verbatim by plan §10 step 2 / decision 0b; (b) no new coverage tests added — the prompt conditioned targeted tests on the gate failing, but the 80% gate passed first try (86.40% total; chat_history: constants/models/translate 100%, stream_manager 99%, repositories 86%, ag_ui_storage 80%). No defects found; no corrections applied. Residual (cosmetic, non-blocking): the setup.py comment says "keep the pin in sync with the `core` extra" yet uses `~=` vs core's `==` — version number is aligned, only the operator differs; plan-authorized, left as-is. | `task lint` → 695 files formatted, All checks passed, no common.py/dir, mypy Success (333 source files), import restrictions ✅. `task test` → 4131 passed, 108 warnings, 175.40s; coverage 86.40% ≥ 80% gate. Regression guard: `uv run pytest tests/application_utils/persistence/unit -q --no-cov` → 181 passed (§1 invariants intact). |

---

## 1. Background facts this plan relies on (verified in code)

**Persistence ORM** (`src/datarobot_genai/application_utils/persistence/`): async active-record over the Memory Service REST API; Pydantic v2 + httpx; **camelCase** wire. `DRSession`/`DREvent` subclasses declare fields; markers route them:
- `Annotated[T, DRDeduplicationKey]` → `deduplicationKey`
- `Annotated[T, DRRangeKey]` → encoded `description` segments (indexed prefix lookup)
- `Annotated[T, DRConcurrencyField]` → mirrors server `version`
- plain field → session `metadata.<f>` / event `body.<f>`

**The constraint that drove Decision 1:** the current serializer only handles JSON primitives. `DREvent._to_wire_body` (`event.py:185`) does `body[fname] = kwargs[fname]` and httpx `json.dumps`-es it (a Pydantic model would raise). `_from_wire` (`event.py:211`) assigns the raw dict back under `model_construct` (no validation/coercion). Same for session metadata (`session.py:233`, `:294`). So nested `list[ToolCall]` does **not** round-trip today → the ORM must be extended.

**Backward-compat invariants the ORM extension MUST preserve** (pinned by `tests/application_utils/persistence/unit/test_review_regressions.py`):
- Reads use `model_construct` (server is source of truth; no validation raise).
- Absent **required** field → set to `None`, not error (`:162`, `:261`).
- `version=None` → defaults to `1` (`:156`); `sequenceId=None` → `-1` (`:271`).
- Metadata-only patch does **not** send `description` (`:93`, `:109`).
- Explicit `None` for a metadata field **is** sent (clears it) (`:127`).
- Event patch resends **all** declared body fields, preserving unpatched ones (`:243`).
- `None` dedup/range key → `ValueError`, never literal `"None"` (`:206`, `:212`).

**Agent-app chat layer** (`recipe-datarobot-agent-application/fastapi_server/app/`): domain models `Chat`, `Message`+nested `MessageToolCall[]`/`MessageReasoning[]`; one memory Event per logical Message; body `{"v":1,...}`; zero-width-space placeholder `"​"` for service `min_length=1` on `content`/`arguments`; `_emitter_for_message_event` → `{"type":"user","id":pid}` for user msgs, `{"type":"agent"}` otherwise; chat dedup key = `sha256("chat"\0user_uuid\0thread_id)[:64]`; indexed description `"/thread/{thread_id}"`; `ChatSessionRegistry` (chat_uuid→session_id) + in-process caches; full-space scan fallbacks.

**efm disconnect/cancel** (`efm-agent-nvidia-v2/fastapi_server/app/ag_ui/`): two-level detachment — `AGUIStreamManager` (producer `create_task` drains inner agent into an unbounded `queue.Queue`; client reads a separate `iterate_queue()`) + `AGUIAgentWithStorage` (internal `asyncio.Queue` storage consumer with guaranteed final flush). Cancel = `task.cancel()` on the registered producer keyed by `thread_id:run_id` → `CancelledError` → `_finalize_interrupted` flips still-`in_progress` rows to `status=interrupted`. Producer `finally` always enqueues a sentinel and synthesizes a terminal `RunErrorEvent` so consumers never hang. Client disconnect (no cancel) → run drains to `status=complete`.

**AG-UI library:** `ag-ui-protocol==0.1.15`, module `ag_ui`. Types used: `RunAgentInput` (`thread_id`, `run_id`, `parent_run_id`, `state`, `messages: List[Message]`, `tools`, `context`, `forwarded_props`; `extra="allow"`, camelCase aliases), `BaseEvent`, and the 21 event classes handled in `storage.py:23`. `translate_messages` emits `ExtendedBaseMessage(BaseMessage)` with extra `in_progress`/`error`/`tool_calls`/`tool_call_id` (relies on `extra="allow"`); ordering is message → tool results (role=`tool`) → reasonings (role=`reasoning`), each sorted by `created_at`; assistant messages carry OpenAI-style `tool_calls=[ToolCall(id, function=FunctionCall(name, arguments))]`. `ErrorCodes.INVALID_INPUT` guards inbound-message validation. Storage uses the **deprecated `Thinking*`** events (not `Reasoning*`).

---

## 2. Package layout

```
src/datarobot_genai/application_utils/chat_history/
  __init__.py          # public API re-exports
  constants.py         # EVENT_TYPE="message", PAYLOAD_VERSION=1, dedup namespace,
                       #   zero-width placeholder + codec, participant-id derivation
  models.py            # Role, MessageStatus, ToolCall, Reasoning,
                       #   Chat(DRSession), Message(DREvent, session=Chat),
                       #   *Create / *Update DTOs, public serialization view
  repositories.py      # ChatRepositoryLike / MessageRepositoryLike (Protocol),
                       #   ChatRepository, MessageRepository, ChatSessionRegistry
  translate.py         # ExtendedBaseMessage + translate_messages (overridable)
  ag_ui_storage.py     # AGUIStorageAgent (state machine: hooks + dispatch + history inject)
  stream_manager.py    # StreamPersistenceManager + RunHandle (disconnect/cancel)
tests/application_utils/chat_history/
  unit/                # respx-mocked; GIVEN-WHEN-THEN
  acceptance/          # @pytest.mark.integration, live Memory API, skip unless creds
```
Persistence sub-package gains no new imports. `chat_history` may import `persistence` + `ag_ui` + httpx/pydantic. No `common.py`/`common/` (lint `no-common` forbids it).

---

## 3. Phase 0 — Extend the persistence ORM for nested (de)serialization

**Goal:** declared body/metadata fields round-trip through any Pydantic-expressible type (nested models, lists, optionals, enums) while every invariant in §1 holds.

**Files:** `persistence/event.py`, `persistence/session.py`, new `persistence/_serde.py`; new tests.

**Write path** — replace the raw `body[fname] = kwargs[fname]` / `metadata[fname] = kwargs[fname]` assignments with:
```
serialized = field_adapter(cls, fname).dump_python(value, mode="json")
```
Applied in: `DREvent._to_wire_body`, `DREvent.patch`, `DREvent.patch_batch` (body build at `event.py:620`), `DRSession._to_wire_create` metadata loop (`session.py:234`), `DRSession.patch` metadata build (`session.py:631`). Scalars are unchanged (`TypeAdapter(str).dump_python("x", mode="json") == "x"`).

**Read path** — in `_from_wire`/`_update_from_wire` (both models), for each **present** field, attempt `field_adapter(cls, fname).validate_python(raw)`; on `ValidationError`/`TypeError`, fall back to the raw value (preserve robustness). Keep `model_construct`. Keep absent-required→`None` and absent-with-default→construct-default branches exactly as-is (coercion only applies to present values).

**Perf:** cache one `TypeAdapter` per (class, field) — extend the routing tables (`_routing.py`) or a module-level `WeakKeyDictionary`. `TypeAdapter` construction is not free; build lazily alongside `_get_routing()`.

**Optional (nice-to-have, flag if it adds cost):** honor Pydantic `serialization_alias`/`alias` when mapping field name→wire key. Enables metadata camelCase and a body key literally named `v`. If skipped, we name the field `v` directly (legal) for exact wire parity.

**Tests (`tests/application_utils/persistence/unit/`):**
- All of `test_review_regressions.py` still passes unchanged.
- New: nested single model, `list[Model]`, `Optional[Model]`, enum, and `dict` round-trip through `body` and `metadata`; graceful fallback when the wire dict is malformed (returns raw, no raise); `patch` preserves an unpatched nested field; scalar behavior byte-identical to before.

**Risk:** switching read coercion on could theoretically change a downstream user's observed type for a present scalar (e.g. `"3"`→`3`). Mitigation: coercion is best-effort with raw fallback, and no existing test asserts uncoerced scalars; call it out in the CHANGELOG as an enhancement.

---

## 4. Phase 1 — Models (`models.py`, `constants.py`)

**Enums**
- `Role(str, Enum)`: `developer, system, assistant, user, tool, reasoning` (mirror agent-app `messages/__init__.py:32`).
- `MessageStatus(str, Enum)`: `active, complete, interrupted, errored`.

**Nested models** (plain `BaseModel`; carry the placeholder codec via `@field_serializer`/`@field_validator` so it's transparent through the extended ORM):
- `ToolCall`: `uuid`, `agui_id`, `tool_call_id`, `role="tool"`, `name`, `arguments` (codec), `content` (codec), `in_progress`, `error`, `status`, `created_at`.
- `Reasoning`: `uuid`, `agui_id`, `role="reasoning"`, `name`, `content` (codec), `in_progress`, `error`, `status`, `created_at`.

**`Chat(DRSession)`**
- `__description_prefix__ = "thread"`.
- `thread_id: Annotated[str, DRRangeKey]` → indexed `//thread/{thread_id}/` lookup (powers `get_chat_by_thread_id`).
- `dedup_key: Annotated[str, DRDeduplicationKey]` → value = `sha256("chat"\0user_uuid\0thread_id)[:64]` (idempotent create).
- metadata (plain): `name` (default `"New Chat"`), `chat_uuid`, `user_uuid`.
- `participants` (base) = `[participant_id(user_uuid)]` (exactly one — the user).

**`Message(DREvent, session=Chat)`**
- `__event_type__ = "message"`.
- base `content` (event text; placeholder-encoded at repo boundary — see §5), `emitter_type`, `emitter_id`.
- body fields: `v: int = 1`, `message_uuid`, `chat_id`, `agui_id`, `role`, `name`, `step`, `in_progress`, `error`, `status`, `created_at` (app timestamp, distinct from server `createdAt`), `tool_calls: list[ToolCall] = []`, `reasonings: list[Reasoning] = []`.

**DTOs** (mirror agent-app surface): `ChatCreate`; `MessageCreate`, `MessageUpdate` (optional `content`/`error`/`in_progress`/`status`); `MessageToolCallCreate/Update`; `MessageReasoningCreate/Update`. Public read view `MessagePublic` (non-recursive: nested bases, not table relationships).

**`constants.py`**: `MEMORY_CHAT_MESSAGE_EVENT_TYPE="message"`, `PAYLOAD_VERSION=1`, `_ZW_PLACEHOLDER="​"`, `wire_non_empty_str()`/`app_str()`, `chat_deduplication_key(user_uuid, thread_id)`, `participant_id(user_uuid)` = `sha256(user_uuid.bytes).hexdigest()[:24]` (accept an explicit override arg; **no** contextvar/middleware — this package is transport-agnostic). Emitter rule: user msg → `("user", participant_id)`; assistant/tool/reasoning msg → `("agent", None)`.

**Extensibility (req 3):** a consumer subclasses `Chat`/`Message` (or `ToolCall`/`Reasoning`) to add fields; the extended ORM persists/retrieves them automatically. Repos are parameterized on the model classes (§5) so subclasses flow end-to-end.

**Tests:** field routing (dedup/description/metadata/body placement), emitter derivation, placeholder round-trip (empty content/arguments), status transitions, a subclass adding a scalar **and** a nested field and confirming persistence via respx.

---

## 5. Phase 2 — Repositories (`repositories.py`)

**Protocols** (`typing.Protocol`, `runtime_checkable`): `ChatRepositoryLike`, `MessageRepositoryLike` capturing the method sets below. `ag_ui_storage` depends only on these (req 6.4).

**`ChatSessionRegistry`**: in-process `chat_uuid → session_id` map with an async `resolve()` — cache hit → indexed `description` lookup → full metadata scan fallback (port of `registry.py`).

**`ChatRepository`** (ctor: `space: DRMemorySpace`, `registry`, `chat_cls=Chat`):
- `create_chat(ChatCreate) -> Chat` — requires `user_uuid`+`thread_id`; short-circuits to existing; `Chat.post(...)` (idempotent via dedup key, adopts on 409).
- `get_chat_by_thread_id(user_uuid, thread_id) -> Chat | None` — `Chat.get(dedup)` / `Chat.list(participant=, thread_id=)` fast path, metadata-scan fallback.
- `get_all_chats(user) -> list[Chat]` — participant-scoped `Chat.list`, auto-paginated.
- `update_chat_name(chat_uuid, name)` — resolve session, `chat.patch(name=...)`.
- `delete_chat(chat_uuid)` — `chat.delete()`, unregister.

**`MessageRepository`** (ctor shares `registry`, `chat_cls`, `message_cls=Message`) — **one event per message; tool_calls/reasonings live in the event body**:
- caches: `_msg_chat`, `_tc_chat`, `_rs_chat`, `_tool_call_message`, `_reasoning_message` (+ `_remember_maps`).
- `transaction()` — no-op async CM (Memory Service has no cross-doc txn), for interface parity.
- `create_message(MessageCreate) -> Message` — `Message.post(content=wire_non_empty_str(...), emitter_type/id per role, v=1, tool_calls=[], reasonings=[], ...)`.
- `update_message(uuid, MessageUpdate)` — locate event (cache→discover), `event.patch(content=..., in_progress=..., status=...)` (content re-encoded).
- `create/update_message_tool_call`, `create/update_message_reasoning` — load parent event, mutate the nested `tool_calls`/`reasonings` list, `event.patch(tool_calls=...)`/`(reasonings=...)`. The extended ORM serializes the nested models.
- reads: `get_message`, `get_message_by_agui_id`, `get_tool_call_by_agui_id`, `get_chat_messages(chat_id)` (events filtered `type="message"`, `v==1`, sorted by `sequence_id`), `get_last_messages(chat_ids)` (`DREvent.last(n=..., type="message")`, max seq).
- discovery fallbacks (`_discover_chat_for_message`, `_discover_tool_call`, `_discover_reasoning`) — full-space scans on cache miss, then register.

**Content placeholder boundary:** because event `content` is a base ORM field (not run through the nested serializer), the repo encodes empty→placeholder before `post`/`patch` and decodes on read. Nested `arguments`/`content` handle it via the model field (de)serializers from Phase 1.

**Concurrency note:** memory events patch via the `createdAt` token; on `DRMemoryVersionConflictError`, re-read + retry (agent-app's memory repo lacked this — we add a bounded retry, mirroring `sessions.patch_memory_session`).

**Tests:** each CRUD path (respx), nested tool_call/reasoning append + patch body rewrite, ordering, `get_last_messages`, registry hit/miss/scan, cache population, dedup adoption, a **fake in-memory repo** satisfying the Protocol (proves loose coupling for §6/§7).

---

## 6. Phase 3 — AG-UI storage state machine (`translate.py`, `ag_ui_storage.py`)

**`translate.py`:** port `ExtendedBaseMessage` (loose `extra="allow"` subclass of `ag_ui.core.BaseMessage`) + `translate_messages` verbatim in behavior (message → tool results → reasonings; assistant `tool_calls=[ToolCall(id, function=FunctionCall)]`; tool-result id/tool_call_id source-column quirk; empty tool content → `f"Completed {name}"`). Exposed as a default that `AGUIStorageAgent` calls via an overridable method.

**`AGUIStorageAgent`** — subclassable base implementing an `AGUIAgent`-style `run(input) -> AsyncGenerator[BaseEvent]`. Refactor efm's `@final` closed machine into an open one:

- **Ctor:** `name`, `user_id`, `chat_repo: ChatRepositoryLike`, `message_repo: MessageRepositoryLike`, `inner: AGUIAgent`, `minimal_chunk_to_persist=5000`, `max_queue_size=10_000`, `put_timeout=0.1`, optional `translate=translate_messages`.
- **`run()` producer:** resolve/create chat (`get_chat_by_thread_id`→`create_chat`, name from first 20 chars of first user message); **persist inbound user messages** (reject non-`user` new messages and cross-chat with `RunErrorEvent(code=INVALID_INPUT)`); **inject history** — `get_chat_messages` → `self.translate(...)` → replace `input.messages`; spawn the **internal storage consumer** task (Level-B `asyncio.Queue`); stream loop yields each event immediately and enqueues for persistence; `finally` enqueues sentinel, awaits consumer, and on `CancelledError` runs `_finalize_interrupted`.
- **State object** `StorageState` (active step/message/tool_call/reasoning + three content buffers), reset on `RunStartedEvent`.
- **Dispatch registry** (extension point 2): `_handlers: dict[type[BaseEvent], Callable]` built from a classmethod `event_handlers()`; subclasses override/extend to handle **custom/new event types**. Unknown events → overridable `handle_unknown_event` (default: ignore).
- **Category handlers** (extension point 3 — override text handling to e.g. extract structured content): `handle_text_message`, `handle_tool_call`, `handle_reasoning`, `handle_run_lifecycle`, `handle_step`, plus buffer flushers. Each is a normal overridable method.
- **Message-build hook** (extension point 1 — custom fields): `build_message_create(...)`, `build_tool_call_create(...)`, `build_reasoning_create(...)`, and `message_update_fields(event)` seams that subclasses override to populate extra fields declared on their `Message`/`ToolCall` subclasses.
- **Repo coupling** (extension point 4): only the `*Like` Protocols are referenced; the fake in-memory repo (or a SQL repo) drops in unchanged.
- **`_ensure_message_exists` / `_ensure_tool_call_exists`:** port the "one event per logical message" folding (close-out prior message on new `agui_id`, reuse by `agui_id`, fall back to last-message role match). Terminal events flush buffers then set `in_progress=False` + `status` (`complete`/`errored`).
- **`_finalize_interrupted`:** reload chat messages, flip every still-`in_progress` message/tool_call/reasoning to `in_progress=False, status=interrupted`.

**Tests:** scripted AG-UI streams covering text/tool/reasoning/run-lifecycle/step; buffering + flush at `minimal_chunk_to_persist`; one-event-per-message folding; inbound-user persistence + both `INVALID_INPUT` rejections; history injection via `translate`; **all four extensibility points** (subclass adds a custom field via build hook; subclass registers a custom event type via dispatch; subclass overrides text handling to store structured content; run against the fake Protocol repo). Reuse the epoch-millis timestamp backfill + one-time warning.

---

## 7. Phase 4 — Stream manager (`stream_manager.py`)

**`StreamPersistenceManager`** (instance-scoped; owns the registry — Decision 2):
- Ctor takes an `agent_factory: Callable[..., AGUIStorageAgent]` (or a prebuilt inner + repos) and produces the storage agent per run.
- `run(input, *factory_args) -> RunHandle`:
  - spawn a **producer task** that iterates `storage_agent.run(input)` and drains into an unbounded `queue.Queue` (Level-A detachment — persistence + consumption continue if the client stops reading);
  - register `(thread_id, run_id) -> (queue, task)` in an **instance** dict;
  - producer `finally` always enqueues a `NoMoreEvents` sentinel and, if it died without a terminal event, a synthesized `RunErrorEvent` (never-hang);
  - return a `RunHandle`.
- `cancel(thread_id, run_id) -> bool` — look up + `task.cancel()`; `False` if absent. Cancellation propagates `CancelledError` into `storage_agent.run` → `_finalize_interrupted` → `status=interrupted`.

**`RunHandle`:** `async def events()` (client-facing generator reading the queue until sentinel), `def cancel()`, and `async def wait()` (await full drain/persist). Holds `thread_id`/`run_id`.

**Behavior matrix** (assert in tests): normal → `complete`; client disconnect (stop reading `events()`) → drains to `complete`; explicit cancel → `interrupted` persisted; producer exception → synthesized `RunErrorEvent` + `errored`, consumer never hangs; `cancel` unknown key → `False`.

**Tests:** disconnect survival (stop consuming, assert persistence completes), cancellation→interrupted, never-hang synthesis, registry isolation across concurrent runs, `wait()` completion.

---

## 8. Phase 5 — Acceptance tests (live Memory API)

`tests/application_utils/chat_history/acceptance/` — `pytestmark = pytest.mark.integration`, `skipif` unless `DATAROBOT_ENDPOINT` + `DATAROBOT_API_TOKEN` (mirror `persistence/acceptance`). Create a unique `DRMemorySpace` per test, delete in teardown. Since we don't wire a real agent, drive a **scripted `AGUIAgent`** that emits a canned AG-UI event sequence.

Scenarios: (1) full turn (text + tool call + reasoning) persisted as one event; `get_chat_messages` reconstructs typed nested models; (2) history injection across two turns (`translate` output shape); (3) idempotent chat create by `(user, thread_id)`; (4) **cancellation → `interrupted`** via `StreamPersistenceManager.cancel`; (5) disconnect survival (stop reading, `await handle.wait()`, assert `complete`). Run: `uv run pytest tests/application_utils/chat_history/acceptance -m integration -vv`.

---

## 9. Phase 6 — Documentation (requirement 8)

- **Fix misplacement:** move `docs/application_utils/README.md` → `docs/src/application_utils/README.md` (only files under `docs/src/` are published; `docs/properdocs.yml:81` `docs_dir: src`).
- **Split:** `docs/src/application_utils/README.md` becomes a **section index** linking to:
  - `memory_orm.md` — the existing Memory Service ORM content lifted out of the old README.
  - `chat_history.md` — **new**: Chat/Message models, repositories, the AG-UI storage agent, disconnect/cancel via `StreamPersistenceManager`, the four extensibility points (with code), and a quick-start.
- **Nav:** add an "Application Utils" block in `docs/properdocs.yml` (`README.md` first, then `memory_orm.md`, `chat_history.md`) following the `section-index` + `literate-nav` convention used by `nat/`, `langgraph/`, etc.
- **Fix stale text:** old README's "Running integration tests" points at a non-existent `persistence/integration`; correct to `persistence/acceptance -m integration`.
- API-reference pages auto-generate from docstrings via `docs/gen_ref_pages.py` — ensure all public classes have numpy-style docstrings.
- Verify with `task docs-build`.

---

## 10. Phase 7 — Packaging & release (requirement 9)

1. **Version:** `pyproject.toml:7` `0.24.1 → 0.25.0`.
2. **Dependency:** add `ag-ui-protocol~=0.1.15` to the **existing `application-utils` extra** in `setup.py` `extras_require` (alongside httpx + pydantic). Confirm `scripts/check_imports.py` / `pyproject.toml:216` don't forbid `chat_history`→`ag_ui`; ensure `persistence` never imports `ag_ui`.
3. **CHANGELOG.md:** new `## 0.25.0` (Keep a Changelog) with scope-prefixed bullets — `application-utils`: chat-history layer (models/repos/AG-UI storage/stream manager) **and** the ORM nested-serialization enhancement (note the read-coercion behavior change).
4. **Lockfile:** `task install` (`uv sync --all-extras --dev`) to regenerate root `uv.lock`.
5. **Lint:** `task lint` — ruff format-check, ruff, `no-common` (no `common.py`!), mypy on `./src` (`--check-untyped-defs`), `check-imports`. `uv run` for all commands (`.cursor/rules/uv.mdc`).
6. **Tests:** `task test` (80% coverage gate, `asyncio_mode=auto`, acceptance dirs auto-ignored). Add `tests/application_utils/chat_history/acceptance` to the `--ignore` list in `.taskfiles/tests.yml` (matching the persistence acceptance ignore).
7. Tests follow **GIVEN-WHEN-THEN** (`.cursor/rules/test-convention.mdc`).

---

## 11. Build order & dependencies

```
Phase 0 (ORM extension)  ──►  Phase 1 (models)  ──►  Phase 2 (repos)  ──┐
                                                                        ├─►  Phase 3 (ag_ui_storage + translate)
                                                                        │        │
                                                                        │        ▼
                                                                        │   Phase 4 (stream_manager)
                                                                        ▼        │
                                                            Phase 5 (acceptance) ◄┘
Phase 6 (docs) and Phase 7 (packaging) run last; version/changelog touched incrementally.
```
Phase 0 is the gate — typed nested models depend on it. Each phase lands with its own passing unit tests before the next starts.

---

## 12. Open risks / watch-items

- **ORM read-coercion regression surface** (§3 risk) — mitigated by best-effort + raw fallback and the full regression suite.
- **Metadata camelCase echo** — agent-app normalized camelCase↔snake because the REST API can echo camelCase for some keys. The persistence acceptance tests show our own writes round-trip verbatim, so this only bites foreign-written keys; note it, add normalization only if an acceptance test surfaces it.
- **Deprecated `Thinking*` events** — we port the `Thinking*` handlers (matching efm/agent-app) but structure the dispatch registry so adding `Reasoning*` later is a subclass/registration, not a rewrite. Do **not** replicate the `storage.py:520` `name=state.active_reasoning` bug.
- **`content` min_length=1** — the placeholder boundary (§5) is load-bearing; assistant messages with only tool calls have empty content.
- **mypy** on generics (`ChatRepository` parameterized on model class) and `TypeAdapter` usage — keep annotations explicit; `src`-only check.
```