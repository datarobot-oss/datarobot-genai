# Config injection migration plan (config.py as the single source of truth)

Working plan for making the application's `config.py` the absolute authority for
LLM configuration across `af-component-agent`, `datarobot-genai`, and
`af-component-llm`. Written 2026-07-20, rewritten 2026-07-21 after nailing down
the corrected object model. Prototype (Step 1a) lives on the genai branch
`mattn/BUZZOK-31738-authority`.

Re-read this at the start of each work session. The steps are scoped so each can
be planned and executed on its own.

---

## The problem in one paragraph

Every `af-component-*` ships a `config.py` with `class Config(DataRobotAppFrameworkBaseSettings)`.
Users treat that as the one authoritative config, because that base class is the
only thing guaranteeing a variable resolves the same way across local dev, a
deployment, and a deployment with dynamic env vars. `datarobot-genai` quietly
deviated: it built its OWN `Config` and reads it independently, so the app's
`config.py` is never consulted for LLM routing. When a user hardcodes a value in
`config.py` without also exporting the matching env var (the "hit it with a
hammer" case), genai's private config never sees it and the change is silently
ignored.

## The deeper root cause (why the naive fix is wrong)

`LLMConfig` in `core/config.py` was designed to be a per-LLM value object you
have MANY of. The proof is still in the tree:
`build_litellm_router(primary: LLMConfig, fallbacks: list[LLMConfig])` in
`core/router.py` expects a `list[LLMConfig]`.

Then `LLMConfig` got inherited as a mixin into the singleton
`class Config(LLMConfig, DataRobotAppFrameworkBaseSettings)`. That collapse does
two kinds of damage:

1. It reduces `LLMConfig` to exactly one instance (the singleton app config), so
   a `list[LLMConfig]` is impossible. Fallbacks and multiple LLMs are dead on
   arrival.
2. In the collapse, the two true globals (`datarobot_endpoint`,
   `datarobot_api_token`) landed on `LLMConfig` instead of on the global
   `Config`, and `datarobot_endpoint` ended up duplicated on `Config`.

So the mixin is the bug, not the fix. Any plan that re-adds
`class Config(LLMConfig, ...)` re-commits the original mistake. This is why the
earlier version of this plan, which proposed exactly that mixin, was discarded.

## The naming model (the core insight)

Only two field names are true, fixed, ecosystem-wide absolutes, enforced
identically in Python, Terraform, Pulumi, and LiteLLM:

- `datarobot_endpoint`
- `datarobot_api_token`

Everything else is namespaced by the LLM component instance name. `llm` is just
the DEFAULT instance name, never an absolute. `LLM_DEPLOYMENT_ID` and
`LLM_DEFAULT_MODEL` are the default instance's env vars, not canonical field
names. A user runs `dr component add llm`, names the instance (say "bob"), and
gets `BOB_DEPLOYMENT_ID` and `BOB_DEFAULT_MODEL`, repeatable infinitely. You
cannot hardcode "bob" into a library, so the library cannot treat any per-LLM
name as absolute.

`use_datarobot_llm_gateway` is per-LLM too (`{name}_use_datarobot_llm_gateway`),
because two LLMs in one app can route differently. This also satisfies Anatolii's
open ticket to namespace `USE_DATAROBOT_LLM_GATEWAY`.

The library MAY default. `get_llm()` with no arguments resolves the default
instance ("llm") as a convenience, and defaulting is normal library behavior.
What it must never do is treat the default as the only option. If a flow uses two
LLMs, the user is responsible for disambiguating with `get_llm(name="bob")`. That
is the only answer that can work, and the responsibility rightly sits with the
user.

## The corrected object model

Two distinct objects, cleanly separated.

```python
# The app's config.py — the single global authority (exactly one of these)
class Config(DataRobotAppFrameworkBaseSettings):
    datarobot_endpoint: str = "https://app.datarobot.com/api/v2"  # true absolute
    datarobot_api_token: str | None = None                        # true absolute

    # per-LLM fields, flat and namespaced by instance ("llm" is the default name)
    llm_default_model: str = "..."
    llm_deployment_id: str | None = None
    llm_use_datarobot_llm_gateway: bool = True
    # bob_default_model, bob_deployment_id, bob_use_datarobot_llm_gateway, ...

    def resolve_llm(self, name: str = "llm") -> "LLMConfig":
        # map {name}_* flat fields + the two globals into an LLMConfig
        ...

# Per-LLM value object — you can have many
class LLMConfig(BaseModel):
    model: str | None = None
    deployment_id: str | None = None
    nim_deployment_id: str | None = None
    use_datarobot_llm_gateway: bool = True
    # endpoint/token are owned by Config; see the interim note below
```

The injection seam gives genai a name-aware resolver, not the app `Config`
itself. genai only ever receives a fully-formed `LLMConfig` and never sees
`llm_*` or the app `Config` shape. It ships default-only first; the `name=`
extension is built later.

## Why it is possible under NAT (the mechanism)

NAT's `load_workflow` -> `load_config` (`nat/runtime/loader.py`) does two ordered
steps:

1. `discover_and_register_plugins(CONFIG_OBJECT)` imports every `nat.plugins`
   entry point. The app registers itself here (recipe `agent/pyproject.toml`:
   `[project.entry-points.'nat.plugins'] langgraph_agent = "agent.register"`),
   which imports `agent/__init__.py` -> `from agent.config import Config`.
2. `validate_schema(config_yaml, Config)` eagerly instantiates every `_type`
   block. The empty `datarobot-llm-component` block fires genai's
   `default_factory` fields, which read config.

The app package is imported in step 1, before step 2 reads config. That import
ordering is the injection seam. It holds identically in local dev (`task dev`)
and in a deployment (`start.sh` runs `nat dragent serve --config_file workflow.yaml`
under gunicorn; each worker runs the same load sequence).

---

## Locked decisions (2026-07-21)

- **Enforcement: type + runtime check.** genai requires the injected provider to
  return an `LLMConfig` (isinstance guarantee), plus a light runtime assertion
  that the required fields exist and are correctly typed, raising a clear error
  otherwise. Belt and suspenders, and it directly answers Anatolii's strict-typing
  concern.
- **genai's private `Config()` stays as the no-provider fallback.** Single flow,
  no env-var gate (removed 2026-07-21 at Matt's call). If the app registers a
  provider, genai reads it; if not (standalone genai with no app around it), genai
  builds its own env-only `Config()` exactly as before. Nothing breaks if the user
  defines nothing.
- **Multi-LLM is designed-for now, built late.** The naming model must stay
  namespace-capable from the start, but `get_llm(name=...)` lands after the
  single-default path is proven.
- **Router / fallbacks wrapper is a downstream cleanup.** The real defect is
  `LLMConfig`-collapsed-to-a-singleton. Restoring it as a value object un-breaks
  `fallbacks` for free. Whether to keep the wrapper or move to LiteLLM's native
  router (Anatolii's lean) is decided later, not a gate on this plan.
- **endpoint/token on the resolved `LLMConfig`: interim populate-from-global.**
  For now `resolve_llm()` copies the two globals onto the resolved `LLMConfig` so
  it is self-contained for the client builder. A later cleanup keeps them strictly
  off `LLMConfig` and has the builder take `(llm_config, global_config)`.

---

## The plan, structured on the four steps

### Step 1 — Inject and elevate config.py to authoritative

Make the app's `config.py` the thing genai actually reads, end to end.

- **1a. genai injection seam (DONE, branch `mattn/BUZZOK-31738-authority`).**
  `register_config_provider()` and `resolve_config()`. Routed the user-intent
  helpers (`default_api_key`, `default_model_name`,
  `default_use_datarobot_llm_gateway`, `default_llm_deployment_id`,
  `default_nim_deployment_id`) and the three `get_llm()` facades through
  `resolve_config()`. Config tests pass, ruff clean. Single flow, no env-var gate
  (removed 2026-07-21): a registered provider means injection; no provider means
  genai's own env-only `Config()`.
- **1b. af-component-agent wiring (DONE 2026-07-21, working branch).** The app
  `config.py.jinja` now declares the two globals (`datarobot_endpoint`,
  `datarobot_api_token`), adds `nim_deployment_id`, keeps the namespaced per-LLM
  fields, adds `resolve_llm(name=<llm_app_name>)` mapping them into a genai
  `LLMConfig`, and registers `lambda: Config().resolve_llm()` at import. Validated
  at the library level (hammer case + non-default / dashed instance names). Version
  floor bump still pending a genai release. Remaining below.
  - `template/{{agent_app_name}}/agent/config.py.jinja`: keep
    `class Config(DataRobotAppFrameworkBaseSettings)` (NOT the mixin). Ensure the
    two globals (`datarobot_endpoint`, `datarobot_api_token`) are present, keep
    the per-LLM namespaced flat fields (the existing `{{llm_app_name}}_*` Jinja
    already produces `llm_*` for the default), and add a `resolve_llm(name="llm")`
    method that maps `{name}_*` + globals into an `LLMConfig` (populating
    endpoint/token from the globals, per the interim decision).
  - Register a resolver-backed provider at import (guarded with try/except on the
    genai import for version compatibility). Default-only for now, resolver shaped
    so `name=` is a clean later extension.
  - Bump the genai version floor. Today the agent pins `datarobot-genai...==0.24.0`
    (exact). The seam is unreleased (post-0.26.1), so this step splits: land the
    wiring now against an editable genai, do the floor bump once genai cuts a
    release with the seam.
- **1c. Validate end to end.** Render the template (default `llm_app_name == "llm"`),
  byte-compile, confirm the provider registers. Run the hammer case: hardcode a
  model in `config.py`, no env vars, and confirm `get_llm()` uses it. Then
  `nat dragent serve` locally to prove the load-order seam through real NAT plugin
  discovery. Then confirm in a deployment (`start.sh` under gunicorn).
  (Library-level hammer validation DONE 2026-07-21; the `nat dragent serve` and
  deployment checks remain.)
- **1d. Extend injection to any remaining read site on the default NAT path (DONE 2026-07-21).**
  Routed `default_deployment_url` and `default_datarobot_llm_gateway_url` through
  `resolve_config()` so the endpoint (a true global) is authoritative from the app
  config too, not just the user-intent fields. Extracted `DEFAULT_DATAROBOT_ENDPOINT`
  as the shared default and coalesce fallback. The router path
  (`to_litellm_params`, `core/router.py`) stays intentionally deferred to the
  downstream router cleanup.

### Step 2 — Clean out old and unused config referencing

Remove what the injected resolver now supersedes, and undo the collapse damage.

**Mixin broken + two-resolver split DONE 2026-07-21 (the central untangle).**
`Config` no longer inherits `LLMConfig`. `resolve_config() -> Config` is the global
(the two globals + app-wide + default-instance fields); one LLM's config comes from
`resolve_config().resolve_llm_config(name=...)`, mapped from it and wired into the
`get_llm` dispatch and every `default_*` helper (previously dead code). Nothing in
genai reads a config attribute directly: the globals go through
`resolve_datarobot_endpoint()` / `resolve_datarobot_api_token()`, per-LLM through
the config object's own `resolve_llm_config(name=...)`. The provider now registers
the global `Config` plus a `default_llm_name`, read back by the string-only
`registered_default_llm_name()`; `_validate_global_config` duck-type-checks it.

**No module-level `resolve_llm_config()` wrapper in genai.** One existed through
several iterations of this migration and was removed five times; it keeps returning
on refactors and rebases. It has to stay gone: a free function returning an
`LLMConfig` gets imported, monkeypatched, and extended, and those overrides live in
genai rather than on the app's config object, so they break as soon as an app
registers a real `config.py`. Resolve at the call site with an explicit instance
name instead. The `DO NOT` note in `core/config.py` is the durable version of this.
Still deferred:
3d canonical de-prefixing of `LLMConfig` (it keeps `llm_*` field names for now),
and the router path (`to_litellm_params` / `core/router.py`).

- **2a.** Move `datarobot_endpoint` and `datarobot_api_token` ownership to the
  global `Config` and off `LLMConfig` as authored fields. Remove the duplicate
  `datarobot_endpoint` declaration on genai's `Config`. (Interim: `resolve_llm`
  still populates them onto the resolved object; the strict removal is the
  downstream cleanup.)
- **2b.** Delete genai's env-only reads that the injected resolver replaces on the
  default path, keeping the private `Config()` only as the no-injection fallback.
- **2c.** Retire the hardcoded `llm_*`-as-absolute assumptions in genai
  (`get_llm_type`, the leaf builders) wherever they are now dead.
- **2d.** In `af-component-llm`, inventory the band-aids that injection makes
  redundant, but defer their removal to Step 4 / downstream so the currently
  shipping env-var contract does not break mid-migration.

### Step 3 — Define the default-fallback naming contract

Write down the rules so nobody relitigates them, and make them enforceable.

**Largely DONE 2026-07-21 (universal namespacing landed).** All per-LLM env vars
are now instance-namespaced `{NAME}_*` across the three repos in one move (partial
namespacing silently breaks the standalone path). `resolve_llm` moved into
`datarobot_genai.core.config.resolve_llm(app_config, name)`; the app registers
`lambda: resolve_llm(Config(), "<llm_app_name>")`. genai `LLMConfig` fields are now
`llm_deployment_id / llm_nim_deployment_id / llm_use_datarobot_llm_gateway /
llm_default_model` plus the two globals; af-component-llm emits `{NAME}_*`
everywhere; default gateway flag = True everywhere, `"0"` written per non-gateway
blueprint. Still to do: 3f enforcement (type + runtime check) and 3g observability
logging. The final canonical de-prefixing of the value object (3d, strip `llm_`
off `LLMConfig` by separating it from the env reader) remains a later cleanup.

- **3a.** Codify the two true globals: `datarobot_endpoint`, `datarobot_api_token`,
  fixed names, ecosystem-wide.
- **3b.** Codify per-LLM namespacing: `{name}_default_model`,
  `{name}_deployment_id`, `{name}_nim_deployment_id`,
  `{name}_use_datarobot_llm_gateway`. `llm` is the default instance name, not an
  absolute.
- **3c.** Define the resolver mapping: flat `{name}_*` fields plus the two globals
  produce an `LLMConfig(model, deployment_id, nim_deployment_id,
  use_datarobot_llm_gateway [+ endpoint/token folded in, interim])`.
- **3d.** Canonicalize `LLMConfig` field names to logical, unprefixed forms
  (`model`, `deployment_id`), so the namespace lives only in the env-var names and
  the resolver, never in the value object. This is the correctly-framed "kill the
  `llm_*` coupling" work.
- **3e.** Confirm the gateway flag is per-LLM (`{name}_use_datarobot_llm_gateway`)
  and coordinate with Anatolii's namespace ticket.
- **3f.** Enforcement (the locked type + runtime check): provider must return an
  `LLMConfig`; a runtime assertion checks required fields are present and typed,
  with a clear error message otherwise.
- **3g.** Observability: log at resolve time which source won (injected provider
  vs env fallback) and the resolved LLM identity (redacted). This answers
  Anatolii's traceability / too-many-hops concern.

### Step 4 — Route default-entity access through functions

Make "which LLM" a function argument, never a hardcoded global read.

- **4a.** `get_llm(name="llm")` resolves the default via the resolver;
  `get_llm(name="bob")` selects a non-default instance; `get_llm(config=...)`
  accepts an explicit `LLMConfig`.
- **4b.** Restore `LLMConfig` as a true multi-instance value object (many
  constructible). This is what un-breaks `fallbacks`.
- **4c.** Multi-LLM support (build-late): the seam provider becomes name-aware;
  `af-component-llm` composes each added instance's namespaced fields into the app
  `Config`; `get_llm(name=...)` selects among them.
- **4d.** Audit that no code reads a hardcoded global `llm_*` off a singleton; all
  default-entity access goes through the resolver or `get_llm`.

---

## Downstream cleanups (explicitly deferred, tracked here so they are not lost)

- **Router / fallbacks wrapper.** Decide keep-vs-deprecate in favor of LiteLLM's
  native router. `LLMConfig`-as-value-object un-breaks it either way.
- **endpoint/token strictly off `LLMConfig`.** Final state: owned only by `Config`,
  builder takes `(llm_config, global_config)`.
- **`af-component-llm` band-aid removal.** Drop/shrink `ensure_datarobot_prefix`,
  stop force-exporting the prefix, simplify `verify_llm`; revisit C5 runtime-param
  convergence; finish deferred C3, E4, F1, F3, F4 (see `LLM_FIX_TRACKING.md` in
  af-component-llm).
- **Prefix normalization at the genai client boundary (Carson).** Then components
  stop prefixing. Reference implementations without genai: data-analyst
  `core/constants.py`, recipe-talk-to-my-docs `web/app/api/v1/chat.py`.
- **Shared config schema extraction (dovetail APP-6588).** Extract the LLM config
  schema into a shared place both genai and the app import, so the agent
  `config.py` is genuinely the master and genai depends on the schema, not the
  reverse.
- **Retire the `datarobot-deployed-llm` stub** once a real default model is always
  resolvable.

---

## Backward-compatibility invariant

There is one flow, selected by whether a provider is registered (no env-var gate).
A standalone genai with no app registering a provider builds its own env-only
`Config()` exactly as today. Any app on the new `config.py` template registers a
provider at import, so genai reads that config. This is the intended end state,
not a temporary opt-in. Consequence to watch: adopting the template makes the app
config's defaults authoritative immediately, which surfaces the
`use_datarobot_llm_gateway` default conflict (app False vs genai True) for the
local / no-env case. Real deployments set the env var via af-component-llm, so env
still wins there; the default reconciliation is tracked as Phase 1 work.

## Cross-repo coordination

- **Carson Gee:** APP-6588 (libllm extraction), prefix-at-client-boundary, the
  "baby tech spec" tying libllm and config extraction. The downstream schema and
  prefix cleanups depend on alignment here.
- **Anatolii Stehnii:** endorsed the injection approach as more in-line with other
  components; owns the `USE_DATAROBOT_LLM_GATEWAY` namespace ticket; leans toward
  using LiteLLM's router directly over the wrapper. He was skeptical injection was
  possible under NAT; Step 1a plus the mechanism section are the counter-proof.
- Genai version floors must move together across genai + agent + recipe.

## Validation strategy per step

- **Unit:** genai `tests/core/test_config.py` for the seam and the resolver; add
  framework-level tests as read paths change.
- **Integration:** render the agent template, `nat dragent serve` locally with the
  gate on, exercise the hammer case (hardcode in `config.py`, no env).
- **Deployment:** confirm the `start.sh` load path honors injection under gunicorn.

## Risks and watch items

- **Settings-source alias gotcha (verified 2026-07-21).**
  `DataRobotAppFrameworkBaseSettings` runs three env sources; two of them
  (`GetenvSettingsSource` for Runtime Parameters, `PulumiConfigSettingsSource`)
  iterate by `field_name.upper()` and ignore pydantic `validation_alias`. Only the
  standard env source honors aliases. So do not rely on `validation_alias` for
  prefixed env vars in a deployment; field names must match the env-var names.
- NAT eager-validation order is relied upon. If NAT changes plugin-discovery vs
  validation ordering, the import-time seam needs a different hook.
- genai version floors must move together across repos.
- Squash-merge policy: reference PRs / tickets in durable notes, never
  feature-branch SHAs.
