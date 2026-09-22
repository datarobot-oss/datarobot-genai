<!--
  ~ Copyright 2026 DataRobot, Inc. and its affiliates.
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# Why DRAgent's gunicorn path forces the standard asyncio loop

## Background

A Python 3.13 migration effort surfaced a crash-loop: `nat dragent serve --use_gunicorn
true` failed on Python 3.12+ with

```
ValueError: Can't patch loop of type <class 'uvloop.Loop'>
```

Chasing that down raised a broader question this document answers: **DRAgent didn't
choose to run on uvloop — it fell out of two unrelated defaults stacking — and NAT's own
code already avoids uvloop for exactly this class of problem on the other server path.**
The fix (`_force_gunicorn_worker_asyncio_loop()` in
[`src/datarobot_genai/dragent/frontends/fastapi.py`](../../../src/datarobot_genai/dragent/frontends/fastapi.py))
now forces the gunicorn path onto the plain asyncio loop, same as the direct-uvicorn path.
Recorded here for whoever next touches gunicorn/event-loop config in dragent.

## Where uvloop actually came from

Nobody in `datarobot-genai` or NAT's dragent integration asks for uvloop explicitly.
It fell out of two independent defaults:

1. `nvidia-nat-core` hard-depends on `uvicorn[standard]~=0.38`. The `standard` extra is
   what pulls in `uvloop` as an installed package.
2. NAT's gunicorn path (`nat/front_ends/fastapi/fastapi_front_end_plugin.py`) hardcodes
   `worker_class = "uvicorn.workers.UvicornWorker"`. That worker class's
   `CONFIG_KWARGS = {"loop": "auto", "http": "auto"}` — "auto" means "prefer uvloop if
   it's importable". Since (1) guarantees it is, every `use_gunicorn: true` dragent
   silently ended up on uvloop.

Nobody sat down and picked uvloop for its performance; it was just what these two
defaults produced when stacked.

## NAT already avoided it on the other path — for a documented reason

The *non*-gunicorn path, in the same file, explicitly does the opposite. From
`fastapi_front_end_plugin.py`:

```python
# By default, Uvicorn uses "auto" event loop policy, which prefers `uvloop` if installed. However,
# uvloop's event loop policy for macOS doesn't provide a child watcher (which is needed for MCP server),
# so setting loop="asyncio" forces Uvicorn to use the standard event loop, which includes child-watcher
# support.
if sys.platform == "darwin" or sys.platform.startswith("linux"):
    event_loop_policy = "asyncio"
else:
    event_loop_policy = "auto"
```

NAT itself had already concluded, for the direct-uvicorn path, that uvloop's lack of a
child watcher is a real risk for MCP server subprocess management, and forces plain
asyncio instead. That reasoning was never extended to the gunicorn path — it just
inherited `UvicornWorker`'s stock "auto" behavior. The crash was a symptom of that
inconsistency, not an isolated uvloop bug: gunicorn mode ended up on a loop that
dragent's own code had already decided, on the other path, wasn't safe for what it needs
to do. `_force_gunicorn_worker_asyncio_loop()` closes that gap by mirroring the same
policy onto the gunicorn path.

## uvloop vs. the standard asyncio loop: practical differences

**Performance.** uvloop (a Cython wrapper around libuv) is genuinely faster than the
stdlib loop on raw scheduling/socket-I/O microbenchmarks — commonly cited as 2-4x on
things like echo servers and high-message-rate loops. That gain is largest when
event-loop overhead is a meaningful fraction of total request time: high-QPS, short,
CPU-light request/response cycles.

DRAgent's workload doesn't look like that. A typical request spends its time waiting on
an LLM completion (hundreds of ms to tens of seconds), a downstream tool call, or a
memory/registry round-trip — all dominated by network latency to something outside the
process. The event loop's own per-callback scheduling overhead is a rounding error
against that. It would take a benchmark to say the difference is exactly zero, but
there's no reason to expect it to be perceptible for this shape of workload.

**Compatibility.** This is the side that actually caused a real outage-shaped bug:

- `uvloop.Loop` is not a subclass of `asyncio.BaseEventLoop`. Anything that assumes it is
  — as `nest_asyncio2._patch_loop` does — breaks. `nest_asyncio2` cannot ever patch a
  uvloop loop, gunicorn or not; that's what caused the crash-loop above.
- NAT's own comment documents a second such gap: uvloop's macOS policy has no child
  watcher, which subprocess-based MCP transports need.
- DRAgent composes a large number of third-party async libraries (litellm, crewai,
  langgraph, llama-index, MCP client transports, ...). Any of them doing sync-over-async
  (an internal `asyncio.run()` call from otherwise-sync code) while running on the
  request loop is exposed to the same "uvloop isn't a drop-in `BaseEventLoop`" class of
  failure. We know of one instance; there is no guarantee it was the only one.

**The tradeoff, stated plainly:** uvloop's speed advantage matters most for workloads
this one isn't, and its incompatibility surface (with `nest_asyncio2`, with child
watchers, and potentially with other libraries in this dependency stack) is not
theoretical — it already caused a crash-loop. For an LLM-latency-bound server, forcing
the standard asyncio loop is a small, likely-unmeasurable performance cost in exchange
for removing a whole class of "some library reached for a uvloop internal that doesn't
exist" bugs — the same conclusion NAT already reached for the direct-uvicorn path.

## What changed

`DRAgentFastApiFrontEndPlugin.run()` now calls `_force_gunicorn_worker_asyncio_loop()`
before starting gunicorn, which mutates `uvicorn.workers.UvicornWorker.CONFIG_KWARGS` to
`loop="asyncio"` (gunicorn resolves `worker_class` by importing that class directly, and
the mutation happens pre-fork, so every worker picks it up). Gunicorn mode is now never on
uvloop at all, which also means the `nest_asyncio2` incompatibility above can't occur —
this is a root-cause fix, not a workaround around `nest_asyncio2` itself.
