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

# Evaluation (`nat eval` + moderations)

Evaluate a NAT/DRAgent workflow offline with **`nat eval`**, scoring each row through DataRobot **moderations** exposed as a NAT **custom evaluator**. Scoring runs *inside* the eval framework, so there are no DRUM code paths and no bespoke pytest assertions: you define a dataset and the guards to score, and `nat eval` produces per-row scores.

Working example, in the same folder as the NAT workflow: [`e2e-tests/dragent/nat/`](../../e2e-tests/dragent/nat/) (`eval_workflow.yml`, `moderations_evaluator.py`, `moderation.yaml`, `dataset.json`).

## How it works

`nat eval --config_file <wf>.yml` runs the workflow over the dataset named in its `eval:` block, then scores each row's output with the configured evaluators. The `moderations` evaluator wraps [`ModerationPipeline`](https://github.com/datarobot/moderations) and reports each guard's metric (e.g. `agent_goal_accuracy`) as the per-row score.

| Piece | File | Role |
|---|---|---|
| eval block | `eval_workflow.yml` | dataset + `evaluators.moderations` |
| evaluator | `moderations_evaluator.py` | `@register_evaluator` wrapping `ModerationPipeline` |
| guards | `moderation.yaml` | which moderation guards to score |
| dataset | `dataset.json` | rows: `id` / `question` / `answer` |

## The `eval` block

Add an `eval:` block to the workflow file. The evaluator key (`moderations`) is the output-file prefix; `_type` must match the evaluator's registered name.

```yaml
eval:
  general:
    output_dir: ./.tmp/nat/moderations_eval/
    dataset:
      _type: json
      file_path: ./dataset.json
  evaluators:
    moderations:
      _type: moderations
      moderation_config: ./moderation.yaml
```

## The moderations evaluator

[`moderations_evaluator.py`](../../e2e-tests/dragent/nat/moderations_evaluator.py) registers `_type: moderations` and is discovered when its module is imported from [`register.py`](../../e2e-tests/dragent/nat/register.py) (NAT loads that via the package's `nat.plugins` entry point). Per row it calls `ModerationPipeline.evaluate_response_async(...)` and maps `EvaluationResult.metrics[metric]` to the score. Config fields:

- **`moderation_config`** — path to the guard YAML.
- **`metric`** — key in `EvaluationResult.metrics` to score on (default `agent_goal_accuracy`).

**Trajectory.** `nat eval` exposes NAT `IntermediateStep`s, not the dragent chat-completion's assembled `pipeline_interactions`. The agentic `agent_goal_accuracy` guard needs the latter, so the evaluator rebuilds a `MultiTurnSample` wire dict from `item.trajectory` (input → Human, `LLM_END` → AI, `TOOL_END` → Tool) and passes it as `pipeline_interactions`. A real `pipeline_interactions` on the dataset entry is preferred as-is when present.

## `moderation.yaml` (guards)

Standard moderation config. `agent_goal_accuracy` is an LLM-as-judge, agentic guard (`is_agentic: true`). Use the LLM gateway as the judge to avoid standing up a deployment:

```yaml
guards:
  - name: Agent Goal Accuracy
    type: ootb
    ootb_type: agent_goal_accuracy
    stage: response
    is_agentic: true
    llm_type: llmGateway
    llm_gateway_model_id: "azure/gpt-4o-2024-11-20"
    intervention: {action: report, conditions: []}
```

Swap to `llm_type: datarobot` + `deployment_id: <24-hex>` to judge with a specific DataRobot LLM deployment instead.

## Run it

The dragent-native `datarobot-llm-component` is only registered when the dragent front end loads, which `nat eval` does not do. Point the agent's LLM at the OpenAI-compatible DataRobot gateway with the stock `openai` type, supplying creds via the environment:

```bash
cd e2e-tests/dragent/nat
export OPENAI_API_KEY="$DATAROBOT_API_TOKEN"
export OPENAI_BASE_URL="$DATAROBOT_ENDPOINT/genai/llmgw"
nat eval --config_file eval_workflow.yml
# score pre-generated answers without running the agent:
nat eval --config_file eval_workflow.yml --skip_workflow --dataset dataset_negative.json
```

Output lands in `output_dir` as `moderations_output.json`: an `average_score` plus one `eval_output_items[]` entry per row with `score` and `reasoning` (the guard's `blocked` flag, `trajectory_used`, and the full `metrics` map).

## Requirements and caveats

- **moderations version.** Trajectory-aware scoring needs `datarobot-moderations >= 11.2.45` (adds the `pipeline_interactions` parameter). On older versions the evaluator feature-detects this and scores the single prompt/response pair instead (`reasoning.trajectory_used: false`).
- **Score, not block.** In the standalone `evaluate_response` path the score is the signal (`average_score`, per-row `metrics`); a guard's `intervention` block flag is not applied the way it is on live serving. Use `action: report` for pure measurement.
- **Judge is separate from the agent LLM.** The judge (`moderation.yaml`) and the workflow's `datarobot_llm` are independent; a model scores its own output leniently, so prefer a different judge model.
