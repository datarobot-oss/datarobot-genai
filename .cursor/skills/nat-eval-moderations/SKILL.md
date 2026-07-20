# Skill: NAT eval with DataRobot moderations

**When to use this skill:** Use when a user wants to evaluate a NAT/DRAgent agent offline with
`nat eval`, scoring outputs through DataRobot moderation guards (`agent_goal_accuracy`,
`faithfulness`, `task_adherence`) instead of the retired DRUM + pytest local-evaluation flow.

Reference guide: `docs/src/nat/eval.md`. Complete, copy-paste-ready versions of every file below
are in the `examples/` folder alongside this skill.

## Prerequisites

1. `datarobot-moderations` is installed (`>= 11.2.45` for trajectory-aware `agent_goal_accuracy`;
   older versions score the single prompt/response pair — the evaluator feature-detects this).
2. A judge LLM: either the DataRobot LLM gateway (no deployment needed) or a 24-hex LLM
   deployment id.
3. `DATAROBOT_ENDPOINT` and `DATAROBOT_API_TOKEN` are set in the environment.

## Step-by-step implementation guide

1. **Add the evaluator.** Copy `examples/moderations_evaluator.py` into your agent package next to
   the workflow, and import it from your `register.py` so NAT discovers `_type: moderations`:
   ```python
   from <your_package> import moderations_evaluator  # noqa: F401
   ```

2. **Configure the guards.** Copy `examples/moderation.yaml` and set the judge — either the gateway
   (no deployment) or a deployment:
   ```yaml
   llm_type: llmGateway
   llm_gateway_model_id: "azure/gpt-4o-2024-11-20"   # or: llm_type: datarobot + deployment_id: <24-hex>
   ```
   Keep `is_agentic: true` for `agent_goal_accuracy`, and `intervention: {action: report, conditions: []}`
   for pure measurement.

3. **Add the eval block.** Copy `examples/eval_workflow.yml`, adapt its `functions`/`workflow` to your
   agent, and keep the `eval:` block wiring `evaluators.moderations: {_type: moderations,
   moderation_config: ./moderation.yaml}`. Point the agent LLM at the gateway with stock
   `_type: openai` — the dragent-native `datarobot-llm-component` is not registered under `nat eval`.

4. **Add cases.** Copy `examples/dataset.json` and add rows (`id` / `question` / `answer`).

5. **Run.** Set `OPENAI_API_KEY="$DATAROBOT_API_TOKEN"` and
   `OPENAI_BASE_URL="$DATAROBOT_ENDPOINT/genai/llmgw"`, then `nat eval --config_file eval_workflow.yml`.
   Read `moderations_output.json`: `average_score` plus per-row `score` and `reasoning`
   (`blocked`, `trajectory_used`, `metrics`).
