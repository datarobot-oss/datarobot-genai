# Skill: NAT eval with DataRobot moderations

Use when a user wants to evaluate a NAT/DRAgent agent offline with `nat eval`, scoring outputs
through DataRobot moderation guards (e.g. `agent_goal_accuracy`, `faithfulness`, `task_adherence`)
instead of the retired DRUM + pytest local-evaluation flow.

Reference guide: `docs/src/nat/eval.md`. Working example: `e2e-tests/dragent/nat/`
(`moderations_evaluator.py`, `moderation.yaml`, `eval_workflow.yml`, `dataset.json`).

1. Add `moderations_evaluator.py` next to the workflow (copy from the example): a
   `ModerationsEvaluatorConfig(EvaluatorBaseConfig, name="moderations")` + `@register_evaluator`
   wrapping `ModerationPipeline.evaluate_response_async`. Import it from the package's `register.py`
   (`from <pkg> import moderations_evaluator  # noqa: F401`) so NAT discovers it.

2. Add `moderation.yaml` with the guards to score. For the agentic `agent_goal_accuracy` guard set
   `is_agentic: true` and a judge LLM — either `llm_type: llmGateway` + `llm_gateway_model_id`
   (no deployment) or `llm_type: datarobot` + `deployment_id`. Use `intervention: {action: report,
   conditions: []}` for pure measurement.

3. Add an `eval:` block to the workflow: `eval.general.dataset` (json `id`/`question`/`answer`) and
   `eval.evaluators.moderations: {_type: moderations, moderation_config: ./moderation.yaml}`.

4. Under `nat eval` the dragent-native `datarobot-llm-component` is not registered (front-end only),
   so set the agent's LLM to stock `_type: openai` pointed at the gateway via `OPENAI_API_KEY` /
   `OPENAI_BASE_URL` (`$DATAROBOT_ENDPOINT/genai/llmgw`).

5. Run: `nat eval --config_file <workflow>.yml`. Read `moderations_output.json`: `average_score`
   plus per-row `score` and `reasoning` (`blocked`, `trajectory_used`, `metrics`).

Requires `datarobot-moderations >= 11.2.45` for trajectory-aware `agent_goal_accuracy`
(the `pipeline_interactions` parameter); older versions score the single prompt/response pair.
