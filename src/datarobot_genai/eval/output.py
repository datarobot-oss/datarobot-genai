# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import warnings
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

PASS_THRESHOLD = 0.5


def _find_artifact(output_dir: str, filename: str) -> Path | None:
    matches = list(Path(output_dir).rglob(filename))
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def normalize_output(
    output_dir: str,
    dataset: list[dict[str, Any]],
    endpoint: str,
    pipeline: str,
    run_id: str,
) -> dict[str, Any]:
    dataset_by_id: dict[str, dict[str, Any]] = {}
    for c in dataset:
        if c["id"] in dataset_by_id:
            warnings.warn(
                f"Duplicate dataset id {c['id']!r}; later entry overwrites earlier",
                UserWarning,
                stacklevel=2,
            )
        dataset_by_id[c["id"]] = c

    predictions_path = _find_artifact(output_dir, "byob_predictions.jsonl")
    results_path = _find_artifact(output_dir, "byob_results.json")

    cases: list[dict[str, Any]] = []
    if predictions_path and predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            if not line.strip():
                continue
            pred: dict[str, Any] = json.loads(line)
            meta: dict[str, Any] = pred.get("metadata", {})
            case_id: str = meta.get("id", "unknown")
            original = dataset_by_id.get(case_id, {})

            scores: dict[str, Any] = pred.get("scores") or {}
            quality_score: Any = scores.get("score")
            grade: str = scores.get("judge_grade", "")
            reason: str = scores.get("reason", "")
            status: str = pred.get("status", "")

            scored_ok = isinstance(quality_score, (int, float))
            passed: bool | None = quality_score >= PASS_THRESHOLD if scored_ok else None

            # A sample is "inconclusive" when the agent answered but the judge
            # call failed (no numeric score). See BUGS.md #3.
            # Judge-free benchmarks emit a human-readable ``reason`` (e.g.
            # "canary present", "found EMAIL"); judge-based ones report the grade.
            if scored_ok:
                judge_reason = reason or f"judge grade: {grade}"
            elif status == "scored":
                detail = reason or grade or "returned no score"
                judge_reason = f"inconclusive — {detail}"
            else:
                judge_reason = f"inconclusive — agent {status}"

            cases.append(
                {
                    "id": case_id,
                    "input": original.get("input", meta.get("input", "")),
                    "expected_behavior": meta.get(
                        "expected_behavior", original.get("expected_behavior")
                    ),
                    "agent_response": pred.get("response") or "",
                    "quality_score": quality_score,
                    "judge_reason": judge_reason,
                    "passed": passed,
                    "answer_match_score": None,
                    "notes": original.get("notes", meta.get("notes", "")),
                    "source": original.get("source", meta.get("source", "")),
                }
            )

    # Dataset cases with no prediction entry are surfaced as inconclusive so
    # partial runs don't silently appear complete in the summary.
    predicted_ids = {c["id"] for c in cases}
    for c in dataset:
        if c["id"] not in predicted_ids:
            cases.append(
                {
                    "id": c["id"],
                    "input": c.get("input", ""),
                    "expected_behavior": c.get("expected_behavior"),
                    "agent_response": "",
                    "quality_score": None,
                    "judge_reason": "inconclusive — no prediction",
                    "passed": None,
                    "answer_match_score": None,
                    "notes": c.get("notes", ""),
                    "source": c.get("source", ""),
                }
            )

    # Aggregate scores straight from the BYOB results.json.
    nemo_aggregate: dict[str, Any] = {}
    if results_path and results_path.exists():
        raw: dict[str, Any] = json.loads(results_path.read_text())
        for task_name, task_data in raw.get("tasks", {}).items():
            for metric_name, metric_data in task_data.get("metrics", {}).items():
                for score_name, score_data in metric_data.get("scores", {}).items():
                    nemo_aggregate[f"{task_name}.{metric_name}.{score_name}"] = score_data.get(
                        "value"
                    )

    scored = [c for c in cases if isinstance(c["quality_score"], (int, float))]
    inconclusive = len(cases) - len(scored)
    mean_score = sum(c["quality_score"] for c in scored) / len(scored) if scored else None
    pass_rate = sum(1 for c in scored if c["passed"]) / len(scored) if scored else None
    good = [c for c in scored if c["expected_behavior"] == "good"]
    bad = [c for c in scored if c["expected_behavior"] == "bad"]

    return {
        "run_id": run_id,
        "completed_at": datetime.now(UTC).isoformat(),
        "agent_endpoint": endpoint,
        "pipeline": pipeline,
        "total_cases": len(dataset),
        "summary": {
            "scored_cases": len(scored),
            "inconclusive_cases": inconclusive,
            "mean_quality_score": round(mean_score, 4) if mean_score is not None else None,
            "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
            "good_case_pass_rate": (
                round(sum(1 for c in good if c["passed"]) / len(good), 4) if good else None
            ),
            "bad_case_pass_rate": (
                round(sum(1 for c in bad if c["passed"]) / len(bad), 4) if bad else None
            ),
            "nemo_aggregate": nemo_aggregate,
        },
        "cases": cases,
    }
