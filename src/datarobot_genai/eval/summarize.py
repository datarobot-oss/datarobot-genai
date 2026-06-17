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
from pathlib import Path
from typing import Any


class ResultsSummarizer:
    def __init__(self, path: Path) -> None:
        self.results_path = self._resolve(path)
        self.data: dict[str, Any] = json.loads(self.results_path.read_text())

    @staticmethod
    def _resolve(path: Path) -> Path:
        if path.is_file():
            return path
        candidate = path / "eval_results.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No eval_results.json found in {path}")

    def print_summary(self) -> None:
        data = self.data

        print(f"\nRun ID:    {data.get('run_id', '?')}")
        print(f"Completed: {data.get('completed_at', '?')}")
        print(f"Agent:     {data.get('agent_endpoint', '?')}")
        print(f"Pipeline:  {data.get('pipeline', '?')}")
        print(f"Cases:     {data.get('total_cases', '?')}")

        s: dict[str, Any] = data.get("summary", {})
        print("\nSummary")
        print(f"  Mean quality score : {s.get('mean_quality_score')}")
        print(f"  Pass rate          : {s.get('pass_rate')}")
        print(f"  Good case pass     : {s.get('good_case_pass_rate')}")
        print(f"  Bad case pass      : {s.get('bad_case_pass_rate')}")

        nemo_agg: dict[str, Any] = s.get("nemo_aggregate", {})
        if nemo_agg:
            print("\nNeMo Aggregate Metrics")
            for key, value in nemo_agg.items():
                print(f"  {key}: {value}")

        cases: list[dict[str, Any]] = data.get("cases", [])
        if cases:
            print("\nPer-case Results")
            print(f"  {'ID':<15} {'Expect':<8} {'Score':<7} {'Pass':<6} Reason")
            print(f"  {'-' * 15} {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 45}")
            for c in cases:
                score = c.get("quality_score")
                score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
                passed = "✓" if c.get("passed") else ("✗" if c.get("passed") is False else "?")
                reason = (c.get("judge_reason") or "")[:50]
                print(
                    f"  {(c.get('id') or '?'):<15} {(c.get('expected_behavior') or '?'):<8} "
                    f"{score_str:<7} {passed:<6} {reason}"
                )

        print()
