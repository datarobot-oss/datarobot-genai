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
import os
import sys
import tempfile
from pathlib import Path

from datarobot_genai.eval.dataset import load_dataset
from datarobot_genai.eval.dataset import to_byob_jsonl
from datarobot_genai.eval.output import normalize_output
from datarobot_genai.eval.runner import run_byob
from datarobot_genai.eval.status import write_status
from datarobot_genai.eval.utils import make_run_id
from datarobot_genai.eval.validation import load_pipeline
from datarobot_genai.eval.validation import preflight_judge
from datarobot_genai.eval.validation import validate_inputs


class EvalRunner:
    """Orchestrates a single batch evaluation run.

    ``repo_root`` is the component directory that holds ``user_pipelines/`` and
    ``output/``. It is required: this class ships inside the installed package,
    so there is no meaningful filesystem default — the caller (the component's
    CLI wrapper) supplies its own location.
    """

    def __init__(
        self,
        endpoint: str,
        pipeline: str,
        dataset: str,
        repo_root: Path,
    ) -> None:
        self.endpoint = endpoint
        self.pipeline = pipeline
        self.dataset = dataset
        self.repo_root = repo_root
        self.pipelines_dir = self.repo_root / "user_pipelines"
        self.output_dir = self.repo_root / "output"

    def run(self, dry_run: bool = False) -> int:
        run_id = make_run_id()

        # 1. Validate
        print("Validating inputs...")
        errors = validate_inputs(
            self.endpoint,
            self.pipeline,
            self.dataset,
            self.pipelines_dir,
            self.repo_root,
        )
        if errors:
            print("Validation failed:", file=sys.stderr)
            for e in errors:
                print(f"  ✗ {e}", file=sys.stderr)
            return 1
        print(f"  ✓ Endpoint reachable: {self.endpoint}")
        print(f"  ✓ Pipeline found:     user_pipelines/{self.pipeline}")
        print(f"  ✓ Dataset found:      {self.dataset}")

        cfg = load_pipeline(self.pipelines_dir / self.pipeline)

        if dry_run:
            print("\nDry run — all inputs valid. Would run BYOB benchmark:")
            print(f"  module:  {cfg['benchmark']['module']}")
            judge = cfg.get("judge")
            if judge:
                print(f"  judge:   {judge['model_id']} @ {judge['url']}")
            else:
                print("  judge:   none (judge-free benchmark)")
            print("  output → output/eval_results.json")
            return 0

        # Preflight the judge before doing anything expensive. Catches missing /
        # invalid tokens, wrong model_id, and gateway outages up front instead of
        # producing a run full of CALL_ERRORs.
        judge_cfg = cfg.get("judge")
        if judge_cfg:
            try:
                preflight_judge(judge_cfg)
            except RuntimeError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                write_status(
                    "failed",
                    run_id,
                    self.pipeline,
                    self.endpoint,
                    self.output_dir,
                    error=str(e),
                )
                return 1
            print(f"  ✓ Judge reachable:    {judge_cfg['model_id']}")

        # 2. Mark as running
        write_status("running", run_id, self.pipeline, self.endpoint, self.output_dir)
        print("\nStatus: running  (output/eval_status.json)")

        dataset = load_dataset(self.dataset)
        print(f"Loaded {len(dataset)} test cases")

        # 3. Convert dataset to BYOB JSONL
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix="byob_dataset_"
        ) as f:
            dataset_jsonl = f.name
        to_byob_jsonl(dataset, dataset_jsonl)

        nemo_output_dir = str(self.output_dir / "raw" / run_id)

        # 4. Run BYOB
        print("Running NeMo Evaluator (BYOB, in-process)...")
        try:
            run_byob(cfg, self.endpoint, dataset_jsonl, nemo_output_dir, self.repo_root)
        except RuntimeError as e:
            write_status(
                "failed",
                run_id,
                self.pipeline,
                self.endpoint,
                self.output_dir,
                error=str(e),
            )
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        finally:
            if Path(dataset_jsonl).exists():
                os.unlink(dataset_jsonl)

        # 5. Normalize output
        try:
            normalized = normalize_output(
                nemo_output_dir, dataset, self.endpoint, self.pipeline, run_id
            )
        except Exception as e:  # noqa: BLE001
            write_status(
                "failed",
                run_id,
                self.pipeline,
                self.endpoint,
                self.output_dir,
                error=f"Output normalization failed: {e}",
            )
            print(f"ERROR normalizing output: {e}", file=sys.stderr)
            return 3

        # 6. Write to fixed output locations (external CLI relies on these paths)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "eval_results.json").write_text(json.dumps(normalized, indent=2))
        write_status("complete", run_id, self.pipeline, self.endpoint, self.output_dir)

        s = normalized["summary"]
        print("\nStatus: complete")
        print("Results: output/eval_results.json")
        print(f"  Total cases:        {normalized['total_cases']}")
        print(f"  Scored / inconcl.:  {s['scored_cases']} / {s['inconclusive_cases']}")
        print(f"  Mean quality score: {s['mean_quality_score']}")
        print(f"  Pass rate:          {s['pass_rate']}")
        print(f"  Good case pass:     {s['good_case_pass_rate']}")
        print(f"  Bad case pass:      {s['bad_case_pass_rate']}")

        return 0
