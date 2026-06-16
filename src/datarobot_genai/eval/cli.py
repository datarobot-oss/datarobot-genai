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
"""Command-line entrypoints for the evaluation component.

These functions back the thin ``run.py`` / ``generate.py`` / ``summarize.py``
wrappers shipped in an evaluation component. Each accepts an optional ``argv``
(for testing) and, where relevant, a ``repo_root`` — the component directory
that holds ``user_pipelines/``, ``user_datasets/``, and ``output/``. ``repo_root``
defaults to the current working directory so a bare invocation still works.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from datarobot_genai.eval.converter import convert_csv_to_cases
from datarobot_genai.eval.converter import save_cases
from datarobot_genai.eval.eval import EvalRunner
from datarobot_genai.eval.generator import CaseGenerator
from datarobot_genai.eval.validation import preflight_judge

_RUN_EPILOG = """\
The external CLI passes three things:
  --endpoint   Base URL of the agent's OpenAI-compatible API
  --pipeline   Filename of a pipeline YAML in user_pipelines/
  --dataset    Path to a test case JSON file (defaults to user_datasets/sample_answer_quality.json)

Fixed output locations (always the same — external CLI can rely on these paths):
  output/eval_status.json     current run status
  output/eval_results.json    normalized results (written on success)

Exit codes:
  0  success
  1  validation error (bad endpoint, missing pipeline/dataset)
  2  evaluator subprocess failed
  3  output normalization failed
"""


def _resolve_repo_root(repo_root: Path | None) -> Path:
    return repo_root if repo_root is not None else Path.cwd()


def run_main(
    argv: Sequence[str] | None = None,
    repo_root: Path | None = None,
) -> None:
    """Run a NeMo Evaluator batch evaluation (BYOB, in-process). Exits with the run's code."""
    repo_root = _resolve_repo_root(repo_root)
    default_dataset = str(repo_root / "user_datasets" / "sample_answer_quality.json")

    parser = argparse.ArgumentParser(
        description="Run NeMo Evaluator batch evaluation (BYOB, in-process)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_RUN_EPILOG,
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Base URL of the agent's OpenAI-compatible API (e.g. http://localhost:8842/v1)",
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        help="Pipeline YAML filename in user_pipelines/ (e.g. answer_quality.yaml)",
    )
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help=f"Path to test case JSON file (default: {default_dataset})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print what would run, without executing",
    )
    args = parser.parse_args(argv)

    runner = EvalRunner(
        endpoint=args.endpoint,
        pipeline=args.pipeline,
        dataset=args.dataset,
        repo_root=repo_root,
    )
    sys.exit(runner.run(dry_run=args.dry_run))


def generate_main(
    argv: Sequence[str] | None = None,
    repo_root: Path | None = None,
) -> None:
    """Generate synthetic evaluation test cases or convert a CSV dataset to JSON."""
    repo_root = _resolve_repo_root(repo_root)

    parser = argparse.ArgumentParser(
        description="Generate synthetic evaluation test cases or convert a CSV dataset to JSON"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--agent-description",
        help=(
            "Description of what the agent does "
            "(triggers synthetic generation via a DataRobot-hosted model)"
        ),
    )
    mode.add_argument(
        "--convert",
        metavar="CSV_FILE",
        help=(
            "Path to a CSV file to convert to JSON "
            "(columns: id, source, input required; others optional)"
        ),
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Total number of cases to generate (split evenly good/bad, default: 10)",
    )
    parser.add_argument(
        "--n-good",
        type=int,
        help="Number of good cases (overrides --n split)",
    )
    parser.add_argument(
        "--n-bad",
        type=int,
        help="Number of bad cases (overrides --n split)",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output JSON file path. "
            "Defaults to user_datasets/generated_cases.json for generation, "
            "or <csv_stem>.json in the same directory for --convert."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing file instead of overwriting (generation only)",
    )
    parser.add_argument(
        "--pipeline",
        metavar="YAML_FILE",
        help=(
            "Pipeline YAML (e.g. user_pipelines/answer_quality.yaml). "
            "When provided, the generator tailors good/bad criteria and required "
            "fields to match the benchmark."
        ),
    )
    parser.add_argument(
        "--url",
        help="DataRobot endpoint URL (overrides DATAROBOT_ENDPOINT env var)",
    )
    parser.add_argument(
        "--model-id",
        help="Model ID to use for generation (overrides LLM_DEFAULT_MODEL env var)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the generation model (overrides DATAROBOT_API_TOKEN env var)",
    )
    args = parser.parse_args(argv)

    if args.convert:
        csv_path = Path(args.convert)
        if not csv_path.exists():
            parser.error(f"CSV file not found: {csv_path}")

        output_path = Path(args.output) if args.output else csv_path.with_suffix(".json")

        print(f"Converting {csv_path} -> {output_path} ...")
        cases = convert_csv_to_cases(csv_path)
        save_cases(cases, output_path)
        print(f"Wrote {len(cases)} cases to {output_path}")
        print()
        print("Review and edit before using in evaluations:")
        for case in cases:
            print(f"  {case['id']}: {str(case['input'])[:70]}")
        return

    # --- generation mode ---
    n_good = args.n_good if args.n_good is not None else args.n // 2
    n_bad = args.n_bad if args.n_bad is not None else args.n - n_good
    output_path = (
        Path(args.output) if args.output else repo_root / "user_datasets" / "generated_cases.json"
    )

    benchmark_name: str | None = None
    if args.pipeline:
        pipeline_path = Path(args.pipeline)
        if not pipeline_path.is_absolute():
            pipeline_path = repo_root / pipeline_path
        if not pipeline_path.exists():
            parser.error(f"Pipeline file not found: {pipeline_path}")
        pipeline_cfg: dict[str, Any] = yaml.safe_load(pipeline_path.read_text())
        benchmark_name = (pipeline_cfg.get("benchmark") or {}).get("name")
        if benchmark_name:
            print(f"Tailoring cases for benchmark: {benchmark_name}")
        judge_cfg = pipeline_cfg.get("judge")
        if judge_cfg:
            try:
                preflight_judge(judge_cfg)
            except RuntimeError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                raise SystemExit(1) from e
            print(f"Judge reachable: {judge_cfg['model_id']}")

    print(f"Generating {n_good} good + {n_bad} bad test cases...")

    generator = CaseGenerator(url=args.url, model_id=args.model_id, api_key=args.api_key)
    cases = generator.generate(args.agent_description, n_good, n_bad, benchmark_name=benchmark_name)
    written = generator.save(cases, output_path, append=args.append)

    print(f"Wrote {len(written)} cases to {output_path}")
    print()
    print("Review and edit before using in evaluations:")
    for case in written:
        behavior_label = "✓" if case["expected_behavior"] == "good" else "✗"
        print(f"  [{behavior_label}] {case['id']}: {case['input'][:70]}")


def summarize_main(argv: Sequence[str] | None = None) -> None:
    """Pretty-print a normalized eval_results.json from a completed run."""
    # Imported lazily so `summarize` stays usable even in minimal installs.
    from datarobot_genai.eval.summarize import ResultsSummarizer

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(summarize_main.__doc__)
        sys.exit(1)
    path = Path(argv[0])
    if not path.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)
    ResultsSummarizer(path).print_summary()
