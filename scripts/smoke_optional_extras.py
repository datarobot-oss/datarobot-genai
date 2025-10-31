# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from collections.abc import Iterable
from typing import Any


def parse_extras_arg(value: str) -> set[str]:
    if value == "all":
        return {"crewai", "langgraph", "llamaindex", "drmcp"}
    if value == "none" or not value:
        return set()
    return {v.strip() for v in value.split(",") if v.strip()}


def expect_import(module: str, should_succeed: bool) -> None:
    try:
        importlib.import_module(module)
        if not should_succeed:
            sys.exit(f"Unexpected: {module} imported without its extras")
        print(f"Imported OK: {module}")
    except ModuleNotFoundError as exc:
        if should_succeed:
            raise
        print(f"Expected missing: {module} -> {type(exc).__name__}")


def run_smoke(extras: Iterable[str]) -> None:
    extras_set = set(extras)

    print("Top-level import...")
    import datarobot_genai as _drg
    from datarobot_genai.agents.base import (
        BaseAgent,
        extract_user_prompt_content,
        make_system_prompt,
    )
    print("Top-level OK")
    # Touch imported symbols to avoid unused-import warnings
    _ = (_drg.__name__, BaseAgent, extract_user_prompt_content, make_system_prompt)

    # Validate import behavior per extras, but permit import if underlying deps are present.
    def dep_present(modname: str) -> bool:
        return importlib.util.find_spec(modname) is not None

    allow_crewai = ("crewai" in extras_set) or dep_present("crewai")
    allow_langgraph = ("langgraph" in extras_set) or dep_present("langgraph")
    allow_llamaindex = ("llamaindex" in extras_set) or dep_present("llama_index")
    allow_drmcp = ("drmcp" in extras_set) or dep_present("fastmcp")

    expect_import("datarobot_genai.agents.crewai", allow_crewai)
    expect_import("datarobot_genai.agents.langgraph", allow_langgraph)
    expect_import("datarobot_genai.agents.llamaindex", allow_llamaindex)
    expect_import("datarobot_genai.drmcp", allow_drmcp)

    # Minimal functional smoke per installed extra
    if "crewai" in extras_set:
        from datarobot_genai.agents.crewai import (
            build_llm,
            create_pipeline_interactions_from_messages,
        )
        from ragas.messages import HumanMessage

        _ = build_llm(
            api_base="https://tenant.datarobot.com/api/v2",
            api_key="tok",
            model="mistral",
            deployment_id="dep-1",
            timeout=1,
        )
        sample = create_pipeline_interactions_from_messages([HumanMessage(content="hi")])
        assert sample is not None
        print("crewai smoke OK")

    if "langgraph" in extras_set:
        from datarobot_genai.agents.langgraph import (
            create_pipeline_interactions_from_events as create_events_langgraph,
        )
        from langchain_core.messages import (
            AIMessage as LC_AIMessage,
            HumanMessage as LC_HumanMessage,
            ToolMessage as LC_ToolMessage,
        )

        events: list[dict[str, Any]] = [
            {
                "node1": {
                    "messages": [
                        LC_ToolMessage(content="tool", tool_call_id="tc_1"),
                        LC_HumanMessage(content="hi"),
                    ]
                }
            },
            {"node2": {"messages": [LC_AIMessage(content="ok")] }},
        ]
        sample = create_events_langgraph(events)
        assert sample is not None and len(sample.user_input) == 2
        print("langgraph smoke OK")

    if "llamaindex" in extras_set:
        from datarobot_genai.agents.llamaindex import (
            DataRobotLiteLLM,
            create_pipeline_interactions_from_events as create_events_llamaindex,
        )

        assert create_events_llamaindex(None) is None
        # Instantiate the LlamaIndex adapter to ensure basic construction works
        llm = DataRobotLiteLLM(model="dr/model")
        assert llm.metadata.is_chat_model is True
        print("llamaindex smoke OK")

    if "drmcp" in extras_set:
        from datarobot_genai.drmcp import (
            DataRobotMCPServer,
        )

        # Just verify the module can be imported and basic classes exist
        assert DataRobotMCPServer is not None
        print("drmcp smoke OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test optional extras behavior")
    parser.add_argument(
        "--extras",
        default="none",
        help=(
            "Extras to enable: one of 'none', 'all', or comma-separated list "
            "(e.g., 'crewai,llamaindex')"
        ),
    )
    args = parser.parse_args()
    extras = parse_extras_arg(args.extras)
    run_smoke(extras)


if __name__ == "__main__":
    main()
