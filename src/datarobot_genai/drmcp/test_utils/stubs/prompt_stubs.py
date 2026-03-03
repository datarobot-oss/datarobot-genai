# Copyright 2026 DataRobot, Inc.
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

"""Stub prompt templates and versions for integration tests (test_dynamic_prompts)."""

from types import SimpleNamespace

# Names and text used by test_dynamic_prompts; keep in sync with conftest fixture names.
STUB_PROMPT_WITHOUT_VERSION = "drmcp-integration-test-prompt-without-version"
STUB_PROMPT_VERSION_NO_VARS = "drmcp-integration-test-prompt-with-version-without-variables"
STUB_PROMPT_VERSION_WITH_VARS = "drmcp-integration-test-prompt-with-variables"
STUB_PROMPT_VERSION_NO_VARS_TEXT = "Prompt text without any variables."
STUB_PROMPT_VERSION_WITH_VARS_TEXT = "Prompt text to greet {{name}} in max {{sentences}} sentences."
# Same name for two templates to test duplicate-name handling (suffix in list_prompts).
STUB_PROMPT_DUPLICATE_NAME = "drmcp-integration-test-prompt-duplicate-stub"


def _var(name: str, description: str = "") -> SimpleNamespace:
    return SimpleNamespace(name=name, description=description)


def _template(tid: str, name: str, description: str = "") -> SimpleNamespace:
    t = SimpleNamespace(id=tid, name=name, description=description or name)
    return t


def _version(
    vid: str,
    ptid: str,
    prompt_text: str,
    variables: list,
    version: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=vid,
        prompt_template_id=ptid,
        prompt_text=prompt_text,
        commit_comment="",
        version=version,
        variables=variables,
        creation_date="",
        creation_user_id="",
        user_name="",
    )


def get_stub_prompt_templates() -> list[SimpleNamespace]:
    """Return stub prompt templates for register_prompts_from_datarobot_prompt_management."""
    # Exclude STUB_PROMPT_WITHOUT_VERSION so it is not registered (test expects it not in list).
    return [
        _template("stub-pt-1", STUB_PROMPT_VERSION_NO_VARS),
        _template("stub-pt-2", STUB_PROMPT_VERSION_WITH_VARS),
        _template("stub-pt-dup-1", STUB_PROMPT_DUPLICATE_NAME),
        _template("stub-pt-dup-2", STUB_PROMPT_DUPLICATE_NAME),
    ]


def get_stub_prompt_template_versions(
    prompt_template_ids: list[str],
) -> dict[str, list[SimpleNamespace]]:
    """Return stub versions for each template id (for stub mode)."""
    result: dict[str, list[SimpleNamespace]] = {}
    if "stub-pt-1" in prompt_template_ids:
        result["stub-pt-1"] = [
            _version(
                "stub-v-1",
                "stub-pt-1",
                STUB_PROMPT_VERSION_NO_VARS_TEXT,
                [],
            )
        ]
    if "stub-pt-2" in prompt_template_ids:
        result["stub-pt-2"] = [
            _version(
                "stub-v-2",
                "stub-pt-2",
                STUB_PROMPT_VERSION_WITH_VARS_TEXT,
                [_var("name"), _var("sentences")],
            )
        ]
    if "stub-pt-dup-1" in prompt_template_ids:
        result["stub-pt-dup-1"] = [
            _version("stub-v-dup-1", "stub-pt-dup-1", "Duplicate prompt 1.", [])
        ]
    if "stub-pt-dup-2" in prompt_template_ids:
        result["stub-pt-dup-2"] = [
            _version("stub-v-dup-2", "stub-pt-dup-2", "Duplicate prompt 2.", [])
        ]
    return result
