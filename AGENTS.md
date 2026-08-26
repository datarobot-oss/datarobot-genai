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

1. Read README.md
2. Then read .cursor/rules
3. Bump the version for each change that ships. Update pyproject.toml and run `task install` to do so.
   For a stack of dependent PRs, bump once on the PR that lands last: `version-check` reads the whole
   stack, so the tip's bump satisfies every PR in it. See
   [CONTRIBUTING.md](CONTRIBUTING.md#versioning-and-releases).
