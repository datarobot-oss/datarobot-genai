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

"""Reusable glue for solving optimization problems with a cuOpt NIM deployment.

The cuOpt NIM is a tool-tagged GPU deployment surfaced through dynamic tool
discovery -- there is deliberately **no** monolithic ``cuopt_solve`` MCP tool
here. This package is the shared plumbing around calling that deployment:

* :mod:`~datarobot_genai.drtools.cuopt.schemas` -- the ``cuopt.*`` panel
  schemas (registered with the panels :class:`SchemaRegistry` at import time).
* :mod:`~datarobot_genai.drtools.cuopt.payload` -- payload normalization
  (high-level ``VRPData`` -> native cuOpt format, problem-type inference).
* :mod:`~datarobot_genai.drtools.cuopt.tables` -- solver output -> tabular
  solution tables + summary text.
* :mod:`~datarobot_genai.drtools.cuopt.persistence` -- solution tables ->
  panels, written through the shared :class:`PanelStore`.
* :mod:`~datarobot_genai.drtools.cuopt.deployment` -- deployment resolution
  (``CUOPT_DEPLOYMENT_ID`` env/runtime param, or tool-tagged lookup) and the
  submit/poll client built on the shared deployment URL/auth seam.
* :mod:`~datarobot_genai.drtools.cuopt.solve` -- the high-level
  ``solve_with_cuopt_deployment`` entry point composing all of the above.

Like the sibling panels package, this package intentionally re-exports nothing
to avoid a public-API compatibility surface; import from the submodules, e.g.::

    from datarobot_genai.drtools.cuopt.solve import solve_with_cuopt_deployment
"""
