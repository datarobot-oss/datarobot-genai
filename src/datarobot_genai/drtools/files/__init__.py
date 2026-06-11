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

"""DataRobot Files API integration for drtools (block storage).

Exposes a small :class:`~datarobot_genai.drtools.files.store.BlobStore` Protocol
plus a Files-API-backed implementation, so higher-level domains (e.g. panels)
depend on the storage *seam* rather than a concrete backend.

Import directly from :mod:`datarobot_genai.drtools.files.store`; this package
intentionally re-exports nothing to avoid a public-API compatibility surface.
"""
