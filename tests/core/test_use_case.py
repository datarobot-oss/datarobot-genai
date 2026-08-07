# Copyright 2026 DataRobot, Inc. and its affiliates.
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

from dataclasses import dataclass
from dataclasses import field

from datarobot_genai.core import use_case as use_case_mod
from datarobot_genai.core.use_case import find_or_create_use_case


@dataclass
class _FakeUseCase:
    id: str
    name: str


@dataclass
class _FakeUseCaseAPI:
    """Fakes ``dr.UseCase``'s ``list``/``create`` classmethods."""

    existing: list[_FakeUseCase] = field(default_factory=list)
    created: _FakeUseCase | None = None
    create_calls: list[dict] = field(default_factory=list)

    def list(self, search_params=None):
        return list(self.existing)

    def create(self, name=None, description=None):
        self.create_calls.append({"name": name, "description": description})
        return self.created


def _install_fake_dr(monkeypatch, use_case_api) -> None:
    fake_dr = type("_FakeDr", (), {"UseCase": use_case_api})()
    monkeypatch.setattr(use_case_mod, "dr", fake_dr)


class TestFindOrCreateUseCase:
    def test_exact_match_returns_id_without_creating(self, monkeypatch):
        api = _FakeUseCaseAPI(existing=[_FakeUseCase(id="existing-uc", name="Quickstart")])
        _install_fake_dr(monkeypatch, api)

        assert find_or_create_use_case("Quickstart") == "existing-uc"
        assert api.create_calls == []

    def test_no_match_creates_use_case(self, monkeypatch):
        api = _FakeUseCaseAPI(created=_FakeUseCase(id="new-uc", name="Quickstart"))
        _install_fake_dr(monkeypatch, api)

        assert find_or_create_use_case("Quickstart") == "new-uc"
        assert len(api.create_calls) == 1
        # A blank description leaves an unexplained use case in the org.
        assert api.create_calls[0]["description"]

    def test_near_miss_name_does_not_count_as_match(self, monkeypatch):
        api = _FakeUseCaseAPI(
            existing=[_FakeUseCase(id="v2-uc", name="Quickstart v2")],
            created=_FakeUseCase(id="new-uc", name="Quickstart"),
        )
        _install_fake_dr(monkeypatch, api)

        assert find_or_create_use_case("Quickstart") == "new-uc"
        assert len(api.create_calls) == 1

    def test_list_raises_returns_none_without_propagating(self, monkeypatch):
        class _RaisingUseCaseAPI:
            def list(self, search_params=None):
                raise RuntimeError("boom")

        _install_fake_dr(monkeypatch, _RaisingUseCaseAPI())

        assert find_or_create_use_case("Quickstart") is None

    def test_create_raises_returns_none_without_propagating(self, monkeypatch):
        class _RaisingCreateUseCaseAPI:
            def list(self, search_params=None):
                return []

            def create(self, name=None, description=None):
                raise RuntimeError("boom")

        _install_fake_dr(monkeypatch, _RaisingCreateUseCaseAPI())

        assert find_or_create_use_case("Quickstart") is None
