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

import base64

import pytest
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from a2a.utils.constants import PREV_AGENT_CARD_WELL_KNOWN_PATH
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import BaseRoute
from starlette.routing import Mount
from starlette.routing import Route
from starlette.testclient import TestClient

from datarobot_genai.dragent.constants import A2A_MOUNT_PATH
from datarobot_genai.dragent.frontends.claim_validation import GeneralOAuthClaimValidationMiddleware
from datarobot_genai.dragent.frontends.claim_validation import is_agent_card_path
from datarobot_genai.dragent.inbound_token import OAUTH_ACCESS_TOKEN_HEADER

from ..helpers import make_jwt

EXPECTED_AUDIENCE = "https://app.datarobot.com/org-1/agent-1"
OTHER_AUDIENCE = "https://app.datarobot.com/org-1/agent-2"
SECRET_CLAIM = "super-secret-subject"
# Opaque DataRobot API token: what `authorization` actually carries on the serving routes.
DATAROBOT_API_TOKEN = "NjRiYWE1Njk5NmZiMzZlM2VlZWVmYzQ0"


def _ok(text: str):
    return lambda _request: PlainTextResponse(text)


def _app(
    *,
    routes: list[BaseRoute],
    guarded: bool = True,
    expected_audience: str = EXPECTED_AUDIENCE,
) -> Starlette:
    """Build an app guarded by the middleware; ``guarded=False`` for the unguarded baseline."""
    middleware = (
        [Middleware(GeneralOAuthClaimValidationMiddleware, expected_audience=expected_audience)]
        if guarded
        else []
    )
    return Starlette(routes=routes, middleware=middleware)


def _a2a_routes() -> list[BaseRoute]:
    """Return the routes A2AStarletteApplication.build() registers, as the inner app sees them."""
    return [
        Route("/", _ok("executed"), methods=["POST"]),
        Route(AGENT_CARD_WELL_KNOWN_PATH, _ok("card"), methods=["GET"]),
    ]


class TestAudienceValidation:
    """The audience check itself, exercised over the A2A routes."""

    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(_app(routes=_a2a_routes()))

    def test_agent_card_path_is_exempt(self, client):
        """GIVEN no token WHEN the agent card is fetched THEN the middleware passes it through."""
        response = client.get(AGENT_CARD_WELL_KNOWN_PATH)
        assert response.status_code == 200
        assert response.text == "card"

    def test_agent_card_path_is_exempt_when_mounted(self):
        """GIVEN the app mounted under /a2a WHEN the card is fetched THEN still exempt.

        Guards the suffix match: the mount prefix is present in ``request.url.path``.
        """
        outer = Starlette(
            routes=[
                Mount(
                    f"/{A2A_MOUNT_PATH}",
                    app=_app(routes=_a2a_routes()),
                )
            ]
        )
        with TestClient(outer) as client:
            response = client.get(f"/{A2A_MOUNT_PATH}{AGENT_CARD_WELL_KNOWN_PATH}")
        assert response.status_code == 200
        assert response.text == "card"

    def test_string_audience_claim_matching_passes(self, client):
        """GIVEN aud as a string equal to the expected audience THEN the request is allowed."""
        token = make_jwt(aud=EXPECTED_AUDIENCE, sub="user-1")
        response = client.post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: token})
        assert response.status_code == 200
        assert response.text == "executed"

    def test_list_audience_claim_containing_expected_passes(self, client):
        """GIVEN aud as a list containing the expected audience THEN the request is allowed."""
        token = make_jwt(aud=[OTHER_AUDIENCE, EXPECTED_AUDIENCE])
        response = client.post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: token})
        assert response.status_code == 200

    def test_bare_token_in_datarobot_header_passes(self, client):
        """GIVEN a bare (non-Bearer) token in the DataRobot header THEN it is still read."""
        token = make_jwt(aud=EXPECTED_AUDIENCE)
        response = client.post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: token})
        assert response.status_code == 200

    def test_bearer_prefixed_token_in_datarobot_header_passes(self, client):
        """GIVEN a Bearer-prefixed token in the DataRobot header THEN the prefix is stripped."""
        token = make_jwt(aud=EXPECTED_AUDIENCE)
        response = client.post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: f"Bearer {token}"})
        assert response.status_code == 200

    def test_authorization_bearer_header_is_used_as_fallback(self, client):
        """GIVEN no DataRobot header WHEN authorization carries a Bearer JWT THEN it is read."""
        token = make_jwt(aud=EXPECTED_AUDIENCE)
        response = client.post("/", headers={"authorization": f"Bearer {token}"})
        assert response.status_code == 200

    def test_mismatched_audience_claim_is_rejected(self, client):
        """GIVEN aud naming a different agent THEN the request is rejected with 401."""
        token = make_jwt(aud=OTHER_AUDIENCE)
        response = client.post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: token})
        assert response.status_code == 401
        assert "detail" in response.json()

    def test_missing_audience_claim_is_rejected(self, client):
        """GIVEN a JWT with no aud claim THEN the request is rejected with 401."""
        response = client.post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(sub="user-1")})
        assert response.status_code == 401

    def test_non_string_audience_claim_is_rejected(self, client):
        """GIVEN an aud claim that is neither a string nor a list of strings THEN 401."""
        response = client.post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=42)})
        assert response.status_code == 401

    def test_missing_token_passes_through(self, client):
        """GIVEN no IdP token THEN the request is not rejected -- this is not an auth check.

        A DataRobot API token caller sends none; rejecting here would break them.
        """
        assert client.post("/").status_code == 200

    def test_datarobot_api_token_on_a2a_passes_through(self, client):
        """GIVEN a DataRobot API token on the A2A endpoint THEN it is not rejected.

        A2A accepts one whenever cross_application_access is not the caller's auth method.
        """
        response = client.post("/", headers={"authorization": f"Bearer {DATAROBOT_API_TOKEN}"})
        assert response.status_code == 200

    def test_basic_authorization_header_is_not_treated_as_a_token(self, client):
        """GIVEN a Basic authorization header THEN it is not read as an IdP token."""
        credential = base64.b64encode(b"user:password").decode()
        response = client.post("/", headers={"authorization": f"Basic {credential}"})
        assert response.status_code == 200

    def test_malformed_token_is_rejected_as_unprocessable(self, client):
        """GIVEN a token that is not a JWT THEN the request is rejected with 422."""
        response = client.post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: "not-a-jwt"})
        assert response.status_code == 422
        assert response.json()["detail"].startswith("Malformed authorization token:")

    def test_rejection_body_leaks_neither_token_nor_claims(self, client):
        """GIVEN a rejected token THEN neither it nor its claim values leak into the body."""
        token = make_jwt(aud=OTHER_AUDIENCE, sub=SECRET_CLAIM)
        response = client.post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: token})
        assert response.status_code == 401
        body = response.text
        assert token not in body
        assert SECRET_CLAIM not in body
        assert OTHER_AUDIENCE not in body
        assert EXPECTED_AUDIENCE not in body

    def test_no_middleware_allows_unauthenticated_requests(self):
        """GIVEN the middleware is not installed THEN inbound requests are not validated."""
        with TestClient(_app(routes=_a2a_routes(), guarded=False)) as client:
            assert client.post("/").status_code == 200


class TestExactAudienceMatching:
    """`aud` must equal the expected audience exactly -- no prefix, suffix, case or
    whitespace leniency, and no substring match.
    """

    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(_app(routes=_a2a_routes()))

    @pytest.mark.parametrize(
        "aud",
        [
            f"-{EXPECTED_AUDIENCE}",
            f"{EXPECTED_AUDIENCE}-",
            f"{EXPECTED_AUDIENCE}/",
            f" {EXPECTED_AUDIENCE}",
            f"{EXPECTED_AUDIENCE} ",
            EXPECTED_AUDIENCE.upper(),
            EXPECTED_AUDIENCE[:-1],
            f"x{EXPECTED_AUDIENCE}x",
        ],
    )
    def test_near_miss_is_rejected(self, client, aud):
        """GIVEN an aud that merely resembles ours THEN the request is rejected."""
        response = client.post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=aud)})
        assert response.status_code == 401

    def test_exact_value_is_accepted(self, client):
        response = client.post(
            "/", headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=EXPECTED_AUDIENCE)}
        )
        assert response.status_code == 200

    def test_list_of_near_misses_is_rejected(self, client):
        """A list claim is matched entry by entry, not by containment."""
        token = make_jwt(aud=[f"-{EXPECTED_AUDIENCE}", f"{EXPECTED_AUDIENCE}-"])
        assert client.post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: token}).status_code == 401


class TestServingRoutes:
    """The same instance also covers the non-A2A routes.

    NAT copies inbound headers into the workflow context on every route, so a token sent to
    /chat/completions is exchanged just as one sent to /a2a/ would be.
    """

    @pytest.fixture
    def client(self) -> TestClient:
        inner_a2a = Starlette(routes=[Route(AGENT_CARD_WELL_KNOWN_PATH, _ok("card"))])
        return TestClient(
            _app(
                routes=[
                    Route("/health", _ok("healthy"), methods=["GET"]),
                    Route("/chat/completions", _ok("completion"), methods=["POST"]),
                    Mount(f"/{A2A_MOUNT_PATH}", app=inner_a2a),
                ],
            )
        )

    def test_datarobot_api_token_request_is_untouched(self, client):
        """GIVEN a DataRobot API token and no IdP token THEN the request proceeds.

        Treating it as an IdP token would 422 every legitimate chat completion.
        """
        response = client.post(
            "/chat/completions", headers={"authorization": f"Bearer {DATAROBOT_API_TOKEN}"}
        )
        assert response.status_code == 200
        assert response.text == "completion"

    def test_request_without_any_token_is_untouched(self, client):
        """GIVEN no credentials at all THEN this middleware does not reject the request."""
        assert client.post("/chat/completions").status_code == 200

    def test_health_probe_is_untouched(self, client):
        """GIVEN a k8s-style probe with no headers THEN it is not rejected."""
        assert client.get("/health").status_code == 200

    def test_idp_token_naming_this_agent_passes(self, client):
        """GIVEN an IdP token whose aud is this agent THEN the request proceeds."""
        token = make_jwt(aud=EXPECTED_AUDIENCE)
        response = client.post("/chat/completions", headers={OAUTH_ACCESS_TOKEN_HEADER: token})
        assert response.status_code == 200

    def test_idp_token_naming_another_agent_is_rejected(self, client):
        """GIVEN a token minted for another agent THEN /chat/completions rejects it too.

        The bypass this closes: the token used to be refused at /a2a/ and accepted here.
        """
        token = make_jwt(aud=OTHER_AUDIENCE)
        response = client.post("/chat/completions", headers={OAUTH_ACCESS_TOKEN_HEADER: token})
        assert response.status_code == 401
        assert "detail" in response.json()

    def test_idp_token_without_audience_is_rejected(self, client):
        response = client.post(
            "/chat/completions", headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(sub="user-1")}
        )
        assert response.status_code == 401

    def test_malformed_idp_token_is_rejected(self, client):
        response = client.post(
            "/chat/completions", headers={OAUTH_ACCESS_TOKEN_HEADER: "not-a-jwt"}
        )
        assert response.status_code == 422

    def test_wrong_audience_jwt_in_authorization_is_rejected(self, client):
        """GIVEN a wrong-audience JWT in `authorization` THEN it is rejected here too.

        Both sides read the same carriers now (``dragent.inbound_token``); this header used to
        be skipped here while the XAA provider still exchanged from it.
        """
        token = make_jwt(aud=OTHER_AUDIENCE)
        response = client.post("/chat/completions", headers={"authorization": f"Bearer {token}"})
        assert response.status_code == 401

    def test_matching_audience_jwt_in_authorization_passes(self, client):
        """GIVEN a correct-audience JWT in `authorization` THEN the request proceeds."""
        token = make_jwt(aud=EXPECTED_AUDIENCE)
        response = client.post("/chat/completions", headers={"authorization": f"Bearer {token}"})
        assert response.status_code == 200

    def test_a2a_subtree_is_exempt(self, client):
        """GIVEN a wrong-audience token at the mounted agent card THEN it is still exempt.

        Pre-empting it would override _handle_get_agent_card's redacted-vs-401 decision.
        """
        token = make_jwt(aud=OTHER_AUDIENCE)
        response = client.get(
            f"/{A2A_MOUNT_PATH}{AGENT_CARD_WELL_KNOWN_PATH}",
            headers={OAUTH_ACCESS_TOKEN_HEADER: token},
        )
        assert response.status_code == 200
        assert response.text == "card"


class TestFallbackHeaderClassification:
    """`authorization` is shared with the DataRobot API token, so only a real JWT counts.

    Asks the parser, not a dot count: opaque tokens can contain two dots (``v2.local.xxx``).
    """

    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(_app(routes=_a2a_routes()))

    @pytest.mark.parametrize(
        "value",
        [
            "NjRiYWE1Njk5NmZiMzZlM2VlZWVmYzQ0",  # opaque DataRobot API token
            "abc.def.ghi",  # opaque, but two dots - the dot-count heuristic misread this
            "v2.local.k4r3ZXlz",  # segmented opaque token, also two dots
        ],
    )
    def test_opaque_value_in_fallback_header_is_left_alone(self, client, value):
        """GIVEN a non-JWT in `authorization` THEN it is neither validated nor rejected."""
        assert client.post("/", headers={"authorization": f"Bearer {value}"}).status_code == 200

    def test_real_jwt_in_fallback_header_is_validated(self, client):
        """GIVEN a decodable JWT in `authorization` THEN its audience is checked."""
        response = client.post(
            "/", headers={"authorization": f"Bearer {make_jwt(aud=OTHER_AUDIENCE)}"}
        )
        assert response.status_code == 401

    def test_malformed_value_in_the_dedicated_header_still_reports_422(self, client):
        """The dedicated header carries nothing else, so a non-JWT there is an error."""
        response = client.post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: "abc.def.ghi"})
        assert response.status_code == 422


class TestPathPredicates:
    @pytest.mark.parametrize(
        "path,expected",
        [
            (AGENT_CARD_WELL_KNOWN_PATH, True),
            (PREV_AGENT_CARD_WELL_KNOWN_PATH, True),
            (f"/{A2A_MOUNT_PATH}{AGENT_CARD_WELL_KNOWN_PATH}", True),
            # The A2A RPC endpoint is "/", which must never be exempt on the A2A app.
            ("/", False),
            ("/chat/completions", False),
        ],
    )
    def test_is_agent_card_path(self, path, expected):
        assert is_agent_card_path(path) is expected


class TestMountPrefixRobustness:
    """Validation must not depend on recognising the mount path.

    ``scope["path"]`` keeps every prefix -- the mount's and any deployment's -- so the
    agent-card exemption matches on suffix rather than equality.
    """

    # Neither the default `/a2a` mount nor anything the serving exemption recognises.
    UNRECOGNISED_MOUNT = "/deployments/abc123/a2a"

    def _client(self) -> TestClient:
        inner = _app(routes=_a2a_routes())
        return TestClient(
            _app(
                routes=[Mount(self.UNRECOGNISED_MOUNT, app=inner)],
            )
        )

    def test_wrong_audience_is_still_rejected_under_an_unrecognised_mount(self):
        """GIVEN an unexpected mount prefix THEN a wrong-audience token is still 401."""
        with self._client() as client:
            response = client.post(
                f"{self.UNRECOGNISED_MOUNT}/",
                headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=OTHER_AUDIENCE)},
            )
        assert response.status_code == 401
        assert response.text != "executed"

    def test_agent_card_keeps_its_own_auth_decision_under_an_unrecognised_mount(self):
        """GIVEN a wrong-audience token on the card route under any prefix THEN still exempt."""
        with self._client() as client:
            response = client.get(
                f"{self.UNRECOGNISED_MOUNT}{AGENT_CARD_WELL_KNOWN_PATH}",
                headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=OTHER_AUDIENCE)},
            )
        assert response.status_code == 200
        assert response.text == "card"

    @pytest.mark.parametrize(
        "path",
        [
            AGENT_CARD_WELL_KNOWN_PATH,
            f"/deployments/abc123/a2a{AGENT_CARD_WELL_KNOWN_PATH}",
            f"/some/other/prefix/a2a{PREV_AGENT_CARD_WELL_KNOWN_PATH}",
        ],
    )
    def test_agent_card_is_exempt_under_any_prefix(self, path):
        assert is_agent_card_path(path) is True
