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
import logging

import pytest
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import BaseRoute
from starlette.routing import Mount
from starlette.routing import Route
from starlette.testclient import TestClient

from datarobot_genai.dragent.constants import A2A_MOUNT_PATH
from datarobot_genai.dragent.frontends.claim_validation import GeneralOAuthClaimValidationMiddleware
from datarobot_genai.dragent.frontends.probe_paths import DATAROBOT_EXPECTED_HEALTH_ROUTES
from datarobot_genai.dragent.inbound_token import OAUTH_ACCESS_TOKEN_HEADER
from datarobot_genai.dragent.inbound_token import _authorization_carries_idp_token

from ..helpers import make_jwt

EXPECTED_AUDIENCE = "https://app.datarobot.com/org-1/agent-1"
OTHER_AUDIENCE = "https://app.datarobot.com/org-1/agent-2"
SECRET_CLAIM = "super-secret-subject"
# Opaque DataRobot API token: what `authorization` actually carries on the serving routes.
DATAROBOT_API_TOKEN = "NjRiYWE1Njk5NmZiMzZlM2VlZWVmYzQ0"

# A representative DataRobot-issued platform credential: equivalent in trust to a DataRobot
# API key, and a perfectly decodable JWT -- which is why it cannot be classified by shape.
# It arrives in `authorization`, never in the gateway's own header.  `aud` is empty because
# it was never minted with this (or any) agent in mind.
DATAROBOT_ISSUED_CLAIMS = {
    "aud": [],
    "client_id": "11111111-1111-1111-1111-111111111111",
    "iss": "https://issuer.example.com",
    "scope": "",
    "sub": "11111111-1111-1111-1111-111111111111",
}

# An external IdP token whose `aud` names the installation rather than an agent: already
# validated and forwarded by the gateway, and exactly what audience binding exists to catch.
INSTALLATION_PREFIX = "https://app.example.com"
IDP_AGENT_ID = "example-agent-principal-id"
PLATFORM_SCOPED_CLAIMS = {
    "iss": "https://idp.example.com/oauth2/default",
    "aud": INSTALLATION_PREFIX,
    "cid": "example-client-id",
    "sub": "example-client-id",
}

# The same token minted for one agent instead.
AGENT_BOUND_AUDIENCE = f"{INSTALLATION_PREFIX}/agents/{IDP_AGENT_ID}"
AGENT_BOUND_CLAIMS = {**PLATFORM_SCOPED_CLAIMS, "aud": AGENT_BOUND_AUDIENCE}


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

    def test_unauthenticated_agent_card_passes_through(self, client):
        """GIVEN no token WHEN the agent card is fetched THEN the middleware passes it through.

        Auth on that route is optional; _handle_get_agent_card decides via
        enable_unauthenticated_well_known_route.
        """
        response = client.get(AGENT_CARD_WELL_KNOWN_PATH)
        assert response.status_code == 200
        assert response.text == "card"

    def test_agent_card_with_wrong_audience_is_rejected(self, client):
        """GIVEN a token naming another agent THEN the card route rejects it too.

        No route is exempt: once a token is presented, it is fully authorized before the
        handler gets to decide what card to serve.
        """
        response = client.get(
            AGENT_CARD_WELL_KNOWN_PATH,
            headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=OTHER_AUDIENCE)},
        )
        assert response.status_code == 401

    def test_agent_card_with_matching_audience_passes_through(self, client):
        response = client.get(
            AGENT_CARD_WELL_KNOWN_PATH,
            headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=EXPECTED_AUDIENCE)},
        )
        assert response.status_code == 200
        assert response.text == "card"

    def test_unauthenticated_agent_card_passes_through_when_mounted(self):
        """GIVEN the app mounted under /a2a WHEN the card is fetched with no token THEN allowed."""
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

    def test_authorization_bearer_jwt_is_not_read_by_default(self, client):
        """GIVEN a JWT in `authorization` and no opt-in THEN it is not read as an IdP token.

        `authorization` carries DataRobot's own credentials, several of which are JWTs.  The
        gateway puts the caller's IdP token in its own header; only that one is in scope.
        """
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

    @pytest.mark.parametrize("aud", [OTHER_AUDIENCE, EXPECTED_AUDIENCE])
    def test_jwt_in_authorization_is_not_read_by_default(self, client, aud):
        """GIVEN a JWT in `authorization` and no opt-in THEN its `aud` is never inspected.

        Parametrized over both audiences deliberately: they take the same path for the same
        reason, because the header is not an IdP carrier and the claim is never decoded.  Not
        a bypass -- with the fallback off the XAA provider does not exchange from this header
        either, so nothing is validated here but exchanged there.  ``TestAuthorizationOptIn``
        covers the local-dev configuration where the two audiences do diverge.
        """
        token = make_jwt(aud=aud)
        response = client.post("/chat/completions", headers={"authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.text == "completion"

    def test_mounted_a2a_subtree_is_checked_too(self, client):
        """GIVEN a wrong-audience token inside the /a2a mount THEN it is rejected.

        One instance covers the mounted app as well; nothing under /a2a is exempt.
        """
        response = client.get(
            f"/{A2A_MOUNT_PATH}{AGENT_CARD_WELL_KNOWN_PATH}",
            headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=OTHER_AUDIENCE)},
        )
        assert response.status_code == 401

    def test_unauthenticated_request_into_the_a2a_mount_passes_through(self, client):
        response = client.get(f"/{A2A_MOUNT_PATH}{AGENT_CARD_WELL_KNOWN_PATH}")
        assert response.status_code == 200
        assert response.text == "card"


class TestCarrierScoping:
    """Which credential is an IdP token at all -- the question that precedes the audience check.

    Only ``x-datarobot-external-access-token`` carries one.  The gateway populates it with the
    external token it already validated, so a value there is in scope by construction;
    ``authorization`` carries DataRobot's own credentials and is out of scope by default.
    """

    def _client(self, expected_audience: str = EXPECTED_AUDIENCE) -> TestClient:
        return TestClient(_app(routes=_a2a_routes(), expected_audience=expected_audience))

    def test_datarobot_issued_jwt_in_authorization_passes_through(self):
        """GIVEN DataRobot's own platform credential in `authorization` THEN it is untouched.

        The bug this fixes: it happens to decode as a JWT, so reading it as an IdP token
        rejected DataRobot-authenticated calls to an agent with the flag on.
        """
        token = make_jwt(**DATAROBOT_ISSUED_CLAIMS)
        response = self._client().post("/", headers={"authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.text == "executed"

    def test_datarobot_issued_jwt_is_rejected_under_the_opt_in(
        self, authorization_carries_idp_token
    ):
        """GIVEN the opt-in THEN the same token is read as an IdP token and fails on `aud`.

        Local-dev configuration only: it says "there is no gateway in front, so treat a JWT in
        `authorization` as the caller's IdP token".  A deployment that turned it on would
        reject its own platform traffic, which is precisely why it defaults off.
        """
        token = make_jwt(**DATAROBOT_ISSUED_CLAIMS)
        response = self._client().post("/", headers={"authorization": f"Bearer {token}"})
        assert response.status_code == 401

    @pytest.mark.parametrize("opt_in", [False, True])
    def test_opaque_datarobot_api_token_in_authorization_passes_through(self, request, opt_in):
        """GIVEN an opaque DataRobot API token THEN it is never an IdP token, opt-in or not.

        It does not decode, so the fallback declines it even when it is switched on.
        """
        if opt_in:
            request.getfixturevalue("authorization_carries_idp_token")
        response = self._client().post(
            "/", headers={"authorization": f"Bearer {DATAROBOT_API_TOKEN}"}
        )
        assert response.status_code == 200
        assert response.text == "executed"

    def test_platform_scoped_idp_token_is_rejected(self):
        """GIVEN an IdP token whose `aud` names the installation THEN this agent refuses it.

        Already validated and forwarded by the gateway; `aud` just does not name us.
        """
        token = make_jwt(**PLATFORM_SCOPED_CLAIMS)
        response = self._client().post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: token})
        assert response.status_code == 401

    def test_agent_bound_idp_token_is_accepted_by_the_matching_agent(self):
        """GIVEN an `aud` naming this agent specifically THEN that agent accepts it."""
        token = make_jwt(**AGENT_BOUND_CLAIMS)
        response = self._client(expected_audience=AGENT_BOUND_AUDIENCE).post(
            "/", headers={OAUTH_ACCESS_TOKEN_HEADER: token}
        )
        assert response.status_code == 200
        assert response.text == "executed"

    def test_agent_bound_idp_token_is_refused_by_another_agent(self):
        """GIVEN the same token THEN an agent with a different principal still refuses it."""
        token = make_jwt(**AGENT_BOUND_CLAIMS)
        other = f"{INSTALLATION_PREFIX}/agents/some-other-agent-principal"
        response = self._client(expected_audience=other).post(
            "/", headers={OAUTH_ACCESS_TOKEN_HEADER: token}
        )
        assert response.status_code == 401

    def test_empty_audience_list_in_the_dedicated_header_is_rejected(self):
        """GIVEN `aud: []` in the gateway's header THEN it is rejected.

        Asserted separately from a missing claim below: ``_audience_claim`` collapses both to
        ``[]``, and neither may be read as "no audience requirement, let it through".
        """
        response = self._client().post("/", headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=[])})
        assert response.status_code == 401

    def test_absent_audience_claim_in_the_dedicated_header_is_rejected(self):
        """GIVEN no `aud` claim at all THEN it is rejected, by the same rule."""
        response = self._client().post(
            "/", headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(sub="user-1")}
        )
        assert response.status_code == 401

    def test_unreadable_opt_in_does_not_break_pass_through(self, monkeypatch, caplog):
        """GIVEN an unparseable value for the opt-in THEN requests still pass through.

        The settings read sits on the path of *every* request, including the ones carrying no
        IdP token at all.  Letting a config typo out of it would turn the traffic this library
        promises to leave alone into 500s, so it resolves to off -- the field default, and what
        a deployment wants -- and says so in the log.
        """
        monkeypatch.setenv("DRAGENT_ALLOW_IDP_TOKEN_IN_AUTHORIZATION", "perhaps")
        _authorization_carries_idp_token.cache_clear()
        token = make_jwt(aud=OTHER_AUDIENCE)
        with caplog.at_level(logging.WARNING):
            response = self._client().post("/", headers={"authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.text == "executed"
        assert "DRAGENT_ALLOW_IDP_TOKEN_IN_AUTHORIZATION" in caplog.text


class TestAuthorizationOptIn:
    """``DRAGENT_ALLOW_IDP_TOKEN_IN_AUTHORIZATION`` -- local runs with no gateway in front.

    The carrier set is shared with the XAA provider (``dragent.inbound_token``), so whatever
    audience validation reads is exactly what gets exchanged.  With the opt-in on, a JWT in
    `authorization` is exchangeable, so it must also be validated -- the bypass this covers.
    """

    @pytest.fixture
    def client(self, authorization_carries_idp_token) -> TestClient:
        return TestClient(
            _app(
                routes=[
                    Route("/chat/completions", _ok("completion"), methods=["POST"]),
                ],
            )
        )

    def test_wrong_audience_jwt_in_authorization_is_rejected(self, client):
        """GIVEN the opt-in and a wrong-audience JWT in `authorization` THEN it is rejected."""
        token = make_jwt(aud=OTHER_AUDIENCE)
        response = client.post("/chat/completions", headers={"authorization": f"Bearer {token}"})
        assert response.status_code == 401

    def test_matching_audience_jwt_in_authorization_passes(self, client):
        """GIVEN the opt-in and a correct-audience JWT THEN the request proceeds."""
        token = make_jwt(aud=EXPECTED_AUDIENCE)
        response = client.post("/chat/completions", headers={"authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.text == "completion"

    def test_opaque_api_token_in_authorization_still_passes(self, client):
        """GIVEN the opt-in THEN an opaque DataRobot API token is still left alone."""
        response = client.post(
            "/chat/completions", headers={"authorization": f"Bearer {DATAROBOT_API_TOKEN}"}
        )
        assert response.status_code == 200


class TestFallbackHeaderClassification:
    """Under the opt-in, `authorization` is shared with the DataRobot API token, so only a
    real JWT counts.

    Asks the parser, not a dot count: opaque tokens can contain two dots (``v2.local.xxx``).
    Local-dev configuration only -- ``DRAGENT_ALLOW_IDP_TOKEN_IN_AUTHORIZATION`` is off in a
    deployment, where the header is not read at all.
    """

    @pytest.fixture
    def client(self, authorization_carries_idp_token) -> TestClient:
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


class TestHealthProbeExemption:
    """Health/readiness routes are exempt, so a probe is never 401'd out of readiness.

    The platform's probe carries whatever token the gateway attaches; its ``aud`` names the
    platform, not this agent.  Checking it leaves the workload permanently un-ready.
    """

    def _client(self, root_path: str = "") -> TestClient:
        routes = [
            Route(path, _ok("healthy"), methods=["GET", "HEAD"])
            for path in DATAROBOT_EXPECTED_HEALTH_ROUTES
        ]
        routes.append(Route("/chat/completions", _ok("completion"), methods=["POST"]))
        return TestClient(_app(routes=routes), root_path=root_path)

    @pytest.mark.parametrize("path", DATAROBOT_EXPECTED_HEALTH_ROUTES)
    def test_wrong_audience_probe_is_allowed(self, path):
        """GIVEN a probe carrying a token for another agent THEN it still succeeds."""
        with self._client() as client:
            response = client.get(
                path, headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=OTHER_AUDIENCE)}
            )
        assert response.status_code == 200, path
        assert response.text == "healthy"

    @pytest.mark.parametrize("path", DATAROBOT_EXPECTED_HEALTH_ROUTES)
    def test_wrong_audience_probe_is_allowed_under_a_mount_prefix(self, path):
        """GIVEN the deployment's ``--root_path`` THEN the prefixed probe is still exempt."""
        root_path = "/6a983b0b73f5f93c12b3be0c/6a983c7931cd39434aacda20"
        with self._client(root_path=root_path) as client:
            response = client.get(
                f"{root_path}{path}",
                headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=OTHER_AUDIENCE)},
            )
        assert response.status_code == 200, path
        assert response.text == "healthy"

    @pytest.mark.parametrize("path", DATAROBOT_EXPECTED_HEALTH_ROUTES)
    def test_malformed_token_on_a_probe_is_allowed(self, path):
        """GIVEN a probe carrying an undecodable JWT THEN it is not 422'd either.

        The exemption runs before the token is read, so no probe can fail on its credential.
        """
        with self._client() as client:
            response = client.get(path, headers={OAUTH_ACCESS_TOKEN_HEADER: "not.a.jwt"})
        assert response.status_code == 200, path

    @pytest.mark.parametrize("path", DATAROBOT_EXPECTED_HEALTH_ROUTES)
    def test_wrong_audience_bearer_probe_is_allowed(self, path):
        """GIVEN the token arrives as ``authorization: Bearer`` THEN the probe still succeeds.

        The exemption short-circuits before the token is read, so it holds whichever of the two
        headers ``find_idp_token`` would have taken it from.
        """
        with self._client() as client:
            response = client.get(
                path, headers={"authorization": f"Bearer {make_jwt(aud=OTHER_AUDIENCE)}"}
            )
        assert response.status_code == 200, path
        assert response.text == "healthy"

    def test_serving_route_is_still_checked(self):
        """GIVEN the exemption THEN a real serving route still rejects a foreign token."""
        with self._client() as client:
            response = client.post(
                "/chat/completions",
                headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=OTHER_AUDIENCE)},
            )
        assert response.status_code == 401

    def test_non_probe_method_on_a_health_path_is_still_checked(self):
        """GIVEN a POST to a health path THEN the exemption does not apply.

        Only GET/HEAD are registered as health routes, so the exemption is scoped to them --
        a mounted app whose execute endpoint sits at ``/`` stays guarded.
        """
        inner = _app(routes=_a2a_routes())
        with TestClient(inner) as client:
            response = client.post(
                "/", headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=OTHER_AUDIENCE)}
            )
        assert response.status_code == 401


class TestMountPrefixRobustness:
    """Validation must hold under any mount or deployment path prefix.

    ``scope["path"]`` keeps every prefix, so nothing here may depend on recognising one.
    """

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

    def test_agent_card_is_checked_under_an_unrecognised_mount(self):
        """GIVEN a wrong-audience token on the card route THEN it is rejected, prefix or not."""
        with self._client() as client:
            response = client.get(
                f"{self.UNRECOGNISED_MOUNT}{AGENT_CARD_WELL_KNOWN_PATH}",
                headers={OAUTH_ACCESS_TOKEN_HEADER: make_jwt(aud=OTHER_AUDIENCE)},
            )
        assert response.status_code == 401

    def test_unauthenticated_agent_card_still_reaches_the_handler(self):
        """GIVEN no token THEN the card route is untouched, whatever the prefix.

        ``enable_unauthenticated_well_known_route`` stays authoritative for that case.
        """
        with self._client() as client:
            response = client.get(f"{self.UNRECOGNISED_MOUNT}{AGENT_CARD_WELL_KNOWN_PATH}")
        assert response.status_code == 200
        assert response.text == "card"
