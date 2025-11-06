from typing import Any
import pytest


@pytest.fixture
def agent_auth_context_data() -> dict[str, Any]:
    """Return sample authorization context data with required AuthCtx fields."""
    return {
        "user": {"id": "123", "name": "foo", "email": "foo@example.com"},
        "identities": [
            {"id": "id123", "type": "user", "provider_type": "github", "provider_user_id": "123"}
        ],
    }
