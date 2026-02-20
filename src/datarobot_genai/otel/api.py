import os

import requests

from datarobot_genai.otel.enums import OtelEntity


def _get_headers() -> dict:
    token = os.environ["DATAROBOT_API_TOKEN"]
    return {"Authorization": f"Bearer {token}"}


def _base_url() -> str:
    return os.environ["DATAROBOT_ENDPOINT"].rstrip("/") + "/api/v2"


def list_traces(entity: OtelEntity) -> list:
    url = f"{_base_url()}/tracing/{entity.entity_type}/{entity.entity_id}/"
    response = requests.get(url, headers=_get_headers())
    response.raise_for_status()
    return response.json()


def get_trace(entity: OtelEntity, trace_id: str) -> dict:
    url = f"{_base_url()}/tracing/{entity.entity_type}/{entity.entity_id}/{trace_id}/"
    response = requests.get(url, headers=_get_headers())
    response.raise_for_status()
    return response.json()
