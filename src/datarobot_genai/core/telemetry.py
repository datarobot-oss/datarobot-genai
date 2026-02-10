from typing import Literal

from pydantic import BaseModel


class OtelEntity(BaseModel):
    entity_type: Literal["use_case"]
    entity_id: str