from typing import Literal

from pydantic import BaseModel

class OtelEntity(BaseModel):
    entity_type: Literal[
        "deployment",
        "use_case",
        "experiment_container",
        "custom_application",
        "workload",
        "workload_deployment"
    ]
    entity_id: str
