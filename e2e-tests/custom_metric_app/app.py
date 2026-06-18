from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="Exclamation Counter")


class ScoringRequest(BaseModel):
    df: dict
    stage: str
    metric_column_name: str
    input_column: str


@app.post("/")
async def count_exclamations(body: ScoringRequest) -> dict:
    df = pd.DataFrame(body.df)
    df[body.metric_column_name] = df[body.input_column].astype(str).str.count("!")
    return {"df": df.to_dict()}


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}
