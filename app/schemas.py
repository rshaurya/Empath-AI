from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    text: str
    top_k: int = 3


class PredictBatchRequest(BaseModel):
    texts: List[str]
    top_k: int = 3
