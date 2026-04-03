from pydantic import BaseModel

class PredictionResponse(BaseModel):
    xgbPred: str # "down" | "flat" | "up"
    xgbProbs: dict[str, float] # {"down": 0.42, "flat": 0.31, "up": 0.27}
    nnPred: str
    nnProbs: dict[str, float]
    timestamp: str
    xgbModelVersion: str
    nnModelVersion: str

class CandleInfo(BaseModel):
    open: str
    high: str
    low: str
    close: str
    rsi: float
    timestamp: str