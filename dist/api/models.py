from pydantic import BaseModel

class PredictionResponse(BaseModel):
    xgbPred: str # "down" | "flat" | "up"
    xgbProbs: dict[str, float] # {"0": 0.42, "1": 0.31, "2": 0.27}
    nnPred: str
    nnProbs: dict[str, float]
    xgbH1Pred: str
    xgbH1Probs: dict[str, float]
    timestamp: str
    xgbModelVersion: str
    nnModelVersion: str
    xgbH1ModelVersion: str

class CandleInfo(BaseModel):
    open: float
    high: float
    low: float
    close: float
    rsi: float
    timestamp: str