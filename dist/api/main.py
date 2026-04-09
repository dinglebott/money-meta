from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import json

from api.inference import loadModels, predict
from api.models import PredictionResponse, CandleInfo
from api.data_processing import getData, parseData

logger = logging.getLogger(__name__)
ARTIFACTS = Path("artifacts")
xgbVersion = 10
nnVersion = 5.4
xgbH1Version = 10.1

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup — load models into memory
    logger.info("Loading models...")
    loadModels()
    logger.info("Models loaded.")
    yield
    # Runs at shutdown (cleanup if needed)

app = FastAPI(
    title="Tree Trader Inference API",
    version="1.3.1",
    lifespan=lifespan
)

# Allow your PWA frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dinglebott.github.io"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# guard against silent failure
assert (ARTIFACTS / f"xgbFeatures_v{xgbVersion}.json").exists(), \
    f"XGB feature list not found for version {xgbVersion}"
assert (ARTIFACTS / f"nnFeatures_v{nnVersion}.json").exists(), \
    f"NN feature list not found for version {nnVersion}"
assert (ARTIFACTS / f"xgbFeatures_v{xgbH1Version}.json").exists(), \
    f"XGB feature list not found for version {xgbH1Version}"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict", response_model=PredictionResponse)
def getPrediction():
    with open(ARTIFACTS / f"xgbFeatures_v{xgbVersion}.json", "r") as file:
        xgbFeatureList = json.load(file)["features"]
    with open(ARTIFACTS / f"nnFeatures_v{nnVersion}.json", "r") as file:
        nnFeatureList = json.load(file)["features"]
    with open(ARTIFACTS / f"xgbFeatures_v{xgbH1Version}.json", "r") as file:
        xgbH1FeatureList = json.load(file)["features"]

    try:
        jsonData, timestamp = getData("EUR_USD", "H4", 400)
        featuresDf = parseData(jsonData)
        xgbFeaturesDf = featuresDf[xgbFeatureList]
        nnFeaturesDf = featuresDf[nnFeatureList]

        jsonDataH1, timestampH1 = getData("EUR_USD", "H1", 400)
        featuresDfH1 = parseData(jsonDataH1)
        xgbH1FeaturesDf = featuresDfH1[xgbH1FeatureList]

        result = predict(xgbFeaturesDf, nnFeaturesDf, xgbH1FeaturesDf)
        return PredictionResponse(
            **result,
            timestamp=timestamp,
            xgbModelVersion=f"{xgbVersion}",
            nnModelVersion=f"{nnVersion}",
            xgbH1ModelVersion=f"{xgbH1Version}"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candle", response_model=CandleInfo)
def getCandleInfo():
    try:
        jsonData, timestamp = getData("EUR_USD", "H4", 400)
        df = parseData(jsonData) # incomplete candle dropped already
        lastCompleteCandle = df.iloc[-1]
        return CandleInfo(
            open=lastCompleteCandle["open"].item(),
            high=lastCompleteCandle["high"].item(),
            low=lastCompleteCandle["low"].item(),
            close=lastCompleteCandle["close"].item(),
            rsi=lastCompleteCandle["rsi_14"].item() + 50.0,
            atr=lastCompleteCandle["raw_atr"].item(),
            timestamp=timestamp
        )
    except Exception as e:
        logger.error(f"Candle retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candle/h1", response_model=CandleInfo)
def getCandleInfoH1():
    try:
        jsonData, timestamp = getData("EUR_USD", "H1", 500)
        df = parseData(jsonData)
        lastCompleteCandle = df.iloc[-1]
        return CandleInfo(
            open=lastCompleteCandle["open"].item(),
            high=lastCompleteCandle["high"].item(),
            low=lastCompleteCandle["low"].item(),
            close=lastCompleteCandle["close"].item(),
            rsi=lastCompleteCandle["rsi_14"].item() + 50.0,
            atr=lastCompleteCandle["raw_atr"].item(),
            timestamp=timestamp
        )
    except Exception as e:
        logger.error(f"Candle H1 retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))