from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from api.inference import loadModels, predict
from api.models import PredictionResponse, CandleInfo
from api.data_processing import getData, parseData

logger = logging.getLogger(__name__)

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
    version="1.0.0",
    lifespan=lifespan
)

# Allow your PWA frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dinglebott.github.io/money-meta"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict", response_model=PredictionResponse)
def getPrediction():
    xgbFeatureList = [
        "adx_direction", "ema_cross", "macd_hist", "bb_width", "bb_position",
        "smooth_return", "dist_smooth", "upper_wick", "lower_wick", "volatility_regime",
        "atr_14", "vol_ratio", "vol_return", "dist_ema15"
    ]
    nnFeatureList = [
        "vol_ratio", "volatility_regime", "ema_cross", "upper_wick", "rsi_14",
        "atr_14", "open_return", "dist_ema15", "lower_wick", "smooth_return",
        "vol_momentum", "adx_direction", "vol_return", "close_return"
    ]
    try:
        jsonData, timestamp = getData("EUR_USD", "H4", 500)
        featuresDf = parseData(jsonData)
        xgbFeaturesDf = featuresDf[xgbFeatureList]
        nnFeaturesDf = featuresDf[nnFeatureList]
        result = predict(xgbFeaturesDf, nnFeaturesDf)
        return PredictionResponse(
            **result,
            timestamp=timestamp,
            xgbModelVersion="9",
            nnModelVersion="5.1"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candle", response_model=CandleInfo)
def getCandleInfo():
    try:
        jsonData, timestamp = getData("EUR_USD", "H4", 500)
        df = parseData(jsonData)
        lastCandle = df.iloc[-1]
        return CandleInfo(
            open=lastCandle["open"].item(),
            high=lastCandle["high"].item(),
            low=lastCandle["low"].item(),
            close=lastCandle["close"].item(),
            rsi=lastCandle["rsi_14"].item(),
            timestamp=timestamp
        )
    except Exception as e:
        logger.error(f"Candle retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))