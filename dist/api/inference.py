import torch
import joblib
import xgboost as xgb
import numpy as np
from pathlib import Path
import json

ARTIFACTS = Path("artifacts")
CLASSES = ["down", "flat", "up"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xgbVersion = 10
nnVersion = 5.4
xgbH1Version = 10.1
nnH1Version = 5.5

class ForexHybrid(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, lstm_dropout, output_size,
                num_filters, kernel_size):
        super(ForexHybrid, self).__init__()
        # CNN layers
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_size, out_channels=num_filters,
                    kernel_size=kernel_size, padding=kernel_size//2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_filters),
            torch.nn.Dropout(dropout)
        )
        # LSTM layers
        self.lstm = torch.nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0
        )
        # Output layer
        self.fc = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        # LSTM
        lstmOutput, _ = self.lstm(x)
        lastTimestep = lstmOutput[:, -1, :]
        return self.fc(lastTimestep)
    
# Loaded once at startup
xgbModel = None
nnModel = None
xgbH1Model = None
nnH1Model = None
scaler = None
scalerH1 = None

def loadModels():
    global xgbModel, nnModel, xgbH1Model, nnH1Model, scaler, scalerH1
    # load H4 xgb
    xgbModel = xgb.XGBClassifier()
    xgbModel.load_model(ARTIFACTS / f"XGBoost_EUR_USD_H4_2026_v{xgbVersion}.json")

    # load H1 xgb
    xgbH1Model = xgb.XGBClassifier()
    xgbH1Model.load_model(ARTIFACTS / f"XGBoost_EUR_USD_H1_2026_v{xgbH1Version}.json")

    # load H4 nn
    with open(ARTIFACTS / f"nnFeatures_v{nnVersion}.json", "r") as file:
        nnFeatures = json.load(file)["features"]
    with open(ARTIFACTS / f"nnHyperparameters_v{nnVersion}.json", "r") as file:
        nnHyperparams = json.load(file)["modelParams"]
    
    nnModel = ForexHybrid(
        **nnHyperparams,
        input_size=len(nnFeatures),
        output_size=3,
        lstm_dropout=nnHyperparams["dropout"]
    ).to(DEVICE)
    nnModel.load_state_dict(torch.load(ARTIFACTS / f"NN_EUR_USD_H4_2026_v{nnVersion}.pth", map_location=DEVICE))
    nnModel.eval()
    scaler = joblib.load(ARTIFACTS / f"scaler_v{nnVersion}.pkl")

    # load H1 nn
    with open(ARTIFACTS / f"nnFeatures_v{nnH1Version}.json", "r") as file:
        nnH1Features = json.load(file)["features"]
    with open(ARTIFACTS / f"nnHyperparameters_v{nnH1Version}.json", "r") as file:
        nnH1Hyperparams = json.load(file)["modelParams"]
    
    nnH1Model = ForexHybrid(
        **nnH1Hyperparams,
        input_size=len(nnH1Features),
        output_size=3,
        lstm_dropout=nnH1Hyperparams["dropout"]
    ).to(DEVICE)
    nnH1Model.load_state_dict(torch.load(ARTIFACTS / f"NN_EUR_USD_H1_2026_v{nnH1Version}.pth", map_location=DEVICE))
    nnH1Model.eval()
    scalerH1 = joblib.load(ARTIFACTS / f"scaler_v{nnH1Version}.pkl")

def predict(xgbFeaturesDf, nnFeaturesDf, xgbH1FeaturesDf, nnH1FeaturesDf) -> dict:
    # xgb prediction
    latestCandle = xgbFeaturesDf.iloc[[-1]]
    xgbPred = xgbModel.predict(latestCandle)[0]
    xgbProbs = xgbModel.predict_proba(latestCandle)[0]
    xgbProbsDict = {
        "0": xgbProbs[0],
        "1": xgbProbs[1],
        "2": xgbProbs[2]
    }

    # build nn sequence
    nnFeatures = scaler.transform(nnFeaturesDf)
    X = []
    for i in range(len(nnFeatures) - 45 + 1):
        X.append(nnFeatures[i : i + 45])
    X = torch.tensor(np.array(X), dtype=torch.float32, device=DEVICE)
    # run inferences
    with torch.no_grad():
        logits = nnModel(X[-2:])
        allNnProbs = torch.softmax(logits, dim=1).cpu().numpy()
        allNnPreds = torch.argmax(logits, dim=1).cpu().numpy()
    nnPred = allNnPreds[-1]
    nnProbs = allNnProbs[-1]
    nnProbsDict = {
        "0": nnProbs[0],
        "1": nnProbs[1],
        "2": nnProbs[2]
    }

    # H1 XGBoost prediction
    latestCandleH1 = xgbH1FeaturesDf.iloc[[-1]]
    xgbH1Pred = xgbH1Model.predict(latestCandleH1)[0]
    xgbH1Probs = xgbH1Model.predict_proba(latestCandleH1)[0]
    xgbH1ProbsDict = {
        "0": xgbH1Probs[0],
        "1": xgbH1Probs[1],
        "2": xgbH1Probs[2]
    }

    # build H1 nn sequence
    nnH1Features = scalerH1.transform(nnH1FeaturesDf)
    XH1 = []
    for i in range(len(nnH1Features) - 40 + 1):
        XH1.append(nnH1Features[i : i + 40])
    XH1 = torch.tensor(np.array(XH1), dtype=torch.float32, device=DEVICE)
    # run inferences
    with torch.no_grad():
        logitsH1 = nnH1Model(XH1[-2:])
        allNnProbsH1 = torch.softmax(logitsH1, dim=1).cpu().numpy()
        allNnPredsH1 = torch.argmax(logitsH1, dim=1).cpu().numpy()
    nnH1Pred = allNnPredsH1[-1]
    nnH1Probs = allNnProbsH1[-1]
    nnH1ProbsDict = {
        "0": nnH1Probs[0],
        "1": nnH1Probs[1],
        "2": nnH1Probs[2]
    }

    return {
        "xgbPred": str(int(xgbPred)),
        "xgbProbs": xgbProbsDict,
        "nnPred": str(int(nnPred)),
        "nnProbs": nnProbsDict,
        "xgbH1Pred": str(int(xgbH1Pred)),
        "xgbH1Probs": xgbH1ProbsDict,
        "nnH1Pred": str(int(nnH1Pred)),
        "nnH1Probs": nnH1ProbsDict
    }