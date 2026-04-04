import torch
import joblib
import xgboost as xgb
import numpy as np
from pathlib import Path

ARTIFACTS = Path("artifacts")
CLASSES = ["down", "flat", "up"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
scaler = None

def loadModels():
    global xgbModel, nnModel, scaler
    # load xgb
    xgbModel = xgb.XGBClassifier()
    xgbModel.load_model(ARTIFACTS / "XGBoost_EUR_USD_H4_2026_v9.json")

    # load nn
    nnModel = ForexHybrid(
        hidden_size=512,
        num_layers=1,
        dropout=0.1,
        num_filters=24,
        kernel_size=3,
        input_size=14,
        output_size=3,
        lstm_dropout=0.1
    ).to(DEVICE)
    nnModel.load_state_dict(torch.load(ARTIFACTS / "NN_EUR_USD_H4_2026_v5.2.pth", map_location=DEVICE))
    nnModel.eval()
    scaler = joblib.load(ARTIFACTS / "scaler_v5.2.pkl")

def predict(xgbFeaturesDf, nnFeaturesDf) -> dict:
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
    for i in range(len(nnFeatures) - 20 + 1):
        X.append(nnFeatures[i : i + 20])
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

    return {
        "xgbPred": str(int(xgbPred)),
        "xgbProbs": xgbProbsDict,
        "nnPred": str(int(nnPred)),
        "nnProbs": nnProbsDict
    }