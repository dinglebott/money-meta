from custom_modules import datafetcher, dataparser
import xgboost as xgb
import torch
import joblib
import numpy as np
import os
import json

# GLOBAL VARIABLES
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, xgbVersion, nnVersion = globalVars.values()

# use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FETCH AND PARSE CURRENT DATA
jsonPath = datafetcher.getData(instrument, granularity, 200, "live")
df = dataparser.parseData(jsonPath)

# XGBOOST SUBMODEL
xgbModel = xgb.XGBClassifier()
xgbModel.load_model(os.path.join("submodels", "XGB", f"XGBoost_{instrument}_{granularity}_{yearNow}_v{xgbVersion}.json"))
# get features
with open(os.path.join("submodels", "XGB", f"features_v{xgbVersion}.json")) as file:
    xgbFeatures = json.load(file)["features"]

# GET XGBOOST PREDICTION
latestCandle = df[xgbFeatures].iloc[[-1]] # slice out last row (last candle)
xgbPred = xgbModel.predict(latestCandle)[0] # gets the only element of the 1D numpy array [n_samples]
xgbProbs = xgbModel.predict_proba(latestCandle)[0] # gets the only row of the 2D numpy array [n_samples, n_classes]

# RNN SUBMODEL
class ForexHybrid(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, lstm_dropout, output_size,
                num_filters, kernel_size):
        super(ForexHybrid, self).__init__()
        # CNN layers: takes 3D tensor as input (batch_size, channels, length)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_size, out_channels=num_filters,
                    kernel_size=kernel_size, padding=kernel_size//2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_filters), # normalise before passing to LSTM
            torch.nn.Dropout(dropout)
        )
        # LSTM layers
        self.lstm = torch.nn.LSTM(
            input_size=num_filters, # takes CNN output
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0
        )
        # Output layer (maps final pattern produced by LSTM to actual prediction) (fully connected)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x is 3D tensor (batch_size, timesteps, features)
        # CNN
        x = x.permute(0, 2, 1) # (batch, features, timesteps)
        x = self.cnn(x)
        x = x.permute(0, 2, 1) # (batch, timesteps, num_filters)
        # LSTM
        lstmOutput, (hidden, cell) = self.lstm(x)
        # lstmOutput: hidden state of EVERY timestep for the LAST layer only (batch_size, timesteps, hidden_size)
        # hidden: final hidden state (LAST timestep) for EVERY layer (layers, batch_size, hidden_size)
        # cell: similar to hidden but contains cell state instead of hidden state
        # FC
        lastTimestep = lstmOutput[:, -1, :] # slice out last timestep across all samples and neurons
        return self.fc(lastTimestep) # map to prediction (batch_size, output size)
# get features
with open(os.path.join("submodels", "NN", f"features_v{nnVersion}.json")) as file:
    nnFeatures = json.load(file)["features"]
# get hyperparams
with open(os.path.join("submodels", "NN", f"hyperparameters_v{nnVersion}.json")) as file:
    nnHyperparams = json.load(file)
    nnParams = nnHyperparams["modelParams"]
    nnLookback = nnHyperparams["lookback"]
# build model
nnModel = ForexHybrid(
    **nnParams,
    input_size=len(nnFeatures),
    output_size=3,
    lstm_dropout=nnParams["dropout"]
).to(device)
# load model
nnFilepath = os.path.join("submodels", "NN", f"NN_{instrument}_{granularity}_{yearNow}_v{nnVersion}.pth")
nnModel.load_state_dict(torch.load(nnFilepath, map_location=device))
nnModel.eval()
scaler = joblib.load(os.path.join("submodels", "NN", f"scaler_v{nnVersion}.pkl"))

# GET RNN PREDICTION
# construct sequences
features = scaler.transform(df[nnFeatures])
X = []
for i in range(len(features) - nnLookback + 1):
    X.append(features[i : i + nnLookback])
X = torch.tensor(np.array(X), dtype=torch.float32, device=device)
# run inferences
with torch.no_grad():
    logits = nnModel(X[-2:])
    allNnProbs = torch.softmax(logits, dim=1).cpu().numpy()
    allNnPreds = torch.argmax(logits, dim=1).cpu().numpy()
nnProbs = allNnProbs[-1]
nnPred = allNnPreds[-1]

# DISPLAY RESULTS
def getLabel(num):
    match num:
        case 0:
            return "DOWN"
        case 1:
            return "FLAT"
        case 2:
            return "UP"

print(f"XGB Version: {xgbVersion}")
print(f"LSTM Version: {nnVersion}")
print("")

for idx, prob in enumerate(xgbProbs): # round probabilities to 2sf, format with "%"
    print(f"{getLabel(idx)}: {prob*100:.2f}%") # label each probability and print
print(f"XGBoost prediction: {getLabel(xgbPred)}")
print("")

for idx, prob in enumerate(nnProbs):
    print(f"{getLabel(idx)}: {prob*100:.2f}%")
print(f"LSTM prediction: {getLabel(nnPred)}")