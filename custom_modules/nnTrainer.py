import torch
from custom_modules import dataparser
import pandas as pd
import numpy as np
import joblib
import os
import json

# GLOBAL VARIABLES
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, _, nnVersion = globalVars.values()

# use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BUILD CLASS
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

# GET FEATURES
filepath = os.path.join("submodels","NN", f"features_v{nnVersion}.json")
with open(filepath, "r") as file:
    featureList = json.load(file)["features"]

# GET PARAMETERS
filepath = os.path.join("submodels","NN", f"hyperparameters_v{nnVersion}.json")
with open(filepath, "r") as file:
    hyperparams = json.load(file)
    params = hyperparams["modelParams"]
    lookback = hyperparams["lookback"]

# LOAD MODEL AND SCALER
model = ForexHybrid(
    **params,
    input_size=len(featureList),
    output_size=3,
    lstm_dropout=params["dropout"]
).to(device)

filepath = os.path.join("submodels", "NN", f"NN_{instrument}_{granularity}_{yearNow}_v{nnVersion}.pth")
model.load_state_dict(torch.load(filepath, map_location=device))
model.eval()

filepath = os.path.join("submodels", "NN", f"scaler_v{nnVersion}.pkl")
scaler = joblib.load(filepath)

# PARSE DATA TO PREDICT (val set cos i'm lazy)
df = dataparser.parseData(os.path.join("json_data", f"{instrument}_{granularity}_{yearNow - 2}-01-01_{yearNow}-01-01.json"))
timestamps = df["time"] # separate timestamps to avoid scaling
df.drop(columns=["time"], inplace=True)

features = df[featureList]
labels = df["target"]

labels = labels[features.index] # align indexes
timestamps = timestamps[features.index]

# SCALE FEATURES, CREATE DATA SEQUENCES
features = scaler.transform(features)

def createSequences(fts, lbls, lookback):
    X, y = [], []
    for i in range(len(fts) - lookback):
        X.append(fts[i : i + lookback])
        y.append(lbls[i + lookback])
    return np.array(X), np.array(y)

X, y = createSequences(features, labels, lookback)

# CONVERT TO TENSORS
X = torch.tensor(X, dtype=torch.float32, device=device)
y = torch.tensor(y, dtype=torch.long, device=device)
# shape of X: (samples, timesteps, features)
# shape of y: (samples, output_classes)

# RUN INFERENCES
with torch.no_grad():
    logits = model(X)
    probs = torch.argmax(logits, dim=1).cpu().numpy()

# EXPORTS
nnProbs = pd.DataFrame(probs, columns=["nn_0", "nn_1", "nn_2"])