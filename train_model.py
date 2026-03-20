from custom_modules import dataparser
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import numpy as np
import joblib
import os
import json

# GLOBAL VARIABLES
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, xgbVersion, nnVersion = globalVars.values()

# GET XGBOOST PREDICTIONS

# GET RNN PREDICTIONS
from custom_modules.nnTrainer import nnProbs

# CREATE SCALER

# TRAIN MODEL

# TEST MODEL

# PRINT RESULTS AND SAVE MODEL