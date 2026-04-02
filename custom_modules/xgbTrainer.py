# EXPORTS:
# xgbProbs contains xgb predictions for 2024-2026
# xgbTimestamps contains contains timestamps to align indexes with nn
# labels contains true 0, 1, 2 for each sample
import xgboost as xgb
from custom_modules import dataparser
import os
import json

# GLOBAL VARIABLES
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, xgbVersion, _ = globalVars.values()

# LOAD TRAINING DATA
df = dataparser.parseData(f"json_data/{instrument}_{granularity}_{yearNow - 2}-01-01_{yearNow}-01-01.json")
timestamps = df["time"]

# GET FEATURES
filepath = os.path.join("submodels", "XGB", f"features_v{xgbVersion}.json")
with open(filepath, "r") as file:
    features = json.load(file)["features"]

# DEFINE DATASETS
X = df[features]
y = df["target"]

# BUILD MODEL
model = xgb.XGBClassifier()
filepath = os.path.join("submodels", "XGB", f"XGBoost_{instrument}_{granularity}_{yearNow}_v{xgbVersion}.json")
model.load_model(filepath)

# EXPORTS
xgbProbs = model.predict_proba(X)
xgbTimestamps = timestamps[X.index].str.replace("Z", "", regex=False) # strip timezone for compatibility
labels = y