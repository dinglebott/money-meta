# EXPORTS:
# xgbProbs contains xgb predictions for 2024-2026
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

# GET FEATURES
filepath = os.path.join("submodels", "XGB", f"features_v{xgbVersion}.json")
with open(filepath, "r") as file:
    features = json.load(file)["features"]

# GET HYPERPARAMETERS
filepath = os.path.join("submodels", "XGB", f"hyperparameters_v{xgbVersion}.json")
with open(filepath, "r") as file:
    params = json.load(file)
params["max_depth"] = int(params["max_depth"])
params["min_child_weight"] = int(params["min_child_weight"])

# DEFINE DATASETS
X = df[features]
y = df["target"]

# BUILD MODEL
model = xgb.XGBClassifier(
    **params, eval_metric="mlogloss",
    n_estimators=1000, # high ceiling
    early_stopping_rounds=50, # stop after metric plateaus for 50 rounds
    random_state=42,
    device="cpu", # avoid transferring data to gpu
    tree_method="hist"
)

# EXPORTS
xgbProbs = model.predict_proba(X)
labels = y