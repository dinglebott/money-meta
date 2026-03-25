# EXPORTS:
# xgbProbs contains xgb predictions for 2024-2026
# labels contains true 0, 1, 2 for each sample
import xgboost as xgb
from custom_modules import dataparser
from datetime import datetime
import os
import json

# GLOBAL VARIABLES
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, xgbVersion, _ = globalVars.values()

# LOAD TRAINING DATA
df = dataparser.parseData(f"json_data/{instrument}_{granularity}_{yearNow - 16}-01-01_{yearNow}-01-01.json")
dfTrain = dataparser.splitByDate(df, datetime(yearNow - 16, 1, 1), datetime(yearNow - 3, 1, 1))
dfVal = dataparser.splitByDate(df, datetime(yearNow - 3, 1, 1), datetime(yearNow - 2, 1, 1))
dfPred = dataparser.splitByDate(df, datetime(yearNow - 2, 1, 1), datetime(yearNow, 1, 1))

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
X_train = dfTrain[features]
y_train = dfTrain["target"]
X_val = dfVal[features]
y_val = dfVal["target"]
X_pred = dfPred[features]
y_pred = dfPred["target"]

# BUILD MODEL
model = xgb.XGBClassifier(
    **params, eval_metric="mlogloss",
    n_estimators=1000, # high ceiling
    early_stopping_rounds=50, # stop after metric plateaus for 50 rounds
    random_state=42,
    device="cpu", # avoid transferring data to gpu
    tree_method="hist"
)

# TRAIN MODEL
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# EXPORTS
xgbProbs = model.predict_proba(X_pred)
labels = y_pred