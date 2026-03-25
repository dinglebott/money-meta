from custom_modules import dataparser
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import pandas as pd
import joblib
import os
import json

# GLOBAL VARIABLES
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, _, _ = globalVars.values()

# GET XGBOOST PREDICTIONS
from custom_modules.xgbTrainer import xgbProbs, labels
xgbDf = pd.DataFrame(xgbProbs, index=labels.index, columns=["xgb_0", "xgb_1", "xgb_2"])

# GET RNN PREDICTIONS
from custom_modules.nnTrainer import nnProbs, nnTimestamps
nnDf = pd.DataFrame(nnProbs, index=nnTimestamps, columns=["nn_0", "nn_1", "nn_2"])

# SCALE DATA
labelsDf = pd.Series(labels.values, index=labels.index)
merged = xgbDf.join(nnDf, how="inner").join(labelsDf.rename("label"), how="inner")
merged.dropna(inplace=True)
X = merged[["xgb_0", "xgb_2", "nn_0", "nn_2"]].values
y = merged["label"].values

X_train = X[:(len(X)//3)*2]
X_test = X[(len(X)//3)*2:]
y_train = y[:(len(y)//3)*2]
y_test = y[(len(y)//3)*2:]
print(f"XGB rows: {len(xgbDf)} | NN rows: {len(nnDf)} | Merged rows: {len(merged)}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TRAIN MODEL
model = LogisticRegression(
    C=0.1, # low C = high regularisation
    max_iter=1000,
    multi_class="multinomial",
    solver="lbfgs",
    class_weight="balanced"
)
model.fit(X_train, y_train)

# TEST MODEL
preds = model.predict(X_test)
trainPreds = model.predict(X_train)
probs = model.predict_proba(X_test)
f1Score = f1_score(y_test, preds, average="macro", zero_division=0)
trainF1Score = f1_score(y_train, trainPreds, average="macro", zero_division=0)
rocAucScore = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")

# PRINT RESULTS AND SAVE MODEL
print(f"F1 score (macro-averaged): {f1Score:.4f}")
print(f"Train F1 score: {trainF1Score:.4f}")
print(f"ROC-AUC score: {rocAucScore:.4f}")