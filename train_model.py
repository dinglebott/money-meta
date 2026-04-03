from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import joblib
import os
import json

# GLOBAL VARIABLES
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, _, _ = globalVars.values()

# GET XGBOOST PREDICTIONS
from custom_modules.xgbTrainer import xgbProbs, xgbTimestamps, labels
xgbDf = pd.DataFrame(xgbProbs, index=xgbTimestamps, columns=["xgb_0", "xgb_1", "xgb_2"])
xgbDf.index = pd.to_datetime(xgbDf.index) # ensure proper datetime format for aligning

# GET RNN PREDICTIONS
from custom_modules.nnTrainer import nnProbs, nnTimestamps
nnDf = pd.DataFrame(nnProbs, index=nnTimestamps, columns=["nn_0", "nn_1", "nn_2"])
nnDf.index = pd.to_datetime(nnDf.index)

# ALIGN DATA
labelsDf = pd.Series(labels, index=pd.to_datetime(xgbTimestamps))
merged = xgbDf.join(nnDf, how="inner").join(labelsDf.rename("label"), how="inner")
# inner join takes intersection of indexes
merged.dropna(inplace=True)
X = merged[["xgb_0", "xgb_2", "nn_0", "nn_2"]].values
y = merged["label"].values

# SPLIT DATA
X_train = X[:(len(X)//3)*2]
X_test = X[(len(X)//3)*2:]
y_train = y[:(len(y)//3)*2]
y_test = y[(len(y)//3)*2:]

# SCALE DATA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TRAIN MODEL
model = LogisticRegression(
    C=0.1, # low C = high regularisation (use np.inf for none)
    solver="lbfgs",
    max_iter=1000,
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
cmatrix = confusion_matrix(y_test, preds)
cmatrixDf = pd.DataFrame(cmatrix, index=["Real -", "Real ~", "Real +"], columns=["Pred -", "Pred ~", "Pred +"])
cmatrixDf["Count"] = cmatrixDf.sum(axis=1)
cmatrixDf.loc["Count"] = cmatrixDf.sum(axis=0)

# PRINT RESULTS AND SAVE MODEL
print(f"F1 score (macro-averaged): {f1Score:.4f}")
print(f"Train F1 score: {trainF1Score:.4f}")
print(f"ROC-AUC score: {rocAucScore:.4f}")
print(f"Confusion matrix:\n{cmatrixDf}")

joblib.dump(model, os.path.join("models", f"metamodel_{instrument}_{granularity}_{yearNow}.pkl"))
joblib.dump(scaler, os.path.join("models", "metascaler.pkl"))