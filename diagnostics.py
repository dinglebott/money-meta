# To get a rough probability threshold at which to trade
import numpy as np
import json

# GLOBAL VARIABLES
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, _, _ = globalVars.values()

# GET XGBOOST PREDICTIONS
from custom_modules.xgbTrainer import xgbProbs, labels

# GET RNN PREDICTIONS
from custom_modules.nnTrainer import nnProbs

# DIAGNOSTIC FUNCTION
def probabilityDiagnostic(probs, labels, modelName):
    classes = {0: "down", 1: "flat", 2: "up"}
    print(f"\n{modelName} probability distributions:")
    print("-" * 80)
    
    preds = np.argmax(probs, axis=1)
    
    for trueClass, trueName in classes.items():
        trueMask = labels == trueClass
        
        for predClass, predName in classes.items():
            # cell (trueClass, predClass) of the confusion matrix
            cellMask = trueMask & (preds == predClass)
            n = cellMask.sum()
            
            if n == 0:
                continue
            
            avgProbs = probs[cellMask].mean(axis=0)
            correct = "✓" if trueClass == predClass else "✗"
            
            print(f"{correct} True={trueName:<8} Pred={predName:<8} "
                  f"P(down)={avgProbs[0]:.3f}  "
                  f"P(flat)={avgProbs[1]:.3f}  "
                  f"P(up)={avgProbs[2]:.3f}  "
                  f"n={n}")
        print()

probabilityDiagnostic(xgbProbs, labels, "XGBoost")
probabilityDiagnostic(nnProbs, labels[20:], "LSTM")