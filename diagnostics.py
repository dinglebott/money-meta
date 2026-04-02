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
    print("-" * 60)
    
    preds = np.argmax(probs, axis=1)
    
    for trueClass, className in classes.items():
        mask = labels == trueClass
        classProbs = probs[mask]
        classPreds = preds[mask]
        
        # split into correct and incorrect predictions
        correctMask = classPreds == trueClass
        incorrectMask = ~correctMask
        
        if correctMask.sum() > 0:
            avgCorrect = classProbs[correctMask].mean(axis=0)
            print(f"True={className:<8} {'correct':<8} "
                  f"P(down)={avgCorrect[0]:.3f}  "
                  f"P(flat)={avgCorrect[1]:.3f}  "
                  f"P(up)={avgCorrect[2]:.3f}  "
                  f"n={correctMask.sum()}")
        
        if incorrectMask.sum() > 0:
            avgIncorrect = classProbs[incorrectMask].mean(axis=0)
            print(f"True={className:<8} {'wrong':<8} "
                  f"P(down)={avgIncorrect[0]:.3f}  "
                  f"P(flat)={avgIncorrect[1]:.3f}  "
                  f"P(up)={avgIncorrect[2]:.3f}  "
                  f"n={incorrectMask.sum()}")
        print()

labelsNp = labels.values
probabilityDiagnostic(xgbProbs, labelsNp, "XGBoost")
probabilityDiagnostic(nnProbs, labelsNp[20:], "LSTM")