## READ ME FIRST
This repo was originally intended to be a meta-model that synthesises probabilities from the XGBoost and LSTM models. However, it SUCKS. So I have repurposed it into a deployment repo that pushes it to a website.\
The data at the bottom is from the attempt at the meta-model.\
<br/>

## STRUCTURE AND DEPLOYMENT
Website: https://dinglebott.github.io/money-meta \
`dist/`\
|--`api/`
>|--`data-processing.py` - Data-fetching and data-parsing code, also loads API key for OANDA\
>|--`inference.py` - Functions to load the models and to get their predictions\
>|--`main.py` - Exposes endpoints (/health, /predict, /candle), connects with frontend website

|--`artifacts/` - Contains models and scaler for neural network\
|--`Dockerfile` - Contains instructions for Railway to build Docker container\
|--`requirements.txt` - Lists dependencies for project to be installed by Dockerfile (torch and torchvision have been excluded, CPU-specific version installed separately in Dockerfile to minimise loading time)\
`docs/` - Contains the website files (index, css, js)\
<br/>

*The following sections are legacy (see explanatory note at the top of the docs)*\
<br/>

## META-MODEL TRAINING DATA
Model type: Scikit-learn logistic regression\
Outputs of XGB model: xgb_0, xgb_2 (probabilities of down and up moves respectively)\
Outputs of LSTM model: nn_0, nn_2\
<br/>

## MODEL EVALUATION
**Explanation of metrics:**\
Negative = 0, Flat = 1, Positive = 2\
F1 score (0-1) => Harmonic mean of Precision and Recall\
F1 score (macro-averaged) => Unweighted mean of F1 score calculated for each class (1 and 0)\
ROC-AUC score (0-1) => Probability that a randomly chosen 1 is ranked higher than a randomly chosen 0 by the model\
Precision (0-1) => Correctly predicted 1's / All predicted 1's\
Recall (0-1) => Correctly predicted 1's / All real 1's\
<br/>

### Model 1
**F1 score (macro-averaged):** 0.3850\
**Train F1 score:** 0.4353\
**ROC-AUC score:** 0.5749\
**Confusion matrix:**
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 87 | 108 | 124 |
| Real ~ | 68 | 179 | 104 |
| Real + | 106 | 104 | 130 |
<br/>
