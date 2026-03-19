import xgboost
import torch
import joblib
from custom_modules import datafetcher, dataparser
import numpy as np
import os
import json

# GLOBAL VARIABLES
yearNow = 2026
instrument = "EUR_USD"
granularity = "H4"

# use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD XGBOOST SUBMODEL

# LOAD RNN SUBMODEL

# FETCH AND PARSE CURRENT DATA

# CONSTRUCT SEQUENCES

# RUN INFERENCES

# DISPLAY RESULTS
