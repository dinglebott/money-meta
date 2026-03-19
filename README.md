## About project
Goal: Predict the direction of future price movements in forex markets by training a neural network\
This is part 3 of the whole forex prediction project, where I ensemble the models from parts 1 and 2 with a meta-model. See below for the rest of the project.\
The meta-model is a simple logistic regressor that synthesises the predictions of the XGBoost and RNN and produces a final prediction.\
*See DOCS.md for detailed results and workflow*\
<br/>
Part 1: [trading-trees](https://github.com/dinglebott/trading-trees), using a tree-based architecture (XGBoost)\
Part 2: [noisy-neurons](https://github.com/dinglebott/noisy-neurons), using neural networks (LSTM + CNN)

## Outline of methodology
Phase 1:
*See DOCS.md for detailed testing methodology*

## Project structure
The main code is all in the top-level scripts.\
The `custom_modules` folder contains helper functions to fetch and manipulate the data.\
It also contains fully-trained XGBoost and RNN models, and scripts to get leak-free predictions for training the meta-model.

## How to build a model
The top-level scripts contain global variables for the current year, the desired instrument, and granularity. I built my model for 2026 EUR/USD at H4 granularity. For other options, set these to the appropriate values.\
#### IMPORTANT:
You need an OANDA API key to pull historical data (or you can use the data I pulled already).\
If you have a key, set it as an environment variable `API_KEY` in a local `.env` file.\
The code fetches from the api-fxtrade.oanda.com live server, so if your key is from a demo account, change this to api-fxpractice.oanda.com.

## How to use a model
Run `use_model.py`, the prediction and confidence will be printed to the terminal.\
Remember to set the correct global variables. Obviously make sure you have the correct models trained for your use case first.

## Why?????
this better make me some money