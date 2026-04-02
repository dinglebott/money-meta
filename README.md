## About project
Goal: Predict the direction of future price movements in forex markets with machine learning\
This is part 3 of the whole forex prediction project, where I deploy the models from parts 1 and 2 in a coherent package. See below for the rest of the project.\
*See DOCS.md for detailed results and workflow*\
<br/>
Part 1: [trading-trees](https://github.com/dinglebott/trading-trees), using a tree-based architecture (XGBoost)\
Part 2: [noisy-neurons](https://github.com/dinglebott/noisy-neurons), using neural networks (LSTM + CNN)\

#### IMPORTANT:
This repo was originally intended to be a meta-model that synthesises probabilities from the XGBoost and LSTM models. However, it SUCKS. So I have repurposed it into a deployment repo that pushes it to a website. Many of the files in the repo are relics of that attempt.\
<br/>

## Project structure
The main code is all in the top-level scripts.\
The `custom_modules` folder contains helper functions to fetch and manipulate the data.\
`nnTrainer.py`, `xgbTrainer.py`, and `train_model` are deprecated (see explanatory note above). They were originally used to get leak-free predictions for training the meta-model.\
<br/>

## Setup
Inside the `env.json` file, set the current year, the desired instrument, and granularity. I built my model for 2026 EUR/USD at H4 granularity. For other options, set these to the appropriate values.\
#### IMPORTANT:
You need an OANDA API key to pull historical data (or you can use the data I pulled already).\
If you have a key, set it as an environment variable `API_KEY` in a local `.env` file.\
The code fetches from the api-fxtrade.oanda.com live server, so if your key is from a demo account, change this to api-fxpractice.oanda.com.\
<br/>

## How to use the models
Run `predict.py`, the predictions and probability breakdowns are printed to the terminal.\
Remember to set the correct global variables. Obviously make sure you have the correct models trained for your use case first.\
<br/>

## Why?????
this BETTER make me some money