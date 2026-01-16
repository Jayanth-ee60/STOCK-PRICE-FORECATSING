# Stock Price Forecasting using Machine Learning

## Problem Statement
Forecasting stock prices is challenging due to noise, non-stationarity, and regime changes.
This project predicts the *next-day direction* and *closing price magnitude* of the NIFTY 50 index using machine learning.

## Dataset
- Source: Yahoo Finance (yfinance)
- Instrument: NIFTY 50 Index (^NSEI)
- Period: 2007–2026
- Features: OHLCV(Open, High, Low, Close, Volume) data

## Methodology
- Data Cleaning: Rows with zero volume or missing OHLCV data are purged. In finance, guessing (imputing) missing data is inconsistent in stock price predictions.
- Log-Normal Returns: We use log returns to ensure mathematical consistency across different price levels
- Time-based train-test split(80 : 20) to avoid data leakage, i.e., to make sure that the model never "peeks into the future"
- Logistic Regression for direction classification
- Linear Regression for return magnitude

- Price reconstructed using exponential return transformation

## Features Used (The DNA of the Model) 
- ret_1: 1-day log return
- ret_5: 5-day log return
- volatility_5: Rolling volatility
- vol_chg: Volume percentage change
- oc_return: Intraday strength
- ma_ratio: Trend momentum indicator

## Performane at a Glance
- The model is evaluated on its ability to predict the unseen 20% of recent market history.
| Metric | Value | Description |
| :--- | :--- | :--- |
| *Directional Accuracy* | ~53% | How often the model correctly guessed "Up" vs "Down" |
| *Return MAE* | ~0.005 | Mean Absolute Error of the predicted log-returns |
| *Confidence Threshold* | 0.55 | Minimum probability required to trigger a price prediction |
| *Instrument* | ^NSEI | NIFTY 50 Index (Yahoo Finance) |

![Prediction Plot](results/prediction_plot.png)

## How to Run
bash
git clone https://github.com/Jayanth-ee60/STOCK-PRICE-FORECASTING.git
cd YOUR_REPO_NAME
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
python src/forecast.py
n.


##⚠️ Disclaimer
- This project is for educational purposes only. The stock market involves significant risk. This model is a research tool, not financial advice.
