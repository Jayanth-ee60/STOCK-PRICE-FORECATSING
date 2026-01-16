import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler


import yfinance as yf

df = yf.download(
    "^NSEI",
    start="2007-09-17",
    end="2026-01-14",
    group_by="column",
    auto_adjust=False,
    progress=False
)

# Hard failure if data didn't load
if df.empty:
    raise RuntimeError("NIFTY data download failed. Yahoo Finance issue, not model issue.")

# To Flatten MultiIndex columns immediately
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Sort the date and use it as the index(already an index)
df = df.sort_index()
df.to_csv("data/NIFTY50.csv", index=True)

'''DATA CLEANING'''
# Drop rows with missing core values
df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

# Remove zero-volume rows (bad data days)
df = df[df["Volume"] > 0]

'''Rows with missing OHLCV were removed because price and volume are fundamental market 
observables.Imputation would introduce synthetic price dynamics and distort volatility, 
returns, and direction labels in a time-series setting.'''

#FEATURES(Mathematical Constructs for prediction)

#1. RETURN (Logarithmic)
#Single day returns
df["ret_1"] = np.log(df['Close']/df["Close"].shift(1))

#Week long returns
df["ret_5"] = np.log(df['Close']/df["Close"].shift(5))

#2. VOLATILITY (RISK REGIME - INTER DAY)
df["volatility_5"] = df["ret_1"].rolling(5).std()

#3. INTRADAY CONTROL(/Dominance)
df["oc_return"] = (df["Close"] - df["Open"]) / df["Open"]

#VOLUME CHANGE (PARTICIPATION AND CONVICTION)
df["vol_chg"] = df["Volume"].pct_change()
df["vol_chg"] = df["vol_chg"].fillna(0)

#MOMENTUM 
df["ma_5"] = df["Close"].rolling(5).mean()
df["ma_10"] = df["Close"].rolling(10).mean()

df["ma_ratio"] = df["ma_10"] / df["ma_5"] - 1

#TARGET VARIABLES
#Direction target
df["dir_target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

#Return Target(Regression)
df["ret_target"] = np.log(df["Close"].shift(-1) / df["Close"])

df.dropna(inplace=True)

#FEATURE MATRIX
features = ["ret_1", "ret_5", "volatility_5", "vol_chg", "oc_return", "ma_ratio"]

X = df[features]
Y_dir = df["dir_target"]
Y_ret = df["ret_target"]

'''TEST-TRAIN SPLIT:
    Time-based splitting as required by stock market predictions'''
split = int(0.8 * len(df))

X_train, X_test = X.iloc[:split], X.iloc[split:]
Y_dir_train, Y_dir_test = Y_dir.iloc[:split], Y_dir.iloc[split:]
Y_ret_train, Y_ret_test = Y_ret.iloc[:split], Y_ret.iloc[split:]

'''FEATURE SCALING:'''
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''MODEL 1: DIRECTION CLASSIFER'''
clf = LogisticRegression()
clf.fit(X_train_scaled, Y_dir_train)

dir_pred = clf.predict(X_test_scaled)
dir_prob = clf.predict_proba(X_test_scaled)[:, 1]

print("Direction Accuracy:", accuracy_score(Y_dir_test, dir_pred))
print(classification_report(Y_dir_test, dir_pred))

'''MODEL 2: RETURN REGRESSION (Magnitude)'''
reg = LinearRegression()
reg.fit(X_train_scaled, Y_ret_train)

ret_pred = reg.predict(X_test_scaled)

print("Return MAE:", mean_absolute_error(Y_ret_test, ret_pred))

confidence_threshold = 0.55

filtered_ret_pred = np.where(
    dir_prob > confidence_threshold,
    ret_pred,
    0
)

close_today = df["Close"].iloc[split:].values

predicted_close = close_today * np.exp(filtered_ret_pred)

actual_close = df["Close"].shift(-1).iloc[split:].values

mask = ~np.isnan(actual_close)
actual_close = actual_close[mask]
predicted_close = predicted_close[mask]

price_mae = np.mean(np.abs(actual_close - predicted_close))
print("Predicted Close MAE:", price_mae)

plt.figure(figsize=(10,6))
plt.plot(actual_close[:100], label="Actual Close")
plt.plot(predicted_close[:100], label="Predicted Close")
plt.legend()
plt.title("Actual vs Predicted Close Prices")

plt.savefig("results/prediction_plot.png", dpi=300, bbox_inches="tight")
plt.show()
