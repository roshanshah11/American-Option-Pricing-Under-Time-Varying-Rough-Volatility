#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

def compute_rolling_hurst(returns, power):
    n = 2**power
    hursts = []
    for t in range(n, len(returns)+1):
        data = returns[t-n:t]
        X = np.arange(2, power+1)
        Y = []
        for p in X:
            m = 2**p
            s = 2**(power-p)
            rs_vals = []
            for i in range(s):
                subs = data[i*m:(i+1)*m]
                dev = np.cumsum(subs - subs.mean())
                R = dev.max() - dev.min()
                S = subs.std()
                rs_vals.append(R/S if S!=0 else 0)
            Y.append(np.log2(np.mean(rs_vals)))
        coeff = np.polyfit(X, Y, 1)
        hursts.append(coeff[0])
    return np.array(hursts)


def make_lag_features(series, k):
    X, y, dates = [], [], []
    for i in range(k, len(series)):
        # use positional indexing to avoid future deprecation
        window = series.iloc[i-k:i].values
        X.append(window)
        y.append(series.iloc[i])
        dates.append(series.index[i])
    return np.array(X), np.array(y), dates

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
# Remove argparse config and replace with interactive menu
def interactive_menu():
    print("Select action:")
    print("0: Train model")
    print("1: Evaluate model")
    print("2: Forecast Hurst values")
    choice = input("Enter choice (0,1,2): ")
    return choice

# gather user inputs
mode = interactive_menu()
train = (mode == '0')
evaluate = (mode == '1')
forecast = 0
if mode == '2':
    forecast = int(input("Enter number of days to forecast: ") or 14)

# common parameters
power = int(input("Enter power for rolling Hurst (default 6): ") or 6)
k = 10
# ticker for training
ticker = ''
if mode != '0':
    # ticker for evaluate/forecast
    ticker = input("Enter ticker symbol (default AAPL): ") or 'AAPL'
num_tickers = 0
if mode == '0':
# num_tickers only for training
    num_tickers = int(input("Enter number of tickers for training (default 150): ") or 150)
# train/test split fraction
train_frac = float(input("Enter train split fraction (default 0.9): ") or 0.9)
# ────────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Config -> ticker={ticker}, power={power}, k={k}, num_tickers={num_tickers}, train={train}, evaluate={evaluate}, forecast={forecast}")

    data_dir = Path(__file__).parent / 'data'
    print("Loading data from", data_dir / 'dataset3.csv')
    # ensure consistent date parsing
    df = pd.read_csv(data_dir / 'dataset3.csv', parse_dates=['date'],infer_datetime_format=True)
    df = df.sort_values('date')
    df_all = df.copy()
    print(f"Data loaded: {len(df_all)} rows, {df_all['ticker'].nunique()} unique tickers")

    # Global model path
    model_path = Path(__file__).parent / 'models' / 'hurst_model.json'
    model_path.parent.mkdir(exist_ok=True)

    # Train
    if train:
        print("Starting training...")
        n = 2**power
        tickers = df_all['ticker'].unique().tolist()
        selected = random.sample(tickers, min(num_tickers, len(tickers)))
        print(f"Selected {len(selected)} tickers for training.")
        X_list, y_list = [], []
        for t in selected:
            df_t = df_all[df_all['ticker']==t].sort_values('date')
            returns_t = df_t['return'].values
            hurst_vals = compute_rolling_hurst(returns_t, power)
            start = n - 1
            hurst_series_t = pd.Series(hurst_vals, index=df_t['date'].values[start:])
            X_t, y_t, _ = make_lag_features(hurst_series_t, k)
            X_list.append(X_t)
            y_list.append(y_t)
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        print(f"Aggregated training data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y_train)
        model.save_model(str(model_path))
        print("Model training completed and saved to", model_path)

    # Evaluate
    if evaluate:
        print(f"Starting evaluation for ticker {ticker}...")
        df_t = df_all[df_all['ticker']==ticker].sort_values('date')
        returns = df_t['return'].values
        dates = df_t['date']
        hurst_vals = compute_rolling_hurst(returns, power)
        start = 2**power - 1
        hurst_series = pd.Series(hurst_vals, index=dates[start:])
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        X, y, dates_feat = make_lag_features(hurst_series, k)
        # split data by user‑defined fraction instead of hardcoded 0.8
        cutoff = int(len(X) * train_frac)
        X_test, y_test = X[cutoff:], y[cutoff:]
        dates_test = dates_feat[cutoff:]
        preds = model.predict(X_test)
        print(f"Evaluation on {ticker}:")
        print(f"Test set size: {len(X_test)}")
        print(f"Predictions: {preds}") 
        print(f"Actual values: {y_test}")
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        print(f"Evaluation metrics -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        print(dates_test[0], dates_test[-1])
        plt.figure(figsize=(10,5))
        plt.plot(dates_test, y_test, label='Actual')
        plt.plot(dates_test, preds, label='Predicted')
        plt.title(f'Evaluate Hurst Prediction for {ticker}')
        plt.legend(); plt.grid(True); plt.show()

    # Forecast
    if forecast > 0:
        print(f"Starting forecasting for next {forecast} days on ticker {ticker}...")
        df_t = df_all[df_all['ticker']==ticker].sort_values('date')
        returns = df_t['return'].values
        dates = df_t['date']
        hurst_vals = compute_rolling_hurst(returns, power)
        start = 2**power - 1
        hurst_series = pd.Series(hurst_vals, index=dates[start:])
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        history = list(hurst_series.values[-k:])
        future = []
        for i in range(forecast):
            x_in = np.array(history[-k:]).reshape(1, -1)
            y_hat = model.predict(x_in)[0]
            future.append(y_hat)
            history.append(y_hat)
        last_date = hurst_series.index[-1]
        future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast, freq='B')
        future_series = pd.Series(future, index=future_idx)
        print("Forecasted values:")
        print(future_series)
        plt.figure(figsize=(10,5))
        plt.plot(hurst_series.index, hurst_series.values, label='History')
        plt.plot(future_series.index, future_series.values, '--', label='Forecast')
        plt.title(f'14-Day Hurst Forecast for {ticker}')
        plt.legend(); plt.grid(True); plt.show()
        print("Forecast completed.")

if __name__ == '__main__':
    main()