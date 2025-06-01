#!/usr/bin/env python3
# This script computes the rolling Hurst exponent of a time series
# and trains/evaluates an XGBoost model to forecast its future values.

import numpy as np                # numerical operations (arrays, math)
import pandas as pd               # data manipulation (DataFrame)
import matplotlib.pyplot as plt   # plotting library for visualization
from pathlib import Path          # convenient filesystem path handling
from sklearn.metrics import mean_squared_error, mean_absolute_error # error metrics for model evaluation
import logging                    # logging status and debug messages
import joblib                     # saving/loading Python objects (models)
from xgboost import XGBRegressor  # XGBoost regression model

# Forecast horizon depth (how many days ahead to predict)
HORIZON = 14

# Configure logger to display INFO messages with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths for raw CSV and cached pickle files
DATA_PATH   = Path(__file__).parent / 'data' / 'dataset3.csv'
DATA_PICKLE = Path(__file__).parent / 'data' / 'dataset3.pkl'

# Directory to store trained models
MODELS_DIR = Path(__file__).parent / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def compute_rolling_hurst(returns: np.ndarray, power: int) -> np.ndarray:
    """
    Calculate the Hurst exponent using R/S analysis over
    a rolling window of size 2^power.
    """
    # here is a source i used: https://en.wikipedia.org/wiki/Hurst_exponent
    # Ensure returns are in numpy array form
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)
    # Validate 'power' parameter
    if not isinstance(power, int) or power < 1:
        raise ValueError("power must be a positive integer")
    # Window length = 2^power
    n = 2**power
    if len(returns) < n:
        raise ValueError(f"Need at least {n} data points for power={power}")
    hursts = []                          # list to collect Hurst exponents
    exponents = np.arange(2, power+1)    # scales for R/S calculation
    # Slide the window through the data
    for t in range(n, len(returns) + 1):
        window = returns[t-n:t]          # current data chunk
        rs_log = []                      # log2 of rescaled range values
        for exp in exponents:
            # split window into segments and compute R/S per segment
            m = 2**exp
            s = n // m
            segments = window.reshape(s, m)
            dev = np.cumsum(
                segments - segments.mean(axis=1, keepdims=True),
                axis=1
            )
            R = dev.max(axis=1) - dev.min(axis=1)
            S = segments.std(axis=1)
            rs = np.where(S != 0, R/S, 0)
            rs_log.append(np.log2(rs.mean()))
        # linear fit: slope ≈ Hurst exponent
        hursts.append(np.polyfit(exponents, rs_log, 1)[0])
    return np.array(hursts)

def load_data() -> pd.DataFrame:
    """
    Load the price dataset from CSV or cache, sort by date,
    and return a pandas DataFrame.
    """
    # here is a source i used: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    if DATA_PICKLE.exists():
        df = pd.read_pickle(DATA_PICKLE)
        logger.info(f"Loaded data from cache: {DATA_PICKLE}")
    else:
        df = pd.read_csv(
            DATA_PATH,
            parse_dates=['date'],
            low_memory=False
        )
        df.sort_values('date', inplace=True)
        df.to_pickle(DATA_PICKLE)
        logger.info(f"Saved data to cache: {DATA_PICKLE}")
    return df

def make_lagged_features(series: pd.Series, k: int):
    """Generate lagged features matrix X and target vector y from a series."""
    vals = series.values
    X, y = [], []
    for i in range(k, len(vals)):
        X.append(vals[i-k:i])
        y.append(vals[i])
    return np.array(X), np.array(y)

def train_xgb_only(hurst_series: pd.Series, train_frac: float, k: int, xgb_params: dict):
    """Train a pure XGBoost model on lagged Hurst series."""
    # here is a source i used: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor
    split = int(len(hurst_series) * train_frac)
    train = hurst_series.iloc[:split]
    X_tr, y_tr = make_lagged_features(train, k)
    xgb_only = XGBRegressor(**xgb_params)
    xgb_only.fit(X_tr, y_tr)
    joblib.dump(xgb_only, MODELS_DIR / 'xgb_only.pkl')
    logger.info("Pure XGBoost model saved to %s", MODELS_DIR)
    return xgb_only

def evaluate_xgb_only(hurst_series: pd.Series, xgb_only, k: int, test_frac: float):
    """Evaluate pure XGBoost model on test split and plot results."""
    # here is a source i used: https://scikit-learn.org/stable/modules/model_evaluation.html
    split = int(len(hurst_series) * (1 - test_frac))
    test = hurst_series.iloc[split:]
    X_te, y_te = make_lagged_features(test, k)
    y_pred = xgb_only.predict(X_te)
    mse = mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    logger.info("XGBoost-only results – MSE: %.4f, MAE: %.4f", mse, mae)
    # Plot actual vs prediction
    plt.figure(figsize=(12,6))
    dates_te = test.index[k:]
    plt.plot(dates_te, y_te, 'k-', label='Actual Hurst')
    plt.plot(dates_te, y_pred, 'g--', label='XGB-only Forecast')
    plt.title('XGBoost-only: Actual vs Forecast Hurst Exponent')
    plt.xlabel('Date'); plt.ylabel('Hurst Exponent')
    plt.legend(); plt.grid(True); plt.xticks(rotation=45); plt.tight_layout(); plt.show()
    return mse, mae

def forecast_xgb_only(hurst_series: pd.Series, xgb_only, periods: int, k: int):
    """Generate future forecasts using pure XGBoost model recursively."""
    # here is a source i used: https://pandas.pydata.org/docs/reference/api/pandas.date_range.html
    hist = list(hurst_series.values)
    preds = []
    for _ in range(periods):
        if len(hist) < k:
            p = 0
        else:
            p = xgb_only.predict(np.array(hist[-k:]).reshape(1, -1))[0]
        preds.append(p)
        hist.append(p)
    dates = pd.date_range(hurst_series.index[-1] + pd.Timedelta(days=1), periods=periods, freq='B')
    return pd.DataFrame({'ds': dates, 'yhat': np.array(preds)})

def train_multi_step_xgb(hurst_series: pd.Series, train_frac: float, k: int, horizon: int, xgb_params: dict):
    """Train multiple XGBoost models to predict all steps in the forecast horizon at once."""
    split = int(len(hurst_series) * train_frac)
    train = hurst_series.iloc[:split]
    values = train.values
    
    # Create separate models for each step in the forecast horizon
    models = []
    
    for step in range(horizon):
        X, y = [], []
        # For each forecast step, create appropriate training examples
        for i in range(k, len(values) - step):
            X.append(values[i-k:i])
            y.append(values[i+step])  # Target is 'step' days ahead
            
        X = np.array(X)
        y = np.array(y)
        
        # Train model for this specific forecast step
        model = XGBRegressor(**xgb_params)
        model.fit(X, y)
        models.append(model)
        logger.info(f"Trained model for step {step+1}/{horizon}")
    
    # Save the models
    joblib.dump(models, MODELS_DIR / f'xgb_multi_step_{horizon}.pkl')
    logger.info(f"Multi-step XGBoost models saved to {MODELS_DIR}")
    
    return models

def forecast_multi_step_xgb(hurst_series: pd.Series, models: list, periods: int, k: int):
    """Generate all future forecasts at once using the multi-step models."""
    # Get the most recent window of data
    last_window = hurst_series.values[-k:]
    
    # Generate predictions for each step using its own model
    predictions = []
    for step in range(min(periods, len(models))):
        model = models[step]
        pred = model.predict(last_window.reshape(1, -1))[0]
        predictions.append(pred)
    
    # If we need more predictions than we have models for
    while len(predictions) < periods:
        predictions.append(predictions[-1])  # Just repeat the last prediction
    
    dates = pd.date_range(hurst_series.index[-1] + pd.Timedelta(days=1), 
                         periods=periods, freq='B')
    
    return pd.DataFrame({'ds': dates, 'yhat': np.array(predictions)})

def evaluate_multi_step_xgb(hurst_series: pd.Series, models: list, k: int, test_frac: float, horizon: int):
    """Evaluate multi-step XGBoost models on test split and plot results."""
    split = int(len(hurst_series) * (1 - test_frac))
    test = hurst_series.iloc[split:]
    
    # Calculate the actual values for each forecast step
    actuals = []
    for step in range(horizon):
        if step < len(test) - k:
            actual_vals = test.iloc[k+step:].values
            actuals.append(actual_vals)
        else:
            actuals.append([])
    
    # Calculate predictions for each step
    predictions = []
    for step, model in enumerate(models):
        if step < horizon:
            X_test = []
            for i in range(k, len(test) - step):
                X_test.append(test.iloc[i-k:i].values)
            
            if X_test:
                X_test = np.array(X_test)
                preds = model.predict(X_test)
                predictions.append(preds)
            else:
                predictions.append([])
    
    # Calculate error metrics for each step
    mse_values = []
    mae_values = []
    for step in range(horizon):
        if len(actuals[step]) > 0 and len(predictions[step]) > 0:
            min_len = min(len(actuals[step]), len(predictions[step]))
            mse = mean_squared_error(actuals[step][:min_len], predictions[step][:min_len])
            mae = mean_absolute_error(actuals[step][:min_len], predictions[step][:min_len])
            mse_values.append(mse)
            mae_values.append(mae)
            logger.info(f"Step {step+1} - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # Plot results for multiple steps
    plt.figure(figsize=(14, 7))
    dates_test = test.index[k:]
    plt.plot(dates_test, test.iloc[k:].values, 'k-', label='Actual Hurst')
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for step in range(min(5, horizon)):  # Plot up to 5 steps for clarity
        if len(predictions[step]) > 0:
            forecast_dates = dates_test[:len(predictions[step])]
            plt.plot(forecast_dates, predictions[step], 
                    f'{colors[step % len(colors)]}--', 
                    label=f'Step {step+1} Forecast')
    
    plt.title('Multi-Step XGBoost: Actual vs Forecast Hurst Exponent')
    plt.xlabel('Date')
    plt.ylabel('Hurst Exponent')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return np.mean(mse_values), np.mean(mae_values)

def interactive_menu():
    # here is a source i used: https://docs.python.org/3/library/functions.html#input
    print("Select action:")
    print("0: Train XGBoost model (interactive)")
    print("1: Evaluate XGBoost model (interactive)")
    print("2: Forecast XGBoost model (interactive)")
    print("3: Train Multi-Step XGBoost model (interactive)")
    print("4: Quick train XGBoost with defaults")
    print("5: Quick evaluate XGBoost with defaults")
    print("6: Quick forecast XGBoost with defaults")
    print("7: Evaluate Multi-Step XGBoost model")
    print("8: Forecast with Multi-Step XGBoost model")
    choice = input("Enter choice (0,1,2,3,4,5,6,7,8): ")
    return choice

def main():
    """
    Main entry point: show menu, get user choice, compute Hurst series,
    and train/evaluate/forecast the XGBoost model as requested.
    """
    # here is a source i used: https://docs.python.org/3/library/logging.html
    logger.info("Starting main execution...")
    mode = interactive_menu().strip()
    if mode not in ['0','1','2','3','4','5','6','7','8']:
        logger.error("Invalid mode selected. Exiting.")
        return

    # Interactive input vs. quick defaults
    if mode in ['0','1','2','3','7','8']:
        power     = int(input("Enter power for rolling Hurst (default 6): ") or 6)
        ticker    = input("Enter ticker (default AAPL): ") or 'AAPL'
        train_frac= float(input("Enter train fraction (default 0.8): ") or 0.8)
        test_frac = 1 - train_frac
        k         = int(input("Enter lag depth k (default 5): ") or 5)
        xgb_params= {
            'n_estimators': int(input("Enter n_estimators (100): ") or 100),
            'learning_rate': float(input("Enter learning_rate (0.1): ") or 0.1)
        }
        periods   = int(input("Enter forecast days (default 14): ") or HORIZON)
    else:
        power, ticker      = 4, 'AAPL'
        train_frac, test_frac = 0.8, 0.2
        k                  = 5
        xgb_params         = {'n_estimators':100, 'learning_rate':0.1}
        periods            = HORIZON

    # Load data and filter by ticker symbol
    df_all = load_data()
    df_t   = df_all[df_all['ticker']==ticker].sort_values('date')
    returns= df_t['return'].values

    # Compute rolling Hurst exponent series
    hurst_vals   = compute_rolling_hurst(returns, power)
    dates        = df_t['date']
    start_index  = 2**power - 1
    hurst_series = pd.Series(hurst_vals, index=dates[start_index:])

    # Execute requested action
    if mode in ['0','4']:
        train_xgb_only(hurst_series, train_frac, k, xgb_params)
    elif mode in ['1','5']:
        xgb_only = joblib.load(MODELS_DIR / 'xgb_only.pkl')
        evaluate_xgb_only(hurst_series, xgb_only, k, test_frac)
    elif mode in ['2','6']:
        xgb_only = joblib.load(MODELS_DIR / 'xgb_only.pkl')
        df_fc    = forecast_xgb_only(hurst_series, xgb_only, periods, k)
        logger.info("XGBoost-only forecast:\n%s", df_fc)
        plt.figure(figsize=(12,6))
        plt.plot(hurst_series.index, hurst_series.values, 'k-', label='Historical Hurst')
        plt.plot(df_fc['ds'], df_fc['yhat'], 'g--', label='Forecast')
        plt.title(f'{periods}-Day Hurst Forecast for {ticker}')
        plt.xlabel('Date'); plt.ylabel('Hurst Exponent')
        plt.legend(); plt.grid(True)
        plt.xticks(rotation=45); plt.tight_layout(); plt.show()
    elif mode == '3':
        # Train multi-step model
        multi_models = train_multi_step_xgb(hurst_series, train_frac, k, periods, xgb_params)
        logger.info(f"Trained {len(multi_models)} models for multi-step forecasting")
    elif mode == '7':
        # Evaluate multi-step model
        try:
            multi_models = joblib.load(MODELS_DIR / f'xgb_multi_step_{periods}.pkl')
            mse, mae = evaluate_multi_step_xgb(hurst_series, multi_models, k, test_frac, periods)
            logger.info(f"Multi-step XGBoost results - Average MSE: {mse:.4f}, Average MAE: {mae:.4f}")
        except FileNotFoundError:
            logger.error(f"Multi-step model file not found. Please train the model first.")
    elif mode == '8':
        # Forecast with multi-step model
        try:
            multi_models = joblib.load(MODELS_DIR / f'xgb_multi_step_{periods}.pkl')
            df_fc = forecast_multi_step_xgb(hurst_series, multi_models, periods, k)
            logger.info("Multi-step XGBoost forecast:\n%s", df_fc)
            
            plt.figure(figsize=(12,6))
            plt.plot(hurst_series.index, hurst_series.values, 'k-', label='Historical Hurst')
            plt.plot(df_fc['ds'], df_fc['yhat'], 'r--', label='Multi-step Forecast')
            plt.title(f'{periods}-Day Multi-step Hurst Forecast for {ticker}')
            plt.xlabel('Date'); plt.ylabel('Hurst Exponent')
            plt.legend(); plt.grid(True)
            plt.xticks(rotation=45); plt.tight_layout(); plt.show()
        except FileNotFoundError:
            logger.error(f"Multi-step model file not found. Please train the model first.")

if __name__ == '__main__':
    main()