#!/usr/bin/env python3
# This script computes the rolling Hurst exponent of a time series
# and trains/evaluates an LSTM model to forecast its future values.

import numpy as np                # numerical operations (arrays, math)
import pandas as pd               # data manipulation (DataFrame)
import matplotlib.pyplot as plt   # plotting library for visualization
from pathlib import Path          # convenient filesystem path handling
from sklearn.metrics import mean_squared_error, mean_absolute_error # error metrics for model evaluation
from sklearn.preprocessing import MinMaxScaler  # for normalizing data
import logging                    # logging status and debug messages
import joblib                     # saving/loading Python objects (models)
import tensorflow as tf           # deep learning framework
from tensorflow.keras.models import Sequential, load_model  # Neural network model components
from tensorflow.keras.layers import LSTM, Dense, Dropout    # Neural network layers
from tensorflow.keras.callbacks import EarlyStopping        # Prevent overfitting

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

def create_sequences(data, seq_length):
    """
    Create sequences for LSTM model training - reshapes data into
    3D format expected by LSTM: [samples, time steps, features]
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_lstm_model(seq_length, dropout_rate=0.2):
    """
    Create an LSTM model with specified sequence length and dropout rate
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(hurst_series: pd.Series, train_frac: float, seq_length: int, epochs: int, batch_size: int):
    """Train an LSTM model on the Hurst series."""
    # here is a source i used: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    
    # Split the data into training and testing sets
    split = int(len(hurst_series) * train_frac)
    train = hurst_series.values[:split]
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.reshape(-1, 1))
    
    # Create sequences for LSTM
    X_train, y_train = create_sequences(train_scaled, seq_length)
    
    # Build and train the LSTM model
    model = build_lstm_model(seq_length)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    logger.info("Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save the model and scaler
    model.save(MODELS_DIR / 'lstm_model.h5')
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    logger.info("LSTM model and scaler saved to %s", MODELS_DIR)
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model, scaler

def evaluate_lstm_model(hurst_series: pd.Series, model, scaler, seq_length: int, test_frac: float):
    """Evaluate the LSTM model on test data and plot results."""
    # here is a source i used: https://scikit-learn.org/stable/modules/model_evaluation.html
    
    # Split the data
    split = int(len(hurst_series) * (1 - test_frac))
    test = hurst_series.values[split-seq_length:]  # Include seq_length previous points for first prediction
    test_scaled = scaler.transform(test.reshape(-1, 1))
    
    # Create sequences for testing
    X_test, y_test = create_sequences(test_scaled, seq_length)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    logger.info("LSTM model results – MSE: %.4f, MAE: %.4f", mse, mae)
    
    # Plot actual vs prediction
    plt.figure(figsize=(12, 6))
    dates_te = hurst_series.index[split:split+len(y_pred)]
    plt.plot(dates_te, y_test_orig, 'k-', label='Actual Hurst')
    plt.plot(dates_te, y_pred, 'r--', label='LSTM Forecast')
    plt.title('LSTM: Actual vs Forecast Hurst Exponent')
    plt.xlabel('Date')
    plt.ylabel('Hurst Exponent')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return mse, mae

def forecast_lstm(hurst_series: pd.Series, model, scaler, periods: int, seq_length: int):
    """Generate future forecasts using the LSTM model recursively."""
    # here is a source i used: https://pandas.pydata.org/docs/reference/api/pandas.date_range.html
    
    # Get the most recent data points for initial sequence
    last_sequence = hurst_series.values[-seq_length:].reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence)
    
    # Make recursive forecasts
    curr_sequence = last_sequence_scaled.reshape(1, seq_length, 1)
    forecasts = []
    
    for _ in range(periods):
        # Predict the next value
        next_pred_scaled = model.predict(curr_sequence)
        forecasts.append(next_pred_scaled[0, 0])
        
        # Update the sequence for the next prediction (slide window)
        curr_sequence = np.append(curr_sequence[:, 1:, :], 
                                 [[next_pred_scaled]], 
                                 axis=1)
    
    # Convert scaled predictions back to original scale
    forecasts_array = np.array(forecasts).reshape(-1, 1)
    forecast_values = scaler.inverse_transform(forecasts_array).flatten()
    
    # Create forecast dataframe with dates
    dates = pd.date_range(hurst_series.index[-1] + pd.Timedelta(days=1), periods=periods, freq='B')
    return pd.DataFrame({'ds': dates, 'yhat': forecast_values})

def interactive_menu():
    # here is a source i used: https://docs.python.org/3/library/functions.html#input
    print("Select action:")
    print("0: Train LSTM model (interactive)")
    print("1: Evaluate LSTM model (interactive)")
    print("2: Forecast LSTM model (interactive)")
    print("4: Quick train LSTM with defaults")
    print("5: Quick evaluate LSTM with defaults")
    print("6: Quick forecast LSTM with defaults")
    choice = input("Enter choice (0,1,2,4,5,6): ")
    return choice

def main():
    """
    Main entry point: show menu, get user choice, compute Hurst series,
    and train/evaluate/forecast the LSTM model as requested.
    """
    # here is a source i used: https://docs.python.org/3/library/logging.html
    logger.info("Starting main execution...")
    mode = interactive_menu().strip()
    if mode not in ['0','1','2','4','5','6']:
        logger.error("Invalid mode selected. Exiting.")
        return

    # Interactive input vs. quick defaults
    if mode in ['0','1','2']:
        power = int(input("Enter power for rolling Hurst (default 6): ") or 6)
        ticker = input("Enter ticker (default AAPL): ") or 'AAPL'
        train_frac = float(input("Enter train fraction (default 0.8): ") or 0.8)
        test_frac = 1 - train_frac
        seq_length = int(input("Enter sequence length (default 10): ") or 10)
        epochs = int(input("Enter epochs (default 50): ") or 50)
        batch_size = int(input("Enter batch size (default 32): ") or 32)
        periods = int(input("Enter forecast days (default 14): ") or HORIZON)
    else:
        power, ticker = 4, 'AAPL'
        train_frac, test_frac = 0.8, 0.2
        seq_length = 10
        epochs = 50
        batch_size = 32
        periods = HORIZON

    # Load data and filter by ticker symbol
    df_all = load_data()
    df_t = df_all[df_all['ticker']==ticker].sort_values('date')
    returns = df_t['return'].values

    # Compute rolling Hurst exponent series
    hurst_vals = compute_rolling_hurst(returns, power)
    dates = df_t['date']
    start_index = 2**power - 1
    hurst_series = pd.Series(hurst_vals, index=dates[start_index:])

    # Execute requested action
    if mode in ['0','4']:
        train_lstm_model(hurst_series, train_frac, seq_length, epochs, batch_size)
    elif mode in ['1','5']:
        model = load_model(MODELS_DIR / 'lstm_model.h5')
        scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        evaluate_lstm_model(hurst_series, model, scaler, seq_length, test_frac)
    elif mode in ['2','6']:
        model = load_model(MODELS_DIR / 'lstm_model.h5')
        scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        df_fc = forecast_lstm(hurst_series, model, scaler, periods, seq_length)
        logger.info("LSTM forecast:\n%s", df_fc)
        
        # Plot historical data and forecast
        plt.figure(figsize=(12, 6))
        plt.plot(hurst_series.index, hurst_series.values, 'k-', label='Historical Hurst')
        plt.plot(df_fc['ds'], df_fc['yhat'], 'r--', label='LSTM Forecast')
        plt.title(f'{periods}-Day Hurst Forecast for {ticker} using LSTM')
        plt.xlabel('Date')
        plt.ylabel('Hurst Exponent')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()