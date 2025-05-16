import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime

# Set paths
repo_root = os.path.abspath('.')
data_dir = Path(repo_root) / 'XGboost_Roshan' / 'src' / 'data'

print("Loading dataset3...")
# Load dataset3 using pickle for efficiency
with open(data_dir / 'dataset3.pkl', 'rb') as f:
    df3 = pickle.load(f)

# Choose a ticker that exists in dataset3
chosen_ticker = 'AAPL'
if 'ticker' in df3.columns:
    tickers3 = df3['ticker'].unique()
    if chosen_ticker not in tickers3:
        chosen_ticker = tickers3[0]  # Default to first ticker if AAPL not found
    
    print(f"\nUsing {chosen_ticker} for analysis")
    
    # Process stock price data from dataset3
    ticker_data_full = df3[df3['ticker'] == chosen_ticker].copy()
    ticker_data_full['date'] = pd.to_datetime(ticker_data_full['date'])
    ticker_data_full = ticker_data_full.sort_values(by='date')

    # Initialize parameters that might be overwritten by dataset1
    X0 = None
    latest_date = None
    strike = None
    log_returns = np.array([]) # Ensure it's an array
    
    # --- Attempt to load X0 and strike from dataset1 ---
    option_loaded_from_dataset1 = False
    MIN_PRICE_POINTS_FOR_PARAMS = 22 # For Hurst and rolling volatility

    print("\nAttempting to load option data from dataset1.csv...")
    try:
        dataset1_path = data_dir / 'dataset1.csv'
        if dataset1_path.exists():
            df1 = pd.read_csv(dataset1_path)
            
            # Ensure correct column names as per user spec
            # date, secid, exdate, cp_flag, strike_price
            required_cols_d1 = ['date', 'secid', 'exdate', 'cp_flag', 'strike_price']
            if not all(col in df1.columns for col in required_cols_d1):
                print(f"Warning: dataset1.csv is missing one of required columns: {required_cols_d1}")
            else:
                df1_ticker = df1[df1['secid'] == chosen_ticker].copy()
                if not df1_ticker.empty:
                    df1_ticker['date'] = pd.to_datetime(df1_ticker['date'], errors='coerce')
                    df1_ticker['exdate'] = pd.to_datetime(df1_ticker['exdate'], errors='coerce')
                    df1_ticker.dropna(subset=['date', 'exdate', 'strike_price', 'cp_flag'], inplace=True)
                    
                    # Filter for valid options (exdate > date)
                    df1_ticker = df1_ticker[df1_ticker['exdate'] > df1_ticker['date']]
                    df1_ticker = df1_ticker.sort_values(by='date', ascending=False)

                    selected_option = None
                    # Prefer Call options
                    call_options = df1_ticker[df1_ticker['cp_flag'] == 'C']
                    if not call_options.empty:
                        selected_option = call_options.iloc[[0]]
                    else:
                        # Fallback to Put options
                        put_options = df1_ticker[df1_ticker['cp_flag'] == 'P']
                        if not put_options.empty:
                            selected_option = put_options.iloc[[0]]
                    
                    if selected_option is not None and not selected_option.empty:
                        option_record_date = selected_option['date'].iloc[0]
                        strike_price_from_d1 = selected_option['strike_price'].iloc[0] / 1000.0
                        
                        # Find stock price (X0) from df3 on option_record_date
                        stock_data_for_X0 = ticker_data_full[ticker_data_full['date'] == option_record_date]
                        
                        if not stock_data_for_X0.empty:
                            X0 = stock_data_for_X0['close'].iloc[0]
                            latest_date = option_record_date
                            strike = strike_price_from_d1
                            option_loaded_from_dataset1 = True
                            print(f"Successfully loaded option from dataset1 for {chosen_ticker} on {option_record_date.strftime('%Y-%m-%d')}:")
                            print(f"  Underlying Price (X0) from df3: {X0:.4f}")
                            print(f"  Strike Price from dataset1: {strike:.4f}")
                        else:
                            print(f"Warning: Could not find stock price in dataset3 for {chosen_ticker} on {option_record_date.strftime('%Y-%m-%d')}. Using defaults for X0 and strike.")
                    else:
                        print(f"No suitable options found for {chosen_ticker} in dataset1.csv after filtering. Using defaults for X0 and strike.")
                else:
                    print(f"{chosen_ticker} not found in dataset1.csv (secid column). Using defaults for X0 and strike.")
        else:
            print(f"dataset1.csv not found at {dataset1_path}. Using defaults for X0 and strike.")
    except Exception as e:
        print(f"Error processing dataset1.csv: {e}. Using defaults for X0 and strike.")

    # Determine data for calculating log returns (for eta, xi, rho)
    prices_for_params_calc = np.array([])
    if option_loaded_from_dataset1:
        # Use data up to the option's date for parameter calculation if sufficient
        data_for_params = ticker_data_full[ticker_data_full['date'] <= latest_date].copy()
        if len(data_for_params['close'].values) >= MIN_PRICE_POINTS_FOR_PARAMS:
            prices_for_params_calc = data_for_params['close'].values
            print(f"Using historical data up to {latest_date.strftime('%Y-%m-%d')} for eta, xi, rho calculation.")
        else:
            prices_for_params_calc = ticker_data_full['close'].values
            print(f"Warning: Historical data up to {latest_date.strftime('%Y-%m-%d')} (len: {len(data_for_params['close'].values)}) is too short for robust eta, xi, rho. Using full history.")
    else:
        # Fallback: use full history if option not loaded or X0 not found
        prices_for_params_calc = ticker_data_full['close'].values
        print("Using full available historical data for eta, xi, rho calculation.")

    if len(prices_for_params_calc) > 1:
        log_returns = np.log(prices_for_params_calc[1:] / prices_for_params_calc[:-1])
    else:
        log_returns = np.array([]) # Ensure it is empty if no data
        print("Warning: Not enough price data to calculate log returns for parameters.")

    # Set X0 and strike if not loaded from dataset1
    if not option_loaded_from_dataset1:
        if not ticker_data_full.empty:
            X0 = ticker_data_full['close'].values[-1]
            latest_date = ticker_data_full['date'].values[-1]
            strike = X0 * 1.05
            print("Using default X0 (latest price from df3) and strike (X0 * 1.05).")
        else:
            # This case should ideally not happen if chosen_ticker is in df3
            print("Critical Error: ticker_data_full is empty. Cannot determine X0 or strike.")
            # Set to NaN or raise error to prevent further calculation with bad data
            X0, latest_date, strike = np.nan, pd.NaT, np.nan 
    
    # Ensure all parameters are not None before proceeding
    if X0 is None or latest_date is None or strike is None:
        print("Critical Error: X0, latest_date, or strike could not be determined. Aborting parameter calculation.")
        # Exit or handle error appropriately
        # For now, we'll let it proceed and likely fail at print statements, but this should be handled.
    else:
        print(f"Latest price date considered: {latest_date.strftime('%Y-%m-%d') if pd.notnull(latest_date) else 'N/A'}")
        print(f"Initial price (X0): {X0:.4f}")
        print(f"Strike price: {strike:.4f}")

    # Calculate eta (roughness parameter) using Hurst exponent
    # Ensure log_returns is not empty and long enough for Hurst calculation
    H = np.nan # Default to NaN
    eta = np.nan # Default to NaN
    if len(log_returns) >= 8: # Minimum length for Hurst function (approx)
        def calculate_hurst(time_series):
            """Calculate Hurst exponent using R/S analysis"""
            lags = range(2, min(21, len(time_series)//4))
            tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
            reg = LinearRegression().fit(np.log(lags).reshape(-1, 1), np.log(tau))
            H = reg.coef_[0]
            return H
        
        H = calculate_hurst(log_returns)
        eta = 2 - 2*H
        print(f"Hurst exponent (H): {H:.4f}")
        print(f"Roughness parameter (eta): {eta:.4f}")
    
    # 4. Calculate volatility (xi) - annualized volatility
    xi = np.nan # Default to NaN
    if len(log_returns) >= 2: # Minimum for std deviation
        xi = np.std(log_returns) * np.sqrt(252)
    print(f"Volatility (xi): {xi:.4f}")
    
    # 5. Calculate correlation (rho) between returns and volatility changes
    rho = -0.7  # Default value
    # Ensure log_returns is long enough for rolling window
    if len(log_returns) >= rolling_window: # rolling_window is 21
        rolling_vol = pd.Series(log_returns).rolling(window=rolling_window).std().values
        rolling_vol = rolling_vol[~np.isnan(rolling_vol)]
        
        if len(rolling_vol) > 1: # Need at least two points to calculate diff
            vol_changes = np.diff(rolling_vol)
            # Align returns: log_returns starts from index 0 (price[1]/price[0])
            # rolling_vol starts effectively after 'rolling_window - 1' initial log_returns
            # vol_changes starts one step after that.
            # So, aligned_returns should correspond to log_returns that generated the first part of vol_changes.
            # If log_returns has length N, rolling_vol has N - (rolling_window - 1) elements after removing NaNs from start.
            # vol_changes has len(rolling_vol) - 1 elements.
            # The first element of rolling_vol corresponds to log_returns[0:rolling_window-1].std()
            # The first element of vol_changes is rolling_vol[1] - rolling_vol[0].
            # The returns should align with the period *before* the vol change.
            # The original code aligns log_returns[rolling_window : rolling_window+len(vol_changes)]
            # This means it takes returns from the (rolling_window)-th log_return onwards.
            # Let's re-verify alignment logic.
            # Original: aligned_returns = log_returns[rolling_window:rolling_window+len(vol_changes)]
            # A common approach is to correlate returns with subsequent volatility or concurrent.
            # Let's use returns that lead to the *first* volatility in the pair used for vol_changes.
            # rolling_vol[i] is std of log_returns[i : i + rolling_window -1] (conceptually, after handling NaNs)
            # vol_changes[k] = rolling_vol[k+1] - rolling_vol[k]
            # We need to correlate log_returns with vol_changes.
            # If we correlate r_t with (vol_t+1 - vol_t), then r_t is from period t.
            # Original alignment might be offset.
            # For simplicity, let's stick to original alignment for now but note this could be refined.
            
            # Ensure lengths match for correlation
            start_index_returns = rolling_window -1 # Start of the first full window for rolling_vol
                                                   # vol_changes starts effectively one period later.
            
            # We need to align log_returns with vol_changes.
            # vol_changes[i] is based on rolling_vol[i+1] and rolling_vol[i]
            # rolling_vol[i] is std over log_returns up to index i + (rolling_window -1)
            # A common definition for leverage effect is correlation between r_t and sigma_{t+1} or (sigma_{t+1} - sigma_t)
            # If vol_changes[k] = rolling_vol[k+1] - rolling_vol[k],
            # and rolling_vol[k] summarizes log_returns up to some point j,
            # then log_return at j or j-1 could be correlated.
            # The original code: aligned_returns = log_returns[rolling_window:rolling_window+len(vol_changes)]
            # This means log_returns from index 21 up to 21+len(vol_changes).
            # This attempts to align returns that occur *concurrently* with the window producing the *first* part of the vol_change.
            
            # Let's use a slightly more robust way to slice, ensuring indices are valid
            # The first element of rolling_vol (non-NaN) corresponds to log_returns[0...rolling_window-1].
            # The second element of rolling_vol corresponds to log_returns[1...rolling_window].
            # vol_changes[0] = rolling_vol[1] - rolling_vol[0].
            # The return contemporaneous with rolling_vol[0] is log_returns[rolling_window-1].
            # The return contemporaneous with rolling_vol[1] is log_returns[rolling_window].
            # So, to align with vol_changes[0], we could use log_returns[rolling_window].
            
            # The original code `log_returns[rolling_window : rolling_window+len(vol_changes)]` seems plausible.
            # It aligns the return at the end of the second window of the pair forming the first vol_change.
            
            if len(vol_changes) > 0:
                # The returns that are input to the std calculation for rolling_vol[k] are log_returns[k : k+rolling_window]
                # The returns that are input to the std calculation for rolling_vol[k+1] are log_returns[k+1 : k+1+rolling_window]
                # vol_changes[k] = std(log_returns[k+1 : k+1+rolling_window]) - std(log_returns[k : k+rolling_window])
                # We want to correlate this with a log_return. Which one? Typically log_return[k] or log_return[k+1].
                # The provided alignment: log_returns[rolling_window : rolling_window+len(vol_changes)]
                # log_returns[21] aligns with vol_changes[0].
                # log_returns[21] is the 22nd log return.
                # rolling_vol[0] is std(log_returns[0]...log_returns[20])
                # rolling_vol[1] is std(log_returns[1]...log_returns[21])
                # vol_changes[0] = rolling_vol[1] - rolling_vol[0]
                # Correlating with log_returns[21] (the last return in the second window) seems reasonable for contemporaneous correlation.
                
                # Make sure the slice for aligned_returns is valid and has same length as vol_changes
                num_vol_changes = len(vol_changes)
                # The first element of rolling_vol comes from log_returns[:rolling_window]
                # The first vol_change involves rolling_vol[0] and rolling_vol[1]
                # rolling_vol[1] involves log_returns[1:rolling_window+1]
                # So vol_changes[0] is related to log_returns up to index rolling_window.
                # The relevant return for vol_changes[i] could be log_returns[i + rolling_window]
                
                # The original logic was:
                # aligned_returns = log_returns[rolling_window:rolling_window+len(vol_changes)]
                # This slice produces len(vol_changes) elements, starting from log_returns[rolling_window]
                
                # Let's make it safer:
                # We need to make sure this slice does not go out of bounds for log_returns
                # And that it produces the same number of elements as vol_changes
                
                # The first value of rolling_vol (non-NaN) is calculated from log_returns[0:rolling_window].
                # The first value of vol_changes is rolling_vol[1] - rolling_vol[0].
                # rolling_vol[1] uses log_returns[1:rolling_window+1].
                # A contemporaneous return for vol_changes[0] could be log_returns[rolling_window] (the last return in the window for rolling_vol[1]).
                
                slice_start = rolling_window 
                # This corresponds to log_returns[21] if rolling_window is 21.
                # This is the last return in the window that gives rolling_vol[1].
                
                if slice_start + num_vol_changes <= len(log_returns):
                    aligned_returns = log_returns[slice_start : slice_start + num_vol_changes]
                    if len(aligned_returns) == num_vol_changes and num_vol_changes > 0: # Ensure we have data for correlation
                         rho_corr = np.corrcoef(aligned_returns, vol_changes)
                         if not np.isnan(rho_corr[0,1]):
                            rho = rho_corr[0,1]
                         else:
                            print("Warning: Could not calculate correlation for rho (NaN result). Using default.")
                    else:
                        print("Warning: Length mismatch or no data for rho calculation after slicing. Using default.")
                else:
                     print("Warning: Not enough log_returns for specified alignment with vol_changes for rho. Using default.")
            else:
                print("Warning: Not enough volatility changes to calculate rho. Using default.")
        else:
            print("Warning: Not enough rolling volatility data to calculate rho. Using default.")
    else:
        print(f"Warning: Log returns length ({len(log_returns)}) is less than rolling window ({rolling_window}). Cannot calculate rho. Using default.")
    print(f"Return-volatility correlation (rho): {rho:.4f}")
    
    # 6. Risk-free rate (r) - use standard value
    r = 0.05  # Standard risk-free rate
    
    # 7. Strike price - set at 5% above current price (standard for slightly OTM option)
    strike = X0 * 1.05
    print(f"Strike price (5% OTM): {strike:.4f}")
    
    # Try to load dataset1 sample to check if we can get better parameters
    print("\nChecking dataset1 for options data...")
    try:
        # Load a small sample of dataset1
        df1_sample = pd.read_csv(data_dir / 'dataset1.csv', nrows=1000)
        
        # Check if our ticker exists in the sample
        if 'secid' in df1_sample.columns and chosen_ticker in df1_sample['secid'].values: # Changed 'ticker' to 'secid'
            print(f"Found {chosen_ticker} in dataset1 sample, full data was processed if available.")
            # The logic for using dataset1 is now above. This is just a confirmation message.
        else:
            print(f"Note: {chosen_ticker} not found in dataset1 sample (secid column).")
    except Exception as e:
        print(f"Note: Could not check/load dataset1 sample: {e}")
    
    # Summary of calculated parameters
    print("\nFinal option parameters:")
    print(f"eta = {eta:.4f}")
    print(f"X0 = {X0:.4f if pd.notnull(X0) else 'N/A'}")
    print(f"r = {r:.4f}")
    print(f"rho = {rho:.4f}")
    print(f"xi = {xi:.4f}")
    print(f"strike = {strike:.4f if pd.notnull(strike) else 'N/A'}")
    
    # Save parameters to a file
    if pd.notnull(X0) and pd.notnull(strike) and pd.notnull(eta) and pd.notnull(rho) and pd.notnull(xi) and pd.notnull(r):
        params = {
            'eta': eta,
            'X0': X0,
            'r': r,
            'rho': rho,
            'xi': xi,
            'strike': strike,
            'ticker': chosen_ticker,
            'calculation_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        with open('option_parameters.pkl', 'wb') as f:
            pickle.dump(params, f)
        print("\nParameters saved to option_parameters.pkl")
        
        # Create a Python file with the parameters as constants
        with open('option_params.py', 'w') as f:
            f.write(f"# Option parameters for {chosen_ticker} calculated on {datetime.now().strftime('%Y-%m-%d')}\n\n")
            f.write(f"ETA = {eta:.4f}\n")
            f.write(f"X0 = {X0:.4f}\n")
            f.write(f"R = {r:.4f}\n")
            f.write(f"RHO = {rho:.4f}\n")
            f.write(f"XI = {xi:.4f}\n")
            f.write(f"STRIKE = {strike:.4f}\n")
        print("Parameters also saved to option_params.py for easy import")
    else:
        print("\nSkipping saving parameters due to NaN values in one or more key parameters.")
    
else:
    print("Column 'ticker' not found in dataset3") 