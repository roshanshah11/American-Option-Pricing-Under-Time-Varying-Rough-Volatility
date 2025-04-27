import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as sps
import pandas as pd
from pathlib import Path
import random

power = 4  # This gives 2^7 = 128 datapoints for the rolling window
# The rolling sample length
n = 2**power

# Load data from local file
# Determine data directory relative to script
data_dir = Path(__file__).parent / 'data'
# Load and prepare the dataset
df = pd.read_csv(data_dir / 'dataset3.csv', low_memory=False)

# Convert date to datetime and sort
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Get unique tickers
tickers = df['ticker'].unique()
print(f"Found {len(tickers)} unique ticker(s) in the dataset.")
print(f"Sample tickers: {tickers[:5]}...")

# More efficient way to filter tickers
ticker_counts = df.groupby('ticker')['return'].count()
valid_tickers = list(ticker_counts[ticker_counts == 252].index)
print(valid_tickers)
print(f"Found {len(valid_tickers)} valid ticker(s) with exactly 252 data points.")
print(f"Valid tickers sample: {valid_tickers[:5]}...")

# Process a single ticker (first one by default)
ticker = random.choice(valid_tickers)
print(f"Chosen ticker: {ticker}")
ticker_data = df[df['ticker'] == ticker].copy()
ticker_data.set_index('date', inplace=True)

# Extract returns directly from dataset
returns = ticker_data['return'].values

# Print data length for verification
#print(f"Ticker: {ticker}")
print(f"Number of datapoints: {len(returns)}")
print(f"Date range: {ticker_data.index.min()} to {ticker_data.index.max()}")
print(f"Using rolling window of size: {n}")

# Check if we have enough data
if len(returns) < n:
    raise ValueError(f"Not enough datapoints. Need at least {n}, but have {len(returns)}")

# Initialising arrays
hursts = np.array([])
tstats = np.array([])
pvalues = np.array([])

total_windows = len(returns) - n + 1
print(f"Starting rolling Hurst calculation over {total_windows} windows...")

# Calculating the rolling Hurst exponent
for t in np.arange(n, len(returns)+1):
    idx = t - n + 1
    if idx % 50 == 0 or idx == 1:
        print(f"Computing window {idx}/{total_windows}...")
    # Specifying the subsample
    data = returns[t-n:t]
    X = np.arange(2, power+1)
    Y = np.array([])
    for p in X:
        m = 2**p
        s = 2**(power-p)
        rs_array = np.array([])
        # Moving across subsamples
        for i in np.arange(0, s):
            subsample = data[i*m:(i+1)*m]
            mean = np.average(subsample)
            deviate = np.cumsum(subsample-mean)
            difference = max(deviate) - min(deviate)
            stdev = np.std(subsample)
            if stdev == 0:  # Avoid division by zero
                rescaled_range = 0
            else:
                rescaled_range = difference/stdev
            rs_array = np.append(rs_array, rescaled_range)
        # Calculating the log2 of average rescaled range
        Y = np.append(Y, np.log2(np.average(rs_array)))
    reg = sm.OLS(Y, sm.add_constant(X))
    res = reg.fit()
    hurst = res.params[1]
    hursts = np.append(hursts, hurst)


# Print results summary
print(f"Average Hurst: {np.mean(hursts):.4f}")
print(f"Min Hurst: {np.min(hursts):.4f}, Max Hurst: {np.max(hursts):.4f}")

# Visualising the Hurst exponent
plt.figure(figsize=(12, 6))
plt.title(f'Rolling Hurst Exponent for {ticker}')
plt.ylim(0.3, 1.1)
plt.plot(ticker_data.index[n-1:], hursts, label='Hurst Exponent')
plt.plot(ticker_data.index[n-1:], np.ones(len(hursts))*0.5, 'r--', label='H=0.5 (Random Walk)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()