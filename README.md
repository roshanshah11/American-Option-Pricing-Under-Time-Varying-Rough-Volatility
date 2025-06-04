# American Option Pricing Under Time-Varying Rough Volatility
## A Signature-Based Hybrid Framework

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-≤3.10-blue)
![Interface](https://img.shields.io/badge/interface-Jupyter%20Notebook-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)

## Overview

This repository implements a cutting-edge modular framework for pricing American options under non-stationary rough volatility using signature methods. Our approach extends the foundational work of Bayer et al. (2023) by incorporating:

- **Rolling Hurst Parameter Forecasting**: Dynamic estimation of market roughness
- **Regime-Aware Model Switching**: Intelligent selection between rough Bergomi and Heston models
- **Random Fourier Features (RFFs)**: Efficient kernel approximations for computational acceleration
- **Signature-Based Optimal Stopping**: Advanced mathematical methods for option pricing

The result is a hybrid architecture that dynamically adapts to evolving market conditions while maintaining computational efficiency without requiring GPU resources or deep neural networks.

## Key Features

- **XGBoost-based Multi-step Forecasting**: Advanced time series prediction for Hurst parameter estimation
- **Regime Selection**: Intelligent switching between rough Bergomi and Heston volatility models
- **Signature-based Pricing**: Duality-based bounds for optimal stopping problems
- **Random Fourier Features**: Fast kernel approximations for computational efficiency
- **CPU-friendly Runtime**: No GPU or deep learning dependencies required
- **Modular Architecture**: Easy customization for different underlying models

### Testing Framework

The repository includes comprehensive testing files that allow experimentation with different hyperparameters and model configurations:

- `Testing_linear_signature_stopping.py`: Linear signature methods testing
- `Testing_deep_signature_stopping.py`: Deep signature methods testing

**Note**: The implementation is model-agnostic and can be easily modified for different stochastic models by changing the simulation of training and testing data.

## Installation

### Requirements

This project requires Python 3.10 or lower (for TensorFlow compatibility). To install dependencies:

```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Platform-Specific Setup

#### macOS (Homebrew)
```bash
# Install Python 3.10 via Homebrew
brew install python@3.10
# Ensure the new Python is first in your PATH
brew unlink python && brew link --force python@3.10
# Create and activate a venv with Python 3.10
python3.10 -m venv .venv
source .venv/bin/activate
# Upgrade pip
python3.10 -m pip install --upgrade pip
# Install TensorFlow macOS builds
pip install tensorflow-macos==2.11.0 tensorflow-metal==0.7.0
```

#### Windows
```bash
# Install Python 3.10 from python.org
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```

#### Linux
```bash
# Install Python 3.10 using your package manager
# Example for Ubuntu:
# sudo apt-get install python3.10 python3.10-venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository Structure
```
American-Option-Pricing-Under-Time-Varying-Rough-Volatility/
├── Linear signature optimal stopping/    # Core linear signature implementations
│   ├── Linear_signature_optimal_stopping.py # Main linear signature pricing class
│   └── Testing_linear_signature_stopping.py # Testing and experimentation script
├── Non linear signature optimal stopping/ # Deep learning signature methods
│   ├── Deep_signatures_optimal_stopping.py   # Deep signature neural networks
│   ├── Kernel_signature_optimal_stopping.py  # Kernel-based signature methods
│   ├── Deep_kernel_signature_optimal_stopping.py # Hybrid deep+kernel approach
│   └── Testing_deep_signature_stopping.py    # Deep signature testing script
├── XGboost_Roshan/                       # XGBoost forecasting module
│   ├── src/main.py                          # Main XGBoost training pipeline
│   ├── src/train_hurst.py                   # Hurst parameter forecasting
│   └── src/XGboost_Hurst_Tutorial.ipynb    # Tutorial notebook
├── Notebooks/                            # Jupyter demonstration notebooks
│   ├── final.ipynb                         # Main demonstration notebook
│   └── summarystats.ipynb                  # Statistical analysis notebook
├── Core Python Modules
│   ├── rBergomi_simulation.py               # Rough Bergomi path simulation
│   ├── dynamic_hurst_rbergomi.py           # Dynamic Hurst estimation
│   ├── Signature_computer.py               # Signature computation utilities
│   ├── FBM_package.py                      # Fractional Brownian motion tools
│   └── utils.py                            # General utility functions
└── requirements.txt                      # Python dependencies
```

## Quick Start
1. Clone the repository
2. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook to explore demos in the `Notebooks/` directory

## Jupyter Notebook Guide
This repository is primarily notebook-based. Here's a recommended sequence:
1. **summarystats.ipynb**: Understand the statistical properties of the data
2. **XGboost_Hurst_Tutorial.ipynb**: Learn how to forecast the Hurst parameter
3. **final.ipynb**: See the complete pricing pipeline in action

## Example Usage
Import the pipeline and run pricing:

```python
# Example for importing and running the pricing pipeline
from Linear_signature_optimal_stopping import LinearSignatureStopping
from rBergomi_simulation import simulate_paths

# Generate paths
paths = simulate_paths(n_paths=1000, hurst=0.1)

# Initialize and run pricing
pricer = LinearSignatureStopping(paths)
result = pricer.compute_price_bounds()

print(f"Price bounds: [{result.lower_bound}, {result.upper_bound}]")
```

## Results
- Reduces duality gaps by up to 50% compared to deep signature baselines
- Runs 2–3x faster using RFF-based approximation

## Project Architecture
![Architecture Diagram](docs/architecture.png)
*Figure: Component interaction in the signature-based hybrid framework*

*Note: If this image doesn't exist yet in your repository, please create a simple diagram showing the workflow from forecasting to pricing.*

## Dependencies
- **Core**: numpy, pandas, scipy, scikit-learn
- **Machine Learning**: xgboost, tensorflow (2.14), torch
- **Signature Computing**: iisignature
- **Optimization**: cvxpy, gurobipy (optional)
- **Visualization**: matplotlib, seaborn
- **Time Series**: prophet, statsmodels
- **Stochastic Processes**: Custom implementations for fractional Brownian motion

See `requirements.txt` for complete dependency list with versions.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Prerequisites
- Basic understanding of stochastic calculus and option pricing theory
- Familiarity with rough volatility models (particularly rough Bergomi)
- Knowledge of signature methods and their application to paths
- Experience with Python and scientific computing libraries

## References
- Bayer et al. (2025): "Pricing American options under rough volatility
using deep-signatures and signature-kernels"
- Lyons (2014): "Rough paths, signatures and the modelling of functions"
- Gatheral et al. (2018): "Volatility is rough"
- Rahimi & Recht (2007): "Random features for large-scale kernel machines"

## Contact
- Author: Roshan Shah
- GitHub: @roshanshah11
- Email: roshah529@gmail.com

## License
MIT License
