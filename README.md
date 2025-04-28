# Optimal Stopping with Signatures

This repository contains the implementations related to the numerical section of the paper "Primal and dual optimal stopping with signatures" (https://arxiv.org/abs/2312.03444), as well as extended methods relying on deep and kernel learning methodologies, accompanying a forthcoming paper on "American option pricing in rough volatility models".

## How to use the code

A step-by-step guidance with notebooks is provided for:
- Optimal stopping of fractional Brownian motion (lower and upper bounds) (Example_Optimal_Stopping_FBM.ipynb)
- Pricing American options in rBergomi model (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554754), (lower and upper bounds) (Example_American_Put_Option_rBergomi.ipynb)
- XGBoost model for volatility analysis and prediction (Roshan.ipynb)

Additionally, for American options in the rough Bergomi we provide two files Testing_linear_signature_stopping.py and Testing_deep_signature_stopping.py, where one can play around with different (hyper) parameters and model choices. Notice that the implementation does not depend on the underlying model, and these examples can easily be modified for different models by simply changing the simulation of the training and testing data.

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

### macOS (Homebrew) Setup
If you're on macOS using Homebrew, you can install a Python version compatible with TensorFlow (Python ≤ 3.10) and create your venv as follows:
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

## Project Structure

The repository is organized as follows:

```
Optimal_Stopping_with_signatures/
├── Linear signature optimal stopping/
│   ├── Linear_signature_optimal_stopping.py
│   └── Testing_linear_signature_stopping.py
├── Non linear signature optimal stopping/
│   ├── Deep_signatures_optimal_stopping.py
│   ├── Deep_kernel_signature_optimal_stopping.py
│   └── Testing_deep_signature_stopping.py
├── Notebooks/
│   ├── Example_Optimal_Stopping_FBM.ipynb
│   ├── Example_American_Put_Option_rBergomi.ipynb
│   └── Roshan.ipynb
├── XGboost_Roshan/
│   ├── src/
│   │   ├── main.py
│   │   ├── train_hurst.py
│   │   ├── features/
│   │   ├── models/
│   │   └── utils/
├── Signature_computer.py
├── SignatureKernelComputer.py
├── rBergomi_simulation.py
├── FBM_package.py
├── utils.py
├── requirements.txt
└── README.md
```

## XGBoost Volatility Analysis Extension

The XGboost_Roshan directory contains an extension to the main project that utilizes XGBoost models to analyze and predict volatility based on historical data. The primary focus is on creating volatility paths and implementing windowing techniques to enhance model performance.

### Using the XGBoost Component

1. Dataset: Place your volatility dataset in `XGboost_Roshan/src/data/`
2. Training: Run `python XGboost_Roshan/src/main.py` to execute model training
3. The implementation includes:
   - Windowing functions for feature engineering
   - XGBoost model implementation with training and prediction methods
   - Volatility path calculation and visualization

## Remarks about the code:

- The module Signature_computer.py relies in the package iisignature (https://pypi.org/project/iisignature/), and allows to compute log and standard signatures for various variation of underlying paths.
- The LinearDualSolver in Linear_signature_optimal_stopping.py has the option of choosing Gurobi optimization to solve the linear programs, which requires a free license (an explanation how to install it can be found here https://www.gurobi.com/academia/academic-program-and-licenses/). It is recommended to use it for high-dimensional problems, but alternatively one can set LP_solver ="CVXPY", to use the free cvxpy solvers.
- For the simulation of rBergomi model we use (slightly changed version of) the code from R. McCrickerd (https://github.com/ryanmccrickerd/rough_bergomi)
- For the simulation of fractional Brownian motion we use (slightly changed version of) the package C. Flynn (https://pypi.org/project/fbm/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## Recent Updates

### Bug Fixes and Compatibility Updates

1. **M3/M2/M1 Mac Compatibility**
   - Updated TensorFlow optimizer to use legacy version for better performance on Apple Silicon
   - Changed `tf.keras.optimizers.Adam` to `tf.keras.optimizers.legacy.Adam` in both deep learning implementations

2. **TypeError Fix in AutoGraph Conversion**
   - Modified `_is_known_loaded_type` function in TensorFlow's autograph conversion to handle TypeError gracefully
   - Added try/except blocks around isinstance checks to prevent conversion crashes
   - Updated function signature to accept correct number of arguments (f, module_name, entity_name)

### Installation

// ... existing code ...

