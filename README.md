American Option Pricing Under Time-Varying Rough Volatility
===========================================================

A Signature-Based Hybrid Framework

License: MIT
Python Version: 3.7+
Interface: Jupyter Notebook

Project Summary
---------------
This repository implements a modular framework for pricing American options under non-stationary rough volatility using signature methods. It extends the work of Bayer et al. (2025) by adding:
- Rolling Hurst forecasts
- Regime-aware model switching
- Random Fourier Features (RFFs) for kernel acceleration

The result is a hybrid architecture that adapts to evolving market conditions while avoiding the computational burden of deep neural networks.

Key Features
------------
- XGBoost-based multi-step forecasting
- Regime selection between rough Bergomi and Heston
- Signature-based pricing under duality bounds
- Fast kernel approximations using RFF
- CPU-friendly runtime (no GPU or deep learning required)

Repository Structure
--------------------
- data/: Market data and intermediate outputs
- notebooks/: Jupyter notebooks for demonstration
- src/: Core Python source code
- results/: Output charts and pricing intervals
- requirements.txt: Python dependencies

Quick Start
-----------
1. Clone the repository
2. Create a virtual environment and install from requirements.txt
3. Run Jupyter Notebook to explore demos

Example Usage
-------------
Import the pipeline and run pricing:

from src.pipeline import run_pricing_pipeline

result = run_pricing_pipeline(
    ticker="AAPL",
    option_type="put",
    dte=10,
    as_of="2024-08-31"
)

print(result.price_bounds)

Results
-------
- Reduces duality gaps by up to 50% compared to deep signature baselines
- Runs 2–3x faster using RFF-based approximation

Dependencies
------------
- numpy, pandas, scipy
- xgboost
- iisignature or esig
- matplotlib, seaborn
- Optional: torch or tensorflow

References
----------
1. Bayer et al. (2025)
2. Lyons (2014)
3. Gatheral et al. (2018)
4. Rahimi & Recht (2007)

Contact
-------
Author: Roshan Shah
GitHub: @roshanshah11
Email: rshah25@lawrenceville.org

License
-------
MIT License
