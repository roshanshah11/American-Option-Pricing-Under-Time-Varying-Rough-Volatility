# Volatility XGBoost Project

This project aims to utilize an XGBoost model to analyze and predict volatility based on historical data. The primary focus is on creating a volatility path using a specified dataset and implementing a windowing technique to enhance the model's performance.

## Project Structure

```
volatility-xgboost-project
├── src
│   ├── data
│   │   └── dataset2.csv
│   ├── features
│   │   └── windowing.py
│   ├── models
│   │   └── xgboost_model.py
│   ├── utils
│   │   └── volatility_path.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd volatility-xgboost-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset in `src/data/dataset2.csv`.
2. Run the main script to execute the model training and volatility path calculation:
   ```
   python src/main.py
   ```

## Overview

- **Dataset**: The project uses `dataset2.csv` for training the XGBoost model and generating the volatility path.
- **Windowing**: The `windowing.py` file contains functions to create 30-day windows from the dataset, which helps in feature engineering.
- **Model**: The `xgboost_model.py` file implements the XGBoost model with methods for training and prediction.
- **Volatility Calculation**: The `volatility_path.py` file includes functions to calculate and visualize the volatility path based on the model's predictions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.