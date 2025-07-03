# Gold Price Time-Series Prediction

This project explores and compares various machine learning and statistical models for forecasting gold prices using historical time-series data. The models include ARIMA, XGBoost, LSTM, and Transformer-based neural networks.

## Motivation

Gold price prediction is crucial for investors, traders, and researchers. Accurate forecasting helps in making informed financial decisions and understanding market dynamics.

## Technologies Used

- Python 3.8+
- pandas, numpy, matplotlib, scikit-learn
- xgboost
- statsmodels, pmdarima
- torch (PyTorch)
- tensorflow, keras (for LSTM/Transformer in some notebooks)

## Project Structure

- `data_preprocessing.py`: Data cleaning and preprocessing utilities.
- `gold_arima.py`: ARIMA model for time-series forecasting.
- `xgb.py`: XGBoost regression for gold price prediction.
- `transformer.ipynb`, `project_2.ipynb`, `black-scholes.ipynb`: Deep learning and advanced modeling notebooks.
- `results.ipynb`: Model comparison and visualization.
- `gold_dec24(GC=F)_1d.csv`, `gold_dec24(GC=F)_1wk.csv`, etc.: Input data files.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd gold-price-time-series
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

- Run scripts directly, e.g.:
  ```bash
  python data_preprocessing.py
  python gold_arima.py
  python xgb.py
  ```
- Open and run Jupyter notebooks for deep learning models and analysis.

## Data

Input data files (CSV) are required for running the models. Place them in the project root as provided.

## Results

See `results.ipynb` for model performance comparison and visualizations.

## License

MIT License (or specify your license here)

   
