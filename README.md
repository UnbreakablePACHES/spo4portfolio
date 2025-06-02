# Smart Predict-then-Optimize Plus for Portfolio Optimization

## Project Description  
This project implements an end-to-end portfolio optimization framework based on the Smart Predict-then-Optimize Plus (SPO+) methodology. By integrating deep learning time series forecasting models with SPO+ loss functions, it aims to improve portfolio allocation by jointly optimizing prediction accuracy and investment decisions.

## Key Features  
- Forecast future asset returns using deep learning models such as LSTM and DLinear  
- Train models with SPO+ loss to align prediction with optimization goals  
- Support multiple global stock indices datasets (e.g., S&P 500, Dow Jones Industrial Average)  
- Backtesting module to evaluate portfolio performance  
- Visualization of training losses and backtesting results

## Tech Stack  
- Python 3.8+  
- PyTorch or TensorFlow  
- pandas, numpy, matplotlib for data processing and visualization  
- yfinance API (optional) for financial data retrieval

## Dataset  
Daily closing prices of major global stock indices from 2000 to 2024, sourced from Wind database or yfinance.

## Installation and Usage  
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
python train.py       # Train the model
python backtest.py    # Run portfolio backtesting
