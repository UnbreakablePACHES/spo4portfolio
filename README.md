# Smart Predict-then-Optimize Plus for Portfolio Optimization

## Project Description  
This project implements an end-to-end portfolio optimization framework based on the Smart Predict-then-Optimize + (SPO+) methodology. 
(For more details: [Smart Predict-then-Optimize: Learning Policies for Decision-Making under Uncertainty](https://arxiv.org/abs/1710.08005))

We aims to improve portfolio allocation by jointly optimizing prediction accuracy and investment decisions by integrating deep learning time series forecasting models with SPO+ loss functions.

## Key Features  
- Forecast future asset returns using deep learning models such as LSTM and DLinear  
- Train models with SPO+ loss to align prediction with optimization goals  
- Backtesting module to evaluate portfolio performance  
- Visualization of training losses and backtesting results

## Environment 
- Deep Learning Framework: PyTorch (version >=1.8)
- Required Python Packages:  
  - pandas (>=1.1)  
  - numpy (>=1.18)  
  - matplotlib (>=3.3)  
  - yfinance

## Dataset  
Daily data in MSCI ishares ETF from 2023-01-01 to 2024-12-31, sourced from yfinance.
