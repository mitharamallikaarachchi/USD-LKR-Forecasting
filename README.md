# USD/LKR Exchange Rate Forecasting

## Project Overview

This project develops and evaluates time-series and machine learning approaches for forecasting the USD/LKR exchange rate using historical exchange-rate data.

The project follows an end-to-end forecasting pipeline covering:

- Data collection
- Data preparation
- Exploratory Data Analysis
- Statistical analysis
- Feature engineering
- Forecasting model development
- Model evaluation
- Model comparison
- Best model selection

## Objectives

- Collect and prepare historical USD/LKR exchange-rate data.
- Analyse historical exchange-rate behaviour.
- Analyse returns and volatility.
- Perform stationarity testing using the Augmented Dickey-Fuller test.
- Analyse temporal dependence using ACF and PACF.
- Create lagged and rolling statistical features.
- Develop one-step-ahead forecasting models.
- Compare models using MAE and RMSE.
- Identify the best-performing model based on out-of-sample performance.

## Project Pipeline

```text
Data Collection
      ↓
Data Preparation
      ↓
Exploratory Data Analysis
      ↓
Statistical Analysis
      ↓
Feature Engineering
      ↓
One-Step-Ahead Target Creation
      ↓
Time-Based Train/Test Split
      ↓
Forecasting Models
      ↓
Model Evaluation
      ↓
Model Comparison
      ↓
Best Model Selection

