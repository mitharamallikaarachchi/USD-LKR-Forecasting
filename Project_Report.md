# Comprehensive Research Report: USD/LKR Exchange Rate Forecasting Under Economic Regime Changes

## 1. Executive Summary
This project investigates the predictability of the US Dollar to Sri Lankan Rupee (USD/LKR) exchange rate, with a specific focus on the transition between a period of extreme economic instability (the 2022 crisis) and a subsequent stabilization phase. Through a rigorous pipeline of data collection, statistical analysis, feature engineering, and multi-model forecasting, this study compares traditional statistical methods (ARIMA, GARCH) against modern machine learning (XGBoost) and deep learning (LSTM) architectures. 

The central finding is that the USD/LKR exchange rate exhibits "random walk" characteristics, where a simple Naive forecast consistently outperforms highly complex models in out-of-sample testing.

---

## 2. Project Foundation & Data Pipeline

### 2.1 Data Collection and Cleaning
**Implementation File:** [`notebooks/01_data_collection.ipynb`](notebooks/01_data_collection.ipynb)

The raw data was sourced from the Central Bank of Sri Lanka (CBSL) via the AllRatesToday/Kaggle dataset. The following pipeline was implemented to ensure data integrity:
- **Filtering**: To maintain consistency, only the `indicative` rate for the `USD` base and `LKR` quote was extracted.
- **Temporal Integrity**: Dates were converted to datetime objects and sorted chronologically. Duplicate entries were removed to prevent artificial bias.
- **Final Output**: The processed series was saved as `master_dataset.csv`, containing 3,499 observations from May 2010 to September 2026.

### 2.2 Exploratory Data Analysis (EDA) & Statistical Testing
**Implementation File:** [`notebooks/02_eda_and_statistics.ipynb`](notebooks/02_eda_and_statistics.ipynb)

#### 2.2.1 Visual Trend Analysis
The exchange rate showed relative stability for years, followed by a catastrophic spike in 2022.
- **Overall Trend**: Visualized in `figures/Historical USD-LKR Exchange Rate.png`.
- **Crisis Focus**: The 2022 regime is isolated in `figures/USD-LKR Exchange Rate During 2022.png`, showing the rapid depreciation of the LKR.
- **Regime Comparison**: A side-by-side comparison of the Crisis (2022) and Stabilization (2023-2024) periods is shown in `figures/USD-LKR Exchange Rate- Crisis vs Stabilization.png`.

#### 2.2.2 Log Returns and Volatility
To analyze the *changes* in the rate rather than the *level*, daily log returns were calculated: $r_t = \ln(P_t / P_{t-1})$.
- **Return Distribution**: Visualized in `figures/Daily USD-LKR Log Returns.png`. The returns show extreme outliers during the crisis, as further evidenced by the `figures/Distribution of USD-LKR Daily Log Returns.png` and `figures/Boxplot of USD-LKR Daily Log Returns.png`.
- **Volatility Analysis**: A 20-day rolling standard deviation of log returns was computed.
    - **General Volatility**: See `figures/20-Day Rolling Volatility of USD-LKR Log Returns.png`.
    - **Regime Volatility**: The contrast between the crisis peaks and stabilization troughs is evident in `figures/20-Day Rolling Volatility- Crisis vs Stabilization.png`.
    - **Return Density**: The different distributions of returns between regimes are captured in `figures/Distribution of USD-LKR Log Returns.png`.

#### 2.2.3 Formal Statistical Tests
- **Stationarity (ADF Test)**:
    - **USD/LKR Level**: $p$-value $\approx 0.756$. Result: **Non-Stationary**.
    - **Log Returns**: $p$-value $\approx 0.000$. Result: **Stationary**.
    - *Implication*: The price level contains a unit root; therefore, models must work on differenced data or returns.
- **Serial Dependence (ACF/PACF)**:
    - **ACF**: See `figures/ACF of USD-LKR Daily Log Returns.png`.
    - **PACF**: See `figures/PACF of USD-LKR Daily Log Returns.png`.
    - *Findings*: Minimal significant autocorrelation at higher lags, suggesting that the return series is nearly a white noise process, characteristic of efficient markets.

---

## 3. Feature Engineering
**Implementation File:** [`notebooks/03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb)

To enable the ML models to capture temporal patterns, the following features were engineered from the `usd_lkr_eda.csv` dataset:

| Feature Category | Variables | Purpose |
| :--- | :--- | :--- |
| **Price Lags** | `Lag_1`, `Lag_5`, `Lag_20` | Capture short and medium-term price levels. |
| **Return Lags** | `Return_Lag_1`, `Return_Lag_5`, `Return_Lag_20` | Capture momentum and mean-reversion signals. |
| **Rolling Stats** | `Rolling_Mean_5`, `Rolling_Mean_20`, `Rolling_Std_20` | Identify local trends and volatility regimes. |
| **Target** | `Target` ($USD\_LKR_{t+1}$) | Formulate the one-step-ahead forecasting task. |

**Data Leakage Prevention**: All features were constructed using `shift()` and rolling windows that only look backward in time. The dataset was strictly sorted chronologically.

---

## 4. Model Development & Evaluation
**Implementation File:** [`notebooks/04_model_development_and_forecasting.ipynb`](notebooks/04_model_development_and_forecasting.ipynb)

### 4.1 Experimental Setup
- **Chronological Split**: 80% Training / 20% Testing.
- **Split Visualization**: `figures/Training and Testing Periods of USD-LKR Exchange Rate.png`.
- **Evaluation Metrics**: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### 4.2 Model Implementation & Performance

#### 4.2.1 Baseline: Naive Forecast
- **Logic**: $\hat{y}_{t+1} = y_t$.
- **Result**: MAE: 0.4631, RMSE: 0.8867.
- **Visualization**: `figures/Actual vs Naive USD-LKR Forecast.png`.

#### 4.2.2 Traditional: ARIMA(1,1,1)
- **Logic**: Combines Autoregression, Differencing, and Moving Averages.
- **Result**: MAE: 23.0635, RMSE: 25.7682.
- **Visualization**: `figures/Actual vs ARIMA USD-LKR Forecast.png`.

#### 4.2.3 Volatility: GARCH(1,1)
- **Logic**: Models conditional variance of log returns.
- **Level Forecast Result**: MAE: 0.4650, RMSE: 0.8877.
- **Visualization (Level)**: `figures/Actual vs GARCH USD-LKR Forecast.png`.
- **Visualization (Volatility)**: `figures/GARCH Conditional Volatility of USD-LKR Returns.png` shows how the model tracks risk spikes.

#### 4.2.4 Machine Learning: XGBoost
- **Logic**: Gradient Boosted Trees using all engineered features.
- **Result**: MAE: 1.8577, RMSE: 2.4266.
- **Visualization**: `figures/Actual vs XGBoost USD-LKR Forecast.png`.

#### 4.2.5 Deep Learning: LSTM
- **Logic**: RNN architecture with 20-day look-back window.
- **Scaling**: Min-Max Scaling applied to inputs.
- **Training**: `figures/LSTM Training and Validation Loss.png` shows the convergence of the MSE loss.
- **Result**: MAE: 21.7188, RMSE: 22.0969.
- **Visualization**: `figures/Actual vs LSTM USD-LKR Forecast.png`.

---

## 5. Final Comparison & Synthesis

### 5.1 Performance Summary Table

| Model | MAE | RMSE | Rank |
| :--- | :--- | :--- | :--- |
| **Naive** | **0.4631** | **0.8867** | 1 |
| **GARCH** | 0.4650 | 0.8877 | 2 |
| **XGBoost** | 1.8577 | 2.4266 | 3 |
| **LSTM** | 21.7188 | 22.0969 | 4 |
| **ARIMA** | 23.0635 | 25.7682 | 5 |

**Visual Comparison**: `figures/RMSE Comparison of USD-LKR Forecasting Models.png` and `figures/MAE Comparison of USD-LKR Forecasting Models.png`.

### 5.2 Discussion of Results
1. **The Random Walk Hypothesis**: The extreme proximity of the Naive and GARCH models' performance suggests that the USD/LKR exchange rate behaves almost like a random walk. In such series, the most recent observation is the most reliable predictor for the immediate future.
2. **Overfitting in Complex Models**: LSTM and ARIMA performed poorly on the testing set. This indicates that these models captured noise in the training data (overfitting) rather than generalizable patterns, particularly given the structural break of the 2022 crisis.
3. **XGBoost's Middle Ground**: XGBoost outperformed the deep learning and ARIMA models but failed to beat the Naive baseline, suggesting that while it captured some trend information, the noise in the LKR market is too high for pure regression to solve.
4. **The Role of GARCH**: While its level forecast was essentially "Naive," GARCH provided critical insights into the *risk* (volatility) of the currency, which is vital for hedging and policy making.

---

## 6. Final Conclusion
This research demonstrates that for the USD/LKR pair, simplicity wins. No matter the complexity of the model—be it the sequential memory of LSTMs or the boosting power of XGBoost—none could consistently outperform the simple assumption that tomorrow's price will be today's price.

This highlights the inefficiency of using complex black-box models on highly volatile financial series without incorporating exogenous macroeconomic data (like IMF funding, foreign reserves, or inflation rates).

### Future Directions
- **Exogenous Variables**: Incorporate US Treasury yields, Gold prices, or Brent crude oil.
- **Hybrid Models**: Combine GARCH for volatility and LSTM for mean prediction.
- **Regime-Specific Tuning**: Train separate models for "Crisis" and "Stabilization" periods.
