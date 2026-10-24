# USD/LKR Exchange-Rate Forecasting

Forecasting the next published USD/LKR indicative exchange-rate observation with time-series, machine-learning, and volatility models, and measuring how accuracy changes across horizons and economic regimes.

*Financial Time Series Analysis (CM3023), General Sir John Kotelawala Defence University, Team 07.*

---

## Overview

The project uses **3,499 CBSL USD/LKR observations (7 May 2010 – 25 September 2026)** and compares a Naive benchmark, ARIMA, XGBoost, MLP, and LSTM for one-step forecasting. GARCH(1,1) is evaluated separately as a volatility model. The analysis is built around two difficulties of this series: persistence with irregular observation gaps, and the major regime change in **March 2022**.

**Research question:** How effectively can the next published USD/LKR observation be forecast using traditional time-series and machine-learning methods, and how does performance change across horizons and economic regimes?

The target is the *next available observation*, not a rate assigned to an invented calendar date. Lags and horizons are therefore measured in observations, not calendar days.

## Key results

| Item | Result |
|---|---|
| Best point-forecast model (test MAE) | ARIMA(1,1,1): 0.406721 |
| Naive benchmark (test MAE / RMSE) | 0.424284 / 0.947432 |
| GARCH(1,1) test volatility MSE / QLIKE | 0.411381 / −2.46633 |
| Final one-step forecast | **330.3869** (latest observed: 330.3522 on 2026-09-25) |
| Approximate 95% prediction interval | 327.8478 to 332.9260 |
| Key structural break | March 2022 (floating / shock regime) |

**Takeaways**

- ARIMA(1,1,1) has the lowest test MAE, but the Naive benchmark has the lowest RMSE, so no model wins on every metric.
- Added model complexity did not help: XGBoost, LSTM, and MLP did not beat ARIMA or Naive on MAE.
- Naive beats XGBoost-return at 1, 5, and 20 observations ahead, and during the 2022 stress test.
- Point forecasting and volatility forecasting are different tasks and are evaluated separately.

## Dataset overview

- **Source:** [Central Bank Exchange Rates (Kaggle)](https://www.kaggle.com/datasets/allratestoday/central-bank-exchange-rates), a multi-country repository of central-bank rates. The Sri Lankan series is isolated from it, so the notebooks refer to it as CBSL data.
- **Raw file:** `Data/raw/cbsl.csv` with 190,935 rows and five columns: `date`, `base`, `quote`, `type`, and `value` (the exchange rate).
- **Filter:** only rows with `base == "USD"`, `quote == "LKR"`, and `type == "indicative"` are kept.
- **Cleaned series:** `Data/processed/master_dataset.csv` with 3,499 observations (3,498 log returns) from 2010-05-07 to 2026-09-25.
- **Engineered dataset:** `Data/processed/feature_engineered_dataset.csv` keeps all 3,499 rows, including the latest row used for the final forecast.
- **Final forecast record:** `Data/processed/final_forecast.csv`.
- **Frequency:** irregular, with weekends, holidays, and longer source gaps. Returns and lags are therefore observation-based, not daily measures.
- **Data quality:**
  - no duplicate dates or missing values remain;
  - 41 gaps exceed five calendar days, and the longest is 65 days;
  - 387 zero-change observations;
  - 354 extreme returns were flagged, inspected, and not deleted.
- **Log-return distribution:** mean 0.000305, standard deviation 0.004890, skewness 8.74, excess kurtosis 201.42, minimum −0.040162, maximum 0.127185. The heavy tails mean a few large shocks can dominate squared-error metrics.
- **Economic periods:**
  - Pre-float / managed: 2010-05-07 to 2022-03-08, 2,399 observations, return SD 0.003094;
  - Floating / 2022 shock: 2022-03-09 to 2022-12-30, 198 observations, return SD 0.014723;
  - Post-2022 stabilization: 2023-01-02 to 2026-09-25, 902 observations, return SD 0.004269.
- **Engineered features:**
  - current state: `USD_LKR`, `log_return`;
  - calendar gaps: `days_since_previous_obs`, `long_previous_gap`;
  - lags: `Lag_1`, `Lag_5`, `Lag_20`, `Return_Lag_1`, `Return_Lag_5`, `Return_Lag_20`;
  - rolling statistics: `Rolling_Mean_5`, `Rolling_Mean_20`, `Rolling_Std_20`, `rolling_volatility_20`;
  - targets: `Target_1`, `Target_5`, `Target_20`, and return targets.
- **Leakage control:** no target column appears among the predictors. Rows whose next observation is more than five calendar days away are excluded from the primary one-step comparison, leaving 3,067 eligible observations.
- **Statistical properties:**
  - the level series is non-stationary (ADF p = 0.756), while log returns are stationary (ADF p ≈ 1.75 × 10⁻¹³);
  - Ljung–Box tests show joint serial dependence, and the ARCH-LM test (p ≈ 1.19 × 10⁻¹²³) shows strong conditional heteroskedasticity;
  - a Chow-style test around 9 March 2022 is highly significant (F = 11,416).

## Methodology

```text
Raw CBSL data
      ↓
Filtering and data-quality checks
      ↓
Level, return, distribution, and regime analysis
      ↓
Stationarity, dependence, and ARCH diagnostics
      ↓
Leakage-aware feature engineering
      ↓
Chronological train/validation/test split
      ↓
Model validation and selection
      ↓
Common test-set evaluation
      ↓
Volatility, multi-horizon, and 2022 stress analyses
      ↓
Full-data refit and final one-step forecast
```

**Evaluation design.** The primary one-step comparison uses 3,067 eligible observations from 2014 onward where the next observation is at most five calendar days away. They are split chronologically, never randomly:

| Partition | Observations |
|---|---|
| Training | 2,146 |
| Validation | 460 |
| Test | 461 |

Features use only information available at the forecast origin: level and return lags (1, 5, 20), rolling mean, standard deviation and volatility, and calendar-gap indicators.

**Models**

| Model | Target | Notes |
|---|---|---|
| Naive | Next level | Carries the current rate forward. |
| ARIMA | Next level | Rolling one-step forecasts; orders (0,1,0), (1,1,0), (0,1,1), (1,1,1) compared on validation MAE. |
| XGBoost-return | Next log return | 300 trees, depth 3, learning rate 0.03, subsample 0.8, column subsample 0.8. |
| MLP-return | Next log return | Additional return-based neural comparison. |
| LSTM-return | Next log return | Uses the previous 20 observed returns. |
| GARCH(1,1) | Conditional variance | Scored with volatility MSE and QLIKE, not ranked against level models. |

## Results

### One-step test-set comparison (461 observations)

| Model | MAE | RMSE | Directional accuracy |
|---|---|---|---|
| **ARIMA(1,1,1)** | **0.406721** | 0.974414 | 0.620390 |
| Naive | 0.424284 | **0.947432** | n/a (no-change forecast) |
| XGBoost-return | 0.448500 | 1.100929 | 0.611714 |
| LSTM-return | 0.511477 | 0.982958 | 0.581345 |
| MLP-return | 18.030750 | 22.014662 | 0.568330 |

ARIMA reduces MAE by about 4.1% versus Naive, and a Diebold–Mariano test finds the difference significant (DM = −2.60, p = 0.0094). Its RMSE is about 2.8% higher, so Naive is slightly better when large errors dominate.

### Volatility, horizons, and stress test

- **GARCH(1,1):** α + β = 0.995978, indicating highly persistent volatility.
- **Multi-horizon MAE (Naive vs XGBoost-return):** 5 observations ahead 1.116 vs 1.445; 20 observations ahead 2.749 vs 4.616.
- **2022 stress test (239 observations):** Naive MAE 1.112 vs XGBoost-return 4.680. Crisis periods are substantially harder to forecast.

### Final forecast

ARIMA(1,1,1), selected on validation MAE and refit to the full series, forecasts the **next observed value at 330.3869** (95% interval 327.8478 to 332.9260). The interval is approximate and may under-cover during structural breaks. Output is saved to `Data/processed/final_forecast.csv`.

## Repository structure

```text
USD-LKR-Forecasting/
├── Data/
│   ├── raw/
│   │   └── cbsl.csv
│   └── processed/
│       ├── master_dataset.csv
│       ├── feature_engineered_dataset.csv
│       ├── corrected_model_comparison.csv
│       ├── final_forecast.csv
│       └── other analysis outputs
├── figures_new/
│   └── updated report figures
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_and_statistics.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_develpoment_and_forecasting.ipynb
│   └── 05_final_prediction.ipynb
├── Project_Report.md
├── Formatted_Project_Report_Updated.docx
└── README.md
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_collection.ipynb` | Filters the raw file, checks quality and observation gaps, and saves the cleaned series. |
| `02_eda_and_statistics.ipynb` | Creates returns, evaluates distributions and stationarity, tests dependence and ARCH effects, and compares economic periods. |
| `03_feature_engineering.ipynb` | Creates lag, rolling, calendar-gap, and one-, five-, and twenty-observation target features. |
| `04_model_develpoment_and_forecasting.ipynb` | Performs chronological validation, evaluates models and volatility, and runs supplementary analyses. |
| `05_final_prediction.ipynb` | Refits the validation-selected model on all observations and writes the final forecast record. |

## Reproduction

1. Clone the repository and open it in Jupyter Notebook, JupyterLab, or VS Code.
2. Install the Python libraries imported at the top of the notebooks (the LSTM requires TensorFlow).
3. Run the notebooks in numerical order from the repository root so relative paths resolve.
4. Review outputs in `Data/processed/` and figures in `figures_new/`.

Do not replace the chronological split with random train/test splitting.

<details>
<summary><strong>Report figures (18)</strong></summary>

1. CBSL USD/LKR indicative exchange rate
2. USD/LKR exchange rate and log returns
3. Normal Q-Q plot and 20-observation rolling return volatility
4. Distribution of USD/LKR log returns
5. Log-return distribution by economic period
6. ACF and PACF of log returns
7. USD/LKR across defined economic periods
8. Log-return distribution and return standard deviation by economic period
9. Feature timing alignment versus future targets and trailing rolling metrics
10. Chronological modelling split across feature dates
11. Common-test point forecast MAE and error distributions
12. Naive baseline interval backtest and empirical coverage
13. One-step-ahead forecasts on the common test period
14. XGBoost feature importance
15. Mean absolute error (MAE) by direct forecast horizon
16. 2022 historical stress test forecasts and MAE comparison
17. Recent log returns and final ARIMA(1,1,1) forecast summary
18. Final forecast for the next observed observation

</details>

## Limitations

- Univariate analysis: no interest rates, inflation, reserves, trade flows, or other external predictors.
- The data is an indicative published rate, not an executable market quote.
- Irregular gaps mean an observation-based return can span several calendar days, and the five-day eligibility rule removes long-gap targets from the primary comparison.
- Results depend on the chosen split, features, hyperparameters, and software environment; the LSTM also depends on TensorFlow availability and training behaviour.
- A single test period cannot establish universal model superiority, and prediction intervals may under-cover during structural breaks.
- Level MAE/RMSE, directional accuracy, volatility MSE, and QLIKE answer different questions and should not be merged into one ranking.


