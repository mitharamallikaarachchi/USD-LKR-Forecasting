# Forecasting the USD/LKR Exchange Rate Under Economic Regime Changes

## Abstract

This study evaluates whether the next published USD/LKR indicative exchange-rate observation can be forecast using traditional time-series methods and machine-learning methods. The analysis uses 3,499 observations from 7 May 2010 to 25 September 2026. The workflow includes data validation, return and distribution analysis, stationarity tests, autocorrelation diagnostics, regime analysis around the March 2022 policy change, leakage-aware feature construction, chronological validation, one-step forecasting, volatility forecasting, multi-horizon analysis, and a historical 2022 stress test.

The primary one-step comparison uses 3,067 eligible observations from 2014 onward for which the next observation is no more than five calendar days away. The data is divided chronologically into training (2,146), validation (460), and test (461) observations. ARIMA(1,1,1), selected using validation MAE, achieves the lowest test MAE (0.406721). The Naive benchmark has MAE 0.424284 and RMSE 0.947432; XGBoost-return has MAE 0.448500; LSTM-return has MAE 0.511477; and MLP-return performs substantially worse. GARCH is evaluated separately as a volatility model and obtains test volatility MSE 0.411381 and QLIKE -2.46633.

The final model selected by validation is ARIMA(1,1,1). Refit to the complete available level series, it produces a forecast of **330.3869** for the next observed exchange-rate observation, with an approximate 95% prediction interval of **327.8478 to 332.9260**. The results show that model complexity does not guarantee better performance, that the March 2022 regime change is statistically and economically important, and that point forecasting and volatility forecasting should not be treated as the same task.

**Keywords:** USD/LKR, exchange-rate forecasting, ARIMA, XGBoost, LSTM, GARCH, regime change, volatility, Sri Lanka

---

## 1. Introduction

Exchange-rate movements affect import costs, export earnings, foreign debt payments, tourism receipts, remittances, and business planning. Forecasting the USD/LKR rate is difficult because the series contains persistence, structural breaks, changing volatility, policy effects, and sudden movements.

### 1.1 Research question

> How effectively can the next published USD/LKR observation be forecast using traditional time-series and machine-learning methods, and how does performance change across horizons and economic regimes?

### 1.2 Objectives

1. Obtain and validate a real USD/LKR indicative exchange-rate series.
2. Examine levels, log returns, distributional properties, stationarity, dependence, and volatility.
3. Identify the effect of the March 2022 move away from the managed exchange-rate regime.
4. Construct features using information available at the forecast origin.
5. Compare a Naive benchmark, ARIMA, XGBoost, MLP, and LSTM using chronological evaluation.
6. Evaluate GARCH as a volatility model and examine longer horizons and the 2022 stress period.

### 1.3 Forecasting target

For an observation at time \(t\), the primary target is the rate at the next available observation:

$$
Target_t = USD\_LKR_{t+1}.
$$

The target is observation-based rather than a fabricated calendar date. Lags and horizons therefore represent observations, not necessarily exact calendar days.

---

## 2. Data and preparation

### 2.1 Data source and filtering

The raw file is `data/raw/cbsl.csv`, identified by the notebooks as a Central Bank of Sri Lanka (CBSL) exchange-rate file. The raw file contains 190,935 rows and five columns: `date`, `base`, `quote`, `type`, and `value`. The project retains rows satisfying:

| Filter | Value |
|---|---|
| Base currency | USD |
| Quote currency | LKR |
| Rate type | indicative |

This produces 3,499 observations. The exact source URL and retrieval date are not encoded in the supplied CSV and should be added if required by the submission instructions; they are not guessed here.

### 2.2 Cleaning and coverage

Dates were parsed, observations were sorted chronologically, duplicate dates were removed, and invalid values were excluded. No duplicate dates or missing USD/LKR values remained. The cleaned series runs from **2010-05-07** to **2026-09-25**. There are 41 gaps longer than five calendar days; the longest is 65 calendar days. There are 387 zero-change observations and 354 observations flagged by the notebook’s robust extreme-return diagnostic. These flagged observations were inspected, not automatically deleted.

![CBSL USD/LKR indicative exchange rate](figures_new/CBSL%20USD-LKR%20Indicative%20Exchange%20Rate.png)

**Figure 1. CBSL USD/LKR indicative exchange rate.** The full history shows the long-run movement of the rate and the sharp change around 2022.

### 2.3 Observation-based returns

The log return is:

$$
r_t=\ln\left(\frac{P_t}{P_{t-1}}\right).
$$

Because the data contains weekends, holidays, and longer source gaps, a return is a change between consecutive observations and should not automatically be described as a one-calendar-day return.

---

## 3. Exploratory and statistical analysis

### 3.1 Levels and log returns

![USD/LKR exchange rate and log returns](figures_new/USD-LKR%20Exchange%20Rate%20and%20Log%20Returns%20Between%20Consecutive%20Observations.png)

**Figure 2. USD/LKR exchange rate and log returns between consecutive observations.** The level and return panels answer different questions: the level captures long-run price movement, while returns capture proportional changes between observed rates.

![USD/LKR log returns](figures_new/USD-LKR%20Log%20Returns%20Between%20Consecutive%20Observations.png)

**Figure 3. USD/LKR log returns between consecutive observations.** Most changes are close to zero, but large movements occur during unstable periods.

### 3.2 Return distribution

The return sample contains 3,498 observations. Its mean is 0.000305, standard deviation is 0.004890, skewness is 8.739760, and excess kurtosis is 201.417102. The minimum is -0.040162 and the maximum is 0.127185. The Jarque–Bera statistic is 5,957,441.2456 with p-value 0, providing strong evidence against normality.

![Distribution of USD/LKR log returns](figures_new/Distribution%20of%20USD-LKR%20Log%20Returns.png)

**Figure 4. Distribution of USD/LKR log returns.** The concentration near zero and very long tails show why large shocks can strongly affect squared-error metrics.

![Log-return distribution by economic period](figures_new/Log-Return%20Distribution%20by%20Period.png)

**Figure 5. Log-return distribution by economic period.** Return behaviour differs across the managed, floating/shock, and post-2022 stabilization periods.

### 3.3 Stationarity

Both ADF and KPSS results were considered:

| Series | ADF statistic | ADF p-value | KPSS statistic | KPSS p-value | N |
|---|---:|---:|---:|---:|---:|
| USD/LKR level | -0.992196 | 0.756100 | 7.980349 | 0.01 | 3,499 |
| Log return | -8.442357 | 1.75e-13 | 0.077462 | 0.10 | 3,498 |

The level results are consistent with non-stationarity, while the return results support stationarity. This motivates differencing for ARIMA and return-based modelling for the machine-learning and volatility analyses.

### 3.4 Autocorrelation and ARCH effects

![ACF of USD/LKR log returns](figures_new/ACF%20of%20USD-LKR%20Log%20Returns.png)

**Figure 6. ACF of USD/LKR log returns.** The ACF provides evidence about serial dependence at individual lags.

![PACF of USD/LKR log returns](figures_new/PACF%20of%20USD-LKR%20Log%20Returns.png)

**Figure 7. PACF of USD/LKR log returns.** The PACF helps distinguish additional lag relationships after shorter lags are considered.

The Ljung–Box p-values are \(2.47\times10^{-81}\) at lag 10 and \(1.91\times10^{-89}\) at lag 20, indicating joint serial dependence at the tested lags. The ARCH-LM statistic is 605.4668 with p-value \(1.19\times10^{-123}\), and the corresponding F-test p-value is \(3.93\times10^{-136}\). These results support examining conditional volatility, but they do not guarantee that a GARCH model will produce the best point forecast.

### 3.5 Economic periods and the March 2022 break

The notebooks define three analytical periods:

| Period | Observations | Dates | Mean rate | Rate standard deviation | Mean return | Return standard deviation |
|---|---:|---|---:|---:|---:|---:|
| Pre-float / managed | 2,399 | 2010-05-07 to 2022-03-08 | 154.694578 | 27.172521 | 0.000240 | 0.003094 |
| Floating / 2022 shock | 198 | 2022-03-09 to 2022-12-30 | 350.623253 | 27.416223 | 0.002960 | 0.014723 |
| Post-2022 stabilization | 902 | 2023-01-02 to 2026-09-25 | 312.750934 | 16.719445 | -0.000105 | 0.004269 |

![USD/LKR across defined economic periods](figures_new/USD-LKR%20Across%20Defined%20Economic%20Periods.png)

**Figure 8. USD/LKR across defined economic periods.** The floating/shock period is treated separately from the earlier managed phase and the later stabilization period.

A Chow-style diagnostic around 9 March 2022 gives an F statistic of 11,416.3745 and p-value 0. This is strong diagnostic evidence of a structural change, although the conventional test should be interpreted cautiously because the series may be serially dependent and heteroskedastic.

---

## 4. Feature engineering

The feature-engineering notebook retains all 3,499 rows, including the latest row needed for the final live-style forecast. It creates:

| Feature group | Examples | Purpose |
|---|---|---|
| Current state | `USD_LKR`, `log_return` | Current level and movement |
| Calendar-gap information | `days_since_previous_obs`, `long_previous_gap` | Account for irregular observation spacing |
| Level lags | `Lag_1`, `Lag_5`, `Lag_20` | Observation-based persistence |
| Return lags | `Return_Lag_1`, `Return_Lag_5`, `Return_Lag_20` | Recent movement and reversal |
| Rolling statistics | `Rolling_Mean_5`, `Rolling_Mean_20`, `Rolling_Std_20` | Local level and dispersion |
| Volatility | `rolling_volatility_20` | Recent return risk |
| Targets | `Target_1`, `Target_5`, `Target_20` and return targets | Future observations for separate horizons |

The feature audit confirms that no target column appears in the feature list. The expected missing values occur only at the start of lagged/rolling features and at the final row’s unavailable future target. Rows whose next observation is more than five calendar days away are excluded from the primary one-step comparison, leaving 3,067 eligible observations.

---

## 5. Forecasting methodology

### 5.1 Chronological evaluation

The eligible observations are split without random shuffling:

| Partition | Observations | Coverage |
|---|---:|---|
| Training | 2,146 | 2014-01-01 to target 2022-11-30 |
| Validation | 460 | target dates through 2024-10-28 |
| Test | 461 | target dates through 2026-09-25 |

Validation is used for model/order selection. The test set is reserved for the final comparison on common target rows.

### 5.2 Models

- **Naive:** carries the current rate forward.
- **ARIMA:** rolling one-step level forecasts with candidate orders (0,1,0), (1,1,0), (0,1,1), and (1,1,1). ARIMA(1,1,1) is selected by validation MAE.
- **XGBoost-return:** predicts the next log return and converts it back to a rate level. The selected configuration is 300 trees, depth 3, learning rate 0.03, subsample 0.8, and column subsample 0.8.
- **MLP-return:** a return-based neural network included as an additional machine-learning comparison.
- **LSTM-return:** predicts the next log return using the preceding 20 observed returns.
- **GARCH(1,1):** evaluated separately as a conditional-volatility model using variance MSE and QLIKE, not level MAE/RMSE.

MAE and RMSE measure level forecast error. MASE, directional accuracy, relative RMSE, and a paired Diebold–Mariano diagnostic add context.

---

## 6. Forecasting results

### 6.1 Main test-set comparison

| Model | MAE | RMSE | MASE | Directional accuracy | Relative RMSE vs Naive |
|---|---:|---:|---:|---:|---:|
| **ARIMA(1,1,1)** | **0.406721** | **0.974414** | 1.303866 | 0.620390 | 1.028479 |
| Naive | 0.424284 | 0.947432 | 1.360170 | 0.002169 | 1.000000 |
| XGBoost-return | 0.448500 | 1.100929 | 1.437801 | 0.611714 | 1.162013 |
| LSTM-return | 0.511477 | 0.982958 | 1.639693 | 0.581345 | 1.037497 |
| MLP-return | 16.035584 | 19.975001 | 51.406832 | 0.559653 | 21.083300 |

![One-step-ahead forecasts on the common test period](figures_new/One-Step-Ahead%20USD-LKR%20Forecasts%20on%20the%20Common%20Test%20Period.png)

**Figure 9. One-step-ahead forecasts on the common test period.** ARIMA has the lowest MAE, while the Naive benchmark remains a strong reference because adjacent exchange-rate observations are often close.

The Diebold–Mariano diagnostics against Naive are:

| Model | DM statistic | p-value |
|---|---:|---:|
| ARIMA | -2.597451 | 0.009392 |
| XGBoost-return | 1.240074 | 0.214948 |
| MLP-return | 28.880227 | 0.000000 |

The ARIMA result is statistically distinguishable from Naive under this paired diagnostic, while the XGBoost difference is not significant at conventional levels. These results should still be interpreted with the limitations of one sample and one test design in mind.

### 6.2 GARCH volatility results

The estimated GARCH parameters are \(\mu=0.031017\), \(\omega=0.005061\), \(\alpha=0.213364\), and \(\beta=0.782614\). On the test period, volatility MSE is **0.411381** and QLIKE is **-2.46633** (lower is better). These scores evaluate variance forecasts against realized squared returns; they must not be ranked together with level MAE or RMSE.

### 6.3 Feature importance

![XGBoost feature importance](figures_new/XGBoost%20Feature%20Importance.png)

**Figure 10. XGBoost feature importance.** `Rolling_Std_20` has the largest model-specific importance (0.170780), followed by `Lag_20` (0.112598), `Lag_5` (0.097691), and `rolling_volatility_20` (0.087701). Because many lag and rolling-level variables are highly correlated, importance is diagnostic rather than causal.

### 6.4 Multi-horizon analysis

| Horizon | Model | Test observations | MAE | RMSE |
|---:|---|---:|---:|---:|
| 1 | Naive | 461 | 0.424284 | 0.947432 |
| 1 | XGBoost-return | 461 | 0.432352 | 1.036141 |
| 5 | Naive | 457 | 1.115646 | 1.856010 |
| 5 | XGBoost-return | 457 | 1.444756 | 2.172095 |
| 20 | Naive | 442 | 2.748674 | 4.088248 |
| 20 | XGBoost-return | 442 | 4.616474 | 6.197427 |

Errors increase as the forecast horizon grows. Naive is better than XGBoost-return at all three horizons in this supplementary comparison.

### 6.5 2022 historical stress test

The primary test period is recent and does not contain the March 2022 shock. A separate static-model walk-forward stress test evaluates one-step forecasts during 2022:

| Period | Model | Observations | MAE | RMSE |
|---|---|---:|---:|---:|
| 2022 walk-forward window | Naive | 239 | 1.111523 | 3.618012 |
| 2022 walk-forward window | XGBoost-return | 239 | 4.679923 | 6.154374 |

The shock period is substantially harder to forecast, and the Naive benchmark outperforms XGBoost-return in this stress test.

---

## 7. Final forecast

Validation selects ARIMA(1,1,1), not the model chosen after inspecting the test results. Refit to the complete level series, it uses the latest available observation dated 2026-09-25, whose rate is 330.3522. The next publication date is unknown.

| Item | Value |
|---|---|
| Latest observation date | 2026-09-25 |
| Latest observed USD/LKR | 330.3522 |
| Selected model | ARIMA(1,1,1) |
| Forecast horizon | Next observed exchange-rate observation |
| Forecast | **330.3869** |
| Approximate 95% prediction interval | **327.8478 to 332.9260** |

![Final USD/LKR forecast for the next observed observation](figures_new/Final%20USD-LKR%20Forecast%20(Next%20Observed%20Observation).png)

**Figure 11. Final one-step forecast and approximate prediction interval.** The forecast is shown at a labelled placeholder after the latest observation because the next publication date is not known.

The interval is an approximate model-based interval and is not a guarantee. It may under-cover during policy changes, liquidity shocks, or other structural breaks.

---

## 8. Discussion

ARIMA is the best point-forecast model under the corrected validation and common-test design, but the Naive benchmark remains competitive and has the lowest RMSE among the listed models. This is consistent with short-run persistence: the latest exchange-rate level contains substantial information about the next observed level. XGBoost-return and LSTM-return add nonlinear or sequential capacity, but capacity does not create information about unexpected policy or market shocks. MLP-return is unstable in this configuration and has much larger errors.

The March 2022 break is central to interpretation. The floating/shock period has a much higher mean return and return standard deviation than the managed and post-2022 periods. A model calibrated in one regime may not transfer reliably to another. The 2022 stress test confirms that crisis forecasting is materially more difficult.

GARCH remains useful even though it is not a competing level model. The ARCH-LM result supports conditional-volatility modelling, and the GARCH output provides a risk estimate that a point forecast alone cannot provide. A practical treasury system could therefore combine an ARIMA or Naive point forecast with a volatility forecast and external macroeconomic information.

---

## 9. Limitations

1. The analysis uses one univariate exchange-rate series and does not include interest rates, inflation, reserves, trade flows, political events, intervention data, or international market variables.
2. The data is an indicative published rate and should not automatically be interpreted as an executable market quote.
3. Observation gaps mean that an observation-based return may span several calendar days.
4. The five-day gap rule improves comparability but removes 41 long-gap target rows from the primary comparison.
5. Results depend on the selected chronological split, feature definitions, hyperparameters, and implementation environment.
6. The LSTM result depends on optional TensorFlow availability and training behaviour.
7. A single holdout period cannot establish universal model superiority.
8. Prediction intervals are approximate and may fail during regime changes.
9. Level MAE/RMSE, directional accuracy, volatility MSE, and QLIKE answer different questions and should not be combined into one ranking.

---

## 10. Conclusion

The corrected analysis finds that ARIMA(1,1,1), selected using validation data, provides the lowest MAE on the common one-step test period: 0.406721. The Naive forecast remains a strong and transparent benchmark, while XGBoost-return and LSTM-return provide no improvement over it in this design. MLP-return performs poorly. GARCH should be interpreted separately as a volatility model, with test volatility MSE 0.411381 and QLIKE -2.46633.

The selected final model forecasts **330.3869** for the next observed USD/LKR exchange-rate observation, with an approximate 95% interval of **327.8478–332.9260**. The most important practical conclusion is that forecasting performance is regime-dependent: the March 2022 shock is a clear structural break and is much harder to forecast than ordinary periods. A production system should combine transparent point forecasting, volatility monitoring, external explanatory variables, rolling revalidation, and uncertainty reporting.

---

## 11. Reproduction guide

The analysis is represented by the following notebooks:

1. [`notebooks/01_data_collection.ipynb`](notebooks/01_data_collection.ipynb) — filters, validates, and saves the cleaned series.
2. [`notebooks/02_eda_and_statistics.ipynb`](notebooks/02_eda_and_statistics.ipynb) — creates returns, statistical tests, dependence diagnostics, and regime analysis.
3. [`notebooks/03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb) — creates leakage-aware lag, rolling, gap, and multi-horizon features.
4. [`notebooks/04_model_develpoment_and_forecasting.ipynb`](notebooks/04_model_develpoment_and_forecasting.ipynb) — performs chronological validation, common-test evaluation, volatility analysis, feature diagnostics, multi-horizon analysis, and the 2022 stress test.
5. [`notebooks/05_final_prediction.ipynb`](notebooks/05_final_prediction.ipynb) — refits the validation-selected model and writes the final one-step forecast.

The report figures are the complete updated set supplied in `figures_new.zip`.
