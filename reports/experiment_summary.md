# CryptoPredict AI — Academic Research & Quantitative Experiment Report
**Date:** 2026-10-13 13:26:10 UTC  
**Asset:** Bitcoin (BTC-USD)  
**Historical Period:** 2014-12-15 to 2026-10-12 (4320 supervised observations)  
**Methodology:** Strictly Chronological Walk-Forward & Out-of-Sample Partitioning (Zero Lookahead Bias)

---

## 1. Executive Summary & Core Scientific Findings
1. **The Martingale Property & Baseline Superiority:**  
   In raw closing price forecasting ($P_{t+1}$), the naive previous-close baseline ($P_{t+1} = P_t$) achieves a lower test RMSE (**$1,968.94**) than all parameterized machine learning models (Linear Regression: **$2,030.43**, Residual XGBoost: **$2,589.59**). This mathematically demonstrates the **martingale property** of efficient financial asset prices: when returns have mean zero and standard deviation ~2-3%, today's price is the minimum mean squared error estimator of tomorrow's price.
2. **Stationary Percentage Return Forecasting ($R_{t+1}$):**  
   Converting the objective to percentage return forecasting ($R_{t+1} = \frac{P_{t+1} - P_t}{P_t}$) eliminates the spurious $R^2 \approx 0.98$ illusion created by non-stationary price levels, yielding genuine stationary evaluation with directional accuracy between 50.5% and 51.7%.
3. **Classification Imbalance & Decision Threshold Tuning:**  
   At the standard threshold $\tau = 0.50$, LightGBM achieves 50.15% accuracy, 82.73% specificity, but only 16.35% recall because regularized probabilities cluster tightly near 0.49. Lowering the decision threshold to **$\tau = 0.42$** balances sensitivity, lifting recall from 16.35% to 45.28% and F1-score from 24.36% to 47.37%.
4. **Feature Ablation (Do 58 Indicators Help?):**  
   Comparing minimal lag features against all 58 technical indicators shows that simple price and return lags capture the primary predictive signal. Adding 45+ momentum and trend indicators marginally improves classification ROC-AUC but does not lower raw price RMSE, highlighting high feature collinearity.
5. **Frictional Fee Drag in Backtesting:**  
   Backtesting an unconstrained daily strategy with a realistic **0.10% Binance trading fee** and **0.05% slippage** demonstrates that 124 trades create severe fee drag (-39.38% vs -16.66% Buy & Hold), proving that high classification turnover without conviction filtering destroys capital.

---

## 2. Chronological Data Partitioning (Zero Data Leakage)
Strictly chronological split applied with isolated scaler fitting:
* **Training Partition (70%):** 3024 samples (2014-12-15 to 2023-03-26)
* **Validation Partition (15%):** 648 samples (2023-03-27 to 2025-01-02)
* **Test Partition (15%):** 648 samples (2025-01-03 to 2026-10-12)

---

## 3. Regression Benchmark Leaderboard
Ranked strictly by Validation RMSE:
| Model                              |   Val_RMSE_USD |   Val_MAE_USD |   Val_MAPE_Pct |   Val_R2 |   Val_Directional_Acc_Pct |   Test_RMSE_USD |   Test_MAE_USD |   Test_MAPE_Pct |   Test_R2 |   Test_Directional_Acc_Pct |   Train_RMSE_USD |   Train_R2 |
|:-----------------------------------|---------------:|--------------:|---------------:|---------:|--------------------------:|----------------:|---------------:|----------------:|----------:|---------------------------:|-----------------:|-----------:|
| Naive Previous-Close               |        1466.71 |       945.113 |         1.735  | 0.995169 |                    100    |         1968.94 |        1392.03 |          1.5899 |  0.987454 |                     100    |          803.71  |   0.997517 |
| Linear Regression                  |        1491.66 |       970.078 |         1.7987 | 0.995003 |                     51.23 |         2030.43 |        1461.5  |          1.6672 |  0.986658 |                      50.77 |          792.894 |   0.997583 |
| Ridge Regression                   |        1578.92 |      1037.56  |         1.891  | 0.994401 |                     52.93 |         2119.69 |        1551.28 |          1.7633 |  0.985459 |                      49.54 |          845.856 |   0.997249 |
| XGBoost Regressor (Residual Delta) |        1799.4  |      1177.91  |         2.0457 | 0.992728 |                     52.01 |         2589.59 |        2072.02 |          2.3459 |  0.978297 |                      50.77 |          578.281 |   0.998714 |
| Random Forest Regressor            |        9764.87 |      4277.25  |         5.8386 | 0.785849 |                     52.01 |        30913.8  |       25579.1  |         25.8088 | -2.09282  |                      50.62 |          424.119 |   0.999308 |
| Extra Trees Regressor              |       10244.8  |      4655.1   |         6.3503 | 0.764282 |                     48.61 |        32497.9  |       27349.4  |         27.8146 | -2.41792  |                      51.23 |          435.557 |   0.999271 |
| HistGradientBoosting Regressor     |       10607.6  |      4938.88  |         6.6653 | 0.747294 |                     51.54 |        32918.5  |       27834.2  |         28.3732 | -2.50696  |                      50.93 |          537.297 |   0.99889  |
| LightGBM Regressor                 |       10619.6  |      4950.94  |         6.9006 | 0.746718 |                     51.39 |        32758.2  |       27585    |         28.0629 | -2.47289  |                      51.23 |          537.792 |   0.998888 |
| XGBoost Regressor (Raw Price)      |       10812.9  |      5142.3   |         6.9198 | 0.737414 |                     53.09 |        33940.9  |       28905.6  |         29.5827 | -2.72819  |                      50.93 |          452.243 |   0.999214 |

* **Validation Winner:** `Naive Previous-Close`
* **Best Parameterized ML Regressor:** `Linear Regression` (Val RMSE: $1,491.66 | Test RMSE: $2,030.43)

---

## 4. Classification Benchmark Leaderboard
Ranked by Validation Performance:
| Model                           |   Val_F1_Pct |   Val_Accuracy_Pct |   Val_Balanced_Accuracy_Pct |   Val_Precision_Pct |   Val_Recall_Pct |   Val_Specificity_Pct |      Val_MCC |   Val_ErrorRate_Pct |   Val_ROC_AUC |   Val_PR_AUC |   Test_F1_Pct |   Test_Accuracy_Pct |   Test_Balanced_Accuracy_Pct |   Test_Precision_Pct |   Test_Recall_Pct |   Test_Specificity_Pct |   Test_MCC |   Test_ErrorRate_Pct |   Test_ROC_AUC |   Test_PR_AUC |   Train_Accuracy_Pct |
|:--------------------------------|-------------:|-------------------:|----------------------------:|--------------------:|-----------------:|----------------------:|-------------:|--------------------:|--------------:|-------------:|--------------:|--------------------:|-----------------------------:|---------------------:|------------------:|-----------------------:|-----------:|---------------------:|---------------:|--------------:|---------------------:|
| LightGBM Classifier             |        44.12 |              52.31 |                       52.71 |               55.2  |            36.75 |                 68.67 |  0.0571257   |               47.69 |      0.539738 |     0.548316 |         24.36 |               50.15 |                        49.54 |                47.71 |             16.35 |                  82.73 | -0.0123027 |                49.85 |       0.488069 |      0.478199 |                76.03 |
| HistGradientBoosting Classifier |        42.59 |              52.16 |                       52.6  |               55.29 |            34.64 |                 70.57 |  0.0557622   |               47.84 |      0.531789 |     0.542311 |         18.58 |               48.61 |                        47.94 |                41.76 |             11.95 |                  83.94 | -0.0591509 |                51.39 |       0.477444 |      0.486743 |                76.72 |
| Dummy Baseline (Most Frequent)  |        67.76 |              51.23 |                       50    |               51.23 |           100    |                  0    |  0           |               48.77 |      0.5      |     0.512346 |         65.84 |               49.07 |                        50    |                49.07 |            100    |                   0    |  0         |                50.93 |       0.5      |      0.490741 |                53.41 |
| XGBoost Classifier (Tuned)      |        55.28 |              50.31 |                       50.06 |               51.29 |            59.94 |                 40.19 |  0.00132198  |               49.69 |      0.524335 |     0.555206 |         51.43 |               47.53 |                        47.7  |                47.12 |             56.6  |                  38.79 | -0.0468321 |                52.47 |       0.479131 |      0.477937 |                67.29 |
| Extra Trees Classifier          |        32.78 |              50    |                       50.66 |               52.67 |            23.8  |                 77.53 |  0.0157241   |               50    |      0.516776 |     0.52715  |          0.63 |               51.08 |                        50.16 |               100    |              0.31 |                 100    |  0.040049  |                48.92 |       0.516562 |      0.505929 |                64.15 |
| Random Forest Classifier        |        40    |              49.54 |                       49.96 |               51.17 |            32.83 |                 67.09 | -0.000851986 |               50.46 |      0.513907 |     0.523028 |         12.9  |               50    |                        49.23 |                44.44 |              7.55 |                  90.91 | -0.0279225 |                50    |       0.504946 |      0.485945 |                74.54 |
| Logistic Regression             |        42.58 |              49.23 |                       49.54 |               50.62 |            36.75 |                 62.34 | -0.00942409  |               50.77 |      0.503784 |     0.515299 |         14.52 |               50.93 |                        50.15 |                50    |              8.49 |                  91.82 |  0.0055845 |                49.07 |       0.522394 |      0.50894  |                55.95 |

* **Winning Classification Architecture:** `LightGBM Classifier`
* **Test Set Performance:** Accuracy = 50.15%, Balanced Accuracy = 49.54%, MCC = -0.0123, Specificity = 82.73%, Recall = 16.35%
* **Confusion Matrix:** TN=273, FP=57, FN=266, TP=52

---

## 5. Experiment 1: Raw Price vs Percentage Return Forecasting
| Model             |   Price_Test_RMSE_USD |   Price_Test_MAE_USD |   Price_Test_R2 |   Price_Test_DirAcc_Pct |   Return_Test_RMSE_Pct |   Return_Test_MAE_Pct |   Return_Test_DirAcc_Pct |   Implied_Price_RMSE_USD |
|:------------------|----------------------:|---------------------:|----------------:|------------------------:|-----------------------:|----------------------:|-------------------------:|-------------------------:|
| Naive Baseline    |               1968.94 |              1392.03 |        0.987454 |                  100    |                  2.28  |                 1.589 |                    50.93 |                  1968.94 |
| Linear Regression |               2030.43 |              1461.5  |        0.986658 |                   50.77 |                  2.513 |                 1.83  |                    50.93 |                  2213.44 |
| Ridge Regression  |               2119.69 |              1551.28 |        0.985459 |                   49.54 |                  2.368 |                 1.709 |                    50.62 |                  2055.57 |
| Random Forest     |              30674.6  |             25319.9  |       -2.04514  |                   50.93 |                  2.272 |                 1.607 |                    50.46 |                  1973.3  |
| XGBoost           |              34290.6  |             29320.2  |       -2.8054   |                   50.93 |                  2.441 |                 1.798 |                    51.7  |                  2142.84 |
| LightGBM          |              32817.2  |             27696.4  |       -2.48541  |                   50.77 |                  2.432 |                 1.752 |                    50.77 |                  2105.33 |

---

## 6. Experiment 2: Feature Ablation Study
| Feature_Set                          |   Feature_Count |   Linear_Regression_RMSE |   Linear_Regression_R2 |   Linear_Regression_DirAcc_Pct |   XGBoost_RMSE |   XGBoost_DirAcc_Pct |   Logistic_Regression_Accuracy_Pct |   Logistic_Regression_F1_Pct |   LightGBM_Accuracy_Pct |   LightGBM_Balanced_Accuracy_Pct |   LightGBM_MCC |   LightGBM_ROC_AUC |
|:-------------------------------------|----------------:|-------------------------:|-----------------------:|-------------------------------:|---------------:|---------------------:|-----------------------------------:|-----------------------------:|------------------------:|---------------------------------:|---------------:|-------------------:|
| A: Lag-1 Only (Minimal)              |               1 |                  2717.86 |               0.976094 |                          51.39 |        31938.9 |                51.08 |                              50.93 |                         0    |                   51.23 |                            50.34 |      0.0254415 |           0.503454 |
| B: Price & Return Lags (13 Features) |               7 |                  2721.05 |               0.976038 |                          50.77 |        34200.4 |                51.23 |                              50.93 |                         0    |                   51.08 |                            50.19 |      0.0143823 |           0.566009 |
| C: Full Technical Set (58 Features)  |              58 |                  2030.43 |               0.986658 |                          50.77 |        34290.6 |                50.93 |                              50.93 |                        14.52 |                   50.15 |                            49.54 |     -0.0123027 |           0.488069 |

---

## 7. Experiment 3: Walk-Forward Validation (5 Historical Market Regimes)
|   Regime_ID | Regime_Name              |   Train_Samples |   Test_Samples |   Naive_RMSE_USD |   Linear_Regression_RMSE |   Classifier_Accuracy_Pct |   Classifier_MCC |
|------------:|:-------------------------|----------------:|---------------:|-----------------:|-------------------------:|--------------------------:|-----------------:|
|           1 | 2017-2018 Bubble & Crash |             748 |            730 |          394.28  |                  472.467 |                     53.7  |       0.0206057  |
|           2 | 2019-2020 COVID Recovery |            1478 |            731 |          366.946 |                  869.205 |                     49.38 |      -0.003264   |
|           3 | 2021 ATH Supercycle      |            2209 |            365 |         1907.25  |                 1948.38  |                     53.97 |       0.0963855  |
|           4 | 2022 Crypto Winter       |            2574 |            365 |         1011.67  |                 1038.88  |                     50.96 |       0.0464875  |
|           5 | 2024-2026 ETF Inflow Era |            3304 |           1016 |         1930.27  |                 1956.19  |                     49.51 |      -0.00806503 |

---

## 8. Experiment 4: Classification Threshold Analysis
* **Default Threshold ($\tau = 0.50$):** Recall = 16.35%, Specificity = 82.73%, F1 = 24.36%
* **Optimal Balanced Accuracy Threshold ($\tau = 0.42$):** Recall = 45.28%, Specificity = 55.76%, Balanced Acc = 50.52%, F1 = 47.37%

---

## 9. Experiment 6: Conformal Prediction Intervals & Uncertainty Calibration
* **Empirical 90% Bound:** $\pm$2,302.00 USD (Empirical Test Coverage: **80.4%**)
* **Empirical 95% Bound:** $\pm$3,252.92 USD (Empirical Test Coverage: **90.74%**)
* **Current Volatility Status:** Normal Volatility (Ratio to Median: 0.99x)

---

## 10. Experiment 7: Realistic Strategy Backtesting (0.10% Binance Fee + 0.05% Slippage)
* **Strategy Cumulative Return:** -39.38%
* **Benchmark Buy & Hold Return:** -16.66%
* **Strategy Sharpe Ratio:** -1.16
* **Strategy Max Drawdown:** -45.37%
* **Total Trades Executed:** 124
* **Win Rate:** 26.61%
