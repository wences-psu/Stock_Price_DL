# 04 · Evaluating price forecasts

This document explains `src/evaluate.py` and `notebooks/04_evaluation.ipynb`.

---

## 1. Evaluation units

The network is trained on **scaled log returns**, but bussines deal in **dollars**. Every prediction is therefore converted back before computing metrics:

```
model output (scaled) ──inverse_transform──▶ log return ──C_t · exp(r)──▶ price in $
```

All metrics are computed on the **test set**, a period the model never saw and that played no part in choosing settings.

---

## 2. Regression metrics

Let $y_i$ be the actual next-day close, $\hat y_i$ the prediction, and $n$ the number of test days.

| Metric | Formula | Interpretation | Notes |
|---|---|---|---|
| **MAE** | $\frac{1}{n}\sum \lvert \hat y_i - y_i \rvert$ | Average miss in dollars | Easy to explain |
| **RMSE** | $\sqrt{\frac{1}{n}\sum (\hat y_i - y_i)^2}$ | Like MAE, but big misses count more | Always ≥ MAE; a large gap means occasional big errors |
| **MAPE** | $\frac{100}{n}\sum \left\lvert \frac{\hat y_i - y_i}{y_i} \right\rvert$ | Average miss in % of price | Comparable across price levels and stocks |
| **Directional accuracy** | % of days where $\text{sign}(\hat y_i - C_t) = \text{sign}(y_i - C_t)$ | Did we get up/down right? | 50% ≈ coin flip; compare with the share of up days |

**Directional accuracy?** A model can have a low MAE and still be useless for decisions if it gets the direction wrong. Conversely, a model with correct direction 55% of the time could be valuable even with a mediocre MAE.

**Caveat:** if the test period went up on 53% of days, a model that *always* predicts "up" reaches 53% directional accuracy. The "random walk + drift" baseline behaves like this, because its drift is positive.

---

## 3. Baselines

Baseline provides a reference point for the metrics.

| Baseline | Forecast | Why include it |
|---|---|---|
| **Naive (random walk)** | $\hat C_{t+1} = C_t$ | The theoretical benchmark for efficient markets; very hard to defeat |
| **Random walk + drift** | $\hat C_{t+1} = C_t \cdot e^{\bar r_{train}}$ | Adds the long-run upward trend of the stock |
| **Moving average (5 days)** | $\hat C_{t+1} = \frac{1}{5}\sum_{k=0}^{4} C_{t-k}$ | Technical heuristic; could be worse than naive because it lags |

**Rule:** a model is only worth deploying if it exceeds the best baseline by a margin that is **statistically significant** and **practically meaningful**.

---

## 4. Diebold-Mariano test [1]

With about 630 test days, small differences between models can easily come from chance. The **Diebold-Mariano (DM) test** [1] compares the predictive accuracy of two competing forecasts under the null hypothesis of equal predictive accuracy between them [2].

1. For each day, compute the loss differential $d_t = e_{A,t}^2 - e_{B,t}^2$.
2. Compute $DM = \dfrac{\bar d}{\sqrt{\hat\sigma_d^2 / n}}$, which is approximately $\mathcal{N}(0, 1)$ if the forecasts are equally good.
3. A two-sided p-value below 0.05 means the difference is statistically significant.

| DM statistic | p-value | Conclusion |
|---|---|---|
| negative | < 0.05 | Model A is significantly better than B |
| positive | < 0.05 | Model A is significantly worse than B |
| any | ≥ 0.05 | No evidence of a difference: treat them as equivalent |

The notebook uses the simple 1-step version. For multi-step forecasts, the variance must be corrected for autocorrelation (Newey-West) [3], and the Harvey-Leybourne-Newbold (1997) [2] correction improves small sample behavior.

---

## 5. Reading the evaluation plots

### Actual vs predicted
- **Full test period:** the lines almost overlap. Use statistical significant test to verify.
- **Last 60 days, with the naive baseline:** check whether the prediction simply follows the price one day late. If the model line sits on top of the naive line, the model has learned "tomorrow ≈ today".

### The "lag illusion"
```
price
 │        actual
 │   ╱╲    ╱╲
 │  ╱  ╲  ╱  ╲         prediction = same shape shifted 1 day right
 │ ╱    ╲╱    ╲  ╱╲
 └──────────────────── time
```
Any forecast close to $C_t$ produces a curve that looks like the actual price shifted by one day. Visually it looks great; statistically it is just the naive forecast.

### Residual histogram
- **Centered at 0:** unbiased (no systematic over- or under-prediction).
- **Width:** typical error size.
- **Long tails:** a few very large misses on news days (earnings, macro shocks); no price-based model can foresee them.

### Spread of predicted returns
If the standard deviation of predicted returns is much smaller than that of actual returns, the model is making near average predictions to minimize the loss When the target is mostly noise.

### Metric comparison bars
Models are blue and baselines gray. The best entry is at the top. If a gray bar is at the top, the neural networks did not beat that baseline.

---

## 6. Comparing "with volume" vs "without volume"

To isolate the effect of volume:
1. Compare the same architecture with and without volume.
2. Look at the RMSE difference in dollars.
3. Run the DM test between the two forecasts.
4. Ideally, repeat with several random seeds or walk forward folds since a single run can mislead.

Only when the difference is consistent and significant means the volume improves the prediction.

---

## 7. Evaluation criteria

- Metrics computed on the **test** set, in **dollars**
- Compared against naive, drift and moving average baselines
- Statistical test for the key comparisons
- Charts examined for the lag illusion
- Results checked against the business success criteria defined in phase 1
- Decision documented: deploy, iterate, or stop

---

## 8. Trading value

Accuracy metrics do not measure trading value. A proper backtest would:
- Turn predictions into positions (for example, long if the predicted return exceeds a threshold)
- Subtract transaction costs and slippage
- Report cumulative return, Sharpe ratio, maximum drawdown and turnover
- Compare with buy and hold

---

References:
[1] Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. Journal of Business & Economic Statistics, 13(3), 253–263. 
https://doi.org/10.1080/07350015.1995.10524599

[2] statsmodels Developers. "statsmodels.tsa.stattools.diebold_mariano_test." statsmodels 0.15.1 (+794). Accessed October 26, 2026. https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.diebold_mariano_test.html.

[3] Wikipedia contributors. Newey–West estimator [Internet]. Wikipedia, The Free Encyclopedia; 2026 Aug 24, 02:37 UTC [cited 2026 Oct 26]. Available from: https://en.wikipedia.org/w/index.php?title=Newey%E2%80%93West_estimator&oldid=1370998027.
