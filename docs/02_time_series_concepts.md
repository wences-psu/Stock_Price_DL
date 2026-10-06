# 02 · Time Series concepts for price prediction

This document explains the ideas behind `src/features.py` and `src/preprocessing.py`.

---

## 1. Time Series compared with Tabular Data.

In a typical tabular dataset (for example, house prices), rows are independent therefore shuffling them changes nothing. 

In a time series:
- **Order is important.** Today depends on yesterday.
- **The future is unknown.** A model may only use information available *at the moment of prediction*.
- **The process changes over time.** A pattern from 2012 may not hold in 2026.

Stock prediction is a times series.

---

## 2. Stationarity: why we predict returns, not prices.

A series is **stationary** when its mean and variance stay roughly constant over time.

| Series | Stationary |
|---|---|
| Close price | Not stationary because AAPL went from ~$6 (2010, split-adjusted) to ~$316 (2026); the mean keeps rising |
| Daily log return | Is stationary because it fluctuates around ~0.1% per day in every period |

**Relevance for neural networks.** A network learns a mapping from the input ranges it saw during training. In this project the training prices range from $5.74 to $149.35 while the test prices range from $163.22 to $339.79, so a price predicting network must **extrapolate**, which neural networks do poorly.

**The "tomorrow = today" baseline.** When a network predicts raw prices, the easiest way to get a low error is to output something very close to the last price in the window. The prediction chart looks good, but the model has learned nothing beyond the naive forecast.

### Log returns

$$r_t = \ln\left(\frac{C_t}{C_{t-1}}\right)$$

Why log returns rather than simple returns $(C_t - C_{t-1}) / C_{t-1}$?
- They **add up over time**: the 5-day log return is the sum of 5 daily log returns.
- They are symmetric: +10% followed by −10% in log terms brings you back to the start.
- For small moves, $r_t \approx$ the percentage change.

**Rebuilding the price** from a predicted return:

$$\hat C_{t+1} = C_t \cdot e^{\hat r_{t+1}}$$

This is done in `src/evaluate.py` (`target_to_price`) and in `src/predict.py`.

---

## 3. Patterns of stock returns

These patterns hold for almost every liquid stock. You can see all of them in notebook 01.

| Fact | Observations | Consequence for this project |
|---|---|---|
| **Near zero autocorrelation** | Yesterday's return barely predicts today's | Direction is hard to predict, so baselines are tough to beat |
| **Fat tails** | Crashes and jumps are far more frequent than a normal distribution predicts | The **Huber loss** keeps extreme days from dominating training |
| **Volatility clustering** | Big moves follow big moves | The *size* of moves is more predictable than their *direction* |
| **Volume and volatility link** | High volume goes with large moves | Volume may help predict magnitude more than direction |

### The Efficient Market Hypothesis (EMH) [1]
If past prices reliably predicted future prices, traders would exploit the pattern until it disappeared. In its weak form, the EMH says **past prices and volume are already reflected in the current price**. Real markets are not perfectly efficient, but daily prediction from price data alone is extremely hard. That is why the **random walk** ("tomorrow = today") is the baseline.

---

## 4. Data leakage

**Leakage** means information that would not be available at prediction time sneaks into training. The model looks great in testing and fails in real life.

| Type of leakage | Example | How this project prevents it |
|---|---|---|
| **Random split** | Shuffling rows, so the model trains on 2025 and tests on 2020 | Chronological split (`chronological_split_indices`) |
| **Scaling** | Fitting the scaler on all data, so the training data "knows" the test mean | Scalers fitted on the training rows only (`fit_scalers`) |
| **Feature** | A centered rolling mean (it uses future days), or using $C_{t+1}$ in a feature | Features use only `shift(+k)` and past data (`add_features`) |
| **Target misalignment** | An off by one error, so the model "predicts" today | `shift(-1)` target plus asserts in notebook 02 |
| **Test set reuse** | Tuning settings until the test score looks good | Use validation for decisions; touch test once at the end |

---

## 5. Chronological train / validation / test split

```
|----------------------- train 70% -----------------------|---- val 15% ----|---- test 15% ----|
2010                                                    2021              2024              2026
```

- **Train:** gradient descent adjusts the weights on this data.
- **Validation:** monitored during training (early stopping, learning-rate reduction) and used to compare settings.
- **Test:** an unbiased estimate of future performance. Use it **once**.

### Walk-forward validation (a more robust alternative)
A single split tests one period only (here roughly March 2024 to September 2026). **Walk forward** validation repeats the process several times:

```
Fold 1: [train ........][test]
Fold 2: [train ...............][test]
Fold 3: [train ......................][test]
Fold 4: [train .............................][test]
```

Averaging over folds shows whether results are stable across market regimes. It costs more training time.

---

## 6. Scaling

Neural networks train best when inputs are on similar, small scales (about mean 0, std 1):

$$x_{scaled} = \frac{x - \mu_{train}}{\sigma_{train}}$$

- Large, unscaled inputs (such as log volume ≈ 18) saturate activations and slow down learning.
- We use **two scalers**: one for the features and one for the target. The target scaler lets us convert predictions back to real units with `inverse_transform`.
- The scalers are saved with the model. Prediction must use exactly the transformation used in training.

---

## 7. Sliding windows and 3D tensors

Recurrent and convolutional layers expect input shaped **(samples, timesteps, features)**:

```
LOOKBACK = 4 (for illustration; the project uses 60)

day:      1    2    3    4    5    6    7
feature:  f1   f2   f3   f4   f5   f6   f7

sample for t=4: X = [f1 f2 f3 f4] → y = target of day 5
sample for t=5: X = [f2 f3 f4 f5] → y = target of day 6
sample for t=6: X = [f3 f4 f5 f6] → y = target of day 7
```

- **samples:** how many windows (about 2,900 in training)
- **timesteps:** `LOOKBACK` = 60 days
- **features:** 1 (price only) or 3 (price + volume)

**Choosing the lookback.** Too short and the model misses slower patterns; too long and there are more parameters to fit on noise, plus fewer training samples. 60 trading days (about 3 months) is a balanced starting point; it could be changed during hyperparameter tuning.

**Windows at split boundaries.** The first validation window contains the last 59 training days *as inputs*. That is correct on that date, because those days were already known. What must never happen is a *target* from the future of the training period being used to fit the weights.

---

## 8. Forecast horizon and what "next day" means

- **Horizon h = 1:** we predict the close of the next trading day, using data up to today's close.
- Weekends and holidays are skipped automatically, because rows are trading days.
- In deployment the "next date" is estimated as the next business day, which ignores market holidays. An exchange calendar library fixes this.

---

## Glossary

| Term | Meaning |
|---|---|
| **OHLCV** | Open, High, Low, Close, Volume: the standard daily bar |
| **Adjusted close** | The close corrected for splits (and dividends in some sources) so the series is continuous |
| **Stationary** | Statistical properties do not change over time |
| **Log return** | $\ln(C_t / C_{t-1})$ |
| **Volatility** | The standard deviation of returns (annualized: × √252) |
| **Leakage** | Future information used during training |
| **Lookback / window** | The number of past days fed to the model per prediction |
| **Horizon** | How far ahead we predict (here, 1 day) |
| **Random walk** | A process where the best forecast of tomorrow is today |
| **Drift** | The average tendency of a random walk to move up or down |

## References

[1] Wikipedia contributors, "Efficient-market hypothesis," Wikipedia, The Free Encyclopedia, https://en.wikipedia.org/w/index.php?title=Efficient-market_hypothesis&oldid=1378319980 (accessed October 6, 2026).

