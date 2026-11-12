<!-- Markdown copy of docs/Project_Write_Up_DRAFT_DL_a5155cd.docx with all tracked changes accepted, converted for easy reading and diffs. The .docx is the source; edit it, not this file. -->

# INTRODUCTION

Predicting the next trading day’s closing price is difficult because equity prices react to changing market conditions, volatility, company events, and unpredictable external information. This project investigates whether convolutional and recurrent neural networks can learn useful patterns from Apple Inc. (AAPL) daily observations and improve on simple forecasting rules. Our evaluation deliberately emphasizes held-out chronological forecasts and strong naive baselines rather than relying on training accuracy alone.

In this project methodology, the team uses next-day logarithmic return as the prediction target; estimates are converted back to dollar closing prices before reporting MAE and RMSE.

## Scope and measurable objectives

There are three objectives this project is aiming to achieve as follows:

Objective 1 is to build a defensible next-trading-day AAPL forecasting pipeline using historical Close data, optionally supplemented by trading Volume.

Objective 2 is to compare CNN, LSTM, and GRU architectures on a time-ordered test set using prices reconstructed in US dollars.

Objective 3 is to determine whether any model materially exceeds low-complexity baselines, rather than equating low in-sample loss with practical forecast value.

Success is evaluated by out-of-sample MAE and RMSE and by transparent experimental documentation: feature ordering, target definition, lookback length, split boundaries, scaling fit scope, validation-based model selection, and reproducible data extraction. The formal success criteria, set in the Business and Data Understanding notebook before modeling, are that the selected model achieves a lower test RMSE than the naive persistence baseline and a directional accuracy above the share of up days in the test period. MAE and MAPE provide supporting perspectives.

# PROBLEM STATEMENT

Research question: Given the previous 60 AAPL trading days, do CNN, LSTM, or GRU models predict the next trading day’s closing price more accurately than persistence, random-walk-with-drift, or short moving-average baselines? Also, we consider whether including Volume alongside Close improves performance and whether model selection based on validation results generalizes to unseen test dates.

This is a supervised time-series Regression problem. The desired final result is a closing price in US dollars; a logarithmic return is used as an intermediate training target to help represent relative price movements. The work is a methodological experiment, not a trading system or investment recommendation.

## Challenges

AAPL is non-stationary: price levels shift over years, and modest next-day movements are difficult to distinguish from noise. Time-series leakage is a particular concern: feature scaling, hyperparameter selection, and evaluation must not use knowledge from future test observations. Adding Volume may introduce noise or different scales. Neural networks can overfit small effects, so even apparently good validation loss may not beat a persistence forecast in the market-price metric.

## Specific research gap and positioning

Prior financial deep-learning work includes direction classification, index-constituent prediction, multivariate forecasting, hybrid architectures, and systematic reviews. These studies use different assets, horizons, targets, and evaluation protocols; their reported scores cannot be directly ranked against dollar-based next-day AAPL errors. This project is deliberately narrower: one large-cap asset, a fixed forecasting horizon, two transparent feature configurations, chronological partitions, and simple baselines on aligned test dates.

The team contribution is not a new neural architecture. Instead, it is an applied comparison with explicit controls for target definition and a reproducible chain from a large Kaggle market panel to a small AAPL-specific modeling file. Negative findings are scientifically useful: if a well-defined persistence rule matches or beats a neural predictor, complexity is not justified merely by model sophistication.

## Related Works

Fischer and Krauss (2018) used LSTMs for financial-market prediction and compared them with alternative learners; our single-asset next-day dollar-error setting is narrower and explicitly benchmarks against simple price-persistence rules.

Bao, Yue, and Rao (2017) combined wavelets, stacked autoencoders, and LSTMs; our experiment instead compares accessible CNN/LSTM/GRU architectures using direct AAPL market features. Sezer, Gudelek, and Ozbayoglu (2020) survey deep-learning approaches for financial forecasting, supporting the value of comparing architectures and evaluation designs.

Lim and Zohren (2021) describe modern time-series forecasting architecture choices and limitations.

Bai, Kolter, and Koltun (2018) show that convolutions are credible sequence-model alternatives, motivating a 1-D CNN alongside recurrent models.

These references motivate the design; none establish that a network must beat a naive baseline on this dataset.

Wei, Xu, and He (2022) investigated next-day closing-price prediction using an attention-enhanced LSTM with historical adjusted Close, Open, and trading Volume.

Basma Hameid’s scientific review highlights two lessons for Team 7: historical data preparation matters, and Volume should be tested rather than assumed beneficial. Their study uses different market data and a different architecture; Team 7 instead compares Close-only and Close + Volume for one stock under matched test conditions. In our results, Volume lowers the RMSE of all three architectures, significantly for GRU and LSTM but not for CNN.

Lawi, Mesra, and Amir (2022) compared eight forecasting architectures built around LSTM and GRU variants on grouped stock time-series data. Basma’s review emphasizes that architecture choice affects forecasting quality and that a fair comparison is more useful than assuming one recurrent network is best. Team 7 follows this comparison principle using CNN, GRU, and LSTM with the same 60-trading-day input window, log-return target, chronological partitions, and 632 test forecasts. The eight architectures of Lawi et al. were not implemented in this project.

Li (2026), reviewed by Wenceslao Granados Gomez, proposed a hybrid CNN–LSTM–Transformer pipeline that combines local patterns, temporal dependencies, and global relationships. The review describes richer price, technical, economic, and sentiment inputs, as well as a loss intended to account for direction and asymmetric forecasting risk. Wenceslao identified limitations including the absence of some simple persistence baselines, component-level ablation, and confidence intervals. Team 7 uses separate CNN, LSTM, and GRU regressors rather than a hybrid model, and directly compares their dollar errors with naive persistence, random-walk-with-drift, and five-day moving-average baselines.

Wen, Li, Zhang, and Chen (2019), also reviewed by Wenceslao, examined market-direction classification using higher-order repeated time-series patterns and convolutional networks. Their approach draws on BeatLex pattern summarization (Hooi et al., 2017) and was assessed on stock data that included Apple. Its up/down classification target differs from Team 7’s continuous next-day log-return regression, which is converted into USD prices before MAE and RMSE are calculated. BeatLex and hybrid feature-fusion architectures are related research approaches, not components of Team 7’s tested models.

Across the four scientific readings, the main lessons are to evaluate feature additions empirically, compare architectures under a consistent protocol, and benchmark complex methods against simpler alternatives. The literature also suggests possible research extensions involving attention, richer market features, recurrent-pattern representations, and risk-aware loss functions. Because the reviewed studies use different datasets, forecast targets, and metrics, their published performance figures are not directly comparable with Team 7’s AAPL test RMSE.

## Importance and Impacts

The principal educational and scientific contribution is a reproducible, time-ordered comparison that illustrates why complex models must be tested against economical baselines. Results can help practitioners evaluate whether implementation and maintenance cost are justified by actual forecasting gains. The reported findings do not justify live deployment or financial decisions.

## Data lineage, scope and feature rationale

The original CSV is a multi-asset daily market panel with a two-level header. The verified extraction script reads that header explicitly, selects the AAPL Close and Volume columns, and exports a separate verification copy. The canonical project file and recreated sample each contain 4,210 trading-day observations. This is a content-validation result after loading in pandas, not a claim that the exported CSVs are byte-for-byte identical.

The Date index identifies market sessions and supports chronological splitting. Close is the transaction-price summary that defines the forecast target; Volume is a market-activity proxy that may encode information about participation or liquidity but is not assumed to improve forecasting. Both are continuous or count-valued inputs. A stock split, trading holiday, or extraordinary event can affect observed series; such issues require inspecting the dataset rather than assuming constant calendar-day spacing.

The public Kaggle source improves transparency but does not itself guarantee data completeness or the absence of vendor revisions. The recorded extraction procedure and source reference support reproducibility of the AAPL subset.

# DATA COLLECTION

The dataset was obtained from the Kaggle “S&P500 All Assets – Daily Updated Dataset” (publisher: yash16jr; source file SnP_daily_update.csv). The project extracts the two-tier columns (Close, AAPL) and (Volume, AAPL), retains historical trading dates, and stores the canonical AAPL sample in data/AAPL_only.csv. Dates run from January 4, 2010 through September 29, 2026 and include 4,210 AAPL daily records.

As documented and verified in November 2026, src/extract_aapl.py recreates data/AAPL_only_verified.csv separately from the original Kaggle file, without overwriting the canonical dataset. Comparing loaded pandas values established matching 4,210-row AAPL samples. The source file is read from ~/Downloads/SnP_daily_update.csv on the original Mac; another researcher must download the Kaggle source to reproduce extraction.

| Variable | Type | Interpretation |
|---|---|---|
| Date | Date / index | Trading session, chronological ordering |
| Close | Continuous numeric | AAPL daily closing price, USD |
| Volume | Numeric count | Shares traded that session |
| Next_close | Derived numeric | Following trading session’s closing price |
| Target log return | Derived continuous | ln(Next_close / Close) |

## Exploratory Data Analysis — Evidence and Interpretation

The daily dataset is continuous-valued, not a classification dataset; class imbalance is therefore not a directly applicable diagnostic. Relevant exploratory checks include missing Close/Volume values, duplicate dates, invalid nonpositive closing prices, volume anomalies, return outliers, descriptive statistics (mean, standard deviation, min/max), trend and histogram plots, and Pearson correlation of Close and Volume. A strong level-price trend alone does not prove predictability of next-day returns. Pearson correlation is descriptive association rather than evidence of causality.

The executed Business and Data Understanding notebook supplies the numerical evidence below. It loads all 4,210 AAPL trading days, from January 4, 2010, through September 29, 2026, using the file’s two-row header. An earlier version of the notebook skipped the third CSV row as if it were empty, which dropped the first trading day and produced 4,209 rows; the corrected notebook and the extracted source contain the same observations.

## Data Quality Checks

The executed notebook’s data-quality output reports 4,210 rows, zero missing values, zero duplicate dates, and zero zero-volume sessions. The data frame contains numeric Close (float64) and Volume (int64) columns. The project data loader (src/data_loader.py) also rejects missing values, nonpositive prices, negative volume, duplicate dates, and unsorted dates before any modeling step. Ordinary gaps between sessions are consistent with weekends and market holidays. The notebook identifies an unusual October 2012 closure associated with Hurricane Sandy.

The executed notebook includes a full-history plot, descriptive statistics, daily-log-return distribution, lagged autocorrelations, and correlations linking volume with next-day return and absolute return. These are reported below with their interpretation. The notebook does not compute a Pearson correlation between contemporaneous Close and Volume, so no coefficient for that pair is reported here.

## Executed notebook: reproducible exploratory findings

Data coverage and integrity. The extracted AAPL CSV and the exploratory working data both contain 4,210 daily records with two numerical columns, Close and Volume (2010-01-04 to 2026-09-29). There are 0 missing cells, 0 repeated session dates, and 0 zero-volume trading days.

Descriptive statistics. The exploratory working frame reports an average Close of about $86.66, a standard deviation near $85.33, minimum around $5.74, median approximately $40.82, and maximum around $341.07. Mean daily trading Volume is approximately 211.5 million shares; the observed range covers roughly 17.9 million to 1.881 billion shares. These summaries characterize the historical series, not a prediction or test-set performance estimate.

Growth and return characteristics. In the notebook’s summary, first Close is $6.401 and last Close is approximately $329.400 (growth ratio ~51.461×). Mean daily log return is 0.094%, with daily standard deviation 1.770% and annualized volatility 28.102%. Approximately 53.053% of daily log returns are positive. The observed worst and best single-session percentage changes are −12.865% and +15.329%, respectively.

Outliers and normality. The histogram of daily log returns is sharply centered but visibly more heavy-tailed than a normal curve fitted to the same mean and standard deviation. Skewness is about −0.170 and excess kurtosis about 6.036, showing non-normal extreme-return risk. In response to these tails, all trained networks use the robust Huber loss (delta = 1.0, src/models.py), which behaves like squared error for small errors and like absolute error for large ones.

Temporal dependence. The lag-one autocorrelation of signed daily log returns is about −0.029, versus a notebook reference threshold of approximately ±0.030; signed-return autocorrelations through ten sessions remain small, although lags 3, 7, 8, and 9 slightly exceed the band (the largest in magnitude is −0.073 at lag 8). By comparison, lag-one autocorrelation of absolute returns is approximately +0.193, and absolute-return correlations across lags 1–10 stay positive (about 0.109–0.193). This pattern is consistent with volatility clustering and limited simple linear predictability of return direction.

Volume relationship. The notebook’s volume correlation output and commentary show little linear association between volume information and the direction of next-day returns, while a modest positive relationship appears for the magnitude of next-day returns. For log volume, the displayed next-day absolute-return correlation is approximately +0.119. Thus Volume is motivated as an alternative feature to test, not as proven directional information.

Methodological implication. Rising and non-stationary price levels make the problem harder for models trained on raw prices, especially when later test prices exceed the earlier training range. The notebook therefore motivates a next-session log-return target, time-ordered splits to avoid leakage, separate Close-only and Close-plus-Volume experiments, and strong persistence/drift benchmarks. Predicted returns are mapped back to dollar closing prices before final scoring. The log-return target keeps the output in a stable range, but the model inputs remain the scaled Close (and Volume) levels, so test-period inputs still lie far outside the training range (the test mean of scaled Close is about 6 training standard deviations above the training mean).

# DATA PREPROCESSING

Records are ordered by trading date and used to create a next_close value shifted one trading day ahead. For all six models, the target is calculated as:

r(t+1) = ln(P(t+1) / P(t)), where P(t) is the current AAPL close and P(t+1) the next trading-day close.

In the implementation, add_target() in src/features.py sets target = np.log(next_close / Close) according to TARGET_MODE in config.py, and build_modeling_frame() removes the first row (no previous return) and the last row (no next-day target). For model output r_hat(t+1), the corresponding prediction is P_hat(t+1) = P(t) × exp(r_hat(t+1)). Model errors in USD are evaluated only after this inverse conversion.

Input windows represent the preceding 60 trading days for every model. The stated chronological split is 70% training, 15% validation, and 15% held-out test; after windowing, every configuration has 2,885 training, 630 validation, and 632 test samples. Scaling is based on the training partition and applied to validation and test observations; the held-out test is reserved for evaluation.

Data cleaning and inspection precede model training. Nonpositive closes would invalidate the logarithm and are rejected by the data loader before target creation; extreme observations require investigation rather than automatic removal. No rows were removed as outliers. Standard stock-price modeling does not assume class balancing, and data augmentation was not established as an implemented step.

## Sequence construction and time alignment

For session t, the target is r(t+1) = ln(P(t+1)/P(t)), defined only when both prices are positive and the following session exists. An input example contains an ordered lookback of earlier observations available by the decision time. The model emits a scalar log-return estimate r̂(t+1), and the display/evaluation forecast is P̂(t+1) = P(t) × exp(r̂(t+1)). This inverse formula is essential: comparing raw returns with dollar closing prices would produce meaningless errors.

Splitting by date before fitting preprocessing statistics prevents leakage from the held-out period. The 70/15/15 chronological split is evaluated on the same held-out forecast dates across models and baselines. Input windows may carry earlier context across a split boundary if that context would have been available at forecast time, but targets from later partitions must not influence model fit or tuning. The last training and validation samples, whose targets fall on the first day of the next partition, are therefore dropped, and the target scaler is fitted without them; notebook 02 checks this with assertions.

The 60-session lookback (about three calendar months) is used by every run, so all configurations predict exactly the same dates.

# METHODOLOGY

The work follows a CRISP-DM-inspired workflow: business/data understanding, preprocessing, modeling, evaluation, deployment demonstration, and an improvement iteration. We compare one-dimensional convolutional networks (CNN) for local temporal patterns with recurrent LSTM and GRU networks for gated sequence representation. Close-only and Close+Volume variants are considered. All six configurations (three architectures × two feature sets) use the same log-return target, inputs, and protocol; the selected model is then tuned in a separate improvement iteration.

All six runs share one training configuration: 64 units or filters (kernel size 5 for the CNN), dropout 0.2, Huber loss, Adam optimizer (learning rate 0.001), batch size 32, up to 100 epochs, early stopping on validation loss (patience 10, best weights restored), learning-rate halving when validation loss plateaus, and random seed 42. The inputs are the scaled Close (and Volume) levels over the 60-day window. Log-return prediction is a regression output, not a sigmoid binary-classification output; directional accuracy is an additional diagnostic, not a replacement for dollar error metrics.

Baselines: (1) naive persistence — forecast next close equals the latest observed close; (2) random walk with drift — extrapolate a simple historical price change; and (3) five-day moving average — use the recent mean price. The same dates, available history, and aligned actual next-day targets should be used for fair comparisons. Baseline test performance establishes how much improvement a deep-learning system must earn.

The final six-model experiment uses next-trading-day log return as the target and converts the predicted return back to a dollar closing price before evaluation. Saved prediction files supplied the MAE and RMSE for all six 60-day configurations, and the matching model metadata confirmed target_mode = log_return, the 60-day lookback, the chronological 70/15/15 split, and a common 632-observation held-out period (March 22, 2024–September 29, 2026). The same evaluation protocol was applied to all six configurations.

Model selection and improvement iteration. The network with the lowest validation loss is selected for evaluation and deployment; the test set plays no part in this choice. The selected configuration is then tuned with a KerasTuner random search (O’Malley et al., 2019) over 20 combinations of units, dropout, learning rate, and batch size, scored on validation loss only, and the best combination is retrained with the same training code. The original and tuned versions are compared by validation RMSE in dollars, a rule fixed before the results were seen.

Statistical comparison. Differences in squared forecast error are tested with the Diebold–Mariano test (Diebold & Mariano, 1995) for one-step-ahead forecasts, comparing each network with the naive and drift baselines and each Close + Volume model with its Close-only counterpart.

## Metric definitions and interpretation

On n holdout dates, MAE = (1/n) Σ|P̂−P| and RMSE = sqrt[(1/n) Σ(P̂−P)²]. MAE summarizes a typical absolute dollar miss, while RMSE gives extra penalty to large errors and can reveal episodic model failure. MAPE = 100 × mean(|P̂−P|/|P|) is interpretable as relative error when actual prices are positive. Directional accuracy compares the sign of predicted versus actual price changes from the last observed close.

A fair empirical statement must distinguish (a) validation-based training and selection, (b) one-time test evaluation, and (c) illustrative single-day forecasts. The difference between CNN Close + Volume (RMSE ≈ $4.1735) and the drift baseline (≈ $4.1520) is about two cents; the Diebold–Mariano test finds it not significant (p = 0.38), and it does not establish performance across all market regimes. No repeated-seed robustness test or confidence interval for the RMSE values was run.

# RESULTS

Final verified six-model log-return comparison. The figures below are test-set errors in USD after predicted log returns were converted back to closing prices. Each network uses a 60-trading-day lookback. The 632 test predictions span March 22, 2024, through September 29, 2026. Model configurations were checked against the saved metadata files.

| Model / feature set | MAE ($) | RMSE ($) | MAPE (%) | Dir. acc. (%) |
|---|---|---|---|---|
| CNN — Close only | 2.8748 | 4.1786 | 1.20 | 47.63 |
| CNN — Close + Volume | 2.8663 | 4.1735 | 1.20 | 49.84 |
| GRU — Close only | 3.0590 | 4.3349 | 1.26 | 44.15 |
| GRU — Close + Volume | 2.9259 | 4.2193 | 1.22 | 45.73 |
| LSTM — Close only | 3.0538 | 4.3186 | 1.27 | 44.62 |
| LSTM — Close + Volume (selected by validation loss) | 2.9574 | 4.2378 | 1.23 | 44.78 |
| LSTM — Close + Volume, tuned (improvement) | 3.8322 | 5.0868 | 1.55 | 44.30 |
| Random walk + drift baseline | 2.8363 | 4.1520 | 1.19 | 55.22 |
| Naive persistence baseline | 2.8533 | 4.1577 | 1.19 | n/a |
| 5-day moving average baseline | 4.7336 | 6.4502 | 1.98 | 48.89 |

Table 1. Test-set errors of the six log-return configurations, the tuned version of the selected model, and three baselines (MAE, RMSE, and MAPE: lower is better; directional accuracy: higher is better).

The CNN with Close + Volume produced the lowest test RMSE among the six networks ($4.1735), slightly ahead of CNN Close only ($4.1786). No network beat the random walk with drift ($4.1520, MAE $2.8363) or naive persistence ($4.1577); all six beat the five-day moving average ($6.45). Diebold–Mariano tests show that the two CNN models are not significantly different from the naive and drift baselines (p = 0.19 and 0.30 against naive), while the GRU and LSTM models are significantly worse than naive persistence (p < 0.05). Adding Volume lowered RMSE for every architecture: CNN from $4.1786 to $4.1735, GRU from $4.3349 to $4.2193, and LSTM from $4.3186 to $4.2378; the improvement is significant for GRU and LSTM (p < 0.001) but not for CNN (p = 0.12). No network reached 50% directional accuracy (44.15% to 49.84%), compared with a 55.22% share of up days in the test period.

Model selection and error analysis. LSTM Close + Volume had the lowest validation loss (0.3841) and was selected, but it ranked fourth of six on the test set (RMSE $4.2378); the six validation losses differ only in the third decimal (0.3841 to 0.3857), so validation loss carried little information about the test ranking. The selected model’s predicted daily returns vary far less than the actual returns (standard deviation 0.12% versus 1.76%) and are biased downward: it predicts a rise on only 5.9% of test days, while the price rose on 55.2% of them. All six networks share this bias (mean predicted daily return between −0.06% and −0.35%), which is consistent with the scaled price inputs lying far above the training range.

Improvement iteration. Hyperparameter tuning (20 KerasTuner trials) selected 64 units, no dropout, a learning rate of about 0.0028, and batch size 64; all trials reached similar validation losses (0.3837 to 0.3870). The tuned version improved validation RMSE by less than a tenth of a cent ($2.7048 to $2.7040), so it is preferred under the rule fixed in advance, but the gain did not generalize: its test RMSE rose to $5.0868 (MAE $3.8322). The tuned model predicts a fall on 95% of test days, with a mean predicted daily return of −0.88% versus −0.23% for the original, so it amplified the input-level bias rather than reducing it. Neither version beats the baselines, and the deployment decision is unchanged.

Evaluation metrics: MAE gives average absolute dollar error; RMSE gives added weight to larger errors; MAPE rescales absolute error by actual dollar price; and directional accuracy measures whether predicted direction agrees with the realized next-day movement. None establishes profitability. As this is regression, a confusion matrix, ROC, or classification precision/recall are not primary evaluation measures.

## Practical interpretation and bias/variance considerations

The results caution against equating neural-network sophistication with useful predictive accuracy. Next-day persistence is a strong comparator because daily equity prices frequently differ only modestly from the preceding close. A network can attain low training loss yet underperform persistence in out-of-sample dollars if it smooths abrupt movements, is biased toward historical price levels, or overfits noise. The networks’ systematic downward bias on test prices above the training range is consistent with the level-bias mechanism, although the experiment does not establish it as the only cause.

Bias/variance analysis can compare train-versus-validation residuals, stability across random seeds, and errors in earlier versus later test periods. A large gap between training and validation error would suggest overfitting; uniformly poor errors may suggest target/preprocessing mismatch or insufficient signal. The appropriate improvement strategy is to investigate diagnostics and tune strictly on validation data, preserving the held-out test until model choice is complete. The improvement iteration followed this rule: tuning and the choice between versions used validation data only.

Economic relevance must also distinguish statistical forecast error from executable trading performance. The project does not model transaction fees, bid/ask spread, position-sizing rules, slippage, or risk limits; therefore the numerical predictions are not a validated investment strategy. No real-time deployment or external reproducibility audit is claimed.

# DISCUSSION OF RESULTS

The final six-model experiment illustrates both the importance of choosing a suitable target and the strength of simple financial forecasting baselines. Among neural networks, CNN Close + Volume gave the best held-out dollar RMSE, but its advantage over CNN Close-only was very small. Random walk + drift outperformed every neural model on the reported test dates, including the tuned version of the selected model. Thus the project demonstrates a reproducible comparison, not a proven material forecasting advantage over the baseline.

Trading volume's contribution has the same direction for every architecture but a different size: it lowers RMSE for all three architectures, significantly for GRU and LSTM (Diebold–Mariano p < 0.001) but not for CNN (p = 0.12). These conclusions apply only to the matched log-return experiments on the observed held-out period. The chronological holdout reduces leakage risk but cannot remove market-regime shift, sensitivity to initialization, or the possibility of data-specific findings.

The scientific readings help interpret these findings. Wei et al. (2022) motivate testing trading Volume; here Volume improved every architecture, but not enough for any network to beat the simple baselines. Lawi et al. (2022) support evaluating LSTM and GRU side by side; our CNN, GRU, and LSTM comparisons show that model ranking depends on architecture and feature set. Li (2026) and Wen et al. (2019) describe richer hybrid and pattern-based approaches, but their reported results do not demonstrate an improvement for this AAPL experiment. The random-walk-with-drift result underscores the importance of baseline comparison, a concern also identified in Wenceslao’s reading.

## Reproducibility and Environment

The repository wences-psu/Stock_Price_DL contains the extraction script, the documented dataset verification, all notebooks with their executed outputs, and the saved models, scalers, metrics, tuning trials, and plots. All models were trained and evaluated in the shared environment defined in requirements.txt (Python 3.12.16, TensorFlow 2.16.2, Keras 3.15.1, KerasTuner 1.4.8, scikit-learn 1.6.1) on CPU under Linux (WSL); requirements-linux-gpu.txt adds CUDA libraries for Linux or WSL GPUs. Every run uses random seed 42, and two complete runs of notebooks 01 to 05 produced identical metrics. The earlier macOS files (stock_eval312_history.yml, stock_eval312_pip.txt, stock_eval312_packages.txt) record the original Apple Silicon setup, but they do not include KerasTuner and are not used to reproduce the final run.

## Limitations and Deployment Decision

All six saved models use the same log-return target, 60-day lookback, split ratios, and held-out date interval; the saved evaluations do not show a forecasting advantage over random walk + drift or naive persistence for any model, including the tuned version. The best neural network by held-out RMSE is CNN Close + Volume, while validation-loss selection chose LSTM Close + Volume; the two selection criteria are kept distinct, and the selection was not changed after seeing test results. Market-regime changes and unforeseen events limit generalization, and this educational prototype is not a validated live trading strategy.

Deployment demonstration. The deployment notebook reloads the selected model with its saved scalers. Using data through September 29, 2026 (last close $329.40), it predicted $328.00 for September 30, 2026, a 0.42% decrease. In a 20-day historical replay its MAE was $3.97, compared with $3.74 for the naive forecast on the same days, and it predicted a fall on every day of the replay. The tuned version’s replay MAE was $7.09, and it predicted $323.31 (−1.85%) for September 30, 2026. Because no model outperformed the simple baselines, production deployment is not recommended; this was a historical demonstration, not a current stock-price forecast.

# INDIVIDUAL FEEDBACK — LEARNED LESSONS AND COLLABORATION

Basma Hameid — individual reflection: This project reinforced that a deep-learning experiment is only as reliable as its data lineage, time-aware preprocessing, and evaluation design. Rebuilding the AAPL extract from the original Kaggle file and checking the full 4,210-row sample made reproducibility a concrete task rather than a documentation slogan. Also, I learned to treat the prediction target as part of the model definition: log-return training and direct-dollar training cannot be compared casually without aligned data, inverse conversion, and clear experiment records. The most demanding aspects of this assignment were reconciling multiple notebook revisions, keeping validation selection separate from test reporting, and communicating which results belong to which experiment. The team benefited from shared source control and a division of technical and documentation work, but integration requires disciplined naming, saved outputs, and explicit review of assumptions. A practical improvement would be a shared result manifest for every run recording target mode, lookback, features, seed, split dates, artifact name, and MAE/RMSE. For future teamwork, I would prioritize earlier checkpoint reviews and a consistent experiment export. The project also demonstrated a valuable scientific lesson: it is acceptable—and important—to report that a simple baseline outperforms neural networks. This is a more useful conclusion than overstating performance.

Wenceslao Gomez - TBD

# REFERENCES

Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271. https://doi.org/10.48550/arXiv.1803.01271

Bao, W., Yue, J., & Rao, Y. (2017). A deep learning framework for financial time series using stacked autoencoders and long-short term memory. PLOS ONE, 12(7), e0180944. https://doi.org/10.1371/journal.pone.0180944

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599

Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654–669. https://doi.org/10.1016/j.ejor.2017.11.054

Hooi, B., Liu, S., Smailagic, A., & Faloutsos, C. (2017). BeatLex: Summarizing and forecasting time series with patterns. In Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2017), Lecture Notes in Computer Science, 10535. https://doi.org/10.1007/978-3-319-71246-8_1

Huber, P. J. (1964). Robust estimation of a location parameter. The Annals of Mathematical Statistics, 35(1), 73–101. https://doi.org/10.1214/aoms/1177703732

Lawi, A., Mesra, H., & Amir, S. (2022). Implementation of long short-term memory and gated recurrent units on grouped time-series data to predict stock prices accurately. Journal of Big Data, 9, Article 89. https://doi.org/10.1186/s40537-022-00597-0

Li, X. (2026). Application of deep neural networks in stock price prediction. Proceedings of the 2026 3rd International Conference on Digital Society and Artificial Intelligence, 439–443. https://doi.org/10.1145/3833541.3833614

Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: A survey. Philosophical Transactions of the Royal Society A, 379, 20200209. https://doi.org/10.1098/rsta.2020.0209

O’Malley, T., Bursztein, E., Long, J., Chollet, F., Jin, H., Invernizzi, L., et al. (2019). KerasTuner. https://github.com/keras-team/keras-tuner

Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020). Financial time series forecasting with deep learning: A systematic literature review: 2005–2019. Applied Soft Computing, 90, 106181. https://doi.org/10.1016/j.asoc.2020.106181

Team 7. Stock_Price_DL GitHub project repository. https://github.com/wences-psu/Stock_Price_DL

Wei, J., Xu, Q., & He, C. (2022). Deep learning of predicting closing price through historical adjustment closing price. Procedia Computer Science, 202, 379–384. https://doi.org/10.1016/j.procs.2022.04.052

Wen, M., Li, P., Zhang, L., & Chen, Y. (2019). Stock market trend prediction using high-order information of time series. IEEE Access, 7, 28299–28308. https://doi.org/10.1109/ACCESS.2019.2901842

yash16jr. (2026). S&P500 All Assets – Daily Updated Dataset [Data set]. Kaggle. https://www.kaggle.com/datasets/yash16jr/s-and-p500-daily-update-dataset/data

# APPENDIX A — CODE FUNCTIONALITY AND REPRODUCIBILITY

## Functional overview of project source code

Data extraction (src/extract_aapl.py): reads the source Kaggle multi-asset file with two header rows; selects AAPL Close and Volume; writes data/AAPL_only_verified.csv; reports the number of extracted sessions. The script intentionally leaves data/AAPL_only.csv unchanged. The original extraction used ~/Downloads/SnP_daily_update.csv as the source path.

Preparation (src/preprocessing.py and data-preparation notebook): reads date-indexed stock observations, constructs a shifted next-session Close, derives the next-session log-return target (src/features.py), handles train/validation/test partitions, scales using training data, drops the samples whose targets would cross a partition boundary, and constructs the 60-day lookback windows.

Modeling and tuning (src/models.py, src/train.py, src/tuning.py, and the modeling and improvements notebooks): assemble Close-only or Close-plus-Volume tensors; build and compile the CNN, LSTM, and GRU networks with Huber loss; train them with early stopping, learning-rate reduction, and checkpointing; save each run’s model, scalers, metadata, and history to artifacts/; and run the KerasTuner search for the selected model.

Evaluation and deployment demonstration (src/evaluate.py, src/predict.py and evaluation/deployment notebooks as applicable): reconstruct next-day closing prices from predicted log returns, align forecasts with actual test dates, calculate regression errors, baseline comparators, and Diebold–Mariano tests, and demonstrate a dated historical prediction and a 20-day historical replay.

Environment documentation: requirements.txt defines the shared environment used for the official run (Python 3.12.16, TensorFlow 2.16.2, Keras 3.15.1, KerasTuner 1.4.8); requirements-linux-gpu.txt adds CUDA libraries for Linux or WSL GPUs. The stock_eval312_history.yml, stock_eval312_pip.txt, and stock_eval312_packages.txt files record the earlier macOS (arm64) environment; they omit KerasTuner, and the pip freeze contains locally resolved file references, so they are kept for the record only. All settings (dates, lookback, target, epochs, seed, RETRAIN) are centralized in config.py.

Reproduction procedure: obtain the Kaggle source CSV, clone the project repository, create a Python 3.12.16 environment and install requirements.txt, run src/extract_aapl.py, compare the AAPL extract with the canonical sample, and execute notebooks 01 to 06 in sequence. With RETRAIN = True in config.py the full run takes about 25 minutes on a laptop CPU, mostly for tuning; with RETRAIN = False the saved models and tuning results are reloaded.
