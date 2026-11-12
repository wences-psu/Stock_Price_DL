<!-- Markdown copy of docs/Project_Write_Up_DRAFT_DL_a5155cd.docx, converted from the Word file for easy reading and diffs. The .docx is the source; edit it, not this file. -->

# INTRODUCTION

Predicting the next trading day’s closing price is difficult because equity prices react to changing market conditions, volatility, company events, and unpredictable external information. This project investigates whether convolutional and recurrent neural networks can learn useful patterns from Apple Inc. (AAPL) daily observations and improve on simple forecasting rules. Our evaluation deliberately emphasizes held-out chronological forecasts and strong naive baselines rather than relying on training accuracy alone.

In this project methodology, the team uses next-day logarithmic return as the prediction target; estimates are converted back to dollar closing prices before reporting MAE and RMSE.

## Scope and measurable objectives

There are three objectives this project is aiming to achieve as follows:

Objective 1 is to build a defensible next-trading-day AAPL forecasting pipeline using historical Close data, optionally supplemented by trading Volume.

Objective 2 is to compare CNN, LSTM, and GRU architectures on a time-ordered test set using prices reconstructed in US dollars.

Objective 3 is to determine whether any model materially exceeds low-complexity baselines, rather than equating low in-sample loss with practical forecast value.

Success is evaluated by out-of-sample MAE and RMSE and by transparent experimental documentation: feature ordering, target definition, lookback length, split boundaries, scaling fit scope, validation-based model selection, and reproducible data extraction. Directional accuracy and MAPE provide supporting perspectives but do not replace the primary regression metrics.

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

Basma Hameid’s scientific review highlights two lessons for Team 7: historical data preparation matters, and Volume should be tested rather than assumed beneficial. Their study uses different market data and a different architecture; Team 7 instead compares Close-only and Close + Volume for one stock under matched test conditions. In our results, Volume helps CNN and GRU RMSE but worsens LSTM RMSE.

Lawi, Mesra, and Amir (2022) compared eight forecasting architectures built around LSTM and GRU variants on grouped stock time-series data. Basma’s review emphasizes that architecture choice affects forecasting quality and that a fair comparison is more useful than assuming one recurrent network is best. Team 7 follows this comparison principle using CNN, GRU, and LSTM with the same 60-trading-day input window, log-return target, chronological partitions, and 632 test forecasts. The eight architectures of Lawi et al. were not implemented in this project.

Li (2026), reviewed by Wenceslao Granados Gomez, proposed a hybrid CNN–LSTM–Transformer pipeline that combines local patterns, temporal dependencies, and global relationships. The review describes richer price, technical, economic, and sentiment inputs, as well as a loss intended to account for direction and asymmetric forecasting risk. Wenceslao identified limitations including the absence of some simple persistence baselines, component-level ablation, and confidence intervals. Team 7 uses separate CNN, LSTM, and GRU regressors rather than a hybrid model, and directly compares their dollar errors with a random-walk-with-drift baseline.

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

The executed Business and Data Understanding notebook supplies the numerical evidence below. The extracted source contains 4,210 AAPL dates, while the notebook’s modeling-oriented data starts on 2010-01-05 and contains 4,209 observations through 2026-09-29. The first source observation (2010-01-04) is not present in the displayed 4,209-row notebook frame. The one-row difference reflects the displayed exploratory frame rather than a missing source value; the exact processing step responsible is not established by the available notebook output.

## Data Quality Checks

The executed notebook’s data-quality output reports 4,209 rows, zero missing values, zero duplicate dates, and zero zero-volume sessions. The data frame contains numeric Close (float64) and Volume (int64) columns. The notebook reports no inconsistent high/low-price values; that specific check is recorded in its narrative, although the displayed working frame contains only Close and Volume. Ordinary gaps between sessions are consistent with weekends and market holidays. The notebook identifies an unusual October 2012 closure associated with Hurricane Sandy.

The executed notebook includes a full-history plot, descriptive statistics, daily-log-return distribution, lagged autocorrelations, and correlations linking volume with next-day return and absolute return. These are reported below with their interpretation. The screenshots do not establish a separate Pearson correlation between contemporaneous Close and Volume; no numerical coefficient for that pair is invented here.

## Executed notebook: reproducible exploratory findings

Data coverage and integrity. The original extracted AAPL CSV was verified at 4,210 daily records; the displayed exploratory working data has 4,209 rows and two numerical columns, Close and Volume (2010-01-05 to 2026-09-29). It has 0 missing cells, 0 repeated session dates, and 0 zero-volume trading days. The 4,210-versus-4,209 discrepancy is a processing-stage row-count difference requiring explicit reconciliation, not evidence of a corrupted source file.

Descriptive statistics. The exploratory working frame reports an average Close of about $86.68, a standard deviation near $85.33, minimum around $5.74, median approximately $40.84, and maximum around $341.07. Mean daily trading Volume is approximately 211.4 million shares; the observed range covers roughly 17.9 million to 1.881 billion shares. These summaries characterize the historical series, not a prediction or test-set performance estimate.

Growth and return characteristics. In the notebook’s summary, first Close is $6.412 and last Close is approximately $329.400 (growth ratio ~51.372×). Mean daily log return is 0.094%, with daily standard deviation 1.770% and annualized volatility 28.105%. Approximately 53.042% of daily log returns are positive. The observed worst and best single-session percentage changes are −12.865% and +15.329%, respectively.

Outliers and normality. The histogram of daily log returns is sharply centered but visibly more heavy-tailed than a normal curve fitted to the same mean and standard deviation. Skewness is about −0.170 and excess kurtosis about 6.034, showing non-normal extreme-return risk. The research notebook proposes robust Huber loss as a possible response to such tails; whether the final trained runs actually use that loss must be checked in their configurations.

Temporal dependence. The lag-one autocorrelation of signed daily log returns is about −0.029, versus a notebook reference threshold of approximately ±0.030; most signed-return autocorrelations through ten sessions remain small. By comparison, lag-one autocorrelation of absolute returns is approximately +0.193, and absolute-return correlations across lags 1–10 stay positive (about 0.110–0.193). This pattern is consistent with volatility clustering and limited simple linear predictability of return direction.

Volume relationship. The notebook’s volume correlation output and commentary show little linear association between volume information and the direction of next-day returns, while a modest positive relationship appears for the magnitude of next-day returns. For log volume, the displayed next-day absolute-return correlation is approximately +0.119. Thus Volume is motivated as an alternative feature to test, not as proven directional information.

Methodological implication. Rising and non-stationary price levels make the problem harder for models trained on raw prices, especially when later test prices exceed the earlier training range. The notebook therefore motivates a next-session log-return target, time-ordered splits to avoid leakage, separate Close-only and Close-plus-Volume experiments, and strong persistence/drift benchmarks. Predicted returns are mapped back to dollar closing prices before final scoring.

# DATA PREPROCESSING

Records are ordered by trading date and used to create a next_close value shifted one trading day ahead. For the updated log-return configuration, the target is calculated as:

r(t+1) = ln(P(t+1) / P(t)), where P(t) is the current AAPL close and P(t+1) the next trading-day close.

In the recorded implementation, src/preprocessing.py, load_prepared_data(), assigns frame["target"] = np.log(frame["next_close"] / frame["Close"]) after creation of the modeling frame, then removes rows without a valid next-day target. For model output r_hat(t+1), the corresponding prediction is P_hat(t+1) = P(t) × exp(r_hat(t+1)). Model errors in USD are evaluated only after this inverse conversion.

Input windows represent the preceding 60 trading days for the main comparison. A supplementary CNN Close+Volume experiment used 30 days. The stated chronological split is 70% training, 15% validation, and 15% held-out test; the reported evaluation set has 632 forecasts. Scaling is based on the training partition and applied to validation and test observations; the held-out test is reserved for evaluation.

Data cleaning and inspection precede model training. Nonpositive closes would invalidate the logarithm and must be detected before target creation; extreme observations require investigation rather than automatic removal. The report does not assert unverified counts for missingness or outliers. Standard stock-price modeling does not assume class balancing, and data augmentation was not established as an implemented step.

## Sequence construction and time alignment

For session t, the target is r(t+1) = ln(P(t+1)/P(t)), defined only when both prices are positive and the following session exists. An input example contains an ordered lookback of earlier observations available by the decision time. The model emits a scalar log-return estimate r̂(t+1), and the display/evaluation forecast is P̂(t+1) = P(t) × exp(r̂(t+1)). This inverse formula is essential: comparing raw returns with dollar closing prices would produce meaningless errors.

Splitting by date before fitting preprocessing statistics prevents leakage from the held-out period. The 70/15/15 chronological split is evaluated on the same held-out forecast dates across models and baselines. Input windows may carry earlier context across a split boundary if that context would have been available at forecast time, but targets from later partitions must not influence model fit or tuning.

The 60-session lookback is the principal experimental setting. A 30-session CNN-plus-Volume variant appears in an earlier log-return comparison and is treated as an auxiliary sensitivity experiment, not as one of the six equal-lookback main architectures.

# METHODOLOGY

The work follows a CRISP-DM-inspired workflow: business/data understanding, preprocessing, modeling, evaluation, and deployment demonstration. We compare one-dimensional convolutional networks (CNN) for local temporal patterns with recurrent LSTM and GRU networks for gated sequence representation. Close-only and Close+Volume variants are considered. The earlier six-configuration price-target report and the revised log-return results are distinct experiment sets: numbers from one are not transplanted into the other.

For the recorded log-return runs, the described training configuration is Adam optimizer (learning rate 0.001), batch size 32, dropout 0.2, up to 100 epochs, and early stopping on validation performance. Log-return prediction is a regression output, not a sigmoid binary-classification output; directional accuracy is an additional diagnostic, not a replacement for dollar error metrics.

Baselines: (1) naive persistence — forecast next close equals the latest observed close; (2) random walk with drift — extrapolate a simple historical price change; and (3) five-day moving average — use the recent mean price. The same dates, available history, and aligned actual next-day targets should be used for fair comparisons. Baseline test performance establishes how much improvement a deep-learning system must earn.

The final six-model experiment uses next-trading-day log return as the target and converts the predicted return back to a dollar closing price before evaluation. Saved prediction files supplied the MAE and RMSE for all six 60-day configurations, and the matching model metadata confirmed target_mode = log_return, the 60-day lookback, the chronological 70/15/15 split, and a common 632-observation held-out period (March 22, 2024–September 29, 2026). The same evaluation protocol was applied to all six configurations.

## Metric definitions and interpretation

On n holdout dates, MAE = (1/n) Σ|P̂−P| and RMSE = sqrt[(1/n) Σ(P̂−P)²]. MAE summarizes a typical absolute dollar miss, while RMSE gives extra penalty to large errors and can reveal episodic model failure. MAPE = 100 × mean(|P̂−P|/|P|) is interpretable as relative error when actual prices are positive. Directional accuracy compares the sign of predicted versus actual price changes from the last observed close.

A fair empirical statement must distinguish (a) validation-based training and selection, (b) one-time test evaluation, and (c) illustrative single-day forecasts. The difference between CNN Close+Volume RMSE ≈ $4.1737 and the drift baseline ≈ $4.1520 is about two cents in the documented comparison; this does not establish superior performance for either method across all market regimes. There is no reported uncertainty interval or repeated-seed robustness test supporting a significance claim.

# RESULTS

Final verified six-model log-return comparison. The figures below are test-set errors in USD after predicted log returns were converted back to closing prices. Each network uses a 60-trading-day lookback. The 632 test predictions span March 22, 2024, through September 29, 2026. Model configurations were checked against the saved metadata files.

| Model / feature set | MAE ($) | RMSE ($) |
|---|---|---|
| CNN — Close only | 2.8807 | 4.1837 |
| CNN — Close + Volume | 2.8679 | 4.1737 |
| GRU — Close only | 3.3906 | 4.6419 |
| GRU — Close + Volume | 2.9249 | 4.2178 |
| LSTM — Close only | 3.0385 | 4.3053 |
| LSTM — Close + Volume | 3.1154 | 4.3693 |
| Random walk + drift baseline | 2.8363 | 4.1520 |

Table 1. Final six neural-network configurations and random-walk-with-drift baseline (test-set MAE and RMSE; lower is better).

The CNN with Close + Volume produced the lowest RMSE among the six neural networks ($4.1737), slightly ahead of CNN Close-only ($4.1837). GRU improved from $4.6419 to $4.2178 RMSE when Volume was added, whereas LSTM worsened from $4.3053 to $4.3693. Volume therefore did not have a uniform effect across architectures. The simple random-walk-with-drift baseline remained best overall at $4.1520 RMSE and $2.8363 MAE. CNN Close + Volume was approximately $0.0217 worse in RMSE; this small difference must not be interpreted as statistically significant without uncertainty estimates.

Evaluation metrics: MAE gives average absolute dollar error; RMSE gives added weight to larger errors; MAPE rescales absolute error by actual dollar price; and directional accuracy measures whether predicted direction agrees with the realized next-day movement. None establishes profitability. As this is regression, a confusion matrix, ROC, or classification precision/recall are not primary evaluation measures.

## Practical interpretation and bias/variance considerations

The results caution against equating neural-network sophistication with useful predictive accuracy. Next-day persistence is a strong comparator because daily equity prices frequently differ only modestly from the preceding close. A network can attain low training loss yet underperform persistence in out-of-sample dollars if it smooths abrupt movements, is biased toward historical price levels, or overfits noise. These are plausible mechanisms to investigate, not causes established by the current experiment.

Bias/variance analysis can compare train-versus-validation residuals, stability across random seeds, and errors in earlier versus later test periods. A large gap between training and validation error would suggest overfitting; uniformly poor errors may suggest target/preprocessing mismatch or insufficient signal. The appropriate improvement strategy is to investigate diagnostics and tune strictly on validation data, preserving the held-out test until model choice is complete.

Economic relevance must also distinguish statistical forecast error from executable trading performance. The project does not model transaction fees, bid/ask spread, position-sizing rules, slippage, or risk limits; therefore the numerical predictions are not a validated investment strategy. No real-time deployment or external reproducibility audit is claimed.

# DISCUSSION OF RESULTS

The final six-model experiment illustrates both the importance of choosing a suitable target and the strength of simple financial forecasting baselines. Among neural networks, CNN Close + Volume gave the best held-out dollar RMSE, but its advantage over CNN Close-only was very small. Random walk + drift outperformed every neural model on the reported test dates. Thus the project demonstrates a reproducible comparison, not a proven material forecasting advantage over the baseline.

Trading volume's contribution varies by architecture: it modestly improves CNN and substantially improves GRU RMSE, while the LSTM Close-only run performs better than LSTM Close + Volume. These conclusions apply only to the matched log-return experiments on the observed held-out period. The chronological holdout reduces leakage risk but cannot remove market-regime shift, sensitivity to initialization, or the possibility of data-specific findings.

The scientific readings help interpret these findings. Wei et al. (2022) motivate testing trading Volume, but the mixed results show that more inputs do not automatically improve accuracy. Lawi et al. (2022) support evaluating LSTM and GRU side by side; our CNN, GRU, and LSTM comparisons show that model ranking depends on architecture and feature set. Li (2026) and Wen et al. (2019) describe richer hybrid and pattern-based approaches, but their reported results do not demonstrate an improvement for this AAPL experiment. The random-walk-with-drift result underscores the importance of baseline comparison, a concern also identified in Wenceslao’s reading.

## Reproducibility and Environment

The repository wences-psu/Stock_Price_DL contains the extraction script and documented dataset verification. A dedicated macOS environment was captured as stock_eval312_history.yml and stock_eval312_pip.txt; a portable named-version list was saved as stock_eval312_packages.txt. Verified local versions: Python 3.12.16, TensorFlow 2.16.2, Keras 3.15.1; machine architecture arm64 (Apple Silicon). A clean-room environment recreation on another computer has not been demonstrated, and the pip freeze includes some local build paths.

## Limitations and Deployment Decision

All six saved models use the same log-return target, 60-day lookback, split ratios, and held-out date interval; the saved evaluations do not show a forecasting advantage over random walk + drift. The best neural network is CNN Close + Volume by held-out RMSE, while validation-loss selection may identify a different model; keep the two selection criteria distinct. Market-regime changes and unforeseen events limit generalization, and this educational prototype is not a validated live trading strategy.

# INDIVIDUAL FEEDBACK — LEARNED LESSONS AND COLLABORATION

Basma Hameid — individual reflection: This project reinforced that a deep-learning experiment is only as reliable as its data lineage, time-aware preprocessing, and evaluation design. Rebuilding the AAPL extract from the original Kaggle file and checking the full 4,210-row sample made reproducibility a concrete task rather than a documentation slogan. Also, I learned to treat the prediction target as part of the model definition: log-return training and direct-dollar training cannot be compared casually without aligned data, inverse conversion, and clear experiment records.The most demanding aspects of this assignment were reconciling multiple notebook revisions, keeping validation selection separate from test reporting, and communicating which results belong to which experiment. The team benefited from shared source control and a division of technical and documentation work, but integration requires disciplined naming, saved outputs, and explicit review of assumptions. A practical improvement would be a shared result manifest for every run recording target mode, lookback, features, seed, split dates, artifact name, and MAE/RMSE.For future teamwork, Basma has shared that she would prioritize earlier checkpoint reviews and a consistent experiment export. The project also demonstrated a valuable scientific lesson: it is acceptable—and important—to report that a simple baseline outperforms neural networks. This is a more useful conclusion than overstating performance.

Wenceslao Gomez - TBD

# REFERENCES

Bao, W., Yue, J., & Rao, Y. (2017). A deep learning framework for financial time series using stacked autoencoders and long-short term memory. PLOS ONE, 12(7), e0180944. https://doi.org/10.1371/journal.pone.0180944

Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654–669. https://doi.org/10.1016/j.ejor.2017.11.054

Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020). Financial time series forecasting with deep learning: A systematic literature review: 2005–2019. Applied Soft Computing, 90, 106181. https://doi.org/10.1016/j.asoc.2020.106181

Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: A survey. Philosophical Transactions of the Royal Society A, 379, 20200209. https://doi.org/10.1098/rsta.2020.0209

Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271. https://doi.org/10.48550/arXiv.1803.01271

Kaggle, yash16jr. S&P500 All Assets – Daily Updated Dataset. https://www.kaggle.com/datasets/yash16jr/s-and-p500-daily-update-dataset/data

Team 7. Stock_Price_DL GitHub project repository. https://github.com/wences-psu/Stock_Price_DL

Wei, J., Xu, Q., & He, C. (2022). Deep learning of predicting closing price through historical adjustment closing price. Procedia Computer Science, 202, 379–384. https://doi.org/10.1016/j.procs.2022.04.052

Lawi, A., Mesra, H., & Amir, S. (2022). Implementation of long short-term memory and gated recurrent units on grouped time-series data to predict stock prices accurately. Journal of Big Data, 9, Article 89. https://doi.org/10.1186/s40537-022-00597-0

Li, X. (2026). Application of deep neural networks in stock price prediction. Proceedings of the 2026 3rd International Conference on Digital Society and Artificial Intelligence, 439–443. https://doi.org/10.1145/3833541.3833614

Wen, M., Li, P., Zhang, L., & Chen, Y. (2019). Stock market trend prediction using high-order information of time series. IEEE Access, 7, 28299–28308. https://doi.org/10.1109/ACCESS.2019.2901842

Hooi, B., Liu, S., Smailagic, A., & Faloutsos, C. (2017). BeatLex: Summarizing and forecasting time series with patterns. In Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2017), Lecture Notes in Computer Science, 10535. https://doi.org/10.1007/978-3-319-71246-8_1

# APPENDIX A — CODE FUNCTIONALITY AND REPRODUCIBILITY

## Functional overview of project source code

Data extraction (src/extract_aapl.py): reads the source Kaggle multi-asset file with two header rows; selects AAPL Close and Volume; writes data/AAPL_only_verified.csv; reports the number of extracted sessions. The script intentionally leaves data/AAPL_only.csv unchanged. The original extraction used ~/Downloads/SnP_daily_update.csv as the source path.

Preparation (src/preprocessing.py and data-preparation notebook): reads date-indexed stock observations, constructs a shifted next-session Close, derives the next-session log return for the revised target mode, handles train/validation/test partitions, scales using training data, and constructs the chosen lookback windows.

Feature and model stages (project notebooks and src/features.py, src/models.py, src/train.py as applicable): assemble Close-only or Close-plus-Volume tensors; configure CNN/LSTM/GRU network variants; fit training models using validation-based callbacks; and save training histories/model artifacts where the executed notebooks demonstrate that behavior.

Evaluation and deployment demonstration (src/evaluate.py, src/predict.py and evaluation/deployment notebooks as applicable): reconstruct next-day closing prices from predicted log returns, align forecasts with actual test dates, calculate regression errors and baseline comparators, and demonstrate a dated historical prediction.

Environment documentation: stock_eval312_history.yml records requested Conda dependencies; stock_eval312_pip.txt captures the local pip freeze; stock_eval312_packages.txt offers a simpler version list. Observed local environment: Python 3.12.16, TensorFlow 2.16.2, Keras 3.15.1 on arm64. The pip freeze contains locally resolved file references and may not reconstruct portably without adjustment.

Reproduction procedure: obtain the Kaggle source CSV, clone the project repository, recreate a compatible Python environment, run src/extract_aapl.py, compare the AAPL extract with the canonical sample, and execute the numbered notebooks in sequence using consistent configuration settings.
