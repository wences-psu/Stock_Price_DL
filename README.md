# Next day stock price prediction with deep learning (Keras / TensorFlow)

A project that predicts the **next trading day's closing price** of Apple (AAPL) with LSTM, GRU and 1D-CNN networks, each tested with closing price alone and with closing price plus **trading volume**. Every network predicts the next-day **log return**, which is converted back to a closing price and compared in dollars with simple baselines. The selected model then goes through two improvement iterations: a change of inputs from price levels to daily log returns, and hyperparameter tuning of the better version.

The project follows the **CRISP-DM** framework from start to finish: business understanding, data understanding, data preparation, modeling, evaluation and deployment. Each phase has its own notebook, the reusable code lives in `src/`, and every setting is in `config.py`.

> **Branch note.** This is the `review/a5155cd-log-return` branch. It starts from commit `a5155cd` (the six log-return models described in the original write-up) and fixes the errors found in that version, using the same environment and Git setup as `main`. `main` holds the other version of the project: six price-target models plus a tuning and log-return iteration. This branch adds the same tuning step to the selected log-return model, and a second improvement iteration with stationary inputs.

---

## Project structure

```
Stock_Price_DL/
├── README.md                       <- Project overview
├── config.py                       <- ALL settings (dates, lookback, target, inputs, epochs, tuning trials, RETRAIN...)
├── requirements.txt                <- shared library versions
├── requirements-linux-gpu.txt      <- the same, plus CUDA libraries for Linux or WSL GPUs
├── data/
│   ├── AAPL_only.csv               <- daily AAPL closing price and volume data
│   └── apple_data.pkl              <- cleaned AAPL data saved by notebook 01
├── src/                            <- reusable code used by the notebooks
│   ├── data_loader.py              <- read and validate the CSV
│   ├── features.py                 <- engineered features and the prediction target
│   ├── preprocessing.py            <- split, scale, windows
│   ├── models.py                   <- CNN, LSTM and GRU architectures
│   ├── train.py                    <- training, saving and reloading runs
│   ├── tuning.py                   <- KerasTuner search (notebook 06)
│   ├── evaluate.py                 <- baselines, metrics and comparison tables
│   ├── predict.py                  <- next-day prediction and historical replay
│   ├── plotting.py                 <- shared chart style
│   ├── notebook.py                 <- show_source helper for the notebooks
│   └── extract_aapl.py             <- recreates the AAPL data from the original Kaggle file
├── notebooks/                      <- run these in order
│   ├── 01_business_and_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_deployment.ipynb
│   └── 06_improvements.ipynb
├── docs/                           <- concept and team guides, and the project write-up
├── stock_eval312_*.yml / *.txt     <- record of Basma's original Mac environment
└── artifacts/                      <- models, scalers, metadata, metrics, predictions, plots, tuning trials
```

The data, the artifacts and the notebook outputs are tracked in Git, so every result can be inspected without running anything.

---

## Setup

If you do not have the project on your computer yet, open a terminal and run:

```bash
git clone https://github.com/wences-psu/Stock_Price_DL.git
cd Stock_Price_DL
git switch review/a5155cd-log-return
```

Use the team workflow in [`docs/git-workflow.md`](docs/git-workflow.md) to create a branch, update your local copy, and share changes through a pull request.

Create the shared environment, which uses Python 3.12.16 and the same library versions for both teammates:

```bash
conda create -n stock_eval312 python=3.12.16 -y
conda activate stock_eval312
python -m pip install -r requirements.txt
```

**NVIDIA GPU:** use Linux or WSL and install `requirements-linux-gpu.txt` in the last step instead. TensorFlow 2.10 was the last version with native GPU support on Windows, so TensorFlow 2.16 only uses the CPU on native Windows.

The `stock_eval312_history.yml`, `stock_eval312_pip.txt` and `stock_eval312_packages.txt` files record Basma's Mac environment from before the final run. Install from `requirements.txt` instead.

---

## Quick start

Open the notebooks in order, choose the `stock_eval312` kernel, and run each one from top to bottom:

| Notebook | CRISP-DM phase | What it does |
|---|---|---|
| `01_business_and_data_understanding` | Business and data understanding | Goals, data quality, exploration |
| `02_data_preparation` | Data preparation | Features, log-return target, split, scaling, windows, leakage checks |
| `03_modeling` | Modeling | Trains the six configurations and selects one by validation loss |
| `04_evaluation` | Evaluation | Test metrics in dollars, baselines, volume effect, Diebold-Mariano tests |
| `05_deployment` | Deployment | Reloads the selected model, predicts the next day, replays the last 20 days |
| `06_improvements` | Next iterations | Stationary inputs (log returns) for the selected model, then KerasTuner tuning, compared by validation RMSE |

| Setting in `config.py` | Effect |
|---|---|
| `RETRAIN = True` (default) | Notebooks 03 and 06 train everything from scratch |
| `RETRAIN = False` | They reload the saved models and tuning results from `artifacts/` |

A full run takes about 10 minutes on a laptop CPU, most of it in the tuning step of notebook 06. The official results below come from a CPU run; GPU runs can differ slightly in the last decimals.

---

## Documentation

| Guide |
|---|
| [`docs/01_crisp_dm_workflow.md`](docs/01_crisp_dm_workflow.md) |
| [`docs/02_time_series_concepts.md`](docs/02_time_series_concepts.md) |
| [`docs/03_deep_learning_models.md`](docs/03_deep_learning_models.md) |
| [`docs/04_evaluation_metrics.md`](docs/04_evaluation_metrics.md) |
| [`docs/05_deployment.md`](docs/05_deployment.md) |

The original project write-up for this version is in [`docs/Project_Write_Up_DRAFT_DL_a5155cd.docx`](docs/Project_Write_Up_DRAFT_DL_a5155cd.docx), with a Markdown copy in [`docs/Project_Write_Up_DRAFT_DL_a5155cd.md`](docs/Project_Write_Up_DRAFT_DL_a5155cd.md).

## Collaboration

| Resource | Purpose |
|---|---|
| [`docs/git-workflow.md`](docs/git-workflow.md) | Git workflow, recovery commands, platform notes, and VS Code/PyCharm instructions. |
| [`docs/notebook-maintenance.md`](docs/notebook-maintenance.md) | One time setup and rules for notebooks, outputs and artifacts in Git. |
| [`AGENTS.md`](AGENTS.md) | Shared project rules for agents, coding style, notebooks, and safe Git operations. |
| [`.github/skills/git-workflow/SKILL.md`](.github/skills/git-workflow/SKILL.md) | On demand instructions for asking an agent to safely inspect, sync, branch, commit, or push. |

Each teammate should work on a short lived branch and merge into `main` through a pull request after the other teammate reviews the changes. Do not commit directly to `main`.

## Results

These numbers come from the official CPU run shown in notebooks 03 to 06. All evaluations use the same 632 held-out test days, from March 22, 2024, through September 29, 2026. Every network predicts the next-day log return; the predictions are converted to closing prices before scoring.

| Model | Test MAE (USD) | Test RMSE (USD) | Dir. acc. (%) |
|---|---|---|---|
| Random walk + drift baseline | 2.84 | 4.15 | 55.2 |
| LSTM price + volume, log-return inputs (notebook 06) | 2.86 | 4.16 | 51.7 |
| Naive baseline | 2.85 | 4.16 | n/a |
| CNN price + volume | 2.87 | 4.17 | 49.8 |
| CNN price only | 2.87 | 4.18 | 47.6 |
| GRU price + volume | 2.93 | 4.22 | 45.7 |
| LSTM price + volume (selected by validation loss) | 2.96 | 4.24 | 44.8 |
| LSTM price only | 3.05 | 4.32 | 44.6 |
| GRU price only | 3.06 | 4.33 | 44.1 |
| LSTM price + volume, tuned | 3.83 | 5.09 | 44.3 |
| 5-day moving average baseline | 4.73 | 6.45 | 48.9 |

### Key findings

- None of the six models outperformed the naive or random walk with drift baselines. The two CNN models are statistically indistinguishable from the baselines (Diebold-Mariano p > 0.18); the GRU and LSTM models are significantly worse than the naive baseline (p < 0.05).
- CNN with closing prices and trading volume had the lowest test RMSE among the networks ($4.1735).
- LSTM with closing prices and trading volume was selected by validation loss, not test performance; it ranked fourth of six on the test set. The six validation losses differ only in the third decimal.
- Adding trading volume lowered the test RMSE of all three architectures; the improvement is significant for LSTM and GRU (p < 0.001) but not for CNN (p = 0.12).
- Every network is biased toward predicting a fall: the selected model predicts a rise on 5.9% of test days, while the price rose on 55.2% of them. The inputs are scaled price levels, and test prices lie far above the training range.
- Notebook 06, iteration 1: retraining the selected model with stationary inputs (daily log return, log volume and its change) removes the bias (it predicts a rise on 84.2% of test days) and gives the lowest test RMSE of any network ($4.16, between the drift and naive baselines). On the validation set, however, it scored slightly worse than the original ($2.7175 versus $2.7048), so by the rule set in advance the original went on to tuning.
- Notebook 06, iteration 2: hyperparameter tuning (20 KerasTuner trials) improved the selected model's validation RMSE by less than a tenth of a cent ($2.7048 to $2.7040), but its test RMSE rose from $4.24 to $5.08: the tuned model (no dropout, higher learning rate) amplified the bias. All versions sit within 0.003 of the validation loss of a model that always predicts the average return, so the validation period cannot tell them apart.

### Deployment decision

The historical demonstration predicted a close of $328.00 for September 30, 2026, using data through September 29, 2026, when the last close was $329.40. The selected model's 20-day historical replay had an MAE of $3.97, compared with $3.74 for the naive forecast on the same days; the tuned version's replay MAE was $7.09 and the stationary-input version's $3.87 (it predicted $330.16 for the same day).

Production deployment is not recommended, because no model outperformed the simple forecasting baselines. This was a historical demonstration, not a current stock-price forecast.

## Dataset Source and Reproducibility 
### Original Dataset

The original dataset was obtained from Kaggle:
S&P500 All Assets – Daily Updated Dataset.

Source file: SnP_daily_update.csv

The project uses Apple's (AAPL) historical closing prices
and daily trading volumes from January 4, 2010, through
September 29, 2026.

The AAPL data was extracted from the original dataset
using the Close and Volume columns associated with
the AAPL ticker symbol.

The extracted dataset contains 4,210 trading-day records
and is stored in data/AAPL_only.csv.

### Dataset Extraction Procedure

The AAPL dataset can be reproduced using the Python script
located at `src/extract_aapl.py`.

The original Kaggle file (161 MB) is not stored in Git: it is over
GitHub's 100 MB file limit and can be downloaded again. Save it as
`data/SnP_daily_update.csv` (ignored by Git) or in `~/Downloads`.

The script performs the following steps:

1. Finds SnP_daily_update.csv in `data/`, then in `~/Downloads`
   (or uses the path given with `--source`).
2. Reads it using its two-row header.
3. Selects the Close and Volume columns for ticker AAPL.
4. Keeps the trading days up to `--end-date` (default 2026-09-29, the
   last day of the project dataset). The Kaggle file is updated daily,
   so a newer download contains more days.
5. Creates `data/AAPL_only_verified.csv` (ignored by Git) and compares
   it with `data/AAPL_only.csv`, which it leaves unchanged.

Run the extraction script from the project root:

    python src/extract_aapl.py
    python src/extract_aapl.py --source path/to/SnP_daily_update.csv
    python src/extract_aapl.py --end-date none    # keep every day in the file

### Dataset Verification

The extracted data was compared with the existing
project dataset using pandas.

Verification results with the download used to build the project:

- Original dataset: 4,210 AAPL records.
- Recreated dataset: 4,210 AAPL records.
- Data values match: True.

The comparison confirmed that the recreated dataset
matches the existing project dataset when loaded
using pandas.

An older download of the same Kaggle file (saved on September 21, 2026)
ends on September 18, 2026, so it contains 4,203 of the 4,210 trading
days. On those shared days the closing prices agree to within
$0.0001; the volume differs only on its last day (2026-09-18), which
Kaggle updated afterward.
