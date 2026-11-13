# Next day stock price prediction with deep learning (Keras / TensorFlow)

A project that predicts the **next trading day's closing price** of Apple (AAPL) with 1D-CNN, LSTM and GRU networks, each tested with closing price alone and with closing price plus **trading volume**. The models are compared with simple baselines, and the best raw-price model is then improved with hyperparameter tuning and a log-return target.

The project follows the **CRISP-DM** framework from start to finish: business understanding, data understanding, data preparation, modeling, evaluation and deployment. Each phase has its own notebook, the reusable code lives in `src/`, and every setting is in `config.py`.

The same project is also available as one self-contained notebook, [`notebooks/Team7_Deep_Learning_Final_Project.ipynb`](notebooks/Team7_Deep_Learning_Final_Project.ipynb). It is an alternative deliverable that gives the same results; see [Alternative: single notebook](#alternative-single-notebook).

---

## Project structure

```
Stock_Price_DL/
├── README.md                       <- Project overview
├── AGENTS.md                       <- shared rules for coding agents
├── .github/skills/git-workflow/    <- agent instructions for safe Git operations
├── config.py                       <- ALL settings (dates, lookback, feature sets, epochs, RETRAIN...)
├── requirements.txt                <- shared library versions
├── requirements-linux-gpu.txt      <- the same, plus CUDA libraries for Linux or WSL GPUs
├── data/
│   ├── AAPL_only.csv               <- daily AAPL closing price and volume data
│   └── apple_data.pkl              <- cleaned AAPL data saved by notebook 01
├── src/                            <- reusable code used by the notebooks
│   ├── data_loader.py              <- read and validate the CSV
│   ├── features.py                 <- engineered features and targets
│   ├── preprocessing.py            <- split, scale, windows, tf.data pipelines
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
│   ├── 06_improvements.ipynb
│   └── Team7_Deep_Learning_Final_Project.ipynb   <- optional: the whole project in one notebook
├── docs/                           <- concept and team guides, and the final report (.docx)
├── stock_eval312_*.yml / *.txt     <- record of Basma's original Mac environment
└── artifacts/                      <- models, scalers, metrics, predictions, plots, logs, tuning trials
```

The data, the artifacts and the notebook outputs are tracked in Git, so every result can be inspected without running anything.

---

## Setup

If you do not have the project on your computer yet, open a terminal and run:

```bash
git clone https://github.com/wences-psu/Stock_Price_DL.git
cd Stock_Price_DL
```

You only need to clone the project once on each computer. After cloning, use the team workflow in [`docs/git-workflow.md`](docs/git-workflow.md) to create a branch, update your local copy, and share changes through a pull request.

Create the shared environment, which uses Python 3.12.16 and the same library versions for both teammates:

```bash
conda create -n stock_eval312 python=3.12.16 -y
conda activate stock_eval312
python -m pip install -r requirements.txt
```

**NVIDIA GPU:** use Linux or WSL and install `requirements-linux-gpu.txt` in the last step instead. TensorFlow 2.10 was the last version with native GPU support on Windows, so TensorFlow 2.16 only uses the CPU on native Windows.

The `stock_eval312_history.yml`, `stock_eval312_pip.txt` and `stock_eval312_packages.txt` files record Basma's Mac environment from before the final notebook. Install from `requirements.txt` instead: those files do not include KerasTuner, and their setuptools version is too new for TensorBoard 2.16.

---

## Quick start

Open the notebooks in order, choose the `stock_eval312` kernel, and run each one from top to bottom:

| Notebook | CRISP-DM phase | What it does |
|---|---|---|
| `01_business_and_data_understanding` | Business and data understanding | Goals, data quality, exploration |
| `02_data_preparation` | Data preparation | Features, targets, split, scaling, windows, leakage checks |
| `03_modeling` | Modeling | Trains the six configurations and selects one by validation loss |
| `04_evaluation` | Evaluation | Test metrics in dollars, baselines, volume effect, Diebold-Mariano tests |
| `05_deployment` | Deployment | Reloads the selected model, predicts the next day, replays the last 20 days |
| `06_improvements` | Next iteration | KerasTuner tuning and a log-return target for the selected model |

| Setting in `config.py` | Effect |
|---|---|
| `RETRAIN = True` (default) | Notebooks 03 and 06 train everything from scratch |
| `RETRAIN = False` | They reload the saved models and tuning results from `artifacts/` |

A full run takes about 25 minutes on a laptop CPU, most of it in the tuning step of notebook 06. CPU and GPU runs select the same model and reach the same conclusions, with small differences in the numbers. To view the training logs, run `tensorboard --logdir artifacts/logs` from the project folder.

---

## Alternative: single notebook

[`notebooks/Team7_Deep_Learning_Final_Project.ipynb`](notebooks/Team7_Deep_Learning_Final_Project.ipynb) contains the whole project, code and report text, in one notebook that needs no other project code. It is based on the code template notebook. It has its own copy of the code, follows exactly the same steps as notebooks 01-06, writes the same run folders in `artifacts/`, and gives the same results; its saved outputs are the official CPU run quoted below.

---

## Documentation

| Guide |
|---|
| [`docs/01_crisp_dm_workflow.md`](docs/01_crisp_dm_workflow.md) |
| [`docs/02_time_series_concepts.md`](docs/02_time_series_concepts.md) |
| [`docs/03_deep_learning_models.md`](docs/03_deep_learning_models.md) |
| [`docs/04_evaluation_metrics.md`](docs/04_evaluation_metrics.md) |
| [`docs/05_deployment.md`](docs/05_deployment.md) |

## Collaboration

| Resource | Purpose |
|---|---|
| [`docs/git-workflow.md`](docs/git-workflow.md) | Git workflow, recovery commands, platform notes, and VS Code/PyCharm instructions. |
| [`docs/notebook-maintenance.md`](docs/notebook-maintenance.md) | One time setup and rules for notebooks, outputs and artifacts in Git. |
| [`AGENTS.md`](AGENTS.md) | Shared project rules for agents, coding style, notebooks, and safe Git operations. |
| [`.github/skills/git-workflow/SKILL.md`](.github/skills/git-workflow/SKILL.md) | On demand instructions for asking an agent to safely inspect, sync, branch, commit, or push. |

Each teammate should work on a short lived branch and merge into `main` through a pull request after the other teammate reviews the changes. Do not commit directly to `main`.

## Results

These numbers come from the official CPU run. Notebooks 04 and 06 and the single notebook all show them. All evaluations use the same 632 held-out test days, from March 22, 2024, through September 29, 2026.

| Model | Test RMSE (USD) |
|---|---|
| Random walk + drift baseline | 4.15 |
| Naive baseline | 4.16 |
| GRU price + volume, log-return target | 4.17 |
| 5-day moving average baseline | 6.45 |
| CNN price only | 14.76 |
| CNN price + volume | 16.40 |
| GRU price + volume, tuned | 16.65 |
| GRU price + volume (selected by validation loss) | 32.54 |
| GRU price only | 40.47 |
| LSTM price only | 49.32 |
| LSTM price + volume | 65.33 |

### Key findings

- None of the models outperformed the naive or random walk with drift baselines.
- CNN using closing prices alone had the lowest test RMSE among the six raw-price models.
- GRU with closing prices and trading volume was selected based on validation loss, not test performance. It underestimated the actual close on 602 of the 632 test days.
- Adding trading volume improved GRU performance but worsened CNN and LSTM performance.
- Hyperparameter tuning cut the selected model's test RMSE roughly in half. The log-return target came within a few cents of the baselines, but did not beat them.

### Deployment decision

The historical demonstration predicted a close of $253.65 for September 30, 2026, using data through September 29, 2026, when the last close was $329.40. The selected model's 20-day historical replay had an MAE of $78.24, compared with $3.85 for the log-return version.

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
