# Next day stock price prediction with deep learning (Keras / TensorFlow)

A project that predicts the **next trading day's closing price** of Apple (AAPL) with LSTM, GRU and 1D-CNN networks. It includes a switch to add **trading volume** as an input, to measure whether volume improves the prediction.

The project follows the **CRISP-DM** framework from start to finish: business understanding → data understanding → data preparation → modeling → evaluation → deployment. Every step is explained in commented code, notebooks and Markdown guides.

---

## Project structure

```
Stock_Price/
├── README.md                       ← Project overview
├── requirements.txt                ← pinned library versions
├── config.py                       ← ALL settings (dates, lookback, volume on/off, epochs...)
├── data/
│   ├── AAPL_only.csv               ← daily AAPL closing price and volume data
├── src/                            ← reusable helper code
├── notebooks/                      ← run these in order
│   ├── 01_business_and_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_deployment.ipynb
├── docs/                           ← concept guides
│   ├── 01_crisp_dm_workflow.md
│   ├── 02_time_series_concepts.md
│   ├── 03_deep_learning_models.md
│   ├── 04_evaluation_metrics.md
│   └── 05_deployment.md
└── artifacts/                      ← created by training: models, scalers, metrics, plots
```

---

## Setup

The project was built and tested with Python 3.9, TensorFlow 2.10.1 (the last TensorFlow release with native GPU support on Windows).

The evaluation and deployment notebooks were also successfully tested on macOS using Python 3.12.16 in the stock_eval312 Conda environment.  This environment used a newer TensorFlow version and successfully loaded the saved models.

### First time clone

If you do not have the project on your computer yet, open PowerShell or Git Bash on Windows, Terminal on macOS, or a terminal on Linux and run:

```bash
git clone https://github.com/wences-psu/Stock_Price_DL.git
cd Stock_Price_DL
```

You only need to clone the project once on each computer. After cloning, use the team workflow in [`docs/git-workflow.md`](docs/git-workflow.md) to create a branch, update your local copy, and share changes through a pull request. The commands are the same across Windows, macOS, and Linux.

To create a fresh environment:
```bash
conda create -n stock-dl python=3.9 -y
conda activate stock-dl
python -m pip install -r requirements.txt
```

### macOS Evaluation and Deployment Environment

The evaluation and deployment notebooks were successfully tested on
macOS using Python 3.12.16 in a Conda environment named
`stock_eval312`.

To create and activate this environment:

```bash
conda create -n stock_eval312 python=3.12 -y
conda activate stock_eval312
```

### Verified macOS Package Versions

The following versions were used successfully to execute the
evaluation and deployment notebooks:

| Package | Version |
|---|---|
| Python | 3.12.16 |
| TensorFlow | 2.16.2 |
| NumPy | 1.26.4 |
| Pandas | 2.2.3 |
| Scikit-learn | 1.6.1 |

These versions were verified in the `stock_eval312` Conda environment.

The evaluation and deployment notebooks ran successfully without
retraining or modifying the six saved neural network models.

The original `requirements.txt` remains unchanged to preserve
compatibility with the team's Python 3.9 environment.

---

## Quick start

Open the notebooks in order:

| Notebook |
|---|
| `01_business_and_data_understanding` |
| `02_data_preparation` |
| `03_modeling` |
| `04_evaluation` |
| `05_deployment` |

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
| [`AGENTS.md`](AGENTS.md) | Shared project rules for agents, coding style, notebooks, and safe Git operations. |
| [`.github/skills/git-workflow/SKILL.md`](.github/skills/git-workflow/SKILL.md) | On demand instructions for asking an agent to safely inspect, sync, branch, commit, or push. |

Each teammate should work on a short lived branch and merge into `main` through a pull request after the other teammate reviews the changes. Do not commit directly to `main`.

## Evaluation Results

Six deep learning models were evaluated to predict Apple's (AAPL) next-day closing price. CNN, GRU, and LSTM were tested using closing prices alone and closing prices with trading volume.

### Model Performance (test RMSE, USD)
CNN price only: 14.7613  
CNN price + volume: 16.3988  
GRU price only: 40.4717  
GRU price + volume: 32.5438  
LSTM price only: 49.3220  
LSTM price + volume: 65.3250 
Naive baseline: 4.1577  
Random walk + drift baseline: 4.1520


### Key Findings
CNN using closing prices alone had the lowest test RMSE among the six deep learning models.

Adding trading volume improved GRU performance but worsened CNN and LSTM performance.

None of the six deep learning models outperformed the naive baseline or the random walk with drift baseline.

GRU with closing prices and trading volume was selected based on validation loss, not test performance.

The selected GRU model did not meet the project's overall business success criteria.

### Deployment Results

The deployment notebook successfully generated a prediction for September 30, 2026, using historical data through September 29, 2026.

Last observed closing price: $329.40  
Predicted next-day closing price: $253.65  
Predicted percentage change: -22.99%

A rolling 20-trading-day historical simulation produced a mean absolute error (MAE) of $78.24.

The selected GRU model consistently underestimated actual closing prices during this simulation.

### Deployment Decision

The project successfully completed the CRISP-DM workflow, including data preparation, model training, evaluation, and a deployment demonstration.

However, none of the six deep learning models outperformed the simple forecasting baselines.

Therefore, production deployment is not recommended at this time.

Future improvements should focus on investigating prediction bias, improving model performance, testing additional market indicators, and exploring better feature engineering.