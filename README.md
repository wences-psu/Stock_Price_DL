# Next day stock price prediction with deep learning (Keras / TensorFlow)

A project that predicts the **next trading day's closing price** of Apple (AAPL) with 1D-CNN, LSTM and GRU networks, each tested with closing price alone and with closing price plus **trading volume**. The models are compared with simple baselines, and the best raw-price model is then improved with hyperparameter tuning and a log-return target.

All the code and the report text are in one notebook, [`notebooks/Team7_Deep_Learning_Final_Project.ipynb`](notebooks/Team7_Deep_Learning_Final_Project.ipynb). It needs no other project code.

---

## Project structure

```
Stock_Price_DL/
├── README.md                       <- Project overview
├── requirements.txt                <- shared library versions
├── requirements-linux-gpu.txt      <- the same, plus CUDA libraries for Linux or WSL GPUs
├── data/
│   └── AAPL_only.csv               <- daily AAPL closing price and volume data
├── notebooks/
│   └── Team7_Deep_Learning_Final_Project.ipynb   <- all code, results and report text
├── docs/                           <- concept and team guides
└── artifacts/                      <- created by the notebook: models, scalers, logs (not in Git)
```

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

---

## Quick start

Open the notebook, choose the `stock_eval312` kernel, and select Run All.

| Setting (in the settings cell) | Effect |
|---|---|
| `RETRAIN = True` (default) | Trains everything from scratch |
| `RETRAIN = False` | Reloads the saved models from an earlier complete run, in under a minute |

A full run took about 21 minutes on an RTX 4090 Laptop GPU and about 25 minutes on the CPU of the same laptop. CPU and GPU runs select the same model and reach the same conclusions, with small differences in the numbers. To view the training logs, run `tensorboard --logdir artifacts/logs` from the project folder.

The saved notebook keeps its outputs in Git, unlike other notebooks. See [`docs/notebook-maintenance.md`](docs/notebook-maintenance.md).

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
| [`docs/notebook-maintenance.md`](docs/notebook-maintenance.md) | One time setup and rules for keeping notebooks clean in Git. |
| [`AGENTS.md`](AGENTS.md) | Shared project rules for agents, coding style, notebooks, and safe Git operations. |
| [`.github/skills/git-workflow/SKILL.md`](.github/skills/git-workflow/SKILL.md) | On demand instructions for asking an agent to safely inspect, sync, branch, commit, or push. |

Each teammate should work on a short lived branch and merge into `main` through a pull request after the other teammate reviews the changes. Do not commit directly to `main`.

## Results

These numbers come from the official run saved in the notebook. All evaluations use the same 632 held-out test days, from March 22, 2024, through September 29, 2026.

| Model | Test RMSE (USD) |
|---|---|
| Random walk + drift baseline | 4.15 |
| Naive baseline | 4.16 |
| GRU price + volume, log-return target | 4.17 |
| 5-day moving average baseline | 6.45 |
| CNN price only | 14.94 |
| CNN price + volume | 16.21 |
| GRU price + volume, tuned | 16.65 |
| GRU price + volume (selected by validation loss) | 32.52 |
| GRU price only | 40.47 |
| LSTM price only | 49.33 |
| LSTM price + volume | 64.45 |

### Key findings

- None of the models outperformed the naive or random walk with drift baselines.
- CNN using closing prices alone had the lowest test RMSE among the six raw-price models.
- GRU with closing prices and trading volume was selected based on validation loss, not test performance. It underestimated the actual close on 602 of the 632 test days.
- Adding trading volume improved GRU performance but worsened CNN and LSTM performance.
- Hyperparameter tuning cut the selected model's test RMSE roughly in half. The log-return target came within a few cents of the baselines, but did not beat them.

### Deployment decision

The historical demonstration predicted a close of $253.67 for September 30, 2026, using data through September 29, 2026, when the last close was $329.40. The selected model's 20-day historical replay had an MAE of $78.22, compared with $3.85 for the log-return version.

Production deployment is not recommended, because no model outperformed the simple forecasting baselines. This was a historical demonstration, not a current stock-price forecast.
