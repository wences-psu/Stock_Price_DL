# Next day stock price prediction with deep learning (Keras / TensorFlow)

A project that predicts the **next trading day's closing price** of Apple (AAPL) with LSTM, GRU and 1D-CNN networks. It includes a switch to add **trading volume** as an input, to measure whether volume improves the prediction.

The project follows the **CRISP-DM** framework from start to finish: business understanding → data understanding → data preparation → modeling → evaluation → deployment. Every step is explained in commented code, notebooks and Markdown guides.

---

## Project structure

```
Stock_Price/
├── README.md                       ← you are here
├── requirements.txt                ← pinned library versions
├── config.py                       ← ALL settings (dates, lookback, volume on/off, epochs...)
├── data/
│   ├── SnP_daily_update.csv        ← daily OHLCV S&P 500 data
├── src/                            ← reusable helper code
├── notebooks/                      ← run these in order
│   ├── 01_business_and_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_deployment_demo.ipynb
├── app/
│   └── streamlit_app.py            ← web app for predictions
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

---