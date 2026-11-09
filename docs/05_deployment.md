# 05 · Deployment and operations

Deployment is the last CRISP-DM phase: putting the model where people or systems can use it on **new data**, and keeping it healthy over time.

---

## 1. The artifact folder

Each training run writes a self-contained folder in `artifacts/`:

```
artifacts/lstm_close_volume/
├── model.keras               # network architecture + trained weights
├── feature_scaler.joblib     # StandardScaler fitted on TRAINING features
├── target_scaler.joblib      # StandardScaler fitted on TRAINING targets
├── metadata.json             # features, lookback, target mode, dates, settings
└── history.json              # per-epoch loss (learning curves)
```

**Note**: a model alone cannot make correct predictions. It expects inputs that were:
1. built with the **same features** (`feature_columns` in the metadata),
2. using the **same lookback** (`lookback`),
3. scaled with the **same fitted scalers**,

The model's output must be converted back using the same target mode and target scaler. Saving all of these together prevents a production bug where inference preprocessing differs slightly from training.

> **Format note:** the notebook saves models in the `.keras` format, which needs TensorFlow 2.12 or newer.

---

## 2. The inference pipeline (`predict_next_day` in the notebook)

```mermaid
flowchart LR
    new[(New Close and Volume data)] --> load[load_stock_data<br/>validate]
    load --> feat[add_features_and_target<br/>same features]
    feat --> win[last LOOKBACK rows]
    win --> scale[feature_scaler.transform]
    scale --> model[model.predict]
    model --> inv[target_scaler.inverse_transform]
    inv --> price["price in dollars<br/>to_price"]
```

`predict_next_day()` is the **single source of truth**. The notebook calls it, so what was tested is what runs.

---

## 3. How to use the model

### Python
The notebook's deployment section reloads a saved run and predicts the next day:

```python
reloaded = load_run(selected_run)
print(predict_next_day(reloaded.model, reloaded_data, apple_df))
```

To use this outside the notebook, copy `load_stock_data`, `add_features_and_target`, `load_run`, `to_price` and `predict_next_day` into a `predict.py` file, and add a small `load_data_and_model` function that returns the model and the scaler and settings object the notebook calls `reloaded_data`.

---

## 4. REST API

A web API lets other applications request predictions over HTTP. A minimal FastAPI version (not included in the repo; install `fastapi` and `uvicorn` first):

```python
# api.py: predict.py holds the functions copied from the notebook, as described in section 3
from fastapi import FastAPI
from predict import load_data_and_model, load_stock_data, predict_next_day

app = FastAPI(title="Next-day close API")
MODEL, DATA = load_data_and_model("gru_close_volume")   # load once at startup

@app.get("/health")
def health():
    return {"status": "ok", "model": "gru_close_volume"}

@app.get("/predict")
def predict():
    return predict_next_day(MODEL, DATA, load_stock_data())
```
Run with `uvicorn api:app --port 8000`, then open `http://localhost:8000/docs`.

### Containerizing with Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn
COPY predict.py api.py ./
COPY artifacts/gru_close_volume/ artifacts/gru_close_volume/
COPY data/ data/
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```
A container bundles the exact library versions, so the model behaves the same on any machine or cloud.

---

## 5. Monitoring: models degrade silently

Markets change, and a model trained on 2010-2021 will gradually drift away from reality. Monitor three things:

| What | How | Alert when |
|---|---|---|
| **Data quality** | The loader's validation (nulls, swapped columns, non-positive prices, duplicate dates) | Any validation error, or a missing trading day |
| **Data drift** | Compare recent feature distributions (return volatility, log volume) with the training distribution: mean/std shift, Population Stability Index, or the Kolmogorov-Smirnov test | The feature distribution moves several std away from training |
| **Performance** | Log every prediction; the next day, compute the error. Track a rolling 20-day MAE for **model and naive baseline** | The model's rolling MAE exceeds the naive MAE for a sustained period |

Example monitoring log:

| date | predicted | actual | abs error | naive abs error |
|---|---|---|---|---|
| 2026-09-09 | 316.59 | … | … | … |

---

## 6. Retraining strategy

| Strategy | Description | Pros / cons |
|---|---|---|
| **Scheduled** | Retrain every month or quarter with all data up to that date | Simple and predictable; may retrain when not needed |
| **Triggered** | Retrain when monitoring detects drift or degraded error | Efficient; needs reliable monitoring |
| **Rolling window** | Train only on the last N years | Adapts to new regimes; less data |

**Retraining workflow:** load new data → run the same training pipeline → evaluate on the most recent held-out period → compare with the **current production model** and the baselines → promote only if better → keep the old artifact folder for rollback.

---

## 7. Versioning and reproducibility

- **Code:** git (tag the commit used for each deployed model).
- **Data:** keep a snapshot or hash of the training data.
- **Models:** one artifact folder per run with `metadata.json` (`trained_at`, TF version, data range).
- **Environment:** `requirements.txt` with pinned versions, or a Docker image.

---

## 8. Responsible use

- Show the **uncertainty and limitations**. A single number looks more certain than it is.
- Show the **baseline comparison** next to the prediction.
- Add a clear **"not financial advice"** disclaimer.
- Record which model version produced each prediction (audit trail).

---

## 9. Deployment checklist

- [ ] Artifact folder contains the model, both scalers and the metadata
- [ ] Inference uses the same `predict_next_day()` function that was tested
- [ ] Input data is validated before predicting
- [ ] Predictions are logged with the model version
- [ ] Monitoring of error vs the naive baseline is in place
- [ ] Retraining and rollback procedure is documented
- [ ] Disclaimer is visible to users
