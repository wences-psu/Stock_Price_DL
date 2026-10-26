# 03 · Deep learning models explained

This document explains the building blocks used in `src/models.py` and `src/train.py`.

---

## 1. Neural network

A **dense (fully connected) layer** computes

$$\mathbf{y} = \phi(W\mathbf{x} + \mathbf{b})$$

where $W$ (weights) and $\mathbf{b}$ (biases) are learned, and $\phi$ is an **activation function**. The final layer in this project uses no activation (linear), because a return can be any real number. Hidden layers use `relu`, $\max(0, x)$, which lets the network model non-linear relationships.

### How training works
1. **Forward pass:** feed a batch of windows through the network to get predictions.
2. **Loss:** measure how wrong the predictions are (Huber loss here).
3. **Backpropagation:** compute the gradient of the loss with respect to every weight (the chain rule).
4. **Optimizer step:** move each weight a little in the direction that reduces the loss (Adam here).
5. Repeat for every **batch**. One pass over all training samples is an **epoch**.

| Term | Value in this project | Meaning |
|---|---|---|
| Batch size | 32 | Windows per weight update |
| Epochs | up to 100 | Full passes over the training data (early stopping ends sooner) |
| Learning rate | 0.001 | Size of each optimizer step |

---

## 2. Recurrent neural networks (RNN)

A **dense** network sees the 60-day window as a flat list and has no idea that day 59 comes after day 58. An **RNN** processes the sequence **one step at a time** and carries a **hidden state** $h_t$, a memory of what it has seen so far:

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

```
x_1 → [cell] → h_1 → [cell] → h_2 → ... → [cell] → h_60 → prediction
                 ↑ x_2            ↑ x_3            ↑ x_60
```

The **same weights** are reused at every time step, so the model size does not grow with the window length.

**Problem: vanishing gradients.** During backpropagation, the gradient is multiplied by the recurrent weights at every step. Over 60 steps it tends to shrink toward zero (or explode), so a simple RNN struggles to learn long-range dependencies. LSTM and GRU fix this.

---

## 3. LSTM (Long Short-Term Memory) – the main model

An LSTM cell adds a **cell state** $C_t$, a "conveyor belt" of memory, and three **gates**. Each gate is a sigmoid (0 to 1) that decides how much information flows through:

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) && \text{forget gate: what to erase from memory} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) && \text{input gate: what new information to store} \\
\tilde C_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) && \text{candidate memory} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde C_t && \text{update the cell state} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) && \text{output gate: what to reveal} \\
h_t &= o_t \odot \tanh(C_t) && \text{new hidden state}
\end{aligned}
$$

The cell state is updated by **addition** rather than repeated multiplication, so gradients can flow across many time steps. That lets the LSTM remember information from much earlier in the window.

### The architecture in `build_lstm`

```
Input (60, n_features)
  └─ LSTM(64, return_sequences=True)  → (60, 64)   one hidden vector per day
      └─ Dropout(0.2)
            └─ Dense(1)      → (1,)       the predicted (scaled) next-day return
```

### LSTM parameters

An LSTM has 4 weight sets (3 gates + candidate), each over `[h_{t-1}, x_t]`:

$$\text{params} = 4 \times \big(u \times (u + d) + u\big) \quad (u = \text{units}, d = \text{input dim})$$

| Layer (with 3 features) | Computation | Parameters |
|---|---|---|
| LSTM(64), d=3 | 4 × (64 × 67 + 64) | 17,408 |
| Dense(1) from 64 | 64 + 1 | 65 |
| **Total** | | **17,473** |

---

## 4. GRU (Gated Recurrent Unit)

A GRU is a simplified LSTM with **two gates** and no separate cell state:

$$
\begin{aligned}
z_t &= \sigma(W_z [h_{t-1}, x_t]) && \text{update gate: how much to refresh the memory} \\
r_t &= \sigma(W_r [h_{t-1}, x_t]) && \text{reset gate: how much of the past to use for the candidate} \\
\tilde h_t &= \tanh(W [r_t \odot h_{t-1}, x_t]) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t
\end{aligned}
$$

- Fewer parameters than an LSTM with the same number of units.
- It trains faster and often performs similarly, especially on small datasets.
- Keras's default GRU (`reset_after=True`) uses two bias vectors per gate: params = 3 × (u × (u + d) + 2u).

---

## 5. 1D Convolutional Neural Network (1D-CNN)

A **1D convolution** slides a small filter (kernel) along the time axis. With `kernel_size=5`, each filter looks at 5 consecutive days and computes a weighted sum. It acts as a learned *pattern detector*, for example "three down days then a rebound".

```
days:    d1  d2  d3  d4  d5  d6  d7 ...
filter: [w1  w2  w3  w4  w5]
             [w1  w2  w3  w4  w5]
                 [w1  w2  w3  w4  w5]   → one output per position
```


| | LSTM | GRU | 1D-CNN |
|---|---|---|---|
| Processes steps | sequentially | sequentially | in parallel |
| Long-range memory | strong | strong | limited by kernel size × depth |
| Parameters (size) | large | medium | Small |
| Training speed | slowest | medium | fastest |

---

## 6. Regularization: Dropout

**Dropout(0.2)** randomly sets 20% of a layer's outputs to zero on each training step. The network cannot rely on any single unit, which reduces **overfitting** (memorizing noise in the training data). Dropout is automatically disabled when predicting.

Stock returns are mostly noise, so overfitting is the main danger. Regularization matters more than extra capacity here.

---

## 7. Loss function: Huber

$$
L_\delta(e) =
\begin{cases}
\frac{1}{2}e^2 & \text{if } |e| \le \delta \\
\delta\left(|e| - \frac{1}{2}\delta\right) & \text{otherwise}
\end{cases}
$$

- For small errors it behaves like **MSE** (smooth, strongly penalizes moderate errors).
- For large errors it behaves like **MAE** (grows linearly), so a crash day does not dominate the gradients.
- Targets are standardized, so δ = 1 means errors beyond about one standard deviation of daily returns are treated linearly.

| Loss | Sensitive to outliers | Optimal prediction under noise |
|---|---|---|
| MSE | very | conditional mean |
| MAE | little | conditional median |
| Huber | moderate | between the two |

---

## 8. Optimizer: Adam

**Adam** (Adaptive Moment Estimation) keeps, for every weight, a running average of the gradient (*momentum*) and of the squared gradient (*scale*). Weights with consistently small gradients get relatively larger steps, and noisy ones get smaller steps. It works well with the default learning rate of 0.001 and is the standard first choice.

---

## 9. Callbacks used in training

| Callback | Setting | What it does |
|---|---|---|
| `EarlyStopping` | `patience=10`, `restore_best_weights=True` | Stops when validation loss has not improved for 10 epochs and rolls back to the best epoch |
| `ReduceLROnPlateau` | `factor=0.5`, `patience=5` | Halves the learning rate when validation loss stalls, for finer steps near a minimum |
| `ModelCheckpoint` | `save_best_only=True` | Writes the best model to `model.keras` during training |

### Reading learning curves

```
loss                                   loss
 │╲                                     │╲
 │ ╲___ validation                      │ ╲    ___/ validation  ← rising = overfitting
 │  ╲____ training                      │  ╲__/
 │                                      │     ╲_____ training
 └──────── epochs                       └──────────── epochs
   good fit                               overfitting
```

With noisy financial targets, if the best validation epoch occurs early it means the network quickly learns to "predict roughly the average return" and has little else to learn.

---

## 10. Shapes cheat sheet

| Object | Shape (price only) | Shape (with volume) |
|---|---|---|
| `X_train` | (2876, 60, 1) | (2876, 60, 3) |
| `y_train` | (2876,) | (2876,) |
| Model input | (None, 60, 1) | (None, 60, 3) |
| Model output | (None, 1) | (None, 1) |

`None` is the batch dimension, which can be any size.

---

## 11. Reproducibility

`set_seeds()` fixes the Python, NumPy and TensorFlow random seeds, so weight initialization, dropout masks and shuffling are repeatable. Results can still differ slightly between CPU and GPU or between library versions. With a weak signal, those small differences can change the ranking of models, which is why notebook 04 uses a statistical test and we recommend averaging several seeds.
