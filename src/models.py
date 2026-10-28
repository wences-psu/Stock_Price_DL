"""
models.py - CRISP-DM phase 4: Modeling (network architectures).

All models use the Keras *functional API*: we create an Input tensor, pass
it through layers one by one, and wrap input + output in a keras.Model.
This style makes the data flow explicit and scales to complex architectures
(multiple inputs, skip connections) later.

Input shape for every model: (LOOKBACK, n_features)
    e.g. (60, 1) with price only, (60, 3) with Close + Volume.
Output: 1 number - the (scaled) target for the next day.

See docs/03_deep_learning_models.md for how LSTM, GRU and 1D-CNN work.
"""

from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers

import config


def compile_model(model: keras.Model, learning_rate: float = config.LEARNING_RATE) -> keras.Model:
    """
    Attach the optimizer, loss and metrics to a model.

    Adam  : adaptive optimizer; works well with default settings.
    Huber : behaves like MSE for small errors and like MAE for large ones,
            so a few extreme market days (crashes, earnings jumps) do not
            dominate training the way they would with plain MSE.
    MAE   : mean absolute error, an easy-to-read metric that is tracked but not optimized.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_lstm(
    input_shape: Tuple[int, int],
    learning_rate: float = config.LEARNING_RATE,
    dropout: float = config.DROPOUT,
) -> keras.Model:
    """
    LSTM
    """
    model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(64),
            layers.Dropout(dropout),
            layers.Dense(1)
        ],
            name="lstm")
    return compile_model(model, learning_rate)


def build_gru(
    input_shape: Tuple[int, int],
    learning_rate: float = config.LEARNING_RATE,
    dropout: float = config.DROPOUT,
) -> keras.Model:
    """
    GRU
    """
    model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.GRU(64),
            layers.Dropout(dropout),
            layers.Dense(1)
        ],
            name="gru")
    return compile_model(model, learning_rate)


def build_cnn1d(
    input_shape: Tuple[int, int],
    learning_rate: float = config.LEARNING_RATE,
    dropout: float = config.DROPOUT,
) -> keras.Model:
    """
    1D Convolutional network
    """
    model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(64, kernel_size=5, activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(dropout),
            layers.Dense(1)
        ],
            name="cnn")
    return compile_model(model, learning_rate)


# Registry: lets the rest of the code choose a model by name ("lstm", "gru", "cnn").
MODEL_BUILDERS = {
    "lstm": build_lstm,
    "gru": build_gru,
    "cnn": build_cnn1d,
}


def build_model(
    name: str,
    input_shape: Tuple[int, int],
    learning_rate: float = config.LEARNING_RATE,
    dropout: float = config.DROPOUT,
) -> keras.Model:
    """Build and compile a model by name."""
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_BUILDERS)}.")
    return MODEL_BUILDERS[name](input_shape, learning_rate=learning_rate, dropout=dropout)
