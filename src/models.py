"""
models.py - CRISP-DM phase 4: Modeling (network architectures).

All models are keras.Sequential stacks: an Input layer, the core layer(s)
of the architecture, Dropout, and one Dense output unit.

Input shape for every model: (LOOKBACK, n_features)
    e.g. (60, 1) with price only, (60, 2) with Close + Volume.
Output: 1 number - the (scaled) target for the next day.

See docs/03_deep_learning_models.md for how LSTM, GRU and 1D-CNN work.
"""

from typing import Tuple

import keras
from keras import layers

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


def build_model(
    name: str,
    input_shape: Tuple[int, int],
    units: int = config.BASE_HPARAMS["units"],
    kernel_size: int = config.BASE_HPARAMS["kernel_size"],
    dropout: float = config.DROPOUT,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    """
    Build and compile one of the three architectures:

        lstm : LSTM(units)                                   -> Dropout -> Dense(1)
        gru  : GRU(units)                                    -> Dropout -> Dense(1)
        cnn  : Conv1D(units, kernel_size) + GlobalMaxPooling -> Dropout -> Dense(1)

    The layers are created in the same order as in the single notebook, so the
    same seed gives the same initial weights.
    """
    if name == "lstm":
        core = [layers.LSTM(units)]
    elif name == "gru":
        core = [layers.GRU(units)]
    elif name == "cnn":
        core = [layers.Conv1D(units, kernel_size=kernel_size, activation="relu"), layers.GlobalMaxPooling1D()]
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_BUILDERS)}.")
    model = keras.Sequential([layers.Input(shape=input_shape), *core, layers.Dropout(dropout), layers.Dense(1)], name=name)
    return compile_model(model, learning_rate)


def build_lstm(
    input_shape: Tuple[int, int],
    learning_rate: float = config.LEARNING_RATE,
    dropout: float = config.DROPOUT,
    units: int = config.BASE_HPARAMS["units"],
) -> keras.Model:
    """LSTM: one recurrent layer with 64 units, dropout and a single output."""
    return build_model("lstm", input_shape, units=units, dropout=dropout, learning_rate=learning_rate)


def build_gru(
    input_shape: Tuple[int, int],
    learning_rate: float = config.LEARNING_RATE,
    dropout: float = config.DROPOUT,
    units: int = config.BASE_HPARAMS["units"],
) -> keras.Model:
    """GRU: one recurrent layer with 64 units, dropout and a single output."""
    return build_model("gru", input_shape, units=units, dropout=dropout, learning_rate=learning_rate)


def build_cnn1d(
    input_shape: Tuple[int, int],
    learning_rate: float = config.LEARNING_RATE,
    dropout: float = config.DROPOUT,
    units: int = config.BASE_HPARAMS["units"],
    kernel_size: int = config.BASE_HPARAMS["kernel_size"],
) -> keras.Model:
    """1D convolutional network: 64 filters of width 5, global max pooling, dropout, one output."""
    return build_model("cnn", input_shape, units=units, kernel_size=kernel_size, dropout=dropout,
                       learning_rate=learning_rate)


# Registry: lets the rest of the code list the architectures by name.
MODEL_BUILDERS = {
    "lstm": build_lstm,
    "gru": build_gru,
    "cnn": build_cnn1d,
}
