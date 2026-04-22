"""Custom and built-in Keras callbacks for training."""

import os
import tensorflow as tf
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_callbacks(model_type: str = "lstm") -> list:
    """Return list of Keras callbacks for training."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if model_type == "lstm":
        ckpt_path = config.LSTM_MODEL_PATH
    else:
        ckpt_path = config.TRANSFORMER_MODEL_PATH

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            verbose=1,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.MODEL_DIR, "logs"),
            histogram_freq=0,
        ),
    ]
    return callbacks
