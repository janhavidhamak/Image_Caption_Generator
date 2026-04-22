"""Training utility functions."""

import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def train_val_split(image_names: list, split: float = config.TRAIN_SPLIT, seed: int = config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(image_names).tolist()
    cut = int(len(shuffled) * split)
    return shuffled[:cut], shuffled[cut:]


def steps_per_epoch(image_names: list, sequences: dict, batch_size: int = config.BATCH_SIZE) -> int:
    total = sum(
        sum(len(seq) - 1 for seq in sequences[n])
        for n in image_names
        if n in sequences
    )
    return max(1, total // batch_size)


def save_history(history, path: str = config.HISTORY_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(history.history, f)
    logger.info("Training history saved to %s", path)


def plot_history(history_dict: dict, save_dir: str = config.MODEL_DIR):
    os.makedirs(save_dir, exist_ok=True)
    for metric in ("loss", "accuracy"):
        val_key = f"val_{metric}"
        if metric not in history_dict:
            continue
        plt.figure()
        plt.plot(history_dict[metric], label=f"train {metric}")
        if val_key in history_dict:
            plt.plot(history_dict[val_key], label=f"val {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(f"Training {metric.capitalize()}")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(save_dir, f"{metric}_curve.png")
        plt.savefig(path)
        plt.close()
        logger.info("Saved %s curve to %s", metric, path)
