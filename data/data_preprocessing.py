"""Data loading, cleaning, and preprocessing utilities."""

import os
import re
import pickle
import logging
import time
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# project imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Caption helpers ──────────────────────────────────────────────────────────

def clean_caption(text: str) -> str:
    """Lowercase, strip punctuation/digits, collapse spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_captions(captions_file: str = config.CAPTIONS_FILE) -> dict:
    """Return {image_name: [caption, ...]} with start/end tokens."""
    df = pd.read_csv(captions_file)
    # support both 'image'/'caption' and positional
    if "image" not in df.columns:
        df.columns = ["image", "caption"]

    mapping: dict = {}
    for _, row in df.iterrows():
        img_name = str(row["image"]).strip()
        caption = clean_caption(str(row["caption"]))
        caption = f"startseq {caption} endseq"
        mapping.setdefault(img_name, []).append(caption)
    logger.info("Loaded captions for %d images.", len(mapping))
    return mapping


# ── Vocabulary ───────────────────────────────────────────────────────────────

def build_vocabulary(captions_mapping: dict, min_freq: int = config.MIN_WORD_FREQ):
    """Build word↔index mappings from captions."""
    counter: Counter = Counter()
    for caps in captions_mapping.values():
        for cap in caps:
            counter.update(cap.split())

    vocab = [w for w, c in counter.items() if c >= min_freq]
    vocab = sorted(vocab)

    word_to_idx = {"<pad>": 0, "<unk>": 1}
    for w in vocab:
        word_to_idx[w] = len(word_to_idx)
    idx_to_word = {v: k for k, v in word_to_idx.items()}

    logger.info("Vocabulary size: %d tokens.", len(word_to_idx))
    return word_to_idx, idx_to_word


def get_max_caption_length(captions_mapping: dict) -> int:
    return max(len(cap.split()) for caps in captions_mapping.values() for cap in caps)


# ── Image feature extraction ─────────────────────────────────────────────────

# Keras cache directory where pre-trained weights are stored
_KERAS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".keras", "models")

# InceptionV3 weights file details
_INCEPTION_WEIGHTS_FILENAME = "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"
_INCEPTION_WEIGHTS_URL = (
    "https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/"
    "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"
)


def download_inception_weights(
    dest_dir: str = _KERAS_CACHE_DIR,
    url: str = _INCEPTION_WEIGHTS_URL,
) -> str:
    """Download InceptionV3 ImageNet weights to *dest_dir* and return the local path.

    This is a fallback helper for environments where the automatic Keras
    download fails (e.g. corporate firewalls, flaky connections).  The file is
    placed in the standard Keras model-cache directory so that subsequent calls
    to ``InceptionV3(weights='imagenet')`` will find it automatically.

    Example usage::

        from data.data_preprocessing import download_inception_weights
        download_inception_weights()

    Parameters
    ----------
    dest_dir:
        Directory where the weights file will be saved.
        Defaults to ``~/.keras/models/``.
    url:
        Direct download URL for the weights file.

    Returns
    -------
    str
        Absolute path to the downloaded weights file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, _INCEPTION_WEIGHTS_FILENAME)

    if os.path.exists(dest_path):
        logger.info("InceptionV3 weights already present at %s", dest_path)
        return dest_path

    logger.info("Downloading InceptionV3 weights from %s …", url)
    logger.info("Destination: %s", dest_path)

    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info("Download complete: %s", dest_path)
    except Exception as exc:
        # Remove any partial download so it is not mistaken for a valid file.
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise RuntimeError(
            f"Failed to download InceptionV3 weights from {url}: {exc}\n"
            "Please download the file manually and place it at:\n"
            f"  {dest_path}"
        ) from exc

    return dest_path


def build_feature_extractor(max_retries: int = 3, retry_delay: float = 2.0):
    """Return InceptionV3 model with the classification head removed.

    Automatically retries the weight download on transient network errors,
    using exponential backoff between attempts.

    Parameters
    ----------
    max_retries:
        Maximum number of download attempts before raising the final error.
    retry_delay:
        Base delay in seconds between retries (doubles on each failure).

    Raises
    ------
    RuntimeError
        When all download attempts are exhausted.  The error message includes
        instructions for manually downloading the weights via
        :func:`download_inception_weights`.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    last_exc: Exception = RuntimeError("Unknown error")
    delay = retry_delay

    for attempt in range(1, max_retries + 1):
        try:
            base = InceptionV3(weights="imagenet", include_top=True)
            model = tf.keras.Model(inputs=base.input, outputs=base.layers[-2].output)
            return model
        except (OSError, ConnectionResetError, TimeoutError) as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "Attempt %d/%d failed (%s). Retrying in %.0fs…",
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                logger.error(
                    "All %d download attempts failed. Last error: %s",
                    max_retries,
                    exc,
                )

    weights_path = os.path.join(_KERAS_CACHE_DIR, _INCEPTION_WEIGHTS_FILENAME)
    raise RuntimeError(
        f"Could not download InceptionV3 pre-trained weights after {max_retries} "
        f"attempt(s).\nLast error: {last_exc}\n\n"
        "To fix this, download the weights manually:\n"
        "  1. Run the following Python snippet:\n"
        "       from data.data_preprocessing import download_inception_weights\n"
        "       download_inception_weights()\n"
        "  OR manually download the file from:\n"
        f"       {_INCEPTION_WEIGHTS_URL}\n"
        "  and place it at:\n"
        f"       {weights_path}\n"
        "  2. Then re-run your training command."
    ) from last_exc


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess a single image for InceptionV3."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(config.IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def extract_features(images_dir: str, feature_extractor, save_path: str = config.FEATURES_FILE) -> dict:
    """Extract and cache InceptionV3 features for every image in *images_dir*."""
    if os.path.exists(save_path):
        logger.info("Loading cached features from %s", save_path)
        with open(save_path, "rb") as f:
            return pickle.load(f)

    features = {}
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    logger.info("Extracting features for %d images…", len(image_files))

    for i, fname in enumerate(image_files):
        path = os.path.join(images_dir, fname)
        try:
            arr = preprocess_image(path)
            feat = feature_extractor.predict(arr, verbose=0)
            features[fname] = feat.flatten()
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("Skipping %s: %s", fname, exc)
        if (i + 1) % 500 == 0:
            logger.info("  … %d / %d done", i + 1, len(image_files))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(features, f)
    logger.info("Features saved to %s", save_path)
    return features


# ── Sequence generation ──────────────────────────────────────────────────────

def captions_to_sequences(captions_mapping: dict, word_to_idx: dict) -> dict:
    """Convert caption strings to integer sequences (no padding)."""
    sequences = {}
    unk = word_to_idx["<unk>"]
    for img, caps in captions_mapping.items():
        sequences[img] = [[word_to_idx.get(w, unk) for w in cap.split()] for cap in caps]
    return sequences


def create_data_generator(image_names, features, sequences, word_to_idx, max_len, batch_size):
    """Keras-compatible generator yielding ([img_feat, partial_seq], next_word)."""
    vocab_size = len(word_to_idx)
    X1, X2, y = [], [], []
    n = 0
    while True:
        for img_name in image_names:
            if img_name not in features or img_name not in sequences:
                continue
            feat = features[img_name]
            for seq in sequences[img_name]:
                for i in range(1, len(seq)):
                    in_seq = pad_sequences([seq[:i]], maxlen=max_len, padding="post")[0]
                    out_word = np.zeros(vocab_size, dtype=np.float32)
                    if seq[i] < vocab_size:
                        out_word[seq[i]] = 1.0
                    X1.append(feat)
                    X2.append(in_seq)
                    y.append(out_word)
                    n += 1
                    if n == batch_size:
                        yield [np.array(X1), np.array(X2)], np.array(y)
                        X1, X2, y, n = [], [], [], 0


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_vocab(word_to_idx, idx_to_word, path: str = config.VOCAB_FILE, max_len: int = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"word_to_idx": word_to_idx, "idx_to_word": idx_to_word}
    if max_len is not None:
        payload["max_len"] = max_len
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Vocabulary saved to %s", path)


def load_vocab(path: str = config.VOCAB_FILE):
    with open(path, "rb") as f:
        d = pickle.load(f)
    max_len = d.get("max_len", None)
    return d["word_to_idx"], d["idx_to_word"], max_len
