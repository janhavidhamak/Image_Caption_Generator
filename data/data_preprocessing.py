"""Data loading, cleaning, and preprocessing utilities."""

import os
import re
import pickle
import logging
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

def build_feature_extractor():
    """Return InceptionV3 model with the classification head removed."""
    base = InceptionV3(weights="imagenet", include_top=True)
    model = tf.keras.Model(inputs=base.input, outputs=base.layers[-2].output)
    return model


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
        except (OSError, ValueError, Exception) as exc:  # noqa: BLE001
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
