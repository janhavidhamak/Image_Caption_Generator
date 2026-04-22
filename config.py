"""Centralized configuration for the Image Caption Generator project."""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_files")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CAPTIONS_FILE = os.path.join(BASE_DIR, "captions.txt")
FEATURES_FILE = os.path.join(DATA_DIR, "features.pkl")
TOKENIZER_FILE = os.path.join(DATA_DIR, "tokenizer.pkl")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
TRANSFORMER_MODEL_PATH = os.path.join(MODEL_DIR, "transformer_model.h5")
HISTORY_FILE = os.path.join(MODEL_DIR, "training_history.pkl")

# ── Dataset ──────────────────────────────────────────────────────────────────
IMAGE_SIZE = (299, 299)          # InceptionV3 input size
MIN_WORD_FREQ = 5                # Minimum word frequency for vocabulary
MAX_CAPTION_LENGTH = 35          # Maximum caption length (tokens)
TRAIN_SPLIT = 0.8                # 80 % training, 20 % validation
RANDOM_SEED = 42

# ── InceptionV3 feature extractor ────────────────────────────────────────────
FEATURE_SIZE = 2048              # Output size of InceptionV3 pooling layer

# ── LSTM model ───────────────────────────────────────────────────────────────
EMBEDDING_DIM = 256
LSTM_UNITS = 512
DROPOUT_RATE = 0.5

# ── Transformer model ────────────────────────────────────────────────────────
TRANSFORMER_D_MODEL = 256
TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_DFF = 512
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DROPOUT = 0.1

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# ── Inference ────────────────────────────────────────────────────────────────
BEAM_WIDTH = 3                   # 1 = greedy search
MAX_INFERENCE_LENGTH = MAX_CAPTION_LENGTH

# ── Google Drive dataset link (images.zip) ───────────────────────────────────
GDRIVE_LINK = "https://drive.google.com/drive/u/0/folders/1rfrfcD3WWR28CSqRRQEHcJZRs4vgNO2D"
