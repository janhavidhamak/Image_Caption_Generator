"""Main training script for the image captioning model."""

import os
import sys
import argparse
import logging
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.data_preprocessing import (
    load_captions,
    build_vocabulary,
    get_max_caption_length,
    extract_features,
    build_feature_extractor,
    captions_to_sequences,
    create_data_generator,
    save_vocab,
)
from models.model_lstm import build_lstm_model
from models.model_transformer import build_transformer_model
from training.callbacks import get_callbacks
from training.utils import train_val_split, steps_per_epoch, save_history, plot_history

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train image captioning model")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--images_dir", default=config.IMAGES_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load & preprocess captions
    logger.info("Loading captions…")
    captions_mapping = load_captions()

    # 2. Build vocabulary
    word_to_idx, idx_to_word = build_vocabulary(captions_mapping)
    vocab_size = len(word_to_idx)
    max_len = min(get_max_caption_length(captions_mapping), config.MAX_CAPTION_LENGTH)
    logger.info("Vocab size: %d  |  Max caption length: %d", vocab_size, max_len)

    save_vocab(word_to_idx, idx_to_word, max_len=max_len)

    # 3. Extract image features
    logger.info("Extracting image features…")
    feature_extractor = build_feature_extractor()
    features = extract_features(args.images_dir, feature_extractor)

    # 4. Convert captions to sequences
    sequences = captions_to_sequences(captions_mapping, word_to_idx)

    # Keep only images that have both features and captions
    common = [n for n in features if n in sequences]
    logger.info("Images with both features and captions: %d", len(common))

    # 5. Train / val split
    train_imgs, val_imgs = train_val_split(common)
    logger.info("Train: %d  |  Val: %d", len(train_imgs), len(val_imgs))

    # 6. Build model
    if args.model == "lstm":
        model = build_lstm_model(vocab_size=vocab_size, max_len=max_len)
    else:
        model = build_transformer_model(vocab_size=vocab_size, max_len=max_len)
    model.summary()

    # 7. Data generators
    train_gen = create_data_generator(train_imgs, features, sequences, word_to_idx, max_len, args.batch_size)
    val_gen = create_data_generator(val_imgs, features, sequences, word_to_idx, max_len, args.batch_size)

    train_steps = steps_per_epoch(train_imgs, sequences, args.batch_size)
    val_steps = steps_per_epoch(val_imgs, sequences, args.batch_size)

    # 8. Train
    callbacks = get_callbacks(model_type=args.model)
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # 9. Save history & plots
    save_history(history)
    plot_history(history.history)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
