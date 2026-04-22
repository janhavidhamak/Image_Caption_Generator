"""Generate captions for new images using a trained model."""

import os
import sys
import logging
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.data_preprocessing import preprocess_image, load_vocab, build_feature_extractor
from inference.utils import postprocess_caption

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Load a trained model and vocabulary, then generate captions."""

    def __init__(self, model_type: str = "lstm"):
        self.model_type = model_type
        self.model = None
        self.feature_extractor = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.max_len = config.MAX_INFERENCE_LENGTH

    def load(self):
        """Load model, feature extractor, and vocabulary from disk."""
        model_path = config.LSTM_MODEL_PATH if self.model_type == "lstm" else config.TRANSFORMER_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}. Please train first.")
        if not os.path.exists(config.VOCAB_FILE):
            raise FileNotFoundError(f"Vocabulary file not found at {config.VOCAB_FILE}. Please train first.")

        logger.info("Loading model from %s", model_path)
        self.model = tf.keras.models.load_model(model_path)
        self.word_to_idx, self.idx_to_word, saved_max_len = load_vocab()
        if saved_max_len is not None:
            self.max_len = saved_max_len
        self.feature_extractor = build_feature_extractor()
        logger.info("Model loaded successfully.")

    def _extract_feature(self, image_path: str) -> np.ndarray:
        arr = preprocess_image(image_path)
        feat = self.feature_extractor.predict(arr, verbose=0)
        return feat.flatten()

    def _extract_feature_from_array(self, img_array: np.ndarray) -> np.ndarray:
        """Extract features from a preprocessed image array (already (1, H, W, 3))."""
        feat = self.feature_extractor.predict(img_array, verbose=0)
        return feat.flatten()

    def generate_greedy(self, image_path: str) -> str:
        """Generate caption using greedy search."""
        feat = self._extract_feature(image_path)
        return self._decode_greedy(feat)

    def generate_greedy_from_array(self, img_array: np.ndarray) -> str:
        feat = self._extract_feature_from_array(img_array)
        return self._decode_greedy(feat)

    def _decode_greedy(self, feat: np.ndarray) -> str:
        start_idx = self.word_to_idx.get("startseq", 1)
        end_idx = self.word_to_idx.get("endseq", 2)

        seq = [start_idx]
        for _ in range(self.max_len):
            padded = pad_sequences([seq], maxlen=self.max_len, padding="post")
            feat_in = np.expand_dims(feat, 0)
            preds = self.model.predict([feat_in, padded], verbose=0)

            if self.model_type == "transformer":
                # preds shape: (1, seq_len, vocab_size) — take last position
                next_idx = int(np.argmax(preds[0, len(seq) - 1]))
            else:
                next_idx = int(np.argmax(preds[0]))

            if next_idx == end_idx:
                break
            seq.append(next_idx)

        words = [self.idx_to_word.get(i, "") for i in seq[1:]]
        caption = " ".join(w for w in words if w)
        return postprocess_caption(caption)

    def generate_beam(self, image_path: str, beam_width: int = config.BEAM_WIDTH) -> str:
        """Generate caption using beam search."""
        feat = self._extract_feature(image_path)
        return self._decode_beam(feat, beam_width)

    def _decode_beam(self, feat: np.ndarray, beam_width: int) -> str:
        start_idx = self.word_to_idx.get("startseq", 1)
        end_idx = self.word_to_idx.get("endseq", 2)

        # Each beam: (log_prob, sequence)
        beams = [(0.0, [start_idx])]
        completed = []

        for _ in range(self.max_len):
            candidates = []
            for log_prob, seq in beams:
                if seq[-1] == end_idx:
                    completed.append((log_prob, seq))
                    continue
                padded = pad_sequences([seq], maxlen=self.max_len, padding="post")
                feat_in = np.expand_dims(feat, 0)
                preds = self.model.predict([feat_in, padded], verbose=0)

                if self.model_type == "transformer":
                    probs = preds[0, len(seq) - 1]
                else:
                    probs = preds[0]

                top_k = np.argsort(probs)[-beam_width:]
                for idx in top_k:
                    candidates.append((log_prob + float(np.log(probs[idx] + 1e-9)), seq + [int(idx)]))
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

        completed += beams
        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
        words = [self.idx_to_word.get(i, "") for i in best[1][1:] if i != end_idx]
        caption = " ".join(w for w in words if w)
        return postprocess_caption(caption)
