"""Custom Keras layers shared between model architectures."""

import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    """Additive (Bahdanau) attention used by the LSTM decoder."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features: (batch, feature_size)  → expand to (batch, 1, feature_size)
        features_exp = tf.expand_dims(features, 1)
        hidden_exp = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(features_exp) + self.W2(hidden_exp)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context = attention_weights * features_exp
        context = tf.reduce_sum(context, axis=1)
        return context, attention_weights


class PositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding for transformer decoder."""

    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = self._make_encoding(max_len, d_model)

    @staticmethod
    def _make_encoding(max_len: int, d_model: int):
        positions = tf.cast(tf.range(max_len)[:, None], tf.float32)
        dims = tf.cast(tf.range(d_model)[None, :], tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = positions * angle_rates
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        enc = tf.concat([sines, cosines], axis=-1)[None, :, :]  # (1, max_len, d_model)
        return tf.cast(enc, tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]
